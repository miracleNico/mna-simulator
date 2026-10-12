"""Advanced analysis core for harmonic balance and future extensions."""

from __future__ import annotations

import numpy as np
from math import gcd

from .api.contracts import AnalysisMode, AnalysisOptions, MnaProblem, SimulationResult, SpectrumResult, WaveformResult
from .errors import error_handler
from .mna_builder import build_mna_problem
from .solvers import (
    _krylov_method_name,
    _krylov_route,
    hb_max_nnz,
    reconstruct_time_domain,
    resolve_krylov_rank,
    solve_harmonic_balance,
)
from .table_sources import public_metadata
from .utils import parse_value


def find_gcd_frequency(frequencies: list[float], tolerance: float = 1e-5) -> float | None:
    """Return the floating-point GCD used as the HB fundamental frequency."""

    positive = sorted({freq for freq in frequencies if freq > tolerance})
    if not positive:
        return None

    resolution = tolerance
    scaled = [max(1, int(round(freq / resolution))) for freq in positive]
    gcd_units = scaled[0]
    for value in scaled[1:]:
        gcd_units = gcd(gcd_units, value)
    return gcd_units * resolution


def _num_threads(options: AnalysisOptions) -> int:
    """Worker threads for the native core: the count the service granted, else 1."""

    value = options.num_threads
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return 1
    return max(1, int(value))


def describe_krylov_choice_for_dimension(
    dimension: int,
    rank: int | str | None = "auto",
    fallback_restart: int = 80,
    max_iter: int = 2000,
    method: str = "auto",
) -> dict[str, object]:
    """``describe_krylov_choice(np.zeros((dimension, dimension)), ...)`` without the allocation.

    The reference classified a dense all-zero ``(n K)^2`` matrix only to fill
    the HB metadata defaults (quirk 6.1). Its classification is fixed by the
    dimension: ``symmetric`` (``max |A - A^T| = 0`` and the Cholesky of a zero
    matrix fails), ``positive_definite`` for the empty matrix.
    """

    dimension = int(dimension)
    matrix_kind = "positive_definite" if dimension == 0 else "symmetric"
    route = _krylov_route(matrix_kind, method)
    resolved_rank, rank_mode = resolve_krylov_rank(dimension, rank, fallback_restart)
    effective_restart = min(resolved_rank, max(1, dimension)) if route == "general" else None
    iteration_budget = int(max_iter) if route == "general" else resolved_rank
    return {
        "method": _krylov_method_name(route),
        "requested_method": method,
        "matrix_kind": matrix_kind,
        "route": route,
        "matrix_dimension": dimension,
        "resolved_rank": resolved_rank,
        "rank_mode": rank_mode,
        "iteration_budget": max(1, iteration_budget),
        "effective_restart": effective_restart,
    }


def run_advanced_analysis(problem: MnaProblem, options: AnalysisOptions) -> SimulationResult:
    """Run harmonic balance and reserve extension points for future methods."""

    if options.mode != AnalysisMode.HB:
        raise ValueError(f"Unsupported advanced analysis mode: {options.mode}")

    source_frequencies = []
    for component in problem.components:
        if component.type in {"V", "I"} and component.subtype in {"SIN", "COS"} and component.value2:
            source_frequencies.append(parse_value(component.value2))

    source_frequencies = [freq for freq in source_frequencies if freq]
    if source_frequencies:
        f0 = find_gcd_frequency(source_frequencies)
    else:
        error_handler("HB_ERROR: No sinusoidal sources found.")
        raise RuntimeError("unreachable")

    if f0 is None:
        error_handler("HB_ERROR: Could not determine HB base frequency.")
        raise RuntimeError("unreachable")

    # Only None means "unset" (CPP_MIGRATION_PLAN.md 2.4.1): 0 or a negative count is an error.
    harmonics = 8 if options.hb_harmonics is None else int(options.hb_harmonics)
    if harmonics <= 0:
        error_handler(f"HB_ERROR: harmonics must be a positive integer, got {harmonics}.")
    min_harmonics_needed = int(np.ceil(max(source_frequencies) / f0))
    if harmonics < min_harmonics_needed:
        harmonics = min_harmonics_needed + 4

    omega_0 = 2 * np.pi * f0
    num_vars = int(np.asarray(problem.b_dc).shape[0])
    num_threads = _num_threads(options)
    krylov_stats: list[dict[str, object]] | None = [] if options.use_krylov else None
    hb_stats: dict[str, object] = {}
    x_bar = solve_harmonic_balance(
        problem,
        omega_0=omega_0,
        H=harmonics,
        max_iter=options.max_iter,
        tol=options.f_tol,
        use_krylov=options.use_krylov,
        krylov_tol=options.krylov_tol,
        krylov_max_iter=options.krylov_max_iter,
        krylov_restart=options.krylov_restart,
        krylov_rank=options.krylov_rank,
        krylov_method=options.krylov_method,
        krylov_stats=krylov_stats,
        num_threads=num_threads,
        max_nnz=hb_max_nnz(),
        hb_stats=hb_stats,
    )

    labels = problem.metadata.get("labels", [])
    metadata = public_metadata(problem.metadata)
    metadata.update({"base_frequency_hz": f0, "harmonics": harmonics})
    metadata["hb_solver"] = {
        key: hb_stats[key]
        for key in (
            "dimension",
            "samples",
            "newton_iterations",
            "linear_solves",
            "damping_restarts",
            "limited_steps",
            "damped_steps",
            "undamped_steps",
            "factorizations",
            "y_nnz",
            "device_slots",
            "matrix_nnz",
            "stored_nnz",
            "btf_blocks",
            "dense_blocks",
            "klu_blocks",
            "largest_block",
            "factor_nnz",
            "memory_bytes",
            "threads_used",
        )
        if key in hb_stats
    }
    if options.use_mor:
        from .mor.metadata import disabled_metadata

        metadata.update(disabled_metadata(".hb MOR is not implemented in v1.", options.mor_method))
    if options.use_krylov:
        freq_dim = 2 * harmonics + 1
        choice = describe_krylov_choice_for_dimension(
            num_vars * freq_dim,
            rank=options.krylov_rank,
            fallback_restart=options.krylov_restart,
            max_iter=options.krylov_max_iter,
            method=options.krylov_method,
        )
        last = krylov_stats[-1] if krylov_stats else choice
        methods = sorted({str(entry.get("method", "")) for entry in (krylov_stats or [choice]) if entry.get("method")})
        metadata.update(
            {
                "linear_solver": "krylov",
                "krylov_matrix_kind": last.get("matrix_kind", choice["matrix_kind"]),
                "krylov_policy": "manual override allowed; auto: positive-definite=CG, symmetric=MINRES/CR, general=Arnoldi/GMRES",
                "krylov_requested_method": last.get("requested_method", options.krylov_method),
                "krylov_method": last.get("method"),
                "krylov_engine": last.get("engine", "python"),
                "krylov_methods": methods,
                "krylov_matrix_dimension": last.get("matrix_dimension", choice["matrix_dimension"]),
                "matrix_nnz": last.get("matrix_nnz"),
                "matrix_density": last.get("matrix_density"),
                "krylov_operator_storage": last.get("operator_storage"),
                "krylov_rank_mode": last.get("rank_mode", choice["rank_mode"]),
                "krylov_resolved_rank": last.get("resolved_rank", choice["resolved_rank"]),
                "krylov_iteration_budget": last.get("iteration_budget", choice["iteration_budget"]),
                "krylov_iterations": sum(int(entry.get("iterations", 0) or 0) for entry in (krylov_stats or [])),
                "krylov_solve_count": len(krylov_stats or []),
                "krylov_converged": all(bool(entry.get("converged")) for entry in (krylov_stats or [])) if krylov_stats else None,
                "krylov_used_direct_fallback": any(bool(entry.get("used_direct_fallback")) for entry in (krylov_stats or [])),
            }
        )

    if options.hb_time_window is not None:
        t_plot = np.linspace(0, options.hb_time_window, 2000)
        x_t = reconstruct_time_domain(x_bar, num_vars, harmonics, omega_0, t_plot, num_threads=num_threads)
        waveform = WaveformResult(time=t_plot, values=x_t, labels=labels)
        return SimulationResult(
            mode=options.mode,
            status="ok",
            waveform=waveform,
            harmonic_state=x_bar,
            labels=labels,
            metadata=metadata,
        )

    dim_freq = 2 * harmonics + 1
    freqs = np.arange(harmonics + 1) * f0
    mags = np.zeros((num_vars, harmonics + 1))

    for index in range(num_vars):
        coeffs = x_bar[index * dim_freq : (index + 1) * dim_freq].flatten()
        mags[index, 0] = abs(coeffs[0])
        for h in range(1, harmonics + 1):
            mags[index, h] = np.sqrt(coeffs[2 * h - 1] ** 2 + coeffs[2 * h] ** 2)

    spectrum = SpectrumResult(frequencies=freqs, magnitudes=mags, labels=labels)
    return SimulationResult(
        mode=options.mode,
        status="ok",
        spectrum=spectrum,
        harmonic_state=x_bar,
        labels=labels,
        metadata=metadata,
    )


def build_and_run_advanced(circuit, options: AnalysisOptions, gmin: float = 0.0) -> SimulationResult:
    """Construct the MNA problem and execute an advanced analysis."""

    problem = build_mna_problem(circuit, gmin=gmin)
    return run_advanced_analysis(problem, options)
