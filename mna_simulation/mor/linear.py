"""Linear selected-output MOR using rational Krylov projection.

Option handling, metadata and validation live here; the numerics run in the
native core through :mod:`mna_simulation.native.mor` (CPP_MIGRATION_PLAN.md
section 2.7): the rational Krylov basis on sparse ``G``/``C`` (KLU per
shift), the projections ``V^T G V`` / ``V^T C V`` and the reduced AC and
transient loops. The pure-Python implementation is kept only in the frozen
test oracle ``tests/reference/mna_reference``.
"""

from __future__ import annotations

import numpy as np

from ..api.contracts import MnaProblem, SpectrumResult, WaveformResult
from ..errors import error_handler
from ..native import mor as native_mor
from ..solvers import (
    _native_core,
    _problem_circuit,
    _problem_is_nonlinear,
    _time_sources,
    _transient_time_points,
    is_nonlinear,
    solve_operating_point,
    solve_transient,
)
from .metadata import resolve_mor_order, used_metadata
from .selectors import build_output_selector, validate_probe_subset


def _problem_size(problem: MnaProblem) -> int:
    """Number of unknowns, without materializing the dense ``G``."""

    if problem.native is not None:
        return int(problem.native.n)
    if problem.G_sparse is not None:
        return int(problem.G_sparse.shape[0])
    return int(problem.G.shape[0])


def count_input_directions(problem: MnaProblem) -> int:
    """Count independent source directions used by the MOR order heuristic."""

    directions = _source_directions(problem)
    return max(1, directions.shape[1])


def _nonzero_column(vector: np.ndarray) -> np.ndarray | None:
    column = np.asarray(vector).reshape(-1, 1)
    return column if np.linalg.norm(column) > 0 else None


def _source_directions(problem: MnaProblem) -> np.ndarray:
    n = _problem_size(problem)
    columns: list[np.ndarray] = []
    for vector in (problem.b_dc, np.real(problem.b_ac), np.imag(problem.b_ac)):
        column = _nonzero_column(vector)
        if column is not None:
            columns.append(column.astype(float))
    for row, expr in enumerate(problem.b_time_str.flatten()):
        if str(expr) == "0":
            continue
        unit = np.zeros((n, 1), dtype=float)
        unit[row, 0] = 1.0
        columns.append(unit)
    if not columns:
        return np.zeros((n, 0), dtype=float)
    return np.hstack(columns)


def _safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``np.linalg.solve``; on a singular matrix ``lstsq``; then ``lstsq`` with a ``1e-12 * ||A||_inf`` shift.

    Native dense kernel (``native.mor.safe_solve``).
    """

    return native_mor.safe_solve(matrix, rhs)


def _problem_operands(
    problem: MnaProblem, nonlinear: bool | None = None
) -> tuple[object, float, native_mor.MorOperands]:
    """``(circuit, gmin, operands)``: ``G + gmin * stamps`` and ``C`` in the compiled circuit's pattern.

    The circuit is ``problem.native`` (or the temporary circuit the solver facades
    use when it is missing, see ``solvers._problem_circuit``).
    """

    if nonlinear is None:
        nonlinear = _problem_is_nonlinear(problem)
    circuit, gmin = _problem_circuit(problem, nonlinear)
    return circuit, gmin, native_mor.circuit_operands(circuit, gmin=gmin)


def build_linear_krylov_basis(
    G: np.ndarray,
    C: np.ndarray,
    inputs: np.ndarray,
    selector: np.ndarray,
    shifts: list[complex],
    order: int,
    *,
    num_threads: int | str | None = None,
) -> np.ndarray:
    """Build a one-sided congruence basis from input and output-adjoint moments.

    ``order`` is a true Krylov dimension budget. Earlier versions only solved
    each seed once at each expansion point, so a large distributed RLC network
    could collapse to a handful of vectors and miss later ringing modes. This
    block rational Krylov loop keeps applying ``(G+sC)^-1 C`` to the input and
    output-adjoint directions until the requested independent basis size is
    reached or no new directions remain.

    ``G`` and ``C`` are dense arrays or SciPy sparse matrices; they are merged
    into one CSC pattern and every shifted system ``G + s C`` is factored once by
    the native core (KLU, complex KLU for complex shifts). Candidates are
    orthogonalized with classical Gram-Schmidt twice, so the basis stays
    orthonormal (CPP_MIGRATION_PLAN.md 2.4.1; the reference made one pass).
    """

    operands = native_mor.matrix_operands(G, C)
    return native_mor.krylov_basis(
        operands, inputs, selector, shifts, order, num_threads=native_mor.resolve_threads(num_threads)
    )


def _linearized_operands(
    problem: MnaProblem,
    max_iter: int,
    v_tol: float,
    f_tol: float,
    linear_kwargs: dict[str, object],
) -> tuple[native_mor.MorOperands, dict[str, object]]:
    """``(operands, metadata)``: ``G + J(x_op)`` (or ``G``) and ``C`` in the circuit pattern, plus the op metadata."""

    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_mos_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    metadata: dict[str, object] = {}

    if is_nonlinear(solver_f_str) or level1_mos_devices or level1_bjt_devices:
        dc_solution = solve_operating_point(
            problem,
            problem.b_dc,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            init_cond="DEFAULT",
            **linear_kwargs,
        )
        circuit, gmin = _problem_circuit(problem, True)
        x_op = np.ascontiguousarray(np.asarray(dc_solution, dtype=float).reshape(-1))
        g_linearized = _native_core().linearized_g(circuit, x_op, float(gmin), 1)
        metadata.update(
            {
                "op_used": True,
                "operating_point": {
                    "labels": problem.metadata["labels"],
                    "values": np.asarray(dc_solution).flatten().tolist(),
                },
                "linearized_device_count": len(level1_mos_devices) + len(level1_bjt_devices),
            }
        )
        return native_mor.circuit_operands(circuit, g_values=g_linearized), metadata

    _circuit, _gmin, operands = _problem_operands(problem, nonlinear=False)
    metadata.update({"op_used": False, "linearized_device_count": 0})
    return operands, metadata


def _linearized_system(
    problem: MnaProblem,
    max_iter: int,
    v_tol: float,
    f_tol: float,
    linear_kwargs: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Dense ``G + J(x_op)`` at the ``b_dc`` operating point (``G`` for a linear circuit) and the op metadata.

    ``J`` comes from the native device evaluation (``_core.linearized_g``: ``G +
    gmin * stamps + J(x_op)`` in the circuit pattern); the reduced AC path uses
    the sparse values directly (:func:`_linearized_operands`).
    """

    operands, metadata = _linearized_operands(problem, max_iter, v_tol, f_tol, linear_kwargs)
    return operands.dense("G"), metadata


def solve_ac_mor(
    problem: MnaProblem,
    *,
    f_start: float,
    f_stop: float,
    points: int,
    mor_output_nodes: list[str],
    mor_order: int | str | None,
    requested_method: str,
    validate: bool,
    probe_nodes: list[str],
    max_iter: int,
    v_tol: float,
    f_tol: float,
    linear_kwargs: dict[str, object] | None = None,
    num_threads: int | str | None = None,
) -> SpectrumResult:
    """Solve selected AC outputs through a reduced linearized model.

    ``num_threads`` (a positive integer; ``None``/``"auto"`` mean 1) splits the
    reduced frequency loop; results are identical for every count.
    """

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    operands, spectrum_metadata = _linearized_operands(
        problem,
        max_iter=max_iter,
        v_tol=v_tol,
        f_tol=f_tol,
        linear_kwargs=dict(linear_kwargs or {}),
    )
    n = _problem_size(problem)
    threads = native_mor.resolve_threads(num_threads)
    inputs = _source_directions(problem)
    num_inputs = max(1, inputs.shape[1])  # count_input_directions(problem)
    order, order_mode = resolve_mor_order(n, num_inputs, len(output_labels), mor_order)
    f_mid = float(np.sqrt(max(abs(f_start), 1e-30) * max(abs(f_stop), 1e-30)))
    shifts = [0.0, 1j * 2.0 * np.pi * f_mid, 1j * 2.0 * np.pi * float(f_stop)]
    V = native_mor.krylov_basis(operands, inputs, selector, shifts, order, num_threads=threads)

    Gr, Cr = native_mor.project(operands, V)
    Br = V.T @ np.asarray(problem.b_ac).reshape(-1)
    Sr = selector @ V

    frequencies = np.linspace(f_start, f_stop, points)
    responses = native_mor.reduced_ac(Gr, Cr, Br, Sr, frequencies, num_threads=threads)

    spectrum_metadata.update(
        used_metadata(
            method="linear_krylov",
            requested_method=requested_method,
            original_dimension=n,
            reduced_dimension=V.shape[1],
            resolved_order=order,
            order_mode=order_mode,
            output_labels=output_labels,
            validate=validate,
            extra={
                "mor_basis": "output_aware_rational_krylov",
                "mor_num_inputs": num_inputs,
                "mor_basis_expansion_points": ["0", f"j*{2.0 * np.pi * f_mid:.6g}", f"j*{2.0 * np.pi * float(f_stop):.6g}"],
            },
        )
    )
    return SpectrumResult(frequencies=frequencies, magnitudes=responses, labels=output_labels, metadata=spectrum_metadata)


def solve_transient_mor(
    problem: MnaProblem,
    *,
    t_stop: float,
    t_step: float,
    mor_output_nodes: list[str],
    mor_order: int | str | None,
    requested_method: str,
    validate: bool,
    probe_nodes: list[str],
    max_iter: int,
    v_tol: float,
    f_tol: float,
    init_cond: np.ndarray | None = None,
    linear_kwargs: dict[str, object] | None = None,
    num_threads: int | str | None = None,
) -> tuple[WaveformResult, dict[str, object]]:
    """Solve selected transient outputs through a reduced linear model."""

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    n = _problem_size(problem)
    threads = native_mor.resolve_threads(num_threads)
    inputs = _source_directions(problem)
    num_inputs = max(1, inputs.shape[1])  # count_input_directions(problem)
    order, order_mode = resolve_mor_order(n, num_inputs, len(output_labels), mor_order)
    shifts = [0.0, 1.0 / max(float(t_step), np.finfo(float).tiny)]
    circuit, _gmin, operands = _problem_operands(problem)
    V = native_mor.krylov_basis(operands, inputs, selector, shifts, order, num_threads=threads)
    Gr, Cr = native_mor.project(operands, V)
    Sr = selector @ V
    source_rows, source_programs = _time_sources(problem.b_time_str)

    if init_cond is None:
        # b_dc + b_time(0): every time program added to its row (compile_time_source_func's values).
        b_initial = _native_core().time_source_vector(
            circuit,
            np.ascontiguousarray(np.asarray(problem.b_dc, dtype=float).reshape(-1)),
            source_rows,
            source_programs,
            0.0,
        )
        x0 = solve_operating_point(
            problem,
            b_initial,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            init_cond="DEFAULT",
            **dict(linear_kwargs or {}),
        )
    else:
        x0 = np.asarray(init_cond, dtype=float).reshape(-1, 1)
        if x0.shape[0] != n:
            error_handler(f"NETLIST_FATAL: init_condition has {x0.shape[0]} values, expected {n} (one per unknown).")

    t_points = _transient_time_points(t_stop, t_step)
    values, reduced_failed = native_mor.reduced_transient(
        Gr, Cr, V, Sr, x0, t_points, problem.b_dc, source_rows, source_programs
    )

    metadata = used_metadata(
        method="linear_krylov",
        requested_method=requested_method,
        original_dimension=n,
        reduced_dimension=V.shape[1],
        resolved_order=order,
        order_mode=order_mode,
        output_labels=output_labels,
        validate=validate,
        extra={
            "mor_basis": "output_aware_rational_krylov",
            "mor_num_inputs": num_inputs,
            "mor_basis_expansion_points": ["0", f"{1.0 / max(float(t_step), np.finfo(float).tiny):.6g}"],
        },
    )
    if validate and problem.G_sparse is not None and problem.C_sparse is not None and n >= 64:
        reference_stats: list[dict[str, object]] = []
        reference = solve_transient(
            problem,
            t_stop=t_stop,
            t_step=t_step,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            init_cond=init_cond,
            use_krylov=True,
            krylov_tol=1e-9,
            krylov_max_iter=2000,
            krylov_restart=80,
            krylov_rank="auto",
            krylov_method="auto",
            krylov_stats=reference_stats,
            num_threads=threads,
        )
        reference_values = selector @ np.asarray(reference.values, dtype=float)
        error = values - reference_values
        reference_norm = max(float(np.linalg.norm(reference_values)), 1e-30)
        relative_error = float(np.linalg.norm(error) / reference_norm) if np.all(np.isfinite(error)) else float("inf")
        max_abs_error = float(np.max(np.abs(error))) if np.all(np.isfinite(error)) else float("inf")
        tolerance = 5e-2
        metadata["mor_validation"] = {
            "enabled": True,
            "reference_solver": "sparse_full_order_krylov",
            "relative_error": relative_error,
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
            "reference_iterations": sum(int(entry.get("iterations", 0) or 0) for entry in reference_stats),
            "reference_solve_count": len(reference_stats),
        }
        if reduced_failed or relative_error > tolerance:
            metadata.update(
                {
                    "mor_used": False,
                    "mor_attempted_method": "linear_krylov",
                    "mor_method": "sparse_full_order_krylov",
                    "mor_fallback_reason": (
                        "Linear Krylov MOR validation failed; returned sparse full-order selected outputs."
                    ),
                }
            )
            values = reference_values
    elif validate:
        metadata["mor_validation"] = {"enabled": False, "reason": "no sparse full-order validation path available"}
    return WaveformResult(time=t_points, values=values, labels=output_labels), metadata
