"""Linear selected-output MOR using rational Krylov projection."""

from __future__ import annotations

import numpy as np

from ..api.contracts import MnaProblem, SpectrumResult, WaveformResult
from ..solvers import (
    compile_nl_functions,
    compile_time_source_func,
    is_nonlinear,
    solve_dc_nr,
    solve_linear,
    solve_transient,
    _transient_time_points,
)
from .metadata import resolve_mor_order, used_metadata
from .selectors import build_output_selector, validate_probe_subset


def count_input_directions(problem: MnaProblem) -> int:
    """Count independent source directions used by the MOR order heuristic."""

    directions = _source_directions(problem)
    return max(1, directions.shape[1])


def _nonzero_column(vector: np.ndarray) -> np.ndarray | None:
    column = np.asarray(vector).reshape(-1, 1)
    return column if np.linalg.norm(column) > 0 else None


def _source_directions(problem: MnaProblem) -> np.ndarray:
    n = problem.G.shape[0]
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
    A = np.asarray(matrix)
    b = np.asarray(rhs)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
            return np.linalg.lstsq(A + np.eye(A.shape[0], dtype=A.dtype) * scale * 1e-12, b, rcond=None)[0]


def _append_real_columns(columns: list[np.ndarray], matrix: np.ndarray) -> None:
    array = np.asarray(matrix)
    if array.size == 0:
        return
    if np.iscomplexobj(array):
        columns.append(np.real(array))
        if np.linalg.norm(np.imag(array)) > 0:
            columns.append(np.imag(array))
    else:
        columns.append(array.astype(float))


def _orthonormal_basis(candidates: list[np.ndarray], n: int, order: int) -> np.ndarray:
    if not candidates:
        basis = np.zeros((n, 1), dtype=float)
        basis[0, 0] = 1.0
        return basis
    matrix = np.hstack([np.asarray(candidate).reshape(n, -1) for candidate in candidates])
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 1e-14]
    if matrix.size == 0:
        basis = np.zeros((n, 1), dtype=float)
        basis[0, 0] = 1.0
        return basis
    U, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > max(singular_values[0], 1.0) * 1e-12))
    rank = max(1, min(int(order), rank, U.shape[1]))
    return U[:, :rank]


def _orthogonalize_block(
    basis: np.ndarray,
    block: np.ndarray,
    max_columns: int,
    tolerance: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Append independent block columns and return the accepted normalized part."""

    if basis.shape[1] >= max_columns:
        return basis, np.zeros((basis.shape[0], 0), dtype=float)
    candidate = np.asarray(block)
    if np.iscomplexobj(candidate):
        pieces = [np.real(candidate)]
        if np.linalg.norm(np.imag(candidate)) > 0.0:
            pieces.append(np.imag(candidate))
        candidate = np.hstack(pieces)
    candidate = np.asarray(candidate, dtype=float).reshape(basis.shape[0], -1)
    accepted: list[np.ndarray] = []
    for col in range(candidate.shape[1]):
        vector = candidate[:, [col]]
        if basis.size:
            vector = vector - basis @ (basis.T @ vector)
        norm_value = float(np.linalg.norm(vector))
        if norm_value <= tolerance:
            continue
        accepted_vector = vector / norm_value
        basis = np.hstack([basis, accepted_vector])
        accepted.append(accepted_vector)
        if basis.shape[1] >= max_columns:
            break
    accepted_block = np.hstack(accepted) if accepted else np.zeros((basis.shape[0], 0), dtype=float)
    return basis, accepted_block


def _append_block_if_independent(
    basis: np.ndarray,
    block: np.ndarray,
    max_columns: int,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Append numerically independent block columns to an existing basis."""

    basis, _accepted = _orthogonalize_block(basis, block, max_columns, tolerance=tolerance)
    return basis


def build_linear_krylov_basis(
    G: np.ndarray,
    C: np.ndarray,
    inputs: np.ndarray,
    selector: np.ndarray,
    shifts: list[complex],
    order: int,
) -> np.ndarray:
    """Build a one-sided congruence basis from input and output-adjoint moments.

    ``order`` is a true Krylov dimension budget. Earlier versions only solved
    each seed once at each expansion point, so a large distributed RLC network
    could collapse to a handful of vectors and miss later ringing modes. This
    block rational Krylov loop keeps applying ``(G+sC)^-1 C`` to the input and
    output-adjoint directions until the requested independent basis size is
    reached or no new directions remain.
    """

    n = G.shape[0]
    target_order = max(1, min(int(order), n))
    basis = np.zeros((n, 0), dtype=float)
    input_seeds = inputs if inputs.shape[1] > 0 else np.zeros((n, 0), dtype=float)
    output_seeds = selector.T
    if input_seeds.shape[1] > 0:
        basis = _append_block_if_independent(basis, input_seeds, target_order)
    basis = _append_block_if_independent(basis, output_seeds, target_order)

    systems: list[tuple[np.ndarray, bool]] = []
    for shift in shifts:
        dtype = complex if np.iscomplexobj(shift) else float
        systems.append((np.asarray(G, dtype=dtype) + shift * np.asarray(C), np.iscomplexobj(shift)))

    # Interleave shifts so low- and high-frequency moments both get a chance
    # before any single expansion point consumes the whole order budget.
    state_blocks: list[dict[str, np.ndarray | str]] = []
    for system, _is_complex in systems:
        if input_seeds.shape[1] > 0:
            solved = _safe_solve(system, input_seeds)
            basis, accepted = _orthogonalize_block(basis, solved, target_order)
            if accepted.shape[1] > 0:
                state_blocks.append({"kind": "input", "system": system, "block": accepted})
        solved_adj = _safe_solve(system.T, output_seeds)
        basis, accepted_adj = _orthogonalize_block(basis, solved_adj, target_order)
        if accepted_adj.shape[1] > 0:
            state_blocks.append({"kind": "adjoint", "system": system, "block": accepted_adj})
        if basis.shape[1] >= target_order:
            return basis

    previous_columns = -1
    while basis.shape[1] < target_order and basis.shape[1] != previous_columns:
        previous_columns = basis.shape[1]
        for state in state_blocks:
            system = np.asarray(state["system"])
            block = np.asarray(state["block"])
            if str(state["kind"]) == "adjoint":
                next_block = _safe_solve(system.T, -(np.asarray(C).T @ block))
            else:
                next_block = _safe_solve(system, -(np.asarray(C) @ block))
            basis, accepted = _orthogonalize_block(basis, next_block, target_order)
            state["block"] = accepted if accepted.shape[1] > 0 else block
            if basis.shape[1] >= target_order:
                break

    if basis.shape[1] == 0:
        basis = _orthonormal_basis([], n, target_order)
    return basis


def _linearized_system(
    problem: MnaProblem,
    max_iter: int,
    v_tol: float,
    f_tol: float,
    linear_kwargs: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_mos_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    metadata: dict[str, object] = {}

    if is_nonlinear(solver_f_str) or level1_mos_devices or level1_bjt_devices:
        dc_solution = solve_dc_nr(
            problem.G,
            solver_f_str,
            problem.b_dc,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            init_cond="DEFAULT",
            level1_mos_devices=level1_mos_devices,
            level1_bjt_devices=level1_bjt_devices,
            **linear_kwargs,
        )
        _, Jf_func = compile_nl_functions(
            solver_f_str,
            problem.G.shape[0],
            level1_mos_devices=level1_mos_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
        jacobian = np.asarray(Jf_func(*dc_solution.flatten()), dtype=float)
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
        return problem.G + jacobian, metadata

    metadata.update({"op_used": False, "linearized_device_count": 0})
    return problem.G, metadata


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
) -> SpectrumResult:
    """Solve selected AC outputs through a reduced linearized model."""

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    G_linearized, spectrum_metadata = _linearized_system(
        problem,
        max_iter=max_iter,
        v_tol=v_tol,
        f_tol=f_tol,
        linear_kwargs=dict(linear_kwargs or {}),
    )
    inputs = _source_directions(problem)
    order, order_mode = resolve_mor_order(problem.G.shape[0], count_input_directions(problem), len(output_labels), mor_order)
    f_mid = float(np.sqrt(max(abs(f_start), 1e-30) * max(abs(f_stop), 1e-30)))
    shifts = [0.0, 1j * 2.0 * np.pi * f_mid, 1j * 2.0 * np.pi * float(f_stop)]
    V = build_linear_krylov_basis(G_linearized, problem.C, inputs, selector, shifts, order)

    Vr = V.T
    Gr = Vr @ G_linearized @ V
    Cr = Vr @ problem.C @ V
    Br = Vr @ problem.b_ac
    Sr = selector @ V

    frequencies = np.linspace(f_start, f_stop, points)
    responses = np.zeros((len(output_labels), len(frequencies)), dtype=float)
    for index, frequency in enumerate(frequencies):
        omega = 2.0 * np.pi * float(frequency)
        z = _safe_solve(Gr + 1j * omega * Cr, Br)
        y = Sr @ z
        responses[:, index] = np.abs(y.flatten())

    spectrum_metadata.update(
        used_metadata(
            method="linear_krylov",
            requested_method=requested_method,
            original_dimension=problem.G.shape[0],
            reduced_dimension=V.shape[1],
            resolved_order=order,
            order_mode=order_mode,
            output_labels=output_labels,
            validate=validate,
            extra={
                "mor_basis": "output_aware_rational_krylov",
                "mor_num_inputs": count_input_directions(problem),
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
) -> tuple[WaveformResult, dict[str, object]]:
    """Solve selected transient outputs through a reduced linear model."""

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    inputs = _source_directions(problem)
    order, order_mode = resolve_mor_order(problem.G.shape[0], count_input_directions(problem), len(output_labels), mor_order)
    shifts = [0.0, 1.0 / max(float(t_step), np.finfo(float).tiny)]
    V = build_linear_krylov_basis(problem.G, problem.C, inputs, selector, shifts, order)
    Vr = V.T
    Gr = Vr @ problem.G @ V
    Cr = Vr @ problem.C @ V
    Sr = selector @ V
    b_time_func = compile_time_source_func(problem.b_time_str)

    if init_cond is None:
        x0 = solve_dc_nr(
            problem.G,
            problem.solver_f_str if problem.solver_f_str is not None else problem.f_str,
            problem.b_dc + b_time_func(0.0),
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            **dict(linear_kwargs or {}),
        )
    else:
        x0 = np.asarray(init_cond).reshape(-1, 1)
    z_prev = Vr @ x0

    t_points = _transient_time_points(t_stop, t_step)
    values = np.zeros((len(output_labels), len(t_points)), dtype=float)
    values[:, 0] = (Sr @ z_prev).flatten()
    reduced_failed = False

    for step in range(1, len(t_points)):
        h = float(t_points[step] - t_points[step - 1])
        Cr_h = Cr / h
        rhs = Vr @ (problem.b_dc + b_time_func(float(t_points[step]))) + Cr_h @ z_prev
        z_prev = _safe_solve(Gr + Cr_h, rhs)
        if not np.all(np.isfinite(z_prev)) or float(np.linalg.norm(z_prev)) > 1e12:
            reduced_failed = True
            values[:, step:] = np.nan
            break
        values[:, step] = (Sr @ z_prev).flatten()

    metadata = used_metadata(
        method="linear_krylov",
        requested_method=requested_method,
        original_dimension=problem.G.shape[0],
        reduced_dimension=V.shape[1],
        resolved_order=order,
        order_mode=order_mode,
        output_labels=output_labels,
        validate=validate,
        extra={
            "mor_basis": "output_aware_rational_krylov",
            "mor_num_inputs": count_input_directions(problem),
            "mor_basis_expansion_points": ["0", f"{1.0 / max(float(t_step), np.finfo(float).tiny):.6g}"],
        },
    )
    if validate and problem.G_sparse is not None and problem.C_sparse is not None and problem.G.shape[0] >= 64:
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
