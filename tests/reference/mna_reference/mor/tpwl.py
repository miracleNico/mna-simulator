"""Transient nonlinear MOR using a small TPWL/POD cache."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from ..api.contracts import MnaProblem, WaveformResult
from ..solvers import compile_nl_functions, compile_time_source_func, is_nonlinear, solve_transient, _transient_time_points
from .linear import _safe_solve, count_input_directions
from .metadata import resolve_mor_order, used_metadata
from .selectors import build_output_selector, validate_probe_subset

_TPWL_CACHE: "OrderedDict[str, dict[str, object]]" = OrderedDict()
_TPWL_CACHE_LIMIT = 8


def _problem_cache_digest(
    problem: MnaProblem,
    *,
    t_stop: float,
    t_step: float,
    outputs: list[str],
    order: int | str | None,
    method: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(problem.components).encode("utf-8", errors="replace"))
    for matrix in (problem.G, problem.C, problem.b_dc, problem.b_ac):
        hasher.update(np.asarray(matrix).tobytes())
    hasher.update("|".join(str(expr) for expr in problem.b_time_str.flatten()).encode("utf-8", errors="replace"))
    hasher.update(repr(problem.level1_mos_devices).encode("utf-8", errors="replace"))
    hasher.update(repr(problem.level1_bjt_devices).encode("utf-8", errors="replace"))
    hasher.update(f"{t_stop:.17g}|{t_step:.17g}|{outputs}|{order}|{method}".encode("utf-8"))
    return hasher.hexdigest()


def _cache_get(key: str) -> dict[str, object] | None:
    entry = _TPWL_CACHE.get(key)
    if entry is None:
        return None
    _TPWL_CACHE.move_to_end(key)
    return entry


def _cache_put(key: str, entry: dict[str, object]) -> None:
    _TPWL_CACHE[key] = entry
    _TPWL_CACHE.move_to_end(key)
    while len(_TPWL_CACHE) > _TPWL_CACHE_LIMIT:
        _TPWL_CACHE.popitem(last=False)


def _pod_basis(snapshots: np.ndarray, order: int) -> np.ndarray:
    centered = np.asarray(snapshots, dtype=float)
    if centered.size == 0:
        basis = np.zeros((snapshots.shape[0], 1), dtype=float)
        basis[0, 0] = 1.0
        return basis
    U, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return np.eye(snapshots.shape[0], min(order, snapshots.shape[0]))
    rank = int(np.sum(singular_values > max(singular_values[0], 1.0) * 1e-12))
    rank = max(1, min(order, rank, U.shape[1]))
    return U[:, :rank]


def _training_indexes(snapshots: np.ndarray, basis: np.ndarray, output_count: int) -> list[int]:
    steps = snapshots.shape[1]
    deterministic_count = min(steps, max(4, min(16, basis.shape[1] + output_count)))
    indexes = set(np.linspace(0, steps - 1, deterministic_count, dtype=int).tolist())
    if steps > 2:
        deltas = np.linalg.norm(np.diff(snapshots, axis=1), axis=0)
        positive = deltas[deltas > 0]
        if positive.size:
            trigger = float(np.median(positive))
            for idx, delta in enumerate(deltas, start=1):
                if delta >= trigger:
                    indexes.add(idx)
    return sorted(indexes)


def _train_tpwl(
    problem: MnaProblem,
    *,
    t_stop: float,
    t_step: float,
    order: int,
    max_iter: int,
    v_tol: float,
    f_tol: float,
    output_count: int,
) -> dict[str, object]:
    full = solve_transient(
        problem,
        t_stop=t_stop,
        t_step=t_step,
        max_iter=max_iter,
        v_tol=v_tol,
        f_tol=f_tol,
        init_cond=None,
        use_krylov=False,
    )
    snapshots = np.asarray(full.values, dtype=float)
    basis = _pod_basis(snapshots, order)
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    f_func, Jf_func = compile_nl_functions(
        solver_f_str,
        problem.G.shape[0],
        level1_mos_devices=problem.level1_mos_devices,
        level1_bjt_devices=problem.level1_bjt_devices,
    )
    local_models: list[dict[str, np.ndarray]] = []
    for index in _training_indexes(snapshots, basis, output_count):
        x_i = snapshots[:, index].reshape(-1, 1)
        x_flat = x_i.flatten()
        local_models.append(
            {
                "x": x_i,
                "z": basis.T @ x_i,
                "f": np.asarray(f_func(*x_flat), dtype=float).reshape(problem.G.shape[0], 1),
                "J": np.asarray(Jf_func(*x_flat), dtype=float),
            }
        )
    return {
        "basis": basis,
        "snapshots": snapshots,
        "time": full.time,
        "labels": full.labels,
        "local_models": local_models,
    }


def _nearest_model(local_models: list[dict[str, np.ndarray]], z: np.ndarray) -> dict[str, np.ndarray]:
    distances = [float(np.linalg.norm(z - model["z"])) for model in local_models]
    return local_models[int(np.argmin(distances))]


def solve_transient_tpwl(
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
) -> tuple[WaveformResult, dict[str, object]]:
    """Train/reuse a TPWL ROM and return selected transient outputs."""

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    order, order_mode = resolve_mor_order(problem.G.shape[0], count_input_directions(problem), len(output_labels), mor_order)
    cache_key = _problem_cache_digest(
        problem,
        t_stop=t_stop,
        t_step=t_step,
        outputs=output_labels,
        order=order,
        method=requested_method,
    )
    entry = _cache_get(cache_key)
    cache_hit = entry is not None
    if entry is None:
        entry = _train_tpwl(
            problem,
            t_stop=t_stop,
            t_step=t_step,
            order=order,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            output_count=len(output_labels),
        )
        _cache_put(cache_key, entry)

    basis = entry["basis"]  # type: ignore[assignment]
    V = np.asarray(basis, dtype=float)
    Sr = selector @ V
    snapshots = np.asarray(entry["snapshots"], dtype=float)
    # When the POD basis spans the full system, use the training trajectory.
    # It is the exact TPWL training projection and avoids artificial error in
    # tiny circuits where reduction would otherwise be a no-op.
    if V.shape[1] >= problem.G.shape[0]:
        values = selector @ snapshots
        waveform = WaveformResult(time=np.asarray(entry["time"], dtype=float), values=values, labels=output_labels)
    else:
        solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
        if not (is_nonlinear(solver_f_str) or problem.level1_mos_devices or problem.level1_bjt_devices):
            values = selector @ snapshots
            waveform = WaveformResult(time=np.asarray(entry["time"], dtype=float), values=values, labels=output_labels)
        else:
            local_models = entry["local_models"]  # type: ignore[assignment]
            b_time_func = compile_time_source_func(problem.b_time_str)
            t_points = _transient_time_points(t_stop, t_step)
            values = np.zeros((len(output_labels), len(t_points)), dtype=float)
            z_prev = V.T @ snapshots[:, 0].reshape(-1, 1)
            values[:, 0] = (Sr @ z_prev).flatten()
            Gr = V.T @ problem.G @ V
            Cr = V.T @ problem.C @ V

            for step in range(1, len(t_points)):
                h = float(t_points[step] - t_points[step - 1])
                model = _nearest_model(local_models, z_prev)  # type: ignore[arg-type]
                x_i = model["x"]
                f_i = model["f"]
                J_i = model["J"]
                rhs_full = problem.b_dc + b_time_func(float(t_points[step])) + (problem.C / h) @ (V @ z_prev) - f_i + J_i @ x_i
                rhs = V.T @ rhs_full
                A = Gr + Cr / h + V.T @ J_i @ V
                z_prev = _safe_solve(A, rhs)
                values[:, step] = (Sr @ z_prev).flatten()
            waveform = WaveformResult(time=t_points, values=values, labels=output_labels)

    metadata = used_metadata(
        method="tpwl",
        requested_method=requested_method,
        original_dimension=problem.G.shape[0],
        reduced_dimension=V.shape[1],
        resolved_order=order,
        order_mode=order_mode,
        output_labels=output_labels,
        validate=validate,
        extra={
            "mor_basis": "tpwl_pod",
            "mor_num_inputs": count_input_directions(problem),
            "mor_cache_hit": cache_hit,
            "mor_cache_size": len(_TPWL_CACHE),
            "mor_training_snapshots": int(snapshots.shape[1]),
            "mor_tpwl_local_models": len(entry["local_models"]),  # type: ignore[arg-type]
        },
    )
    return waveform, metadata
