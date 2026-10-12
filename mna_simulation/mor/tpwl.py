"""Transient nonlinear MOR using a small TPWL/POD cache.

The LRU cache, option handling and metadata live here; training (the full
native transient, the POD basis, the training indexes and the local models'
reduced Jacobians) and the reduced TPWL step loop run in the native core
through :mod:`mna_simulation.native.mor` (CPP_MIGRATION_PLAN.md section 2.7).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from scipy import sparse as _sparse

from ..api.contracts import MnaProblem, WaveformResult
from ..native import mor as native_mor
from ..solvers import (
    _problem_circuit,
    _problem_is_nonlinear,
    _time_sources,
    _transient_time_points,
    solve_transient,
)
from .linear import _problem_size, count_input_directions
from .metadata import resolve_mor_order, used_metadata
from .selectors import build_output_selector, validate_probe_subset

_TPWL_CACHE: "OrderedDict[str, dict[str, object]]" = OrderedDict()
_TPWL_CACHE_LIMIT = 8


def _matrix_digest(hasher: "hashlib._Hash", dense: object, sparse_matrix: object) -> None:
    """Feed one matrix into the digest: the canonical CSR form when sparse storage exists, else the dense bytes."""

    if sparse_matrix is not None:
        csr = _sparse.csr_matrix(sparse_matrix, dtype=float, copy=True)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        hasher.update(f"csr{csr.shape}".encode("utf-8"))
        for array in (csr.indptr.astype(np.int64), csr.indices.astype(np.int64), csr.data):
            hasher.update(np.ascontiguousarray(array).tobytes())
        return
    hasher.update(np.asarray(dense).tobytes())


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
    # G and C: canonical sparse form (no dense n x n copy for large circuits).
    _matrix_digest(hasher, problem.G if problem.G_sparse is None else None, problem.G_sparse)
    _matrix_digest(hasher, problem.C if problem.C_sparse is None else None, problem.C_sparse)
    for vector in (problem.b_dc, problem.b_ac):
        hasher.update(np.asarray(vector).tobytes())
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
    """POD basis: leading left singular vectors, rank ``s_i > max(s_0, 1) * 1e-12``, at most ``order`` (native SVD)."""

    return native_mor.pod_basis(snapshots, order)


def _training_indexes(snapshots: np.ndarray, basis: np.ndarray, output_count: int) -> list[int]:
    """Snapshot indexes with a local model: a linspace grid plus every step whose delta reaches the median."""

    return native_mor.training_indexes(snapshots, np.asarray(basis).shape[1], output_count)


def _device_circuit(problem: MnaProblem):
    """The compiled circuit whose devices give ``f(x)`` and ``J(x)`` (``problem.native`` when available)."""

    if problem.native is not None:
        return problem.native
    circuit, _gmin = _problem_circuit(problem, _problem_is_nonlinear(problem))
    return circuit


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
    num_threads: int | str | None = None,
) -> dict[str, object]:
    threads = native_mor.resolve_threads(num_threads)
    full = solve_transient(
        problem,
        t_stop=t_stop,
        t_step=t_step,
        max_iter=max_iter,
        v_tol=v_tol,
        f_tol=f_tol,
        init_cond=None,
        use_krylov=False,
        num_threads=threads,
    )
    snapshots = np.asarray(full.values, dtype=float)
    basis = _pod_basis(snapshots, order)
    indexes = _training_indexes(snapshots, basis, output_count)
    models = native_mor.tpwl_local_models(_device_circuit(problem), snapshots, indexes, basis, num_threads=threads)
    size = snapshots.shape[0]
    local_models: list[dict[str, np.ndarray]] = [
        {
            "x": snapshots[:, index].reshape(-1, 1),
            "z": models["z"][position].reshape(-1, 1),
            "f": models["f"][position].reshape(size, 1),
            # Device Jacobian values in the compiled circuit's CSC pattern (``circuit.pattern()``).
            "J_values": models["jvals"][position],
        }
        for position, index in enumerate(indexes)
    ]
    return {
        "basis": basis,
        "snapshots": snapshots,
        "time": full.time,
        "labels": full.labels,
        "local_models": local_models,
        # Reduced local models for the native step loop: z_i, V^T J_i V, V^T (J_i x_i - f_i).
        "models": {"z": models["z"], "jr": models["jr"], "q": models["q"]},
    }


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
    num_threads: int | str | None = None,
) -> tuple[WaveformResult, dict[str, object]]:
    """Train/reuse a TPWL ROM and return selected transient outputs."""

    labels = list(problem.metadata["labels"])
    selector, output_labels = build_output_selector(labels, mor_output_nodes)
    if validate:
        validate_probe_subset(labels, probe_nodes, output_labels)
    n = _problem_size(problem)
    num_inputs = count_input_directions(problem)
    order, order_mode = resolve_mor_order(n, num_inputs, len(output_labels), mor_order)
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
            num_threads=num_threads,
        )
        _cache_put(cache_key, entry)

    basis = entry["basis"]  # type: ignore[assignment]
    V = np.asarray(basis, dtype=float)
    Sr = selector @ V
    snapshots = np.asarray(entry["snapshots"], dtype=float)
    # When the POD basis spans the full system, use the training trajectory.
    # It is the exact TPWL training projection and avoids artificial error in
    # tiny circuits where reduction would otherwise be a no-op.
    if V.shape[1] >= n:
        values = selector @ snapshots
        waveform = WaveformResult(time=np.asarray(entry["time"], dtype=float), values=values, labels=output_labels)
    else:
        if not _problem_is_nonlinear(problem):
            values = selector @ snapshots
            waveform = WaveformResult(time=np.asarray(entry["time"], dtype=float), values=values, labels=output_labels)
        else:
            circuit, gmin = _problem_circuit(problem, True)
            Gr, Cr = native_mor.project(native_mor.circuit_operands(circuit, gmin=gmin), V)
            source_rows, source_programs = _time_sources(problem.b_time_str)
            t_points = _transient_time_points(t_stop, t_step)
            values = native_mor.tpwl_simulate(
                Gr,
                Cr,
                V,
                Sr,
                entry["models"],  # type: ignore[arg-type]
                snapshots[:, 0],
                t_points,
                problem.b_dc,
                source_rows,
                source_programs,
            )
            waveform = WaveformResult(time=t_points, values=values, labels=output_labels)

    metadata = used_metadata(
        method="tpwl",
        requested_method=requested_method,
        original_dimension=n,
        reduced_dimension=V.shape[1],
        resolved_order=order,
        order_mode=order_mode,
        output_labels=output_labels,
        validate=validate,
        extra={
            "mor_basis": "tpwl_pod",
            "mor_num_inputs": num_inputs,
            "mor_cache_hit": cache_hit,
            "mor_cache_size": len(_TPWL_CACHE),
            "mor_training_snapshots": int(snapshots.shape[1]),
            "mor_tpwl_local_models": len(entry["local_models"]),  # type: ignore[arg-type]
        },
    )
    return waveform, metadata
