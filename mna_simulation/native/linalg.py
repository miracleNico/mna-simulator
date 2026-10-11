"""Linear-algebra facades over the native core (``mna_simulation._core``).

Drop-in equivalents of the reference functions in ``solvers.py``:

* :func:`classify_krylov_matrix`, :func:`resolve_krylov_rank`,
  :func:`describe_krylov_choice`, :func:`krylov_solve_linear` and
  :func:`solve_linear` keep the reference signatures, return shapes, metadata
  keys and exceptions;
* dense ``np.ndarray`` (real or complex) and SciPy sparse matrices are
  accepted and converted to CSC with int32 indices once per call;
* direct solves run through KLU (``_core.direct_solve``) instead of LAPACK
  ``gesv`` on a dense copy and raise ``np.linalg.LinAlgError("Singular
  matrix")`` for a singular system, like ``np.linalg.solve``.

Deliberate differences from the reference (see CPP_MIGRATION_PLAN.md 6.1):

* classification uses the sparse rule for every input: symmetric when
  ``max |a_ij - conj(a_ji)| <= symmetry_tol``, positive definite when
  additionally ``n <= 256`` and a dense Cholesky succeeds. The reference used
  ``np.allclose`` and an unbounded Cholesky for dense inputs;
* Krylov ``info`` always carries ``engine == "mna_core"`` and
  ``threads_used``; iteration counts come from the native kernels (the
  ``n >= 64`` kernels reproduce SciPy's cg/minres/gmres stopping rules, so
  ``converged`` matches the reference, but counts may differ slightly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse as _sparse

from . import require_core

_core = require_core()

__all__ = [
    "LinearSolver",
    "CscArrays",
    "to_csc_arrays",
    "classify_krylov_matrix",
    "resolve_krylov_rank",
    "describe_krylov_choice",
    "krylov_solve_linear",
    "solve_linear",
    "direct_solve",
]

LinearSolver = _core.LinearSolver

_SINGULAR_MESSAGE = "Singular matrix"


@dataclass(frozen=True)
class CscArrays:
    """A square matrix in the CSC form the core takes (int32 indices)."""

    n: int
    col_ptr: np.ndarray
    row_idx: np.ndarray
    values: np.ndarray
    #: Stored entries as the reference counts them: ``A.tocsr().nnz`` for
    #: sparse input (explicit zeros and duplicates included),
    #: ``np.count_nonzero(A)`` for dense input.
    nnz: int
    sparse_input: bool

    @property
    def is_complex(self) -> bool:
        return bool(np.iscomplexobj(self.values))

    def astype(self, dtype: np.dtype) -> CscArrays:
        values = np.ascontiguousarray(self.values, dtype=dtype)
        return CscArrays(self.n, self.col_ptr, self.row_idx, values, self.nnz, self.sparse_input)


def _is_sparse(matrix: object) -> bool:
    return bool(_sparse.issparse(matrix))


def _value_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype(np.complex128) if np.issubdtype(dtype, np.complexfloating) else np.dtype(np.float64)


def to_csc_arrays(A: object) -> CscArrays:
    """Convert a square dense array or SciPy sparse matrix into :class:`CscArrays`.

    Values become float64 (or complex128 for complex input). Raises
    ``np.linalg.LinAlgError`` for a non-square or non-2-D input.
    """

    if _is_sparse(A):
        csr = A.tocsr()
        if csr.ndim != 2 or csr.shape[0] != csr.shape[1]:
            raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")
        nnz = int(csr.nnz)
        csc = csr.tocsc()
        if not csc.has_canonical_format:
            csc = csc.copy()
            csc.sum_duplicates()
        sparse_input = True
    else:
        dense = np.asarray(A)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")
        dense = dense.astype(_value_dtype(dense.dtype), copy=False)
        csc = _sparse.csc_array(dense)
        nnz = int(csc.nnz)
        sparse_input = False
    n = int(csc.shape[0])
    values = np.ascontiguousarray(csc.data, dtype=_value_dtype(csc.dtype))
    col_ptr = np.ascontiguousarray(csc.indptr, dtype=np.int32)
    row_idx = np.ascontiguousarray(csc.indices, dtype=np.int32)
    return CscArrays(n, col_ptr, row_idx, values, nnz, sparse_input)


# ---------------------------------------------------------------------------
# Classification, routing, rank (reference semantics)
# ---------------------------------------------------------------------------


def _classify_csc(op: CscArrays, symmetry_tol: float = 1e-10) -> str:
    return str(_core.classify_matrix(op.col_ptr, op.row_idx, op.values, float(symmetry_tol)))


def classify_krylov_matrix(A: object, symmetry_tol: float = 1e-10) -> str:
    """Classify a matrix for Krylov method selection (sparse rule for every input).

    ``general`` unless ``max |a_ij - conj(a_ji)| <= symmetry_tol``; then
    ``positive_definite`` when ``n <= 256`` and a dense Cholesky succeeds,
    otherwise ``symmetric``. Non-square or non-2-D inputs are ``general``.
    """

    if _is_sparse(A):
        matrix = A.tocsr()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return "general"
    else:
        matrix = np.asarray(A)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return "general"
    return _classify_csc(to_csc_arrays(matrix), symmetry_tol)


def resolve_krylov_rank(matrix_dimension: int, rank: int | str | None = "auto", fallback: int = 80) -> tuple[int, str]:
    """Resolve a UI/API Krylov rank request against an actual matrix size."""

    n = max(1, int(matrix_dimension))
    if rank is None:
        rank = fallback
    if isinstance(rank, str):
        cleaned = rank.strip().lower()
        if cleaned == "auto":
            return max(1, int(np.ceil(0.5 * n))), "auto"
        rank = cleaned
    resolved = int(rank)
    if resolved < 1:
        raise ValueError("krylov_rank must be 'auto' or a positive integer.")
    return resolved, "manual"


def _krylov_route(matrix_kind: str, method: str) -> str:
    requested = method.lower()
    if requested == "auto":
        return matrix_kind
    if requested in {"cg", "conjugate_gradient", "positive_definite", "spd"}:
        return "positive_definite"
    if requested in {"cr", "minres", "conjugate_residual", "symmetric", "hermitian"}:
        return "symmetric"
    if requested in {"arnoldi", "gmres", "arnoldi_gmres", "general"}:
        return "general"
    raise ValueError(f"Unknown Krylov method '{method}'.")


def _krylov_method_name(route: str) -> str:
    if route == "positive_definite":
        return "conjugate_gradient"
    if route == "symmetric":
        return "conjugate_residual"
    return "arnoldi_gmres"


def _matrix_dimension(matrix: object) -> int:
    shape = getattr(matrix, "shape", ())
    return int(shape[0]) if len(shape) >= 1 else 0


def describe_krylov_choice(
    A: np.ndarray,
    rank: int | str | None = "auto",
    fallback_restart: int = 80,
    max_iter: int = 2000,
    method: str = "auto",
) -> dict[str, object]:
    """Describe the auto-selected Krylov method and resolved rank."""

    matrix = A
    dimension = _matrix_dimension(matrix)
    matrix_kind = classify_krylov_matrix(matrix)
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


# ---------------------------------------------------------------------------
# Krylov solve
# ---------------------------------------------------------------------------


def _system_dtype(matrix_dtype: np.dtype, b: object, x0: object | None) -> np.dtype:
    complex_system = (
        np.issubdtype(np.dtype(matrix_dtype), np.complexfloating)
        or np.iscomplexobj(b)
        or (x0 is not None and np.iscomplexobj(x0))
    )
    return np.dtype(np.complex128) if complex_system else np.dtype(np.float64)


def _restore_rhs_shape(x: np.ndarray, b: object) -> np.ndarray:
    rhs = np.asarray(b)
    if rhs.ndim == 2:
        return x.reshape(rhs.shape)
    return x


def krylov_solve_linear(
    A: object,
    b: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 2000,
    restart: int = 80,
    rank: int | str | None = "auto",
    method: str = "auto",
    x0: np.ndarray | None = None,
    *,
    num_threads: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve ``A x = b`` with an auto-selected Krylov subspace method.

    Same routing, rank resolution and ``info`` keys as the reference. Systems
    with ``n >= 64`` run kernels with SciPy's cg/minres/gmres semantics (the
    reference calls SciPy there) and report ``exit_code``; smaller ones (and
    complex symmetric ones) run ports of the reference loops. ``num_threads``
    only matters from ``n >= 4096`` (parallel matrix-vector products; results
    are identical for every thread count).
    """

    sparse_input = _is_sparse(A)
    matrix = A.tocsr() if sparse_input else np.asarray(A)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise np.linalg.LinAlgError("Krylov solve requires a square matrix.")
    n = int(matrix.shape[0])
    op = to_csc_arrays(matrix)
    matrix_kind = _classify_csc(op)
    route = _krylov_route(matrix_kind, method)
    resolved_rank, rank_mode = resolve_krylov_rank(n, restart if rank is None else rank, restart)

    dtype = _system_dtype(op.values.dtype, b, x0)
    rhs = np.ascontiguousarray(np.asarray(b, dtype=dtype).reshape(-1))
    if rhs.shape[0] != n:
        raise ValueError(f"dimension mismatch: right-hand side has {rhs.shape[0]} entries for a {n}x{n} matrix")
    guess = None
    if x0 is not None:
        guess = np.ascontiguousarray(np.asarray(x0, dtype=dtype).reshape(-1))
        if guess.shape[0] != n:
            raise ValueError(f"dimension mismatch: x0 has {guess.shape[0]} entries for a {n}x{n} matrix")
    values = op.values if op.values.dtype == dtype else op.values.astype(dtype)

    solution, core_info = _core.krylov_solve(
        op.col_ptr,
        op.row_idx,
        values,
        rhs,
        route,
        float(tol),
        int(max_iter),
        int(resolved_rank),
        guess,
        int(num_threads),
    )

    density = float(op.nnz / (n * n)) if n > 0 else 0.0
    info: dict[str, Any] = {"method": core_info["method"], "engine": "mna_core"}
    if core_info["scipy_semantics"]:
        # Key set of the reference's SciPy branch (n >= 64).
        info.update(
            {
                "operator_storage": "csr",
                "matrix_nnz": op.nnz,
                "matrix_density": density,
                "converged": bool(core_info["converged"]),
                "iterations": int(core_info["iterations"]),
                "relative_residual": float(core_info["relative_residual"]),
                "iteration_budget": int(core_info["iteration_budget"]),
                "effective_restart": core_info["effective_restart"],
                "exit_code": int(core_info["exit_code"]),
            }
        )
    else:
        info.update(
            {
                "converged": bool(core_info["converged"]),
                "iterations": int(core_info["iterations"]),
                "relative_residual": float(core_info["relative_residual"]),
            }
        )
    info["requested_method"] = method
    info["matrix_kind"] = matrix_kind
    info["route"] = route
    info["matrix_dimension"] = n
    info["matrix_nnz"] = info.get("matrix_nnz", op.nnz)
    info["matrix_density"] = info.get("matrix_density", density)
    info["operator_storage"] = info.get("operator_storage", "csr" if sparse_input else "dense")
    info["resolved_rank"] = resolved_rank
    info["rank_mode"] = rank_mode
    info["iteration_budget"] = info.get("iteration_budget", int(max_iter) if route == "general" else resolved_rank)
    if route == "general":
        info["effective_restart"] = info.get("effective_restart", min(resolved_rank, n))
    info["threads_used"] = int(core_info["threads_used"])
    return _restore_rhs_shape(solution, b), info


# ---------------------------------------------------------------------------
# Direct solve
# ---------------------------------------------------------------------------


def direct_solve(A: object, b: np.ndarray) -> np.ndarray:
    """``np.linalg.solve(A, b)`` through the native direct solver (KLU).

    ``A`` is a square dense array or SciPy sparse matrix, ``b`` has shape
    ``(n,)`` or ``(n, k)``; the result has ``b``'s shape and is complex when
    ``A`` or ``b`` is. Raises ``np.linalg.LinAlgError("Singular matrix")`` for
    a singular matrix.
    """

    op = to_csc_arrays(A)
    rhs = np.asarray(b)
    if rhs.ndim not in (1, 2):
        raise ValueError(f"solve: right-hand side must be 1-D or 2-D, got shape {rhs.shape}")
    if rhs.shape[0] != op.n:
        raise ValueError(f"solve: right-hand side has {rhs.shape[0]} rows for a {op.n}x{op.n} matrix")
    rhs = rhs.astype(_value_dtype(rhs.dtype), copy=False)
    try:
        return _core.direct_solve(op.col_ptr, op.row_idx, op.values, rhs)
    except _core.CoreError as exc:
        if getattr(exc, "code", None) == "JACOBIAN_SINGULAR":
            raise np.linalg.LinAlgError(_SINGULAR_MESSAGE) from exc
        raise


def solve_linear(
    A: np.ndarray,
    b: np.ndarray,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``A x = b``: Krylov first when requested, otherwise (or on fallback) direct.

    The Krylov answer is accepted when it converged or its relative residual is
    at most ``max(10 * krylov_tol, 1e-8)``; otherwise the direct solver runs and
    the stats entry carries ``used_direct_fallback = True``. A
    ``LinAlgError``/``FloatingPointError`` inside the Krylov path records a
    ``describe_krylov_choice`` entry with ``fallback_reason`` and falls back.
    """

    if use_krylov:
        try:
            solution, info = krylov_solve_linear(
                A,
                b,
                tol=krylov_tol,
                max_iter=krylov_max_iter,
                restart=krylov_restart,
                rank=krylov_rank,
                method=krylov_method,
                x0=x0,
            )
            residual = float(info.get("relative_residual", np.inf))
            if bool(info.get("converged")) or residual <= max(krylov_tol * 10, 1e-8):
                if krylov_stats is not None:
                    krylov_stats.append({**info, "used_direct_fallback": False})
                return solution
            if krylov_stats is not None:
                krylov_stats.append({**info, "used_direct_fallback": True})
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            if krylov_stats is not None:
                choice = describe_krylov_choice(
                    A,
                    rank=krylov_rank,
                    fallback_restart=krylov_restart,
                    max_iter=krylov_max_iter,
                    method=krylov_method,
                )
                krylov_stats.append(
                    {
                        **choice,
                        "converged": False,
                        "iterations": 0,
                        "relative_residual": float("inf"),
                        "used_direct_fallback": True,
                        "fallback_reason": str(exc),
                    }
                )

    return direct_solve(A, b)
