"""Model-order-reduction facades over the native core (``mna_simulation._core``).

The numerics of ``mna_simulation/mor/linear.py`` and ``mor/tpwl.py`` run in
``cpp/src/mor`` (bindings ``cpp/bindings/bind_mor.cpp``); those modules keep
option handling, metadata, validation and the TPWL cache and call:

* :func:`krylov_basis`: the output-aware rational Krylov basis of
  ``build_linear_krylov_basis`` on sparse operands (:class:`MorOperands`: ``G``
  and ``C`` values in one shared CSC pattern), every shifted system factored
  once with KLU (real or complex) and reused for every block, the dense
  least-squares fallback of ``_safe_solve`` for a singular shift;
* :func:`project`: ``V^T A V`` with a sparse-dense product (no dense ``A``);
* :func:`reduced_ac` / :func:`reduced_transient`: the reduced frequency and
  step loops of ``solve_ac_mor`` / ``solve_transient_mor``;
* :func:`pod_basis`, :func:`training_indexes`, :func:`tpwl_local_models`,
  :func:`tpwl_simulate`: TPWL training and the reduced TPWL step loop;
* :func:`safe_solve`: the dense ``_safe_solve`` (LU, then least squares, then
  diagonal-regularized least squares).

Operands come from a compiled circuit (:func:`circuit_operands`, the pattern
of ``problem.native``) or from plain matrices (:func:`matrix_operands`: dense
arrays or SciPy sparse matrices, merged into their union pattern).

Errors: malformed shapes raise ``CoreError(NETLIST_FATAL)`` from the core;
the least-squares fallback on a non-finite matrix raises
``numpy.linalg.LinAlgError`` (what ``np.linalg.lstsq`` raised in the Python
implementation).

Threads: ``num_threads`` of :func:`reduced_ac` splits frequencies when
``points >= 2 * threads`` (bitwise identical results for any count); the
shift factorizations and the TPWL device evaluation use more than one thread
only from :data:`PARALLEL_MIN_UNKNOWNS` unknowns, where the work outweighs a
thread start. :func:`resolve_threads` maps ``None``/``"auto"`` to 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import require_core
from .linalg import to_csc_arrays

_core = require_core()

__all__ = [
    "PARALLEL_MIN_UNKNOWNS",
    "MorOperands",
    "circuit_operands",
    "krylov_basis",
    "matrix_operands",
    "pod_basis",
    "project",
    "reduced_ac",
    "reduced_transient",
    "resolve_threads",
    "safe_solve",
    "tpwl_local_models",
    "tpwl_simulate",
    "training_indexes",
]

#: Unknown count from which shift factorizations and TPWL device evaluation run on several threads.
PARALLEL_MIN_UNKNOWNS = int(_core.MOR_PARALLEL_MIN_UNKNOWNS)


@dataclass(frozen=True)
class MorOperands:
    """``G`` and ``C`` (n x n) as values in one shared CSC pattern (int32 indices)."""

    n: int
    col_ptr: np.ndarray
    row_idx: np.ndarray
    g: np.ndarray
    c: np.ndarray

    def with_g(self, g: np.ndarray) -> MorOperands:
        return MorOperands(self.n, self.col_ptr, self.row_idx, np.ascontiguousarray(g, dtype=np.float64), self.c)

    def dense(self, which: str) -> np.ndarray:
        """Dense copy of ``"G"`` or ``"C"`` (tests and diagnostics only)."""

        from scipy import sparse as _sparse

        values = self.g if which == "G" else self.c
        return _sparse.csc_matrix((values, self.row_idx, self.col_ptr), shape=(self.n, self.n)).toarray()


def resolve_threads(num_threads: object) -> int:
    """A positive integer request as given; anything else (``None``, ``"auto"``) is 1."""

    if isinstance(num_threads, (int, np.integer)) and not isinstance(num_threads, bool) and int(num_threads) > 0:
        return int(num_threads)
    return 1


def _translate(exc: Exception) -> None:
    """Re-raise a ``CoreError``: ``JACOBIAN_SINGULAR`` (least squares on a non-finite matrix) as
    ``numpy.linalg.LinAlgError``, every other code as the simulator exception of that code."""

    from ..errors import error_handler

    code = getattr(exc, "code", None) or "UNKNOWN"
    message = getattr(exc, "message", None) or str(exc)
    if code == "JACOBIAN_SINGULAR":
        raise np.linalg.LinAlgError(message) from exc
    error_handler(f"{code}: {message}")


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except _core.CoreError as exc:
        _translate(exc)
        raise


def circuit_operands(circuit: Any, g_values: np.ndarray | None = None, gmin: float = 0.0) -> MorOperands:
    """Operands in a compiled circuit's pattern: ``g_values`` (default ``G + gmin * stamps``) and ``C``."""

    col_ptr, row_idx = circuit.pattern()
    g = circuit.g_effective(float(gmin)) if g_values is None else g_values
    return MorOperands(
        int(circuit.n),
        np.ascontiguousarray(col_ptr, dtype=np.int32),
        np.ascontiguousarray(row_idx, dtype=np.int32),
        np.ascontiguousarray(g, dtype=np.float64),
        np.ascontiguousarray(circuit.values("C"), dtype=np.float64),
    )


def matrix_operands(G: object, C: object) -> MorOperands:
    """Operands from two square matrices (dense arrays or SciPy sparse), merged into their union pattern."""

    from scipy import sparse as _sparse

    if not _sparse.issparse(G) and not _sparse.issparse(C):
        g_dense = np.asarray(G)
        c_dense = np.asarray(C)
        if (
            g_dense.ndim == 2
            and g_dense.shape[0] == g_dense.shape[1]
            and c_dense.shape == g_dense.shape
            and not np.iscomplexobj(g_dense)
            and not np.iscomplexobj(c_dense)
        ):
            # One native pass over both dense arrays (no SciPy conversion of n x n arrays).
            col_ptr, row_idx, g, c = _call(
                _core.mor_dense_operands, g_dense.astype(np.float64, copy=False), c_dense.astype(np.float64, copy=False)
            )
            return MorOperands(int(g_dense.shape[0]), col_ptr, row_idx, g, c)

    g_csc = to_csc_arrays(G)
    c_csc = to_csc_arrays(C)
    if g_csc.n != c_csc.n:
        raise ValueError(f"G is {g_csc.n}x{g_csc.n} but C is {c_csc.n}x{c_csc.n}")
    if g_csc.is_complex or c_csc.is_complex:
        raise TypeError("model-order reduction needs real G and C")
    n = g_csc.n

    def keys(op: Any) -> np.ndarray:
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(op.col_ptr.astype(np.int64)))
        return cols * n + op.row_idx.astype(np.int64)

    g_keys = keys(g_csc)
    c_keys = keys(c_csc)
    union = np.union1d(g_keys, c_keys)  # sorted: column-major (CSC) order
    g = np.zeros(union.size, dtype=np.float64)
    c = np.zeros(union.size, dtype=np.float64)
    g[np.searchsorted(union, g_keys)] = g_csc.values
    c[np.searchsorted(union, c_keys)] = c_csc.values
    counts = np.bincount(union // n, minlength=n) if n > 0 else np.zeros(0, dtype=np.int64)
    col_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=col_ptr[1:])
    row_idx = np.ascontiguousarray(union % n, dtype=np.int32) if n > 0 else np.zeros(0, dtype=np.int32)
    return MorOperands(n, col_ptr, row_idx, g, c)


def krylov_basis(
    operands: MorOperands,
    inputs: np.ndarray,
    selector: np.ndarray,
    shifts: Sequence[complex | float],
    order: int,
    *,
    num_threads: int = 1,
) -> np.ndarray:
    """Orthonormal basis ``V`` (n x k) of ``build_linear_krylov_basis`` (see the module docstring)."""

    n = operands.n
    inputs_array = np.asarray(inputs, dtype=np.float64)
    if inputs_array.ndim != 2:
        inputs_array = inputs_array.reshape(n, -1)
    selector_array = np.asarray(selector, dtype=np.float64)
    if selector_array.ndim == 1:
        selector_array = selector_array.reshape(1, -1)
    shift_values = np.asarray([complex(shift) for shift in shifts], dtype=np.complex128).reshape(-1)
    complex_flags = [bool(np.iscomplexobj(shift)) for shift in shifts]
    basis, _info = _call(
        _core.mor_krylov_basis,
        operands.col_ptr,
        operands.row_idx,
        operands.g,
        operands.c,
        inputs_array,
        selector_array,
        shift_values,
        complex_flags,
        int(order),
        int(num_threads),
    )
    return basis


def project(operands: MorOperands, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(V^T G V, V^T C V)`` for a dense basis ``V`` (n x k)."""

    basis = np.asarray(V, dtype=np.float64)
    Gr = _call(_core.mor_project, operands.col_ptr, operands.row_idx, operands.g, basis)
    Cr = _call(_core.mor_project, operands.col_ptr, operands.row_idx, operands.c, basis)
    return Gr, Cr


def safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``_safe_solve``: ``np.linalg.solve``, then ``lstsq``, then the regularized ``lstsq`` (dense, native)."""

    A = np.asarray(matrix)
    b = np.asarray(rhs)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")
    complex_system = np.iscomplexobj(A) or np.iscomplexobj(b)
    dtype = np.complex128 if complex_system else np.float64
    one_d = b.ndim == 1
    b2 = b.reshape(-1, 1) if one_d else b
    if b2.ndim != 2 or b2.shape[0] != A.shape[0]:
        raise ValueError(f"solve: right-hand side of shape {b.shape} for a {A.shape[0]}x{A.shape[0]} matrix")
    x, _path = _call(_core.mor_safe_solve, A.astype(dtype, copy=False), b2.astype(dtype, copy=False))
    return x.reshape(-1) if one_d else x


def reduced_ac(
    Gr: np.ndarray,
    Cr: np.ndarray,
    Br: np.ndarray,
    Sr: np.ndarray,
    frequencies: np.ndarray,
    *,
    num_threads: int = 1,
) -> np.ndarray:
    """``|Sr _safe_solve(Gr + j*2*pi*f*Cr, Br)|`` for every frequency: (outputs, points)."""

    magnitudes, _info = _call(
        _core.mor_reduced_ac,
        np.asarray(Gr, dtype=np.float64),
        np.asarray(Cr, dtype=np.float64),
        np.ascontiguousarray(np.asarray(Br, dtype=np.complex128).reshape(-1)),
        np.asarray(Sr, dtype=np.float64),
        np.ascontiguousarray(np.asarray(frequencies, dtype=np.float64).reshape(-1)),
        int(num_threads),
    )
    return magnitudes


def reduced_transient(
    Gr: np.ndarray,
    Cr: np.ndarray,
    V: np.ndarray,
    Sr: np.ndarray,
    x0: np.ndarray,
    t_points: np.ndarray,
    b_dc: np.ndarray,
    source_rows: np.ndarray,
    source_programs: Sequence[Any],
) -> tuple[np.ndarray, bool]:
    """Reduced Backward-Euler outputs (outputs, points) and the ``reduced_failed`` flag."""

    values, info = _call(
        _core.mor_reduced_transient,
        np.asarray(Gr, dtype=np.float64),
        np.asarray(Cr, dtype=np.float64),
        np.asarray(V, dtype=np.float64),
        np.asarray(Sr, dtype=np.float64),
        np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(t_points, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(b_dc, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(source_rows, dtype=np.int64),
        list(source_programs),
    )
    return values, bool(info["failed"])


def pod_basis(snapshots: np.ndarray, order: int) -> np.ndarray:
    """POD basis of ``_pod_basis`` (n x r)."""

    basis, _info = _call(_core.mor_pod_basis, np.asarray(snapshots, dtype=np.float64), int(order))
    return basis


def training_indexes(snapshots: np.ndarray, basis_columns: int, output_count: int) -> list[int]:
    """``_training_indexes`` (sorted, unique)."""

    indexes = _call(
        _core.mor_training_indexes, np.asarray(snapshots, dtype=np.float64), int(basis_columns), int(output_count)
    )
    return [int(index) for index in indexes]


def tpwl_local_models(
    circuit: Any,
    snapshots: np.ndarray,
    indexes: Sequence[int],
    V: np.ndarray,
    *,
    num_threads: int = 1,
) -> dict[str, Any]:
    """TPWL local models at ``snapshots[:, indexes]``: ``z``, ``f``, ``jvals``, ``jr``, ``q`` arrays."""

    return _call(
        _core.mor_tpwl_models,
        circuit,
        np.asarray(snapshots, dtype=np.float64),
        np.ascontiguousarray(np.asarray(indexes, dtype=np.int64).reshape(-1)),
        np.asarray(V, dtype=np.float64),
        int(num_threads),
    )


def tpwl_simulate(
    Gr: np.ndarray,
    Cr: np.ndarray,
    V: np.ndarray,
    Sr: np.ndarray,
    models: dict[str, Any],
    x0: np.ndarray,
    t_points: np.ndarray,
    b_dc: np.ndarray,
    source_rows: np.ndarray,
    source_programs: Sequence[Any],
) -> np.ndarray:
    """TPWL reduced outputs (outputs, points) from the local models of :func:`tpwl_local_models`."""

    values, _info = _call(
        _core.mor_tpwl_simulate,
        np.asarray(Gr, dtype=np.float64),
        np.asarray(Cr, dtype=np.float64),
        np.asarray(V, dtype=np.float64),
        np.asarray(Sr, dtype=np.float64),
        np.asarray(models["z"], dtype=np.float64),
        np.asarray(models["jr"], dtype=np.float64),
        np.asarray(models["q"], dtype=np.float64),
        np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(t_points, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(b_dc, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(source_rows, dtype=np.int64),
        list(source_programs),
    )
    return values
