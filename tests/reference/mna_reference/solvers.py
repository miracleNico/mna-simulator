"""Numerical solvers shared by basic and advanced cores."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import numpy as np
import sympy as sp
from numpy.linalg import norm

from .api.contracts import MnaProblem, SpectrumResult, WaveformResult
from .errors import error_handler

try:  # Optional native iterative kernels; pure-Python fallbacks remain below.
    from scipy.sparse import csr_matrix as _scipy_csr_matrix
    from scipy.sparse import issparse as _scipy_issparse
    from scipy.sparse.linalg import cg as _scipy_cg
    from scipy.sparse.linalg import gmres as _scipy_gmres
    from scipy.sparse.linalg import minres as _scipy_minres
except Exception:  # pragma: no cover - depends on optional runtime package.
    _scipy_csr_matrix = None
    _scipy_issparse = None
    _scipy_cg = None
    _scipy_gmres = None
    _scipy_minres = None


def _is_sparse_matrix(matrix: object) -> bool:
    return bool(_scipy_issparse is not None and _scipy_issparse(matrix))


def _matrix_dimension(matrix: object) -> int:
    shape = getattr(matrix, "shape", ())
    return int(shape[0]) if len(shape) >= 1 else 0


def _matrix_dtype(matrix: object) -> np.dtype:
    return np.dtype(getattr(matrix, "dtype", np.float64))


def _matrix_nnz(matrix: object) -> int:
    if _is_sparse_matrix(matrix):
        return int(getattr(matrix, "nnz", 0))
    return int(np.count_nonzero(np.asarray(matrix)))


def _matrix_density(matrix: object) -> float:
    n = _matrix_dimension(matrix)
    return float(_matrix_nnz(matrix) / (n * n)) if n > 0 else 0.0


def _operator_storage(matrix: object) -> str:
    if _is_sparse_matrix(matrix):
        return str(getattr(matrix, "format", "sparse"))
    return "dense"


def lu_decomposition(matrix: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Doolittle LU decomposition without pivoting.

    Kept for callers that explicitly want the (L, U) pair. For solving linear
    systems, prefer :func:`solve_linear`, which uses LAPACK with partial
    pivoting — pivoting is essential for MNA matrices whose diagonals can be
    zero (e.g. nodes with only V-sources and capacitors at DC).
    """

    n = matrix.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1

        for j in range(k, n):
            U[k, j] = matrix[k, j] - np.dot(L[k, :k], U[:k, j])

        if np.abs(U[k, k]) < 1e-18:
            return None, None

        for i in range(k + 1, n):
            L[i, k] = (matrix[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


def classify_krylov_matrix(A: object, symmetry_tol: float = 1e-10) -> str:
    """Classify a matrix for Krylov method selection.

    MNA systems are usually general and often indefinite, so Arnoldi/GMRES is
    the only universally valid Krylov route. Symmetric and positive-definite
    methods are still useful when the assembled matrix proves those properties.
    """

    if _is_sparse_matrix(A):
        matrix = A.tocsr()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return "general"
        diff = matrix - matrix.getH()
        if diff.nnz and float(np.max(np.abs(diff.data))) > symmetry_tol:
            return "general"
        if matrix.shape[0] <= 256:
            try:
                np.linalg.cholesky(matrix.toarray())
                return "positive_definite"
            except np.linalg.LinAlgError:
                pass
        return "symmetric"

    matrix = np.asarray(A)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return "general"
    if not np.allclose(matrix, matrix.conj().T, rtol=symmetry_tol, atol=symmetry_tol):
        return "general"
    try:
        np.linalg.cholesky(matrix)
        return "positive_definite"
    except np.linalg.LinAlgError:
        return "symmetric"


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


def _relative_residual(A: object, x: np.ndarray, b: np.ndarray) -> float:
    b_norm = norm(b)
    scale = b_norm if b_norm > 0 else 1.0
    return float(norm(b - A @ x) / scale)


def _as_krylov_inputs(A: object, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[object, np.ndarray, np.ndarray]:
    dtype = np.result_type(_matrix_dtype(A), b, x0 if x0 is not None else 0.0, np.float64)
    if np.issubdtype(_matrix_dtype(A), np.complexfloating) or np.iscomplexobj(b) or (x0 is not None and np.iscomplexobj(x0)):
        dtype = np.result_type(dtype, np.complex128)
    matrix = A.astype(dtype).tocsr() if _is_sparse_matrix(A) else np.asarray(A, dtype=dtype)
    rhs = np.asarray(b, dtype=dtype).reshape(-1)
    if x0 is None:
        guess = np.zeros_like(rhs)
    else:
        guess = np.asarray(x0, dtype=dtype).reshape(-1)
    return matrix, rhs, guess


def _restore_rhs_shape(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    rhs = np.asarray(b)
    if rhs.ndim == 2:
        return x.reshape(rhs.shape)
    return x


def _arnoldi_gmres(
    A: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    restart: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Restarted GMRES using the Arnoldi process for general matrices."""

    matrix, rhs, x = _as_krylov_inputs(A, b, x0)
    n = len(rhs)
    rhs_norm = norm(rhs)
    if rhs_norm == 0:
        return np.zeros_like(rhs), {
            "method": "arnoldi_gmres",
            "converged": True,
            "iterations": 0,
            "relative_residual": 0.0,
        }

    max_iter = max(1, int(max_iter))
    restart = max(1, min(int(restart) if restart else n, n))
    best_x = x.copy()
    best_residual = _relative_residual(matrix, x, rhs)
    if best_residual <= tol:
        return x, {
            "method": "arnoldi_gmres",
            "converged": True,
            "iterations": 0,
            "relative_residual": best_residual,
        }

    iterations = 0
    breakdown_tol = np.finfo(float).eps * 100
    while iterations < max_iter:
        r = rhs - matrix @ x
        beta = norm(r)
        if beta / rhs_norm <= tol:
            return x, {
                "method": "arnoldi_gmres",
                "converged": True,
                "iterations": iterations,
                "relative_residual": float(beta / rhs_norm),
            }

        m = min(restart, max_iter - iterations)
        V = np.zeros((n, m + 1), dtype=matrix.dtype)
        H = np.zeros((m + 1, m), dtype=matrix.dtype)
        V[:, 0] = r / beta
        candidate = x.copy()

        for j in range(m):
            w = matrix @ V[:, j]
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = norm(w)
            if abs(H[j + 1, j]) > breakdown_tol:
                V[:, j + 1] = w / H[j + 1, j]

            target = np.zeros(j + 2, dtype=matrix.dtype)
            target[0] = beta
            y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], target, rcond=None)
            candidate = x + V[:, : j + 1] @ y
            residual = _relative_residual(matrix, candidate, rhs)
            iterations += 1

            if residual < best_residual:
                best_residual = residual
                best_x = candidate.copy()
            if residual <= tol:
                return candidate, {
                    "method": "arnoldi_gmres",
                    "converged": True,
                    "iterations": iterations,
                    "relative_residual": residual,
                }
            if abs(H[j + 1, j]) <= breakdown_tol:
                x = candidate
                break
        else:
            x = candidate

    return best_x, {
        "method": "arnoldi_gmres",
        "converged": False,
        "iterations": iterations,
        "relative_residual": best_residual,
    }


def _conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conjugate Gradient for Hermitian positive-definite matrices."""

    matrix, rhs, x = _as_krylov_inputs(A, b, x0)
    rhs_norm = norm(rhs)
    if rhs_norm == 0:
        return np.zeros_like(rhs), {
            "method": "conjugate_gradient",
            "converged": True,
            "iterations": 0,
            "relative_residual": 0.0,
        }

    r = rhs - matrix @ x
    p = r.copy()
    rr = np.vdot(r, r)
    best_x = x.copy()
    best_residual = _relative_residual(matrix, x, rhs)
    denom_tol = np.finfo(float).eps * 100

    for iteration in range(1, max(1, int(max_iter)) + 1):
        Ap = matrix @ p
        denom = np.vdot(p, Ap)
        if abs(denom) <= denom_tol:
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        residual = _relative_residual(matrix, x, rhs)
        if residual < best_residual:
            best_residual = residual
            best_x = x.copy()
        if residual <= tol:
            return x, {
                "method": "conjugate_gradient",
                "converged": True,
                "iterations": iteration,
                "relative_residual": residual,
            }
        rr_next = np.vdot(r, r)
        if abs(rr) <= denom_tol:
            break
        p = r + (rr_next / rr) * p
        rr = rr_next

    return best_x, {
        "method": "conjugate_gradient",
        "converged": False,
        "iterations": max(1, int(max_iter)),
        "relative_residual": best_residual,
    }


def _conjugate_residual(
    A: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conjugate Residual for Hermitian/symmetric indefinite matrices."""

    matrix, rhs, x = _as_krylov_inputs(A, b, x0)
    rhs_norm = norm(rhs)
    if rhs_norm == 0:
        return np.zeros_like(rhs), {
            "method": "conjugate_residual",
            "converged": True,
            "iterations": 0,
            "relative_residual": 0.0,
        }

    r = rhs - matrix @ x
    p = r.copy()
    Ap = matrix @ p
    best_x = x.copy()
    best_residual = _relative_residual(matrix, x, rhs)
    denom_tol = np.finfo(float).eps * 100

    for iteration in range(1, max(1, int(max_iter)) + 1):
        denom = np.vdot(Ap, Ap)
        if abs(denom) <= denom_tol:
            break
        alpha = np.vdot(r, Ap) / denom
        x = x + alpha * p
        r_next = r - alpha * Ap
        residual = _relative_residual(matrix, x, rhs)
        if residual < best_residual:
            best_residual = residual
            best_x = x.copy()
        if residual <= tol:
            return x, {
                "method": "conjugate_residual",
                "converged": True,
                "iterations": iteration,
                "relative_residual": residual,
            }
        Ar_next = matrix @ r_next
        beta = np.vdot(Ar_next, Ap) / denom
        p = r_next - beta * p
        Ap = Ar_next - beta * Ap
        r = r_next

    return best_x, {
        "method": "conjugate_residual",
        "converged": False,
        "iterations": max(1, int(max_iter)),
        "relative_residual": best_residual,
    }


def _scipy_krylov_solve(
    A: object,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    route: str,
    resolved_rank: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Use SciPy's native Krylov kernels when they match the selected route."""

    if route == "general" and _scipy_gmres is None:
        return None
    if route == "positive_definite" and _scipy_cg is None:
        return None
    if route == "symmetric" and _scipy_minres is None:
        return None

    matrix, rhs, guess = _as_krylov_inputs(A, b, x0)
    if matrix.shape[0] < 64:
        return None
    if route == "symmetric" and np.issubdtype(_matrix_dtype(matrix), np.complexfloating):
        return None
    operator = matrix.tocsr() if _is_sparse_matrix(matrix) else _scipy_csr_matrix(matrix) if _scipy_csr_matrix is not None else matrix
    iterations = 0

    def count_iteration(_value) -> None:
        nonlocal iterations
        iterations += 1

    if route == "positive_definite":
        solution, exit_code = _scipy_cg(
            operator,
            rhs,
            x0=guess,
            rtol=tol,
            atol=0.0,
            maxiter=max(1, int(resolved_rank)),
            callback=count_iteration,
        )
        method = "conjugate_gradient"
        iteration_budget = resolved_rank
        effective_restart = None
    elif route == "symmetric":
        solution, exit_code = _scipy_minres(
            operator,
            rhs,
            x0=guess,
            rtol=tol,
            maxiter=max(1, int(resolved_rank)),
            callback=count_iteration,
            check=False,
        )
        method = "minres"
        iteration_budget = resolved_rank
        effective_restart = None
    else:
        restart = max(1, min(int(resolved_rank), matrix.shape[0]))
        solution, exit_code = _scipy_gmres(
            operator,
            rhs,
            x0=guess,
            rtol=tol,
            atol=0.0,
            restart=restart,
            maxiter=max(1, int(max_iter)),
            callback=count_iteration,
            callback_type="pr_norm",
        )
        method = "arnoldi_gmres"
        iteration_budget = max_iter
        effective_restart = restart

    residual = _relative_residual(matrix, solution, rhs)
    return _restore_rhs_shape(solution, b), {
        "method": method,
        "engine": "scipy_sparse_linalg",
        "operator_storage": _operator_storage(operator),
        "matrix_nnz": _matrix_nnz(operator),
        "matrix_density": _matrix_density(operator),
        "converged": exit_code == 0,
        "iterations": int(iterations),
        "relative_residual": residual,
        "iteration_budget": int(iteration_budget),
        "effective_restart": effective_restart,
        "exit_code": int(exit_code),
    }


def krylov_solve_linear(
    A: object,
    b: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 2000,
    restart: int = 80,
    rank: int | str | None = "auto",
    method: str = "auto",
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve ``A x = b`` with an auto-selected Krylov subspace method."""

    matrix = A.tocsr() if _is_sparse_matrix(A) else np.asarray(A)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise np.linalg.LinAlgError("Krylov solve requires a square matrix.")
    matrix_kind = classify_krylov_matrix(matrix)
    route = _krylov_route(matrix_kind, method)
    resolved_rank, rank_mode = resolve_krylov_rank(matrix.shape[0], restart if rank is None else rank, restart)

    scipy_result = _scipy_krylov_solve(matrix, b, tol, max_iter, route, resolved_rank, x0=x0)
    if scipy_result is not None:
        solution, info = scipy_result
    elif route == "positive_definite":
        solution, info = _conjugate_gradient(matrix, b, tol=tol, max_iter=resolved_rank, x0=x0)
    elif route == "symmetric":
        solution, info = _conjugate_residual(matrix, b, tol=tol, max_iter=resolved_rank, x0=x0)
    else:
        solution, info = _arnoldi_gmres(matrix, b, tol=tol, max_iter=max_iter, restart=resolved_rank, x0=x0)

    info = dict(info)
    info["requested_method"] = method
    info["matrix_kind"] = matrix_kind
    info["route"] = route
    info["matrix_dimension"] = int(matrix.shape[0])
    info["matrix_nnz"] = info.get("matrix_nnz", _matrix_nnz(matrix))
    info["matrix_density"] = info.get("matrix_density", _matrix_density(matrix))
    info["operator_storage"] = info.get("operator_storage", _operator_storage(matrix))
    info["resolved_rank"] = resolved_rank
    info["rank_mode"] = rank_mode
    info["iteration_budget"] = info.get("iteration_budget", int(max_iter) if route == "general" else resolved_rank)
    if route == "general":
        info["effective_restart"] = info.get("effective_restart", min(resolved_rank, int(matrix.shape[0])))
    return _restore_rhs_shape(solution, b), info


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
    """Solve ``A x = b`` with partial pivoting (LAPACK ``gesv``).

    MNA matrices routinely have zero diagonal entries — any node touching only
    voltage sources and capacitors will have ``G[node, node] == 0`` at DC, and
    a fixed-pivot LU breaks immediately on it. This helper delegates to
    :func:`numpy.linalg.solve`, which performs partial pivoting internally,
    and re-raises a clear ``LinAlgError`` when the system is genuinely
    singular.
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

    if _is_sparse_matrix(A):
        return np.linalg.solve(A.toarray(), b)
    return np.linalg.solve(A, b)


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L*y=b for lower triangular L."""

    n = L.shape[0]
    y = np.zeros((n, 1), dtype=b.dtype)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve U*x=y for upper triangular U."""

    n = U.shape[0]
    x = np.zeros((n, 1), dtype=y.dtype)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def _sympy_locals(t_symbol: sp.Symbol | None = None) -> dict[str, object]:
    """Symbols/functions allowed in generated nonlinear and time expressions."""

    locals_map: dict[str, object] = {
        "sp": sp,
        "np": sp,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "Heaviside": sp.Heaviside,
        "heaviside": sp.Heaviside,
        "Piecewise": sp.Piecewise,
        "Mod": sp.Mod,
        "mod": sp.Mod,
        "Min": sp.Min,
        "Max": sp.Max,
        "pi": sp.pi,
        "True": True,
        "False": False,
    }
    if t_symbol is not None:
        locals_map["t"] = t_symbol
    return locals_map


def _prepare_sympy_expression(expr: str) -> str:
    """Normalize numpy-style generated expressions for SymPy parsing."""

    prepared = str(expr)
    replacements = (
        ("np.heaviside", "Heaviside"),
        ("np.Heaviside", "Heaviside"),
        ("np.mod", "Mod"),
        ("np.Mod", "Mod"),
        ("np.minimum", "Min"),
        ("np.maximum", "Max"),
        ("sp.heaviside", "Heaviside"),
        ("sp.Heaviside", "Heaviside"),
        ("sp.mod", "Mod"),
        ("sp.Mod", "Mod"),
        ("sp.minimum", "Min"),
        ("sp.maximum", "Max"),
    )
    for old, new in replacements:
        prepared = prepared.replace(old, new)
    prepared = prepared.replace("np.", "").replace("sp.", "")
    prepared = re.sub(r"\b(Min|Max)\(\[(.*?)\]\)", r"\1(\2)", prepared)
    return prepared


def _nmos_level1_forward(
    vgs: float,
    vds: float,
    beta: float,
    vth: float,
    lambda_: float,
) -> tuple[float, float, float]:
    """Return IDS, dIDS/dVGS, dIDS/dVDS for an NMOS with VDS >= 0."""

    if vgs <= vth:
        return 0.0, 0.0, 0.0
    vov = vgs - vth
    if vds < vov:
        channel = vov * vds - 0.5 * vds * vds
        modulation = 1.0 + lambda_ * vds
        current = beta * channel * modulation
        d_vgs = beta * vds * modulation
        d_vds = beta * ((vov - vds) * modulation + channel * lambda_)
        return current, d_vgs, d_vds
    current = 0.5 * beta * vov * vov * (1.0 + lambda_ * vds)
    d_vgs = beta * vov * (1.0 + lambda_ * vds)
    d_vds = 0.5 * beta * vov * vov * lambda_
    return current, d_vgs, d_vds


def _nmos_level1_current_and_derivatives(
    vd: float,
    vg: float,
    vs: float,
    beta: float,
    vth: float,
    lambda_: float,
) -> tuple[float, float, float, float]:
    """Return drain-to-source current and derivatives for a symmetric NMOS."""

    if vd >= vs:
        current, d_vgs, d_vds = _nmos_level1_forward(vg - vs, vd - vs, beta, vth, lambda_)
        return current, d_vds, d_vgs, -d_vgs - d_vds

    reverse_current, d_vgs, d_vds = _nmos_level1_forward(vg - vd, vs - vd, beta, vth, lambda_)
    return -reverse_current, d_vgs + d_vds, -d_vgs, -d_vds


def _level1_mos_current_and_derivatives(
    device: dict[str, object],
    vd: float,
    vg: float,
    vs: float,
) -> tuple[float, float, float, float]:
    beta = float(device["beta"])
    vth = float(device["vth"])
    lambda_ = float(device["lambda"])
    if str(device["type"]).upper() == "PMOS":
        current, d_d, d_g, d_s = _nmos_level1_current_and_derivatives(-vd, -vg, -vs, beta, vth, lambda_)
        return -current, d_d, d_g, d_s
    return _nmos_level1_current_and_derivatives(vd, vg, vs, beta, vth, lambda_)


def _evaluate_level1_mos_devices(
    devices: list[dict[str, object]],
    size: int,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f = np.zeros((size, 1), dtype=float)
    jacobian = np.zeros((size, size), dtype=float)

    def voltage(index: int) -> float:
        return 0.0 if index == -1 else float(x_values[index])

    def stamp(row: int, current_sign: float, terminals: tuple[int, int, int], derivatives: tuple[float, float, float]) -> None:
        if row == -1:
            return
        for col, derivative in zip(terminals, derivatives):
            if col != -1 and derivative != 0.0:
                jacobian[row, col] += current_sign * derivative

    for device in devices:
        drain = int(device["drain"])
        gate = int(device["gate"])
        source = int(device["source"])
        current, d_drain, d_gate, d_source = _level1_mos_current_and_derivatives(
            device,
            voltage(drain),
            voltage(gate),
            voltage(source),
        )
        terminals = (drain, gate, source)
        derivatives = (d_drain, d_gate, d_source)
        if drain != -1:
            f[drain, 0] += current
        if source != -1:
            f[source, 0] -= current
        stamp(drain, 1.0, terminals, derivatives)
        stamp(source, -1.0, terminals, derivatives)

    return f, jacobian


def _bjt_level1_npn_currents_and_derivatives(
    vc: float,
    vb: float,
    ve: float,
    is_: float,
    bf: float,
    br: float,
    vaf: float,
    var: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Return NPN terminal currents and dI/d(Vc,Vb,Ve).

    Currents are positive when leaving the external terminal into the compact
    device. This Ebers-Moll-lite model is intentionally small: transport
    currents, finite forward/reverse beta, and simple Early modulation.
    """

    vt = 0.02585
    alpha_f = bf / (bf + 1.0) if bf > 0.0 else 0.0
    alpha_r = br / (br + 1.0) if br > 0.0 else 0.0
    vbe = vb - ve
    vbc = vb - vc
    vce = vc - ve
    vec = ve - vc

    exp_be = np.exp(np.clip(vbe / vt, -80.0, 80.0))
    exp_bc = np.exp(np.clip(vbc / vt, -80.0, 80.0))
    ibe = is_ * (exp_be - 1.0)
    ibc = is_ * (exp_bc - 1.0)
    gbe = is_ * exp_be / vt
    gbc = is_ * exp_bc / vt

    fwd_mod = 1.0 + vce / vaf if vaf > 0.0 else 1.0
    rev_mod = 1.0 + vec / var if var > 0.0 else 1.0
    df_dvc = 1.0 / vaf if vaf > 0.0 else 0.0
    df_dve = -df_dvc
    dr_dvc = -1.0 / var if var > 0.0 else 0.0
    dr_dve = -dr_dvc

    ic = alpha_f * ibe * fwd_mod - ibc * rev_mod
    ib = (1.0 - alpha_f) * ibe + (1.0 - alpha_r) * ibc * rev_mod
    ie = -(ic + ib)

    dic_dvc = alpha_f * ibe * df_dvc + gbc * rev_mod - ibc * dr_dvc
    dic_dvb = alpha_f * gbe * fwd_mod - gbc * rev_mod
    dic_dve = alpha_f * (-gbe * fwd_mod + ibe * df_dve) - ibc * dr_dve

    dib_dvc = (1.0 - alpha_r) * (-gbc * rev_mod + ibc * dr_dvc)
    dib_dvb = (1.0 - alpha_f) * gbe + (1.0 - alpha_r) * gbc * rev_mod
    dib_dve = -(1.0 - alpha_f) * gbe + (1.0 - alpha_r) * ibc * dr_dve

    die_dvc = -(dic_dvc + dib_dvc)
    die_dvb = -(dic_dvb + dib_dvb)
    die_dve = -(dic_dve + dib_dve)

    currents = (ic, ib, ie)
    derivatives = (
        (dic_dvc, dic_dvb, dic_dve),
        (dib_dvc, dib_dvb, dib_dve),
        (die_dvc, die_dvb, die_dve),
    )
    return currents, derivatives


def _level1_bjt_currents_and_derivatives(
    device: dict[str, object],
    vc: float,
    vb: float,
    ve: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    is_ = float(device["is"])
    bf = float(device["bf"])
    br = float(device["br"])
    vaf = float(device["vaf"])
    var = float(device["var"])
    if str(device["type"]).upper() == "QPNP":
        currents, derivatives = _bjt_level1_npn_currents_and_derivatives(-vc, -vb, -ve, is_, bf, br, vaf, var)
        return tuple(-current for current in currents), derivatives
    return _bjt_level1_npn_currents_and_derivatives(vc, vb, ve, is_, bf, br, vaf, var)


def _evaluate_level1_bjt_devices(
    devices: list[dict[str, object]],
    size: int,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f = np.zeros((size, 1), dtype=float)
    jacobian = np.zeros((size, size), dtype=float)

    def voltage(index: int) -> float:
        return 0.0 if index == -1 else float(x_values[index])

    def add_row(row: int, current: float, terminals: tuple[int, int, int], derivatives: tuple[float, float, float]) -> None:
        if row == -1:
            return
        f[row, 0] += current
        for col, derivative in zip(terminals, derivatives):
            if col != -1 and derivative != 0.0:
                jacobian[row, col] += derivative

    for device in devices:
        collector = int(device["collector"])
        base = int(device["base"])
        emitter = int(device["emitter"])
        currents, derivatives = _level1_bjt_currents_and_derivatives(
            device,
            voltage(collector),
            voltage(base),
            voltage(emitter),
        )
        terminals = (collector, base, emitter)
        for row, current, row_derivatives in zip(terminals, currents, derivatives):
            add_row(row, current, terminals, row_derivatives)

    return f, jacobian


def compile_nl_functions(
    f_str_vector: np.ndarray,
    size: int,
    level1_mos_devices: list[dict[str, object]] | None = None,
    level1_bjt_devices: list[dict[str, object]] | None = None,
):
    """Compile nonlinear current expressions and their Jacobian."""

    mos_devices = list(level1_mos_devices or [])
    bjt_devices = list(level1_bjt_devices or [])
    has_symbolic = any(expr != "0" for expr in f_str_vector.flatten())
    symbolic_f_func = None
    symbolic_j_func = None

    if has_symbolic:
        x_vars = sp.symbols(f"x_0:{size}")
        expressions: list[sp.Expr] = []

        for expr_str in f_str_vector.flatten():
            expression = str(expr_str)
            for i in range(size):
                expression = expression.replace(f"x[{i}]", f"x_{i}")
            expressions.append(sp.sympify(_prepare_sympy_expression(expression), locals=_sympy_locals()))

        f_sym = sp.Matrix(expressions)
        J_sym = f_sym.jacobian(x_vars)
        symbolic_f_func = sp.lambdify(x_vars, f_sym, "numpy")
        symbolic_j_func = sp.lambdify(x_vars, J_sym, "numpy")

    def f_func(*args):
        if symbolic_f_func is None:
            f = np.zeros((size, 1), dtype=float)
        else:
            f = np.asarray(symbolic_f_func(*args), dtype=float).reshape(size, 1)
        x_values = np.asarray(args, dtype=float)
        if mos_devices:
            f += _evaluate_level1_mos_devices(mos_devices, size, x_values)[0]
        if bjt_devices:
            f += _evaluate_level1_bjt_devices(bjt_devices, size, x_values)[0]
        return f

    def j_func(*args):
        if symbolic_j_func is None:
            jacobian = np.zeros((size, size), dtype=float)
        else:
            jacobian = np.asarray(symbolic_j_func(*args), dtype=float).reshape(size, size)
        x_values = np.asarray(args, dtype=float)
        if mos_devices:
            jacobian += _evaluate_level1_mos_devices(mos_devices, size, x_values)[1]
        if bjt_devices:
            jacobian += _evaluate_level1_bjt_devices(bjt_devices, size, x_values)[1]
        return jacobian

    return f_func, j_func


def compile_time_source_func(b_time_str_vector: np.ndarray):
    """Compile time-domain source expressions into a callable."""

    size = b_time_str_vector.shape[0]
    t = sp.symbols("t")
    expressions: list[sp.Expr] = []

    for expr_str in b_time_str_vector.flatten():
        expr = _prepare_sympy_expression(str(expr_str))
        expressions.append(sp.sympify(expr, locals=_sympy_locals(t)))

    compiled = sp.lambdify(t, sp.Matrix(expressions), "numpy")

    def evaluate(t_value: float) -> np.ndarray:
        return np.asarray(compiled(t_value), dtype=float).reshape(size, 1)

    return evaluate


def is_nonlinear(f_str_vector: np.ndarray) -> bool:
    """Return True when the symbolic nonlinear vector is non-zero."""

    return any(expr != "0" for expr in f_str_vector.flatten())


def _transient_time_points(t_stop: float, t_step: float) -> np.ndarray:
    """Return monotonic transient sample points ending exactly at ``t_stop``."""

    if t_stop < 0:
        error_handler("TRAN_ERROR: t_stop must be non-negative.")
    if t_step <= 0:
        error_handler("TRAN_ERROR: t_step must be positive.")
    if t_stop == 0:
        return np.array([0.0], dtype=float)

    tolerance = max(abs(t_stop), abs(t_step), np.finfo(float).eps) * 1e-9
    points = [0.0]
    current = 0.0
    while current < t_stop - tolerance:
        next_time = current + t_step
        current = t_stop if next_time >= t_stop - tolerance else next_time
        if current <= points[-1]:
            error_handler("TRAN_ERROR: transient time grid failed to advance.")
        points.append(float(current))
    if points[-1] != t_stop:
        points[-1] = float(t_stop)
    return np.asarray(points, dtype=float)


def _finite_vector_norm(vector: np.ndarray) -> float:
    if not np.all(np.isfinite(vector)):
        return np.inf
    return float(np.linalg.norm(vector))


def _damped_newton_step(
    x: np.ndarray,
    delta_x: np.ndarray,
    current_norm: float,
    residual_at: Callable[[np.ndarray], np.ndarray],
    max_trials: int = 28,
) -> tuple[np.ndarray, float, float]:
    """Pick a finite damped Newton step without committing the full step first."""

    best_x = x
    best_norm = current_norm
    best_scale = 0.0
    scale = 1.0
    for _ in range(max_trials):
        trial = x + delta_x * scale
        try:
            trial_norm = _finite_vector_norm(residual_at(trial))
        except (FloatingPointError, OverflowError, ValueError):
            trial_norm = np.inf
        if np.isfinite(trial_norm) and trial_norm < best_norm:
            best_x = trial
            best_norm = trial_norm
            best_scale = scale
        if np.isfinite(trial_norm) and trial_norm <= current_norm:
            return trial, trial_norm, scale
        scale *= 0.5

    if best_scale > 0.0:
        return best_x, best_norm, best_scale
    return x, current_norm, 0.0


def solve_dc_nr(
    G: np.ndarray,
    f_str_vector: np.ndarray,
    b: np.ndarray,
    max_iter: int = 10000,
    v_tol: float = 1e-9,
    f_tol: float = 1e-9,
    init_cond: np.ndarray | str = "DEFAULT",
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
    level1_mos_devices: list[dict[str, object]] | None = None,
    level1_bjt_devices: list[dict[str, object]] | None = None,
) -> np.ndarray:
    """Solve the DC operating point using linear solve or Newton-Raphson."""

    size = G.shape[0]
    if not is_nonlinear(f_str_vector) and not level1_mos_devices and not level1_bjt_devices:
        try:
            return solve_linear(
                G,
                b,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler(f"NR_CONVERGENCE: Linear system is singular (G matrix): {exc}")
            raise

    f_func, Jf_func = compile_nl_functions(
        f_str_vector,
        size,
        level1_mos_devices=level1_mos_devices,
        level1_bjt_devices=level1_bjt_devices,
    )
    if isinstance(init_cond, str):
        if init_cond == "DEFAULT":
            x = np.zeros((size, 1))
        elif init_cond == "RANDOM":
            x = np.random.uniform(-1.0, 1.0, size=(size, 1))
        else:
            error_handler(f"NETLIST_FATAL: Unsupported init condition '{init_cond}'.")
    else:
        x = init_cond.copy()

    for iteration in range(max_iter):
        x_flat = x.flatten()
        f_k = np.asarray(f_func(*x_flat), dtype=float).reshape(size, 1)
        F_k = (G @ x) + f_k - b

        norm_F_now = _finite_vector_norm(F_k)
        if norm_F_now < f_tol:
            return x

        Jf_k = np.asarray(Jf_func(*x_flat), dtype=float)
        J_k = G + Jf_k

        try:
            delta_x = solve_linear(
                J_k,
                -F_k,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler(f"JACOBIAN_SINGULAR: Jacobian is singular at iteration {iteration}.")
            raise exc

        def residual_at(trial: np.ndarray) -> np.ndarray:
            trial_flat = trial.flatten()
            f_trial = np.asarray(f_func(*trial_flat), dtype=float).reshape(size, 1)
            return (G @ trial) + f_trial - b

        x_next, next_norm, step_scale = _damped_newton_step(x, delta_x, norm_F_now, residual_at)
        accepted_delta = x_next - x
        x = x_next
        if next_norm < f_tol:
            return x
        if step_scale == 0.0:
            error_handler(f"NR_CONVERGENCE: Newton step could not reduce residual at iteration {iteration}.")
        if np.linalg.norm(accepted_delta) < v_tol:
            return x

    error_handler(f"NR_CONVERGENCE: Solver did not converge after {max_iter} iterations.")


def solve_transient(
    problem: MnaProblem,
    t_stop: float,
    t_step: float,
    max_iter: int = 10000,
    v_tol: float = 1e-9,
    f_tol: float = 1e-9,
    init_cond: np.ndarray | None = None,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
) -> WaveformResult:
    """Solve transient response with backward Euler."""

    size = problem.G.shape[0]
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    has_nonlinear_terms = is_nonlinear(solver_f_str) or bool(level1_devices) or bool(level1_bjt_devices)
    if has_nonlinear_terms:
        f_func, Jf_func = compile_nl_functions(
            solver_f_str,
            size,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
    else:
        f_func = lambda *args: np.zeros((size, 1))  # noqa: E731
        Jf_func = lambda *args: np.zeros((size, size))  # noqa: E731

    b_time_func = compile_time_source_func(problem.b_time_str)

    if init_cond is None:
        b_t0 = problem.b_dc + b_time_func(0.0)
        x_prev = solve_dc_nr(
            problem.G,
            solver_f_str,
            b_t0,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            use_krylov=use_krylov,
            krylov_tol=krylov_tol,
            krylov_max_iter=krylov_max_iter,
            krylov_restart=krylov_restart,
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            krylov_stats=krylov_stats,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
    else:
        x_prev = init_cond.copy()

    t_points = _transient_time_points(t_stop, t_step)
    values = np.zeros((size, len(t_points)))
    values[:, 0] = x_prev.flatten()
    sparse_linear_krylov = (
        use_krylov
        and not has_nonlinear_terms
        and problem.G_sparse is not None
        and problem.C_sparse is not None
    )
    cached_step_size: float | None = None
    cached_C_h_sparse = None
    cached_G_be_sparse = None

    for step in range(1, len(t_points)):
        step_size = float(t_points[step] - t_points[step - 1])
        x_k = x_prev.copy()
        b_now = problem.b_dc + b_time_func(float(t_points[step]))

        if sparse_linear_krylov:
            if cached_step_size != step_size:
                cached_step_size = step_size
                cached_C_h_sparse = problem.C_sparse / step_size
                cached_G_be_sparse = problem.G_sparse + cached_C_h_sparse
            b_be = (cached_C_h_sparse @ x_prev) + b_now
            x_next = solve_linear(
                cached_G_be_sparse,
                b_be,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
                x0=x_prev,
            )
            values[:, step] = x_next.flatten()
            x_prev = x_next
            continue

        C_h = problem.C / step_size
        G_be = problem.G + C_h
        b_be = (C_h @ x_prev) + b_now

        if not has_nonlinear_terms:
            x_next = solve_linear(
                G_be,
                b_be,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
                x0=x_prev,
            )
            values[:, step] = x_next.flatten()
            x_prev = x_next
            continue

        converged = False

        for iteration in range(max_iter):
            x_flat = x_k.flatten()
            f_k = np.asarray(f_func(*x_flat), dtype=float).reshape(size, 1)
            F_k = (G_be @ x_k) + f_k - b_be

            norm_F = _finite_vector_norm(F_k)
            if norm_F < f_tol:
                converged = True
                break

            Jf_k = np.asarray(Jf_func(*x_flat), dtype=float)
            J_k = G_be + Jf_k

            try:
                delta_x = solve_linear(
                    J_k,
                    -F_k,
                    use_krylov=use_krylov,
                    krylov_tol=krylov_tol,
                    krylov_max_iter=krylov_max_iter,
                    krylov_restart=krylov_restart,
                    krylov_rank=krylov_rank,
                    krylov_method=krylov_method,
                    krylov_stats=krylov_stats,
                )
            except np.linalg.LinAlgError as exc:
                error_handler(f"JACOBIAN_SINGULAR: Jacobian singular at t={t_points[step]:.2e}, iter={iteration}")
                raise exc

            def residual_at(trial: np.ndarray) -> np.ndarray:
                trial_flat = trial.flatten()
                f_trial = np.asarray(f_func(*trial_flat), dtype=float).reshape(size, 1)
                return (G_be @ trial) + f_trial - b_be

            x_next, next_norm, step_scale = _damped_newton_step(x_k, delta_x, norm_F, residual_at)
            accepted_delta = x_next - x_k
            x_k = x_next
            if next_norm < f_tol:
                converged = True
                break
            if step_scale == 0.0:
                error_handler(f"NR_CONVERGENCE: Newton step could not reduce residual at t={t_points[step]:.2e}, iter={iteration}")
            if np.linalg.norm(accepted_delta) < v_tol:
                converged = True
                break

        if not converged:
            error_handler(f"NR_CONVERGENCE: NR failed to converge at t={t_points[step]:.2e} s")

        values[:, step] = x_k.flatten()
        x_prev = x_k

    return WaveformResult(time=t_points, values=values, labels=problem.metadata["labels"])


def solve_ac(
    problem: MnaProblem,
    f_start: float,
    f_stop: float,
    points: int,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
) -> SpectrumResult:
    """Solve linear small-signal AC response over a frequency sweep."""

    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    spectrum_metadata: dict[str, object] = {}
    if is_nonlinear(solver_f_str) or level1_devices or level1_bjt_devices:
        dc_solution = solve_dc_nr(
            problem.G,
            solver_f_str,
            problem.b_dc,
            use_krylov=use_krylov,
            krylov_tol=krylov_tol,
            krylov_max_iter=krylov_max_iter,
            krylov_restart=krylov_restart,
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            krylov_stats=krylov_stats,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
        f_func, Jf_func = compile_nl_functions(
            solver_f_str,
            problem.G.shape[0],
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
        jacobian = np.asarray(Jf_func(*dc_solution.flatten()), dtype=float)
        G_linearized = problem.G + jacobian
        spectrum_metadata["op_used"] = True
        spectrum_metadata["operating_point"] = {
            "labels": problem.metadata["labels"],
            "values": np.asarray(dc_solution).flatten().tolist(),
        }
        spectrum_metadata["linearized_device_count"] = len(level1_devices) + len(level1_bjt_devices)
    else:
        G_linearized = problem.G
        spectrum_metadata["op_used"] = False
        spectrum_metadata["linearized_device_count"] = 0

    frequencies = np.linspace(f_start, f_stop, points)
    responses = np.zeros((problem.G.shape[0], points))

    for index, frequency in enumerate(frequencies):
        omega = 2 * np.pi * frequency
        system = G_linearized + 1j * omega * problem.C
        try:
            solution = solve_linear(
                system,
                problem.b_ac,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler(f"AC_ERROR: AC solve failed at frequency {frequency:.3e} Hz.")
            raise exc
        responses[:, index] = np.abs(solution.flatten())

    return SpectrumResult(
        frequencies=frequencies,
        magnitudes=responses,
        labels=problem.metadata["labels"],
        metadata=spectrum_metadata,
    )


def create_fourier_matrices(n_harmonics: int, time_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return DFT and inverse DFT matrices for real-valued harmonics."""

    S = 2 * n_harmonics + 1
    K = len(time_points)
    Gamma = np.zeros((S, K))
    Gamma_inv = np.zeros((K, S))

    for k in range(K):
        Gamma_inv[k, 0] = 1.0
        for h in range(1, n_harmonics + 1):
            angle = 2 * np.pi * h * k / K
            Gamma_inv[k, 2 * h - 1] = np.cos(angle)
            Gamma_inv[k, 2 * h] = np.sin(angle)

    Gamma[0, :] = 1.0 / K
    for h in range(1, n_harmonics + 1):
        for k in range(K):
            angle = 2 * np.pi * h * k / K
            Gamma[2 * h - 1, k] = (2.0 / K) * np.cos(angle)
            Gamma[2 * h, k] = (2.0 / K) * np.sin(angle)

    return Gamma, Gamma_inv


def reconstruct_time_domain(x_bar: np.ndarray, num_vars: int, H: int, omega_0: float, t_array: np.ndarray) -> np.ndarray:
    """Reconstruct time-domain waveform from harmonic coefficients."""

    freq_dim = 2 * H + 1
    results = np.zeros((num_vars, len(t_array)))
    reshaped = x_bar.reshape((num_vars, freq_dim))

    for i in range(num_vars):
        values = np.full_like(t_array, reshaped[i, 0], dtype=float)
        for h in range(1, H + 1):
            a_h = reshaped[i, 2 * h - 1]
            b_h = reshaped[i, 2 * h]
            values += a_h * np.cos(h * omega_0 * t_array) + b_h * np.sin(h * omega_0 * t_array)
        results[i, :] = values

    return results


def solve_harmonic_balance(
    problem: MnaProblem,
    omega_0: float,
    H: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
) -> np.ndarray:
    """Solve the harmonic balance system."""

    num_vars = problem.G.shape[0]
    freq_dim = 2 * H + 1
    total_dim = num_vars * freq_dim

    G_bar = np.kron(problem.G, np.eye(freq_dim))
    C_bar = np.kron(problem.C, np.eye(freq_dim))

    Omega_block = np.zeros((freq_dim, freq_dim))
    for h in range(1, H + 1):
        w_h = h * omega_0
        idx_cos = 2 * h - 1
        idx_sin = 2 * h
        Omega_block[idx_cos, idx_sin] = w_h
        Omega_block[idx_sin, idx_cos] = -w_h

    Omega_bar = np.kron(np.eye(num_vars), Omega_block)
    Y_bar = G_bar + (C_bar @ Omega_bar)

    K_samples = 2 * H + 1
    period = 2 * np.pi / omega_0
    t_points = np.linspace(0, period, K_samples, endpoint=False)
    Gamma, Gamma_inv = create_fourier_matrices(H, t_points)

    b_bar = np.zeros((total_dim, 1))
    for index in range(num_vars):
        base = index * freq_dim
        b_bar[base] = problem.b_dc[index]

    b_time_func = compile_time_source_func(problem.b_time_str)
    b_time_matrix = np.zeros((num_vars, K_samples))
    for k, t_value in enumerate(t_points):
        b_time_matrix[:, k] = b_time_func(float(t_value)).flatten()
    for index in range(num_vars):
        base = index * freq_dim
        b_bar[base : base + freq_dim, 0] += Gamma @ b_time_matrix[index, :]

    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    if not is_nonlinear(solver_f_str) and not level1_devices and not level1_bjt_devices:
        try:
            return solve_linear(
                Y_bar,
                b_bar,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler("HB_ERROR: Linear harmonic solve failed.")
            raise exc

    f_func, Jf_func = compile_nl_functions(
        solver_f_str,
        num_vars,
        level1_mos_devices=level1_devices,
        level1_bjt_devices=level1_bjt_devices,
    )
    x_bar = np.zeros((total_dim, 1))

    for _ in range(max_iter):
        x_reshaped = x_bar.reshape((num_vars, freq_dim))
        x_time = x_reshaped @ Gamma_inv.T

        i_nl_time = np.zeros((num_vars, K_samples))
        G_nl_time = np.zeros((num_vars, num_vars, K_samples))

        for k in range(K_samples):
            v_inst = x_time[:, k]
            i_nl_time[:, k] = np.asarray(f_func(*v_inst), dtype=float).flatten()
            G_nl_time[:, :, k] = np.asarray(Jf_func(*v_inst), dtype=float)

        f_bar = (i_nl_time @ Gamma.T).reshape((total_dim, 1))
        residual = (Y_bar @ x_bar) + f_bar - b_bar

        if np.linalg.norm(residual) < tol:
            return x_bar

        J_nonlinear = np.zeros((total_dim, total_dim))
        for r in range(num_vars):
            for c in range(num_vars):
                g_waveform = G_nl_time[r, c, :]
                weighted = g_waveform[:, None] * Gamma_inv
                block = Gamma @ weighted
                r_idx = r * freq_dim
                c_idx = c * freq_dim
                J_nonlinear[r_idx : r_idx + freq_dim, c_idx : c_idx + freq_dim] = block

        try:
            delta_x = solve_linear(
                Y_bar + J_nonlinear,
                -residual,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler("HB_ERROR: Singular HB Jacobian.")
            raise exc

        x_bar = x_bar + delta_x

    error_handler("HB_ERROR: Harmonic balance solver did not converge within limit.")
