"""Native linear algebra (``mna_simulation.native.linalg`` over ``_core``) against the frozen oracle.

Covers ``classify_krylov_matrix``, ``krylov_solve_linear`` and ``solve_linear``
on the matrices of ``tests/test_krylov_solvers.py`` and on the operators
``G + C/h`` and ``G + j*omega*C`` of every linear corpus case (built with
``mna_reference.mna_builder.build_mna_problem``, including the 1035-unknown RLC
mesh), plus the binding surface (``direct_solve``, ``krylov_solve``,
``classify_matrix``, ``LinearSolver``), thread determinism of the parallel
SpMV path and a timing comparison of the direct solver.
"""

from __future__ import annotations

import time
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from corpus import CASES, get_case
from mna_reference import solvers as ref
from mna_reference.mna_builder import build_mna_problem
from mna_reference.netlist import parse_netlist_text
from mna_simulation import native
from mna_simulation.native import linalg

core = native.require_core()

# Keys of the Krylov info that must equal the reference's (absent keys included).
PARITY_KEYS = (
    "converged",
    "method",
    "requested_method",
    "route",
    "matrix_kind",
    "matrix_dimension",
    "matrix_nnz",
    "resolved_rank",
    "rank_mode",
    "iteration_budget",
    "effective_restart",
    "operator_storage",
    "exit_code",
)
STATS_KEYS = PARITY_KEYS + ("used_direct_fallback",)
ABSENT = "<absent>"
DEFAULT_OMEGA = 2 * np.pi * 1e9  # typical f_stop of the corpus AC sweeps


def _csc(A: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = sp.csc_matrix(A)
    m.sum_duplicates()
    return m.indptr.astype(np.int32), m.indices.astype(np.int32), m.data


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale)


def _assert_keys_equal(ref_info: dict, nat_info: dict, keys: tuple[str, ...], context: str) -> None:
    diffs = {k: (ref_info.get(k, ABSENT), nat_info.get(k, ABSENT)) for k in keys if ref_info.get(k, ABSENT) != nat_info.get(k, ABSENT)}
    assert not diffs, f"{context}: reference/native info differ {diffs}"


def _reference_is_chaotic(A: Any, b: np.ndarray, kwargs: dict, converged: bool) -> bool:
    """True when last-bit perturbations of b flip the reference's own ``converged``.

    Restarted GMRES that stagnates right at the tolerance converges or not
    depending on rounding (see the report on rand_resistive_ctrl_s3); such a
    case cannot be held to bitwise-equivalent decisions.
    """

    rng = np.random.default_rng(12345)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            bb = b * (1 + 1e-15 * rng.standard_normal(b.shape))
            _, info = ref.krylov_solve_linear(A, bb, **kwargs)
            if bool(info["converged"]) != converged:
                return True
    return False


def _compare_krylov(A: Any, b: np.ndarray, context: str, **kwargs: Any) -> tuple[dict, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_ref, i_ref = ref.krylov_solve_linear(A, b, **kwargs)
    x_nat, i_nat = linalg.krylov_solve_linear(A, b, **kwargs)
    assert np.shape(x_nat) == np.shape(x_ref), context
    assert i_nat["engine"] == "mna_core"
    if i_ref["converged"] != i_nat["converged"] and _reference_is_chaotic(A, b, kwargs, bool(i_ref["converged"])):
        return i_ref, i_nat
    _assert_keys_equal(i_ref, i_nat, PARITY_KEYS, context)
    assert i_nat["matrix_density"] == pytest.approx(i_ref["matrix_density"], rel=1e-15)
    if np.linalg.norm(b) > 0:
        x0 = kwargs.get("x0")
        if x0 is None or np.linalg.norm(np.asarray(A @ np.asarray(x0).reshape(-1)) - np.asarray(b).reshape(-1)) > 0:
            assert i_nat["iterations"] >= 1, context
    if i_ref["converged"] and i_nat["converged"]:
        assert _rel(x_nat, x_ref) <= 1e-7, f"{context}: solutions differ by {_rel(x_nat, x_ref):.2e}"
    return i_ref, i_nat


def _solve_or_singular(fn: Any, *args: Any, **kwargs: Any) -> np.ndarray | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    except np.linalg.LinAlgError:
        return None


def _compare_solve_linear_krylov(A: Any, b: np.ndarray, context: str, singular_ok: bool = False, **kwargs: Any) -> None:
    """solve_linear(use_krylov=True) parity: stats entry, fallback decision, answer.

    ``singular_ok``: the matrix is numerically singular, so after a fallback
    the direct solvers may legitimately disagree on raising "Singular matrix".
    """

    stats_ref: list[dict] = []
    stats_nat: list[dict] = []
    x_ref = _solve_or_singular(ref.solve_linear, A, b, use_krylov=True, krylov_stats=stats_ref, **kwargs)
    x_nat = _solve_or_singular(linalg.solve_linear, A, b, use_krylov=True, krylov_stats=stats_nat, **kwargs)
    assert len(stats_ref) == len(stats_nat) == 1
    if x_ref is None or x_nat is None:
        assert singular_ok or (x_ref is None and x_nat is None), f"{context}: only one side raised LinAlgError"
        _assert_keys_equal(stats_ref[0], stats_nat[0], STATS_KEYS, context)
        return
    if stats_ref[0]["converged"] != stats_nat[0]["converged"]:
        krylov_kwargs = {
            "tol": kwargs.get("krylov_tol", 1e-9),
            "max_iter": kwargs.get("krylov_max_iter", 2000),
            "restart": kwargs.get("krylov_restart", 80),
            "rank": kwargs.get("krylov_rank", "auto"),
            "method": kwargs.get("krylov_method", "auto"),
        }
        if _reference_is_chaotic(A, b, krylov_kwargs, bool(stats_ref[0]["converged"])):
            return
    _assert_keys_equal(stats_ref[0], stats_nat[0], STATS_KEYS, context)
    assert np.shape(x_nat) == np.shape(x_ref)
    if stats_nat[0]["used_direct_fallback"] and singular_ok:
        return  # both direct answers are rounding noise (|x| ~ 1e17)
    if stats_nat[0]["used_direct_fallback"] or stats_nat[0]["converged"]:
        assert _rel(x_nat, x_ref) <= 1e-7, context


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

GENERAL = np.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
SYMMETRIC = np.array([[4.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, 2.0]])
SPD = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
SMALL_GENERAL = np.array([[2.0, -1.0], [4.0, 3.0]])
COLUMN_B = np.array([[1.0], [2.0], [3.0]])


@lru_cache(maxsize=None)
def _linear_problems() -> tuple[tuple[str, Any, tuple[float, ...], tuple[float, ...]], ...]:
    """(case name, reference MnaProblem, transient steps, AC angular frequencies) of every linear corpus case."""

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case in CASES:
            try:
                problem = build_mna_problem(parse_netlist_text(case.netlist_text))
            except Exception:  # noqa: BLE001 - error cases of the corpus
                continue
            f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
            if ref.is_nonlinear(f_str) or problem.level1_mos_devices or problem.level1_bjt_devices:
                continue
            if problem.G_sparse is None or problem.G.shape[0] == 0:
                continue
            steps = sorted({float(o["t_step"]) for m, o in case.analyses if m == "tran"}) or [1e-9]
            omegas = sorted({2 * np.pi * float(o["f_stop"]) for m, o in case.analyses if m == "ac"}) or [DEFAULT_OMEGA]
            out.append((case.name, problem, tuple(steps[:1]), tuple(omegas[:1])))
    return tuple(out)


@lru_cache(maxsize=None)
def _operators() -> dict[str, tuple[Any, np.ndarray]]:
    """id -> (CSR operator, seeded right-hand side)."""

    ops: dict[str, tuple[Any, np.ndarray]] = {}
    for name, problem, steps, omegas in _linear_problems():
        n = problem.G.shape[0]
        rng = np.random.default_rng(len(ops))
        for h in steps:
            op = (problem.G_sparse + problem.C_sparse / h).tocsr()
            ops[f"{name}-G+C/h"] = (op, rng.standard_normal(n))
        for omega in omegas:
            op = (problem.G_sparse + 1j * omega * problem.C_sparse).tocsr()
            ops[f"{name}-G+jwC"] = (op, rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return ops


def _operator_ids() -> list[str]:
    return list(_operators())


def _mesh_operator() -> tuple[Any, Any]:
    problem = build_mna_problem(parse_netlist_text(get_case("workload_rlc_mesh_23x23").netlist_text))
    return problem, (problem.G_sparse + problem.C_sparse / 10e-9).tocsr()


# ---------------------------------------------------------------------------
# Binding surface
# ---------------------------------------------------------------------------


def test_linear_unit_is_registered() -> None:
    assert "linear" in core.binding_units()
    assert core.DENSE_MAX_N == 0
    assert core.SCIPY_ENGINE_MIN_N == 64
    assert core.CHOLESKY_MAX_N == 256
    assert core.PARALLEL_SPMV_MIN_N == 4096
    assert core.REFACTOR_RCOND_FLOOR == 1e-14


def test_core_classify_matrix_real_and_complex() -> None:
    assert core.classify_matrix(*_csc(GENERAL)) == "general"
    assert core.classify_matrix(*_csc(SYMMETRIC)) == "symmetric"
    assert core.classify_matrix(*_csc(SPD)) == "positive_definite"
    herm = np.array([[2.0, 1 - 1j], [1 + 1j, 3.0]])
    assert core.classify_matrix(*_csc(herm)) == "positive_definite"
    assert core.classify_matrix(*_csc(np.array([[1 + 1j, 1j], [1j, 1 + 1j]]))) == "general"
    assert core.classify_matrix(*_csc(np.array([[2.0, 1.0], [1.0 + 1e-3, 2.0]])), 1e-2) == "positive_definite"
    with pytest.raises(TypeError):
        core.classify_matrix(*_csc(SPD)[:2], np.ones(7, dtype=np.float32))


def test_direct_solve_shapes_and_dtypes() -> None:
    ap, ai, ax = _csc(GENERAL)
    b = np.array([1.0, 2.0, 3.0])
    x = core.direct_solve(ap, ai, ax, b)
    assert x.shape == (3,) and x.dtype == np.float64
    np.testing.assert_allclose(x, np.linalg.solve(GENERAL, b), rtol=1e-14, atol=1e-15)

    rhs = np.arange(12.0).reshape(3, 4)  # (n, k) C-order
    for B in (rhs, np.asfortranarray(rhs), rhs[:, ::2]):
        X = core.direct_solve(ap, ai, ax, B)
        assert X.shape == B.shape
        np.testing.assert_allclose(X, np.linalg.solve(GENERAL, B), rtol=1e-13, atol=1e-14)

    bc = b + 1j * np.array([0.5, -1.0, 2.0])
    xc = core.direct_solve(ap, ai, ax, bc)  # real matrix, complex right-hand side
    assert xc.dtype == np.complex128
    np.testing.assert_allclose(xc, np.linalg.solve(GENERAL, bc), rtol=1e-14, atol=1e-15)

    Ac = GENERAL + 1j * np.eye(3)
    xz = core.direct_solve(*_csc(Ac), b)  # complex matrix, real right-hand side
    np.testing.assert_allclose(xz, np.linalg.solve(Ac, b), rtol=1e-14, atol=1e-15)

    empty = core.direct_solve(np.array([0], dtype=np.int32), np.array([], dtype=np.int32), np.array([]), np.array([]))
    assert empty.shape == (0,)

    with pytest.raises(core.CoreError) as excinfo:
        core.direct_solve(ap, ai, ax, np.ones(4))
    assert excinfo.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError) as excinfo:
        core.direct_solve(*_csc(np.array([[1.0, 2.0], [2.0, 4.0]])), np.ones(2))
    assert excinfo.value.code == "JACOBIAN_SINGULAR"


def test_core_krylov_solve_reports_info() -> None:
    ap, ai, ax = _csc(SPD)
    b = np.array([1.0, 2.0, 3.0])
    x, info = core.krylov_solve(ap, ai, ax, b, "auto", 1e-12, 30, 3)
    np.testing.assert_allclose(SPD @ x, b, atol=1e-10)
    assert info["matrix_kind"] == "positive_definite"
    assert info["route"] == "positive_definite"
    assert info["method"] == "conjugate_gradient"
    assert info["kernel"] == "legacy_cg"
    assert info["engine"] == "mna_core"
    assert info["converged"] is True
    assert info["scipy_semantics"] is False
    assert "exit_code" not in info
    _, info = core.krylov_solve(ap, ai, ax, b, "GMRES", 1e-12, 30, 2)
    assert "matrix_kind" not in info  # explicit methods need no classification
    assert info["route"] == "general" and info["effective_restart"] == 2
    with pytest.raises(ValueError, match="Unknown Krylov method 'foo'."):
        core.krylov_solve(ap, ai, ax, b, "foo", 1e-12, 30, 3)
    with pytest.raises(core.CoreError):
        core.krylov_solve(ap, ai, ax, np.ones(4), "auto", 1e-12, 30, 3)


def test_linear_solver_object_real_complex_dense() -> None:
    ap, ai, ax = _csc(GENERAL)
    b = np.array([1.0, -2.0, 0.5])
    for dense in (False, True):
        solver = core.LinearSolver(ap, ai, complex=False, dense=dense)
        assert solver.kind == ("dense" if dense else "klu")
        assert solver.n == 3 and solver.nnz == 7 and not solver.factored
        assert solver.path_of_last_refactor is None
        solver.factor(ax)
        assert solver.factored and solver.nnz_lu >= 3
        np.testing.assert_allclose(solver.solve(b), np.linalg.solve(GENERAL, b), rtol=1e-14)
        np.testing.assert_allclose(solver.tsolve(b), np.linalg.solve(GENERAL.T, b), rtol=1e-14)
        B = np.c_[b, 2 * b]
        np.testing.assert_allclose(solver.solve(B), np.linalg.solve(GENERAL, B), rtol=1e-14)
        np.testing.assert_allclose(solver.solve(b + 1j * b), np.linalg.solve(GENERAL, b + 1j * b), rtol=1e-14)
        assert 0 < solver.rcond() <= 1
        path = solver.refactor(2 * ax)
        assert path == ("full_factor" if dense else "refactor")
        assert solver.path_of_last_refactor == path
        np.testing.assert_allclose(solver.solve(b), np.linalg.solve(2 * GENERAL, b), rtol=1e-14)

    Ac = GENERAL + 1j * np.diag([1.0, -2.0, 0.5])
    cap, cai, cax = _csc(Ac)
    solver = core.LinearSolver(cap, cai, complex=True)
    assert solver.is_complex
    solver.factor(cax)
    np.testing.assert_allclose(solver.solve(b), np.linalg.solve(Ac, b), rtol=1e-14)
    np.testing.assert_allclose(solver.tsolve(b), np.linalg.solve(Ac.T, b), rtol=1e-14)
    np.testing.assert_allclose(solver.tsolve(b, conjugate=True), np.linalg.solve(Ac.conj().T, b), rtol=1e-14)
    # Pivot order that fails for new values: [[0, 1], [1, 0]] after [[1, 1], [1, 2]].
    two = core.LinearSolver(np.array([0, 2, 4], dtype=np.int32), np.array([0, 1, 0, 1], dtype=np.int32))
    two.factor(np.array([1.0, 1.0, 1.0, 2.0]))
    assert two.refactor(np.array([0.0, 1.0, 1.0, 0.0])) == "full_factor"
    np.testing.assert_allclose(two.solve(np.array([3.0, 5.0])), [5.0, 3.0])
    with pytest.raises(core.CoreError) as excinfo:
        two.factor(np.array([1.0, 2.0, 2.0, 4.0]))
    assert excinfo.value.code == "JACOBIAN_SINGULAR"


# ---------------------------------------------------------------------------
# Facade parity: matrices of test_krylov_solvers.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matrix", "kwargs"),
    [
        (GENERAL, {"tol": 1e-12, "max_iter": 30, "rank": 3}),
        (SYMMETRIC, {"tol": 1e-12, "max_iter": 30, "rank": 3}),
        (SPD, {"tol": 1e-12, "max_iter": 30, "rank": 3}),
        (SPD, {"tol": 1e-12, "max_iter": 30, "rank": 2, "method": "arnoldi_gmres"}),
        (SMALL_GENERAL, {"tol": 1e-12, "rank": 2}),
    ],
    ids=["general", "symmetric", "spd", "spd-arnoldi", "2x2"],
)
def test_krylov_solvers_matrices_match_reference(matrix: np.ndarray, kwargs: dict) -> None:
    b = COLUMN_B[: matrix.shape[0]]
    assert linalg.classify_krylov_matrix(matrix) == ref.classify_krylov_matrix(matrix)
    i_ref, i_nat = _compare_krylov(matrix, b, "test_krylov_solvers", **kwargs)
    assert i_nat["converged"] is True
    x_nat, _ = linalg.krylov_solve_linear(matrix, b, **kwargs)
    assert x_nat.shape == b.shape
    assert np.linalg.norm(matrix @ x_nat - b) < 1e-10
    for method in ("auto", "cg", "spd", "cr", "minres", "hermitian", "gmres", "arnoldi", "general", "CG"):
        _compare_krylov(matrix, b, f"method={method}", **{**kwargs, "method": method})
    x_direct = linalg.solve_linear(matrix, b)
    np.testing.assert_allclose(x_direct, np.linalg.solve(matrix, b), rtol=1e-12, atol=1e-14)
    _compare_solve_linear_krylov(matrix, b, "solve_linear", krylov_tol=1e-12, krylov_rank=kwargs["rank"])


def test_resolve_rank_routing_and_describe_match_reference() -> None:
    for args in [(1, "auto"), (7, "auto"), (8, "auto"), (8, 37), (8, "137"), (8, 251), (5, None, 9), (3, " AUTO "), (4, 2.7)]:
        assert linalg.resolve_krylov_rank(*args) == ref.resolve_krylov_rank(*args)
    for bad in (0, -3, "abc", ""):
        with pytest.raises(ValueError):
            linalg.resolve_krylov_rank(8, bad)
    for matrix in (GENERAL, SYMMETRIC, SPD, sp.csr_matrix(SPD)):
        for method in ("auto", "cg", "cr", "gmres"):
            for rank in ("auto", 2, None):
                choice_ref = ref.describe_krylov_choice(matrix, rank=rank, method=method, max_iter=17)
                assert linalg.describe_krylov_choice(matrix, rank=rank, method=method, max_iter=17) == choice_ref
    with pytest.raises(ValueError, match="Unknown Krylov method 'foo'."):
        linalg.krylov_solve_linear(SPD, COLUMN_B, method="foo")
    with pytest.raises(ValueError, match="Unknown Krylov method 'foo'."):
        linalg.solve_linear(SPD, COLUMN_B, use_krylov=True, krylov_method="foo")
    with pytest.raises(np.linalg.LinAlgError, match="Krylov solve requires a square matrix."):
        linalg.krylov_solve_linear(np.ones((2, 3)), np.ones(2))


def test_classification_edge_cases_match_reference() -> None:
    nan = float("nan")
    tridiag = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(300, 300)).tocsr()
    cases = [
        sp.csr_matrix(np.array([[1e-3]])),
        np.array([[1e-3]]),
        sp.csr_matrix(np.array([[-1e-3]])),
        sp.csr_matrix(np.array([[2.0, 1.0], [1.0 + 1e-11, 2.0]])),
        sp.csr_matrix(np.array([[2.0, 1.0], [1.0 + 1e-9, 2.0]])),
        sp.csr_matrix(np.array([[2.0, nan, 0.0], [1.0, 2.0, 0.0], [0.0, 5.0, 2.0]])),
        sp.csr_matrix(np.array([[nan, 0.0], [0.0, 1.0]])),
        sp.csr_matrix(np.zeros((2, 2))),
        tridiag,  # SPD but n > 256: symmetric on the sparse rule
        tridiag[:256, :256],
        sp.csr_matrix(np.array([[2.0, 1 - 1j], [1 + 1j, 3.0]])),
        sp.csr_matrix(np.array([[1 + 1j, 1j], [1j, 1 + 1j]])),
        np.ones((2, 3)),
        sp.csr_matrix(np.ones((2, 3))),
    ]
    for matrix in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = ref.classify_krylov_matrix(matrix)
        assert linalg.classify_krylov_matrix(matrix) == expected
    # Deliberate difference (CPP_MIGRATION_PLAN 6.1): dense inputs follow the sparse rule.
    assert ref.classify_krylov_matrix(tridiag.toarray()) == "positive_definite"
    assert linalg.classify_krylov_matrix(tridiag.toarray()) == "symmetric"


# ---------------------------------------------------------------------------
# Facade parity: every linear corpus operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_id", _operator_ids())
def test_corpus_operator_parity(op_id: str) -> None:
    A, b = _operators()[op_id]
    n = A.shape[0]
    dense = A.toarray()

    # Classification: sparse input, and dense input where both rules coincide (n <= 256).
    assert linalg.classify_krylov_matrix(A) == ref.classify_krylov_matrix(A), op_id
    if n <= 256:
        assert linalg.classify_krylov_matrix(dense) == ref.classify_krylov_matrix(dense), op_id

    # Numerically singular operators (floating, ungrounded circuits such as
    # qnpn_extended_small_signal_only and mos_small_signal_only): rounding alone
    # decides whether LAPACK or KLU meets an exactly zero pivot.
    singular = n <= 200 and np.linalg.cond(dense) > 1e12

    # krylov_solve_linear (auto) on the sparse operator and on the dense matrix.
    _compare_krylov(A, b, f"{op_id} sparse")
    _compare_krylov(dense, b, f"{op_id} dense")
    # solve_linear with Krylov: same acceptance / fallback decision.
    _compare_solve_linear_krylov(A, b, f"{op_id} solve_linear", singular_ok=singular)

    # Direct path against LAPACK on the dense matrix.
    if singular:
        return
    x_np = np.linalg.solve(dense, b)
    x_nat = linalg.solve_linear(A, b)
    assert _rel(x_nat, x_np) <= 1e-12, f"{op_id}: direct solve differs by {_rel(x_nat, x_np):.2e}"
    x_nat_dense = linalg.solve_linear(dense, b)
    assert _rel(x_nat_dense, x_np) <= 1e-12


def _representatives() -> list[str]:
    """One operator id per (matrix kind, n >= 64, complex)."""

    seen: dict[tuple[str, bool, bool], str] = {}
    for op_id, (A, _b) in _operators().items():
        key = (linalg.classify_krylov_matrix(A), A.shape[0] >= 64, bool(np.iscomplexobj(A.data)))
        seen.setdefault(key, op_id)
    return sorted(seen.values())


@pytest.mark.parametrize("op_id", _representatives())
@pytest.mark.parametrize("method", ["gmres", "cg", "cr", "minres", "spd", "symmetric", "arnoldi_gmres"])
def test_explicit_methods_on_each_matrix_kind(op_id: str, method: str) -> None:
    A, b = _operators()[op_id]
    _compare_krylov(A, b, f"{op_id} method={method}", method=method)
    singular = A.shape[0] <= 200 and np.linalg.cond(A.toarray()) > 1e12
    _compare_solve_linear_krylov(A, b, f"{op_id} solve_linear method={method}", singular_ok=singular, krylov_method=method)


def test_representatives_cover_every_kind() -> None:
    kinds = {(linalg.classify_krylov_matrix(_operators()[op][0]), _operators()[op][0].shape[0] >= 64) for op in _representatives()}
    assert {("general", False), ("symmetric", False), ("positive_definite", False)} <= kinds
    assert ("symmetric", True) in kinds and ("general", True) in kinds


# ---------------------------------------------------------------------------
# Targeted behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "gmres", "cg", "cr"])
def test_zero_rhs_returns_zeros_without_iterating(method: str) -> None:
    _, mesh = _mesh_operator()
    for A in (SPD, SYMMETRIC, GENERAL, mesh, (mesh + 1j * sp.eye(mesh.shape[0])).tocsr()):
        b = np.zeros(A.shape[0])
        x0 = np.ones(A.shape[0])
        i_ref, i_nat = _compare_krylov(A, b, f"zero rhs {method}", method=method, x0=x0)
        x_nat, _ = linalg.krylov_solve_linear(A, b, method=method, x0=x0)
        assert not np.any(x_nat)
        assert i_nat["iterations"] == 0 and i_nat["converged"] is True
        assert i_nat["relative_residual"] == 0.0


def test_krylov_rank_one_on_the_rlc_mesh_falls_back_to_the_direct_solve() -> None:
    _, mesh = _mesh_operator()
    b = np.random.default_rng(7).standard_normal(mesh.shape[0])
    stats_ref: list[dict] = []
    stats_nat: list[dict] = []
    x_ref = ref.solve_linear(mesh, b, use_krylov=True, krylov_rank=1, krylov_stats=stats_ref)
    x_nat = linalg.solve_linear(mesh, b, use_krylov=True, krylov_rank=1, krylov_stats=stats_nat)
    _assert_keys_equal(stats_ref[0], stats_nat[0], STATS_KEYS, "krylov_rank=1")
    assert stats_nat[0]["used_direct_fallback"] is True
    assert stats_nat[0]["method"] == "minres" and stats_nat[0]["resolved_rank"] == 1
    assert stats_nat[0]["exit_code"] == 1
    assert _rel(x_nat, x_ref) <= 1e-12  # x_ref is np.linalg.solve's answer


def test_rlc_mesh_auto_route_matches_the_service_expectations() -> None:
    # tests/test_krylov_solvers.py::test_large_rlc_mesh_routes_to_symmetric_krylov_and_runs
    problem, mesh = _mesh_operator()
    b = (problem.C_sparse / 10e-9) @ np.random.default_rng(3).standard_normal(1035)
    stats: list[dict] = []
    linalg.solve_linear(mesh, b, use_krylov=True, krylov_rank="auto", krylov_stats=stats)
    info = stats[0]
    assert info["matrix_dimension"] == 1035
    assert info["matrix_kind"] == "symmetric"
    assert info["method"] == "minres"
    assert info["resolved_rank"] == 518
    assert info["operator_storage"] == "csr"
    assert info["matrix_nnz"] == 4071
    assert info["matrix_density"] < 0.01
    assert info["used_direct_fallback"] is False
    assert info["effective_restart"] is None and info["exit_code"] == 0


def test_complex_gmres_small_and_large() -> None:
    small = np.array([[4 + 1j, 1, 0], [2j, 3 - 1j, 1 + 1j], [0, 1, 2 + 2j]])
    b = np.array([1.0, 2j, 3 - 1j])
    i_ref, i_nat = _compare_krylov(small, b, "complex 3x3", tol=1e-12)
    assert i_nat["method"] == "arnoldi_gmres" and i_nat["converged"] is True
    assert "exit_code" not in i_nat

    problem, _ = _mesh_operator()
    big = (problem.G_sparse + 1j * DEFAULT_OMEGA * problem.C_sparse).tocsr()
    rhs = problem.b_ac.reshape(-1) + np.random.default_rng(11).standard_normal(1035)
    i_ref, i_nat = _compare_krylov(big, rhs, "complex 1035")
    assert i_nat["route"] == "general" and i_nat["converged"] is True and i_nat["exit_code"] == 0
    # Restarted GMRES that runs out of restart cycles (1 MHz, restart 100, 3 cycles).
    slow = (problem.G_sparse + 1j * 2 * np.pi * 1e6 * problem.C_sparse).tocsr()
    i_ref, i_nat = _compare_krylov(slow, rhs, "complex 1035 budget", rank=100, max_iter=3)
    assert i_nat["converged"] is False and i_nat["exit_code"] == 3
    assert i_nat["iterations"] == i_ref["iterations"] == 300
    assert i_nat["relative_residual"] == pytest.approx(i_ref["relative_residual"], rel=1e-4)


def test_x0_warm_start() -> None:
    problem, mesh = _mesh_operator()
    rng = np.random.default_rng(5)
    b = rng.standard_normal(1035)
    x_exact = linalg.solve_linear(mesh, b)
    warm = x_exact * (1 + 1e-6 * rng.standard_normal(1035))
    for method in ("auto", "gmres", "cg"):
        _compare_krylov(mesh, b, f"x0 {method}", method=method, x0=warm)
        _, cold = linalg.krylov_solve_linear(mesh, b, method=method)
        _, hot = linalg.krylov_solve_linear(mesh, b, method=method, x0=warm)
        if cold["converged"] and hot["converged"]:
            assert hot["iterations"] < cold["iterations"]
    # Small systems run the legacy loops; x0 given as a column like b.
    x0 = np.array([[0.2], [0.6], [1.4]])
    for matrix in (GENERAL, SYMMETRIC, SPD):
        _compare_krylov(matrix, COLUMN_B, "x0 small", tol=1e-12, rank=3, x0=x0)
    # Exact x0 on an SPD 3x3: the legacy CG quirk ("not converged", zero residual) is accepted
    # by solve_linear's residual rule, exactly as in the reference.
    exact = np.array([1.0, 2.0, 3.0])
    rhs = SPD @ exact
    _compare_krylov(SPD, rhs, "exact x0", x0=exact, rank=3)
    _compare_solve_linear_krylov(SPD, rhs, "exact x0 solve_linear", x0=exact, krylov_rank=3)


def test_singular_matrices_raise_numpy_linalg_error() -> None:
    singular = [
        np.array([[1.0, 2.0], [2.0, 4.0]]),
        np.zeros((3, 3)),
        np.array([[1.0, 1j], [1j, -1.0]]),
        # tests/test_edge_cases.py::test_singular_linear_transient_step_raises_raw_numpy_linalg_error:
        # G + C/h for V1 a 0 DC 1, R1 a b 1k, C1 b 0 -1u, h = 1m.
        np.array([[1e-3, -1e-3, 1.0], [-1e-3, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ]
    for matrix in singular:
        for A in (matrix, sp.csr_matrix(matrix)):
            with pytest.raises(np.linalg.LinAlgError) as excinfo:
                linalg.solve_linear(A, np.ones(matrix.shape[0]))
            assert type(excinfo.value) is np.linalg.LinAlgError
            assert str(excinfo.value) == "Singular matrix"
            with pytest.raises(np.linalg.LinAlgError):
                np.linalg.solve(matrix, np.ones(matrix.shape[0]))
    with pytest.raises(np.linalg.LinAlgError, match="Last 2 dimensions of the array must be square"):
        linalg.solve_linear(np.ones((2, 3)), np.ones(2))
    # Krylov on a non-square matrix records the describe_krylov_choice entry, then the direct
    # solve raises, as in the reference.
    stats_ref: list[dict] = []
    stats_nat: list[dict] = []
    with pytest.raises(np.linalg.LinAlgError):
        ref.solve_linear(np.ones((2, 3)), np.ones(2), use_krylov=True, krylov_stats=stats_ref)
    with pytest.raises(np.linalg.LinAlgError):
        linalg.solve_linear(np.ones((2, 3)), np.ones(2), use_krylov=True, krylov_stats=stats_nat)
    assert stats_nat == stats_ref


def test_multiple_rhs_direct_solve() -> None:
    _, mesh = _mesh_operator()
    dense = mesh.toarray()
    rng = np.random.default_rng(9)
    B = rng.standard_normal((1035, 4))
    Bc = B + 1j * rng.standard_normal((1035, 4))
    expected = np.linalg.solve(dense, np.concatenate([B, Bc], axis=1))  # one LAPACK call
    XB, XBc = expected[:, :4].real, expected[:, 4:]
    for rhs, want in ((B, XB), (np.asfortranarray(B), XB), (B[:, ::2], XB[:, ::2]), (B[:, :1], XB[:, :1])):
        x = linalg.solve_linear(mesh, rhs)
        assert x.shape == rhs.shape
        assert _rel(x, want) <= 1e-12
    xc = linalg.solve_linear(mesh, Bc)
    assert xc.dtype == np.complex128
    assert _rel(xc, XBc) <= 1e-12
    small = linalg.solve_linear(GENERAL, np.eye(3))
    np.testing.assert_allclose(small, np.linalg.inv(GENERAL), rtol=1e-13, atol=1e-15)
    assert linalg.solve_linear(np.zeros((0, 0)), np.zeros(0)).shape == (0,)


def _laplacian_2d(side: int, nonsymmetric: bool) -> sp.csr_matrix:
    rng = np.random.default_rng(21)
    T = sp.diags([-1.0, -1.0], [-1, 1], shape=(side, side))
    A = sp.kron(sp.eye(side), T) + sp.kron(T, sp.eye(side)) + 5.0 * sp.eye(side * side)
    A = A.tocsr()
    A.data = A.data * (1 + 0.05 * rng.standard_normal(A.nnz)) if nonsymmetric else A.data
    if not nonsymmetric:
        A = ((A + A.T) * 0.5).tocsr()
    return A


@pytest.mark.parametrize("nonsymmetric", [False, True], ids=["spd", "general"])
def test_parallel_spmv_path_is_deterministic_across_thread_counts(nonsymmetric: bool) -> None:
    A = _laplacian_2d(70, nonsymmetric)  # n = 4900 >= PARALLEL_SPMV_MIN_N
    assert A.shape[0] >= core.PARALLEL_SPMV_MIN_N
    b = np.sin(0.37 * np.arange(A.shape[0])) + 0.1
    x1, i1 = linalg.krylov_solve_linear(A, b, tol=1e-10, rank=60, num_threads=1)
    assert i1["threads_used"] == 1
    assert i1["route"] == ("general" if nonsymmetric else "symmetric")
    assert i1["converged"] is True
    for threads in (2, 4):
        xk, ik = linalg.krylov_solve_linear(A, b, tol=1e-10, rank=60, num_threads=threads)
        assert np.array_equal(xk, x1)
        assert {k: v for k, v in ik.items() if k != "threads_used"} == {k: v for k, v in i1.items() if k != "threads_used"}
        assert ik["threads_used"] == min(threads, core.effective_threads(threads))


def test_direct_solve_timing_against_dense_lapack(capsys: pytest.CaptureFixture[str]) -> None:
    _, mesh = _mesh_operator()
    dense = mesh.toarray()
    b = np.random.default_rng(13).standard_normal(1035)
    op = linalg.to_csc_arrays(mesh)

    def best_of(fn, repeat: int = 5) -> float:
        best = float("inf")
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best

    t_dense = best_of(lambda: np.linalg.solve(dense, b), repeat=3)
    t_klu = best_of(lambda: core.direct_solve(op.col_ptr, op.row_idx, op.values, b))
    t_facade = best_of(lambda: linalg.solve_linear(mesh, b))
    with capsys.disabled():
        print(
            f"\n[native linalg] n=1035 RLC operator: np.linalg.solve (dense LAPACK) {t_dense * 1e3:.2f} ms, "
            f"core.direct_solve (KLU analyze+factor+solve) {t_klu * 1e3:.2f} ms, "
            f"linalg.solve_linear (incl. CSC conversion) {t_facade * 1e3:.2f} ms, "
            f"speedup {t_dense / t_klu:.0f}x"
        )
    x_dense = np.linalg.solve(dense, b)
    assert _rel(core.direct_solve(op.col_ptr, op.row_idx, op.values, b), x_dense) <= 1e-12
    assert t_klu < 0.5 * t_dense
