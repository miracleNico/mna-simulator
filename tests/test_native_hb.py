"""Native harmonic balance (``_core.hb_solve`` behind ``solvers.solve_harmonic_balance``) against the frozen oracle.

Covers the Fourier helpers (bitwise against ``create_fourier_matrices`` /
``reconstruct_time_domain`` / ``numpy.linspace``), every corpus ``hb`` analysis
(``x_bar``, spectrum magnitudes and reconstructed waveforms within 1e-6 of the
oracle, including the 205-harmonic GHz mixer), the analytic phasor of a linear
RC, thread determinism, the memory policy (``MNA_HB_MAX_NNZ``), the
``harmonics <= 0`` ruling (CPP_MIGRATION_PLAN.md 2.4.1), the Krylov option
metadata, the dense-allocation-free Krylov description (quirk 6.1) and the
binding surface.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import warnings
from typing import Any

import numpy as np
import pytest

from corpus import Case, get_case, iter_analyses
from mna_reference import solvers as ref_solvers
from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions
from mna_simulation.core_advanced import describe_krylov_choice_for_dimension, run_advanced_analysis
from mna_simulation.errors import NetlistError, SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.linalg import describe_krylov_choice
from mna_simulation.netlist import parse_netlist_text

core = native.require_core()

# libm's cos/sin are what NumPy uses on Linux; elsewhere allow one rounding step.
BITWISE = sys.platform.startswith("linux")
RTOL = 1e-6
ATOL = 1e-9

MIXER = get_case("hb_ghz_mixer").netlist_text


def _assert_same(a: np.ndarray, b: np.ndarray) -> None:
    if BITWISE:
        assert np.array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-15, atol=1e-15)


def _assert_close(cur: Any, ref: Any, what: str) -> None:
    cur = np.asarray(cur, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert cur.shape == ref.shape, what
    np.testing.assert_allclose(cur, ref, rtol=RTOL, atol=ATOL, err_msg=what)
    scale = max(float(np.max(np.abs(ref), initial=0.0)), 1e-300)
    assert float(np.max(np.abs(cur - ref), initial=0.0)) <= RTOL * scale, what


def _backend(pkg: str) -> Any:
    return importlib.import_module(f"{pkg}.backends.python_backend").PythonBackend()


def _run(pkg: str, netlist: str, options: dict[str, Any]) -> Any:
    """``PythonBackend.run`` (with gmin stepping) of an HB request; returns the result or the exception."""

    backend = _backend(pkg)
    contracts = importlib.import_module(f"{pkg}.api.contracts")
    circuit = backend.parse_text(netlist)
    analysis = backend.build_options(circuit, overrides=dict(options), mode=contracts.AnalysisMode.HB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return backend.run(circuit, analysis)
        except Exception as exc:  # noqa: BLE001 - errors are compared
            return exc


def _hb_params() -> list[Any]:
    params = []
    seen: dict[str, int] = {}
    for case, _index, mode, options in iter_analyses():
        if mode != "hb":
            continue
        base = case.name + ("-window" if "time_window" in options else "") + ("-krylov" if options.get("use_krylov") else "")
        seen[base] = seen.get(base, 0) + 1
        test_id = base if seen[base] == 1 else f"{base}#{seen[base]}"
        marks = [pytest.mark.slow] if "slow" in case.tags else []
        params.append(pytest.param(case, options, id=test_id, marks=marks))
    return params


# ---------------------------------------------------------------------------
# Binding surface and Fourier helpers
# ---------------------------------------------------------------------------


def test_hb_unit_is_registered() -> None:
    assert "hb" in core.binding_units()
    assert core.HB_MAX_DIMENSION == 1_000_000
    assert core.HB_DEFAULT_MAX_NNZ == 30_000_000
    for name in ("hb_solve", "hb_reconstruct", "hb_fourier_matrices", "hb_time_points", "hb_jacobian_block", "btf_solve"):
        assert callable(getattr(core, name))


@pytest.mark.parametrize(("harmonics", "points"), [(0, 1), (1, 3), (3, 7), (8, 17), (205, 411), (5, 4), (2, 11)])
def test_fourier_matrices_are_bitwise_equal_to_the_reference(harmonics: int, points: int) -> None:
    time_points = np.linspace(0.0, 1.0, points, endpoint=False)
    gamma_ref, gamma_inv_ref = ref_solvers.create_fourier_matrices(harmonics, time_points)
    gamma, gamma_inv = solvers.create_fourier_matrices(harmonics, time_points)
    assert gamma.dtype == np.float64 and gamma.shape == gamma_ref.shape
    assert gamma_inv.shape == gamma_inv_ref.shape
    _assert_same(gamma, gamma_ref)
    _assert_same(gamma_inv, gamma_inv_ref)


def test_fourier_matrices_edge_cases_match_the_reference() -> None:
    with pytest.raises(ZeroDivisionError):
        ref_solvers.create_fourier_matrices(2, np.zeros(0))
    with pytest.raises(ZeroDivisionError):
        solvers.create_fourier_matrices(2, np.zeros(0))
    with pytest.raises(ValueError):
        ref_solvers.create_fourier_matrices(-1, np.zeros(3))
    with pytest.raises(ValueError):
        solvers.create_fourier_matrices(-1, np.zeros(3))


@pytest.mark.parametrize("omega0", [2 * np.pi * 5e6, 2 * np.pi * 1000.0000000000001, 1.0, 3.3])
def test_time_points_are_numpy_linspace(omega0: float) -> None:
    for points in (1, 3, 17, 411):
        expected = np.linspace(0, 2 * np.pi / omega0, points, endpoint=False)
        assert np.array_equal(core.hb_time_points(omega0, points), expected)


@pytest.mark.parametrize(("num_vars", "harmonics"), [(3, 205), (2, 3), (1, 0)])
def test_reconstruction_is_bitwise_equal_and_thread_independent(num_vars: int, harmonics: int) -> None:
    rng = np.random.default_rng(num_vars + harmonics)
    x_bar = rng.standard_normal((num_vars * (2 * harmonics + 1), 1))
    t = np.linspace(0, 200e-9, 2000)
    omega0 = 2 * np.pi * 5e6
    expected = ref_solvers.reconstruct_time_domain(x_bar, num_vars, harmonics, omega0, t)
    one = solvers.reconstruct_time_domain(x_bar, num_vars, harmonics, omega0, t)
    four = solvers.reconstruct_time_domain(x_bar, num_vars, harmonics, omega0, t, num_threads=4)
    assert one.shape == (num_vars, t.size)
    _assert_same(one, expected)
    assert np.array_equal(one, four)
    with pytest.raises(ValueError):
        solvers.reconstruct_time_domain(x_bar[:-1], num_vars, harmonics, omega0, t)


def test_jacobian_block_matches_the_reference_triple_product() -> None:
    harmonics = 40
    k = 2 * harmonics + 1
    rng = np.random.default_rng(3)
    g = np.abs(rng.standard_normal(k)) * np.exp(rng.uniform(-30, 25, k))  # diode-like spread
    gamma, gamma_inv = ref_solvers.create_fourier_matrices(harmonics, np.zeros(k))
    expected = gamma @ (g[:, None] * gamma_inv)
    block = core.hb_jacobian_block(harmonics, g)
    assert np.max(np.abs(block - expected)) <= 1e-13 * np.max(np.abs(expected))


def test_btf_solver_matches_numpy_and_reports_singular_matrices() -> None:
    import scipy.sparse as sp

    rng = np.random.default_rng(7)
    n = 60
    dense = np.zeros((n, n))
    dense[:40, :40] = rng.standard_normal((40, 40)) + 40 * np.eye(40)  # dense diagonal block
    dense[40:, 40:] = np.diag(rng.uniform(1, 2, 20)) + np.diag(rng.uniform(-0.5, 0.5, 19), 1)
    dense[:40, 40:] = (rng.random((40, 20)) < 0.1) * rng.standard_normal((40, 20))
    perm = rng.permutation(n)
    matrix = dense[np.ix_(perm, rng.permutation(n))]
    csc = sp.csc_matrix(matrix)
    csc.sort_indices()
    b = rng.standard_normal(n)
    x, info = core.btf_solve(csc.indptr.astype(np.int32), csc.indices.astype(np.int32), csc.data, b)
    np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-12, atol=1e-12)
    assert info["structural_rank"] == n
    assert info["largest_block"] == 40
    assert info["dense_blocks"] >= 1

    singular = sp.csc_matrix(np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]]))
    with pytest.raises(core.CoreError) as exc:
        core.btf_solve(singular.indptr.astype(np.int32), singular.indices.astype(np.int32), singular.data, np.ones(3))
    assert exc.value.code == "JACOBIAN_SINGULAR"


def test_hb_solve_rejects_malformed_inputs() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 1 1k\nR1 a 0 1k\n"))
    with pytest.raises(core.CoreError) as exc:
        core.hb_solve(problem.native, 2 * np.pi * 1e3, 3, np.zeros(5), np.zeros(0, dtype=np.int64), [])
    assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError) as exc:
        core.hb_solve(problem.native, 2 * np.pi * 1e3, 3, np.zeros(2), np.array([7], dtype=np.int64), [None])
    assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError) as exc:
        core.hb_solve(problem.native, -1.0, 3, np.zeros(2), np.zeros(0, dtype=np.int64), [])
    assert exc.value.code == "HB_ERROR"


# ---------------------------------------------------------------------------
# Corpus parity (every hb analysis, direct backend path, harmonic state included)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "options"), _hb_params())
def test_corpus_hb_matches_the_oracle(case: Case, options: dict[str, Any]) -> None:
    if case.deviation_for("hb") is not None:
        pytest.skip("deliberate P3b change (2.4.1): checked by the differential deviation check")
    ref = _run("mna_reference", case.netlist_text, options)
    cur = _run("mna_simulation", case.netlist_text, options)
    if isinstance(ref, Exception) or isinstance(cur, Exception):
        assert type(ref).__name__ == type(cur).__name__, (ref, cur)
        assert getattr(ref, "code", None) == getattr(cur, "code", None)
        return
    assert cur.metadata["harmonics"] == ref.metadata["harmonics"]
    assert cur.metadata["base_frequency_hz"] == ref.metadata["base_frequency_hz"]
    assert cur.metadata["gmin"] == ref.metadata["gmin"]
    assert cur.labels == ref.labels
    _assert_close(cur.harmonic_state, ref.harmonic_state, f"{case.name} x_bar")
    if ref.spectrum is not None:
        assert cur.spectrum is not None
        _assert_close(cur.spectrum.frequencies, ref.spectrum.frequencies, f"{case.name} frequencies")
        _assert_close(cur.spectrum.magnitudes, ref.spectrum.magnitudes, f"{case.name} magnitudes")
    if ref.waveform is not None:
        assert cur.waveform is not None
        assert np.array_equal(cur.waveform.time, ref.waveform.time)
        _assert_close(cur.waveform.values, ref.waveform.values, f"{case.name} waveform")
    stats = cur.metadata["hb_solver"]
    assert stats["dimension"] == len(cur.labels) * (2 * cur.metadata["harmonics"] + 1)


def test_mixer_solver_statistics_and_memory() -> None:
    problem = build_mna_problem(parse_netlist_text(MIXER))
    stats: dict[str, Any] = {}
    solvers.solve_harmonic_balance(problem, 2 * np.pi * 5e6, 205, max_iter=100, tol=1e-6, hb_stats=stats)
    assert stats["dimension"] == 1233
    assert stats["samples"] == 411
    assert stats["newton_iterations"] == 36  # the oracle's count: 36 Newton solves
    assert stats["device_slots"] == 1  # the diode to ground: one dense 411 x 411 block
    assert stats["largest_block"] == 411
    assert stats["matrix_nnz"] == 411 * 411 + (stats["y_nnz"] - 411)  # block replaces the node's diagonal
    assert stats["stored_nnz"] == stats["y_nnz"] + 411 * 411 + 2 * 411 * 411
    assert stats["memory_bytes"] < 20 * 2**20  # the dense reference held (1233^2) x several doubles


# ---------------------------------------------------------------------------
# Analytic and structural checks
# ---------------------------------------------------------------------------


def test_linear_rc_hb_equals_the_analytic_phasor() -> None:
    r, c, amp, f = 1e3, 1e-6, 2.0, 1e3
    problem = build_mna_problem(parse_netlist_text(f"V1 in 0 SIN {amp} {f}\nR1 in out {r}\nC1 out 0 {c}\n"))
    omega = 2 * np.pi * f
    x_bar = solvers.solve_harmonic_balance(problem, omega, 4)
    k = 9
    labels = problem.metadata["labels"]
    out = labels.index("V(out)")
    coeffs = x_bar[out * k : (out + 1) * k, 0]
    # SIN source amp*sin(wt): phasor (sin reference) V_out = amp / (1 + j w R C).
    phasor = amp / (1 + 1j * omega * r * c)
    # x(t) = a cos + b sin = Im(V e^{jwt}) with V = b + j a for a sin reference.
    assert coeffs[2] == pytest.approx(phasor.real, rel=1e-12)
    assert coeffs[1] == pytest.approx(phasor.imag, rel=1e-12)
    assert np.max(np.abs(np.delete(coeffs, [1, 2]))) < 1e-12


def test_thread_count_does_not_change_the_mixer_result() -> None:
    problem = build_mna_problem(parse_netlist_text(MIXER))
    stats1: dict[str, Any] = {}
    stats4: dict[str, Any] = {}
    one = solvers.solve_harmonic_balance(problem, 2 * np.pi * 5e6, 205, hb_stats=stats1, num_threads=1)
    four = solvers.solve_harmonic_balance(problem, 2 * np.pi * 5e6, 205, hb_stats=stats4, num_threads=4)
    assert np.array_equal(one, four)
    assert stats1["threads_used"] == 1
    if core.effective_threads(4) > 1:
        assert stats4["threads_used"] > 1


def test_service_threads_reach_the_native_core() -> None:
    backend = _backend("mna_simulation")
    circuit = backend.parse_text(MIXER.replace(".hb 205", ".hb 205 200n"))
    results = []
    for threads in (1, 4):
        options = backend.build_options(circuit, overrides={"max_iter": 100}, mode=AnalysisMode.HB)
        options.num_threads = threads
        results.append(backend.run(circuit, options))
    assert np.array_equal(results[0].harmonic_state, results[1].harmonic_state)
    assert np.array_equal(results[0].waveform.values, results[1].waveform.values)
    assert results[0].metadata["hb_solver"]["threads_used"] == 1


# ---------------------------------------------------------------------------
# Memory policy and rulings
# ---------------------------------------------------------------------------


def test_memory_policy_raises_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNA_HB_MAX_NNZ", "1000")
    assert solvers.hb_max_nnz() == 1000
    with pytest.raises(SolverConvergenceError) as exc:
        backend = _backend("mna_simulation")
        circuit = backend.parse_text(MIXER)
        backend.run(circuit, backend.build_options(circuit, mode=AnalysisMode.HB))
    assert exc.value.code == "HB_ERROR"
    assert str(exc.value).startswith("HB system too large (N=1233, nnz=")
    assert str(exc.value).endswith("); reduce harmonics")
    # The limit is enforced by the core on the stored entries, not only on N.
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 1 1k\nR1 a 0 1k\n"))
    with pytest.raises(SolverConvergenceError, match=r"HB system too large \(N=202, nnz=20\d+\)"):
        solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, 50, max_nnz=1000)


def test_hb_max_nnz_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNA_HB_MAX_NNZ", raising=False)
    assert solvers.hb_max_nnz() == 30_000_000
    assert solvers.hb_max_nnz({"MNA_HB_MAX_NNZ": "3e7"}) == 30_000_000
    assert solvers.hb_max_nnz({"MNA_HB_MAX_NNZ": " 250000 "}) == 250_000
    for bad in ("abc", "0", "-5", "nan"):
        with pytest.warns(RuntimeWarning):
            assert solvers.hb_max_nnz({"MNA_HB_MAX_NNZ": bad}) == 30_000_000


def test_dimension_limit_is_enforced() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 1 1k\nR1 a 0 1k\n"))
    with pytest.raises(SolverConvergenceError, match=r"HB system too large \(N=2000002, nnz="):
        solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, 500_000, max_nnz=2**62)


@pytest.mark.parametrize("harmonics", [0, -2])
def test_non_positive_harmonics_raise_hb_error(harmonics: int) -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 1 1k\nR1 a 0 1k\n"))
    with pytest.raises(SolverConvergenceError) as exc:
        run_advanced_analysis(problem, AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=harmonics))
    assert exc.value.code == "HB_ERROR"
    assert str(exc.value) == f"harmonics must be a positive integer, got {harmonics}."
    with pytest.raises(SolverConvergenceError, match="harmonics must be a positive integer"):
        solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, harmonics)
    unset = run_advanced_analysis(problem, AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=None))
    assert unset.metadata["harmonics"] == 8


def test_newton_errors_keep_the_reference_texts() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 0.5 1k\nR1 a b 1k\nD1 b 0 1e-14\n"))
    with pytest.raises(SolverConvergenceError, match="^Harmonic balance solver did not converge within limit.$"):
        solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, 4, max_iter=1)
    text = "I1 a 0 SIN 1m 1k\nC1 a 0 1u\n"
    floating = build_mna_problem(parse_netlist_text(text))
    reference_builder = importlib.import_module("mna_reference.mna_builder")
    reference_netlist = importlib.import_module("mna_reference.netlist")
    reference_errors = importlib.import_module("mna_reference.errors")
    with pytest.raises(reference_errors.SolverConvergenceError) as ref_exc:
        ref_solvers.solve_harmonic_balance(
            reference_builder.build_mna_problem(reference_netlist.parse_netlist_text(text)), 2 * np.pi * 1e3, 3
        )
    with pytest.raises(SolverConvergenceError) as exc:
        solvers.solve_harmonic_balance(floating, 2 * np.pi * 1e3, 3)
    assert str(exc.value) == str(ref_exc.value) == "Linear harmonic solve failed."


def test_missing_native_circuit_is_reported() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 SIN 0.5 1k\nR1 a b 1k\nD1 b 0 1e-14\n"))
    with pytest.raises(NetlistError, match="compiled circuit"):
        solvers.solve_harmonic_balance(dataclasses.replace(problem, native=None), 2 * np.pi * 1e3, 3)


# ---------------------------------------------------------------------------
# Krylov option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [0, 1, 7, 64, 257])
@pytest.mark.parametrize("method", ["auto", "gmres", "cg", "minres"])
def test_krylov_choice_from_the_dimension_matches_a_zero_matrix(dimension: int, method: str) -> None:
    expected = describe_krylov_choice(np.zeros((dimension, dimension)), rank="auto", method=method)
    assert describe_krylov_choice_for_dimension(dimension, rank="auto", method=method) == expected
    expected = describe_krylov_choice(np.zeros((dimension, dimension)), rank=5, fallback_restart=9, max_iter=11, method=method)
    assert describe_krylov_choice_for_dimension(dimension, rank=5, fallback_restart=9, max_iter=11, method=method) == expected


KRYLOV_KEYS = (
    "linear_solver",
    "krylov_matrix_kind",
    "krylov_requested_method",
    "krylov_method",
    "krylov_methods",
    "krylov_matrix_dimension",
    "krylov_rank_mode",
    "krylov_resolved_rank",
    "krylov_iteration_budget",
    "krylov_solve_count",
    "krylov_converged",
    "krylov_used_direct_fallback",
)


@pytest.mark.parametrize(
    ("netlist", "options"),
    [
        ("V1 in 0 SIN 1 1k\nV2 in mid SIN 0.5 1.5k\nR1 mid 0 1k\n.hb 3\n", {}),
        ("V1 in 0 SIN 1 1k\nR1 in out 1k\nC1 out 0 1u\n.hb 12\n", {}),
        ("V1 in 0 SIN 1 1k\nR1 in out 1k\nC1 out 0 1u\n.hb 12\n", {"krylov_method": "gmres", "krylov_rank": 6}),
        ("V1 a 0 SIN 0.2 1k\nR1 a b 1k\nD1 b 0 1e-14\nR2 b 0 10k\n.hb 4\n", {"max_iter": 100}),
    ],
    ids=["multitone_linear", "rc_linear_n75", "rc_gmres_rank6", "diode_nonlinear"],
)
def test_krylov_option_metadata_matches_the_oracle(netlist: str, options: dict[str, Any]) -> None:
    request = {"use_krylov": True, **options}
    ref = _run("mna_reference", netlist, request)
    cur = _run("mna_simulation", netlist, request)
    assert not isinstance(ref, Exception) and not isinstance(cur, Exception), (ref, cur)
    _assert_close(cur.harmonic_state, ref.harmonic_state, "x_bar")
    for key in KRYLOV_KEYS:
        assert (key in cur.metadata) == (key in ref.metadata), key
        assert cur.metadata[key] == ref.metadata[key], key
    assert cur.metadata["krylov_engine"] == "mna_core"
    stats: list[dict[str, Any]] = []
    problem = build_mna_problem(parse_netlist_text(netlist))
    harmonics = cur.metadata["harmonics"]
    solvers.solve_harmonic_balance(
        problem,
        2 * np.pi * cur.metadata["base_frequency_hz"],
        harmonics,
        max_iter=options.get("max_iter", 10000),
        tol=1e-9,
        use_krylov=True,
        krylov_method=options.get("krylov_method", "auto"),
        krylov_rank=options.get("krylov_rank", "auto"),
        krylov_stats=stats,
    )
    assert stats and set(stats[0]) >= {
        "method",
        "engine",
        "converged",
        "iterations",
        "relative_residual",
        "requested_method",
        "matrix_kind",
        "route",
        "matrix_dimension",
        "matrix_nnz",
        "matrix_density",
        "operator_storage",
        "resolved_rank",
        "rank_mode",
        "iteration_budget",
        "threads_used",
        "used_direct_fallback",
    }
    assert stats[0]["matrix_dimension"] == len(problem.metadata["labels"]) * (2 * harmonics + 1)


def test_krylov_failure_falls_back_to_the_direct_solve() -> None:
    netlist = "V1 in 0 SIN 1 1k\nR1 in out 1k\nC1 out 0 1u\n"
    problem = build_mna_problem(parse_netlist_text(netlist))
    stats: list[dict[str, Any]] = []
    direct = solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, 30)
    x = solvers.solve_harmonic_balance(
        problem, 2 * np.pi * 1e3, 30, use_krylov=True, krylov_method="gmres", krylov_rank=1, krylov_max_iter=1, krylov_stats=stats
    )
    assert len(stats) == 1 and stats[0]["used_direct_fallback"] is True
    assert stats[0]["converged"] is False
    np.testing.assert_allclose(x, direct, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="Unknown Krylov method"):
        solvers.solve_harmonic_balance(problem, 2 * np.pi * 1e3, 3, use_krylov=True, krylov_method="nope")


def test_core_advanced_never_materializes_dense_matrices() -> None:
    problem = build_mna_problem(parse_netlist_text(MIXER))
    options = AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=205, max_iter=100, f_tol=1e-6, use_krylov=False)
    result = run_advanced_analysis(problem, options)
    assert problem._G_dense is None and problem._C_dense is None  # noqa: SLF001 - lazy caches stay empty
    assert result.metadata["hb_solver"]["newton_iterations"] == 36
