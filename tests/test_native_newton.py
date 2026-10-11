"""Native damped Newton and ``.op`` (``_core.solve_op`` and its facades) against the frozen oracle.

* Every nonlinear corpus ``.op`` converges to the reference operating point (at the gmin
  the reference service settled on) within 1e-8 normalized; failing cases raise the same
  exception class, code and message as the reference.
* Error parity of the ``solve_dc_nr`` facade (``max_iter``, singular Jacobian, singular
  ``G``, no residual reduction, never-converging clamp) with the reference.
* ``init_condition`` handling (2.4.1): DEFAULT/RANDOM/array, the RNG draw of RANDOM, the
  length check and the linear-path diagnostics.
* Krylov stats entries identical to the ``solve_linear`` facade.
* Thread determinism (1 vs 4 threads, bitwise) and GIL release.
* Compile-once gmin stepping in ``PythonBackend.run`` with numerical-only retries.
"""

from __future__ import annotations

import threading
import time
import warnings

import numpy as np
import pytest

from corpus import iter_analyses
from differential.compare import run_request
from mna_reference import solvers as ref_solvers
from mna_reference.api.contracts import AnalysisMode as RefMode
from mna_reference.api.contracts import AnalysisOptions as RefOptions
from mna_reference.core_basic import build_and_run_basic as ref_build_and_run_basic
from mna_reference.mna_builder import build_mna_problem as ref_build
from mna_reference.netlist import parse_netlist_text as ref_parse
from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions
from mna_simulation.backends import python_backend
from mna_simulation.backends.python_backend import PythonBackend, is_numerical_failure, problem_at_gmin
from mna_simulation.core_basic import build_and_run_basic, run_basic_analysis
from mna_simulation.errors import CircuitSimulatorError, NetlistError, SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.linalg import solve_linear
from mna_simulation.netlist import parse_netlist_text
from mna_simulation.utils import DEFAULT_GMIN_STEPS

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

core = native.require_core()

DIODE = "V1 a 0 DC 5\nR1 a b 1k\nD1 b 0 1e-14\n"
DIODE_OP = [5.0, 0.6698509496871571, -0.004330149050312843]


def _diode_fan(count: int, capacitors: bool = False) -> str:
    lines = ["V1 vdd 0 DC 2"]
    for i in range(count):
        lines.append(f"R{i} vdd n{i} {1000 + i}")
        lines.append(f"D{i} n{i} 0 1e-14")
        if capacitors:
            lines.append(f"C{i} n{i} 0 1n")
    return "\n".join(lines) + "\n"


def _inverter_chain(stages: int) -> str:
    lines = ["VDD vdd 0 DC 1.2", "VIN in 0 DC 0"]
    previous = "in"
    for stage in range(1, stages + 1):
        lines += [
            f"MP{stage} o{stage} {previous} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
            f"MN{stage} o{stage} {previous} 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
        ]
        previous = f"o{stage}"
    return "\n".join(lines) + "\n"


def _is_nonlinear(problem) -> bool:
    return bool(ref_solvers.is_nonlinear(problem.solver_f_str) or problem.level1_mos_devices or problem.level1_bjt_devices)


def _nonlinear_op_analyses():
    seen: set[tuple[str, str]] = set()
    params = []
    for case, _index, mode, options in iter_analyses():
        if mode != "op":
            continue
        key = (case.netlist_text, repr(sorted(options.items())))
        if key in seen:
            continue
        try:
            problem = ref_build(ref_parse(case.netlist_text))
        except Exception:  # noqa: BLE001 - structural error cases are not Newton cases
            continue
        if not _is_nonlinear(problem):
            continue
        seen.add(key)
        params.append(pytest.param(case, options, id=f"{case.name}-{len(params)}"))
    return params


NONLINEAR_OPS = _nonlinear_op_analyses()


def _normalized_error(x: np.ndarray, reference: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    return float(np.max(np.abs(x - reference), initial=0.0) / max(1.0, float(np.max(np.abs(reference), initial=0.0))))


def _error_triplet(exc: BaseException) -> tuple[str, str | None, str]:
    return type(exc).__name__, getattr(exc, "code", None), str(exc)


def _raise_from_core(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except core.CoreError as exc:
        solvers._raise_core_error(exc)
        raise


# ---------------------------------------------------------------------------
# Corpus parity
# ---------------------------------------------------------------------------


def test_the_corpus_has_nonlinear_op_cases() -> None:
    assert len(NONLINEAR_OPS) >= 15


@pytest.mark.parametrize(("case", "options"), NONLINEAR_OPS)
def test_native_op_converges_to_the_reference_operating_point(case, options, reference_pkg) -> None:
    if case.deviation_for("op") is not None:
        pytest.skip("deliberate P3b change (2.4.1): checked by the differential deviation check")
    max_iter = int(options.get("max_iter", 10000))
    reference = run_request(reference_pkg, case.netlist_text, "op", options)
    problem = build_mna_problem(parse_netlist_text(case.netlist_text))
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    if reference.error is not None:
        # Same class / code / message as the reference at the last gmin step.
        with pytest.raises(CircuitSimulatorError) as exc:
            _raise_from_core(core.solve_op, problem.native, b, None, DEFAULT_GMIN_STEPS[-1], max_iter)
        assert _error_triplet(exc.value) == _error_triplet(reference.error)
        return
    gmin = float(reference.response.metadata["gmin"])
    x, info = core.solve_op(problem.native, b, None, gmin, max_iter)
    assert x.shape == (problem.native.n,) and x.dtype == np.float64
    assert info["iterations"] >= 0 and info["krylov_stats"] == []
    assert _normalized_error(x, reference.response.dc_solution) <= 1e-8


@pytest.mark.parametrize(
    ("netlist", "options"),
    [
        (DIODE, {"max_iter": 1}),
        (DIODE, {"max_iter": 2}),
        (DIODE, {"max_iter": 0}),
        (DIODE + "R2 x y 1k\n", {}),
        ("VDD d 0 DC 1\nM1 d g 0 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n", {}),
        ("V1 a 0 DC 5\nD1 a 0 1e-14\n", {"max_iter": 3}),
        ("V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\n", {}),
        ("C1 n1 0 1u\n", {}),
        (_inverter_chain(6), {"max_iter": 2}),
    ],
    ids=["max_iter_1", "max_iter_2", "max_iter_0", "jacobian_singular", "mos_floating_gate", "clamped_diode",
         "singular_g", "capacitor_only", "inverter_chain_6"],
)
def test_op_errors_match_the_reference(netlist: str, options: dict) -> None:
    # At gmin 0 (build_and_run_basic) the P3b strategies keep these failures: the junction
    # strategy and the gmin continuation share max_iter, and a structurally singular circuit
    # stays singular. The clamped diode and the 6-stage chain converge with a normal budget
    # (tests/test_native_convergence.py).
    with pytest.raises(Exception) as ref_exc:  # noqa: PT011 - compared below
        ref_build_and_run_basic(ref_parse(netlist), RefOptions(mode=RefMode.OP, **options))
    with pytest.raises(CircuitSimulatorError) as cur_exc:
        build_and_run_basic(parse_netlist_text(netlist), AnalysisOptions(mode=AnalysisMode.OP, **options))
    assert _error_triplet(cur_exc.value) == _error_triplet(ref_exc.value)


def test_solve_dc_nr_facade_on_dense_and_sparse_matrices_matches_the_reference() -> None:
    reference = ref_build(ref_parse(DIODE))
    problem = build_mna_problem(parse_netlist_text(DIODE))
    expected = ref_solvers.solve_dc_nr(reference.G, reference.solver_f_str, reference.b_dc, max_iter=200)
    for G in (problem.G, problem.G_sparse):
        x = solvers.solve_dc_nr(G, problem.solver_f_str, problem.b_dc, max_iter=200)
        assert x.shape == (3, 1)
        assert _normalized_error(x, expected) <= 1e-12
    linear = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 2k\n"))
    x = solvers.solve_dc_nr(linear.G, linear.solver_f_str, linear.b_dc)
    np.testing.assert_allclose(x.reshape(-1), [1.0, 2.0 / 3.0, -1.0 / 3000.0], rtol=1e-14)


# ---------------------------------------------------------------------------
# init_condition (CPP_MIGRATION_PLAN.md 2.4.1)
# ---------------------------------------------------------------------------


def test_binding_x0_is_the_newton_starting_point() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE))
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    cold, cold_info = core.solve_op(problem.native, b, None, 0.0, 200)
    warm, warm_info = core.solve_op(problem.native, b, np.array([5.0, 0.7, -0.004]), 0.0, 200)
    assert warm_info["iterations"] < cold_info["iterations"]
    np.testing.assert_allclose(warm, DIODE_OP, rtol=1e-9)
    np.testing.assert_allclose(cold, DIODE_OP, rtol=1e-9)
    with pytest.raises(core.CoreError) as exc:
        core.solve_op(problem.native, b, np.zeros(2), 0.0, 200)
    assert exc.value.code == "NETLIST_FATAL"


def test_random_init_condition_draws_like_the_reference() -> None:
    reference = ref_build(ref_parse(DIODE))
    problem = build_mna_problem(parse_netlist_text(DIODE))
    np.random.seed(1234)
    current = solvers.solve_dc_nr(problem.G, problem.solver_f_str, problem.b_dc, init_cond="RANDOM")
    after_current = np.random.random()
    np.random.seed(1234)
    expected = ref_solvers.solve_dc_nr(reference.G, reference.solver_f_str, reference.b_dc, init_cond="RANDOM")
    after_reference = np.random.random()
    assert after_current == after_reference  # same number of draws from the global RNG
    assert _normalized_error(current, expected) <= 1e-8


def test_newton_start_rules() -> None:
    diagnostics: list[str] = []
    assert solvers.newton_start(None, 3, True).tolist() == [0.0, 0.0, 0.0]
    assert solvers.newton_start(" default ", 3, True).tolist() == [0.0, 0.0, 0.0]
    start = np.array([[1.0], [2.0], [3.0]])
    copied = solvers.newton_start(start, 3, True)
    copied[0] = 99.0
    assert start[0, 0] == 1.0
    assert solvers.newton_start("RANDOM", 4, False, diagnostics) is None
    assert solvers.newton_start(np.zeros(4), 4, False, diagnostics) is None
    assert solvers.newton_start(None, 4, False, diagnostics) is None
    assert len(diagnostics) == 2 and all("ignored" in line for line in diagnostics)
    with pytest.raises(NetlistError) as exc:
        solvers.newton_start("garbage", 3, True)
    assert exc.value.code == "NETLIST_FATAL" and str(exc.value) == "Unsupported init condition 'garbage'."
    for nonlinear in (True, False):
        with pytest.raises(NetlistError) as length_exc:
            solvers.newton_start([1.0, 2.0], 3, nonlinear)
        assert str(length_exc.value) == "init_condition has 2 values, expected 3 (one per unknown)."


def test_op_diagnostics_reach_the_service_response() -> None:
    from mna_simulation.api.contracts import AnalysisRequest
    from mna_simulation.api.service import SimulationService

    response = SimulationService().run_request(
        AnalysisRequest(netlist_text="V1 a 0 DC 5\nR1 a 0 1k\n", mode=AnalysisMode.OP, options={"init_condition": "random"})
    )
    assert response.dc_solution == pytest.approx([5.0, -5e-3])
    assert response.diagnostics == [
        "init_condition 'RANDOM' ignored: the circuit is linear, its operating point needs no starting point."
    ]


# ---------------------------------------------------------------------------
# Krylov statistics: the solve_linear entries, produced natively
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("netlist", ["V1 n1 0 DC 1\nV2 n3 n4 DC 0\nG1 n2 0 n1 0 2\nF1 n2 0 V1 3\nH1 n3 0 V1 4\nR1 n2 0 1k\nR2 n4 0 1k\n", "rlc8"])
@pytest.mark.parametrize("method", ["auto", "arnoldi_gmres", "conjugate_residual"])
def test_linear_op_krylov_stats_equal_the_solve_linear_entries(netlist: str, method: str) -> None:
    if netlist == "rlc8":
        from corpus import rlc_mesh_netlist

        netlist = rlc_mesh_netlist(8, 8)
    problem = build_mna_problem(parse_netlist_text(netlist))
    kwargs = {"use_krylov": True, "krylov_rank": "auto", "krylov_method": method}
    native_stats: list[dict] = []
    x = solvers.solve_dc_nr(problem.G, problem.solver_f_str, problem.b_dc, krylov_stats=native_stats, **kwargs)
    facade_stats: list[dict] = []
    expected = solve_linear(problem.G, problem.b_dc, krylov_stats=facade_stats, **kwargs)
    assert len(native_stats) == 1
    assert list(native_stats[0]) == list(facade_stats[0])
    assert native_stats == facade_stats
    np.testing.assert_array_equal(x, expected)


def test_newton_krylov_stats_equal_the_solve_linear_entry_of_the_first_iteration() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE))
    kwargs = {"use_krylov": True, "krylov_rank": 2, "krylov_method": "auto"}
    native_stats: list[dict] = []
    solvers.solve_dc_nr(problem.G, problem.solver_f_str, problem.b_dc, max_iter=200, krylov_stats=native_stats, **kwargs)
    f_func, j_func = solvers.compile_nl_functions(problem.solver_f_str, 3)
    x0 = np.zeros(3)
    jacobian = problem.G + j_func(*x0)
    residual = problem.G @ x0.reshape(3, 1) + f_func(*x0) - problem.b_dc
    facade_stats: list[dict] = []
    solve_linear(jacobian, -residual, krylov_stats=facade_stats, **kwargs)
    assert len(native_stats) > 1
    assert native_stats[0] == facade_stats[0]
    assert all(entry["operator_storage"] == "dense" for entry in native_stats)


# ---------------------------------------------------------------------------
# Threads and the GIL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "options"), NONLINEAR_OPS[:12])
def test_op_is_bitwise_identical_for_one_and_four_threads(case, options) -> None:
    problem = build_mna_problem(parse_netlist_text(case.netlist_text))
    results = []
    for threads in (1, 4):
        try:
            result = run_basic_analysis(
                problem, AnalysisOptions(mode=AnalysisMode.OP, num_threads=threads, max_iter=options.get("max_iter", 10000))
            )
        except CircuitSimulatorError as exc:
            results.append(_error_triplet(exc))
            continue
        results.append(np.asarray(result.dc_solution).tobytes())
    assert results[0] == results[1]


def test_parallel_device_evaluation_is_bitwise_identical_on_a_large_circuit() -> None:
    count = int(core.PARALLEL_DEVICE_MIN_COUNT) + 52
    problem = build_mna_problem(parse_netlist_text(_diode_fan(count)))
    assert problem.native.device_counts["diode"] == count
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    serial, serial_info = core.solve_op(problem.native, b, None, 0.0, 200, num_threads=1)
    parallel, parallel_info = core.solve_op(problem.native, b, None, 0.0, 200, num_threads=4)
    assert serial_info["iterations"] == parallel_info["iterations"]
    assert serial.tobytes() == parallel.tobytes()
    assert 0.4 < serial[1] < 0.8


def test_solve_op_releases_the_gil() -> None:
    count = int(core.PARALLEL_DEVICE_MIN_COUNT)
    problem = build_mna_problem(parse_netlist_text(_diode_fan(count)))
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    core.solve_op(problem.native, b, None, 0.0, 200)  # warm-up
    ticks = 0
    stop = threading.Event()

    def spin() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        time.sleep(0.01)
        start_ticks = ticks
        started = time.perf_counter()
        calls = 0
        while time.perf_counter() - started < 0.2:
            core.solve_op(problem.native, b, None, 0.0, 200)
            calls += 1
        during = ticks - start_ticks
    finally:
        stop.set()
        worker.join()
    # With the GIL held for the whole native call the spinning thread could only run
    # between calls; released, it keeps counting while the solves run.
    assert calls >= 1
    assert during > 1000


# ---------------------------------------------------------------------------
# Compile-once gmin stepping (PythonBackend.run)
# ---------------------------------------------------------------------------


def _backend_run(netlist: str, mode: AnalysisMode, **options):
    backend = PythonBackend()
    circuit = backend.parse_text(netlist)
    return backend.run(circuit, AnalysisOptions(mode=mode, gmin_steps=list(DEFAULT_GMIN_STEPS), **options))


def test_gmin_stepping_compiles_the_circuit_once(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[float] = []
    real_build = python_backend.build_mna_problem

    def counting_build(circuit, gmin=0.0):
        builds.append(gmin)
        return real_build(circuit, gmin=gmin)

    monkeypatch.setattr(python_backend, "build_mna_problem", counting_build)
    result = _backend_run("V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n", AnalysisMode.OP)
    assert result.metadata["gmin"] == 1e-12
    assert builds == [0.0]


def test_gmin_stepping_retries_numerical_failures_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    real_run = python_backend.run_basic_analysis

    def counting_run(problem, options):
        calls.append(problem.gmin)
        return real_run(problem, options)

    monkeypatch.setattr(python_backend, "run_basic_analysis", counting_run)
    # An inconsistent voltage-source loop is singular at every gmin (node-to-ground gmin never
    # touches branch rows). Was: a floating resistor island, which P3b node gmin now solves at
    # the first gmin step.
    with pytest.raises(SolverConvergenceError):
        _backend_run("V1 a 0 DC 1\nV2 b 0 DC 2\nV3 a b DC 0.5\nR1 a 0 1k\n", AnalysisMode.OP)
    assert calls == list(DEFAULT_GMIN_STEPS)
    calls.clear()
    assert _backend_run("V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\n", AnalysisMode.OP).metadata["gmin"] == 1e-12
    assert calls == [0.0, 1e-12]
    calls.clear()
    with pytest.raises(NetlistError) as exc:
        _backend_run("V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\n", AnalysisMode.TRAN, tran_stop=-1.0, tran_step=1e-4)
    assert exc.value.code == "TRAN_ERROR" and calls == [0.0]
    calls.clear()
    with pytest.raises(NetlistError) as length_exc:
        _backend_run(DIODE, AnalysisMode.OP, init_condition=np.zeros(2))
    assert length_exc.value.code == "NETLIST_FATAL" and calls == [0.0]


def test_is_numerical_failure_classification() -> None:
    numerical = [
        SolverConvergenceError("x", code="NR_CONVERGENCE"),
        SolverConvergenceError("x", code="JACOBIAN_SINGULAR"),
        SolverConvergenceError("x", code="AC_ERROR"),
        SolverConvergenceError("x", code="LU_CONVERGENCE"),
        SolverConvergenceError("x"),
        SolverConvergenceError("Harmonic balance solver did not converge within limit.", code="HB_ERROR"),
        SolverConvergenceError("Singular HB Jacobian.", code="HB_ERROR"),
        SolverConvergenceError("Linear harmonic solve failed.", code="HB_ERROR"),
        np.linalg.LinAlgError("Singular matrix"),
    ]
    structural = [
        SolverConvergenceError("No sinusoidal sources found.", code="HB_ERROR"),
        SolverConvergenceError("Could not determine HB base frequency.", code="HB_ERROR"),
        NetlistError("x"),
        NetlistError("t_step must be positive.", code="TRAN_ERROR"),
        NetlistError("x", code="MOR_CONFIG"),
        ZeroDivisionError("float division by zero"),
        TypeError("x"),
        ValueError("x"),
    ]
    assert all(is_numerical_failure(exc) for exc in numerical)
    assert not any(is_numerical_failure(exc) for exc in structural)


def test_problem_at_gmin_matches_a_build_at_that_gmin() -> None:
    netlist = DIODE + "C1 b 0 1n\nQ1 a b 0 QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5\n"
    circuit = parse_netlist_text(netlist)
    base = build_mna_problem(circuit, gmin=0.0)
    assert problem_at_gmin(base, circuit, 0.0) is base
    # The view adds the native gmin pattern, which holds one node-to-ground stamp per node on
    # top of the builder's device-terminal stamps (P3b, 2.4.1); build_mna_problem(gmin) only
    # stamps the latter into its G.
    nodes = base.metadata["num_nodes"]
    node_diagonal = np.diag((np.arange(base.native.n) < nodes).astype(float))
    for gmin in DEFAULT_GMIN_STEPS[1:]:
        view = problem_at_gmin(base, circuit, gmin)
        rebuilt = build_mna_problem(circuit, gmin=gmin)
        assert view.native is base.native and view.gmin == gmin
        np.testing.assert_allclose(view.G, rebuilt.G + gmin * node_diagonal, rtol=1e-15, atol=0.0)
        assert view.metadata["G_nnz"] == rebuilt.metadata["G_nnz"]
        np.testing.assert_array_equal(view.C, rebuilt.C)


def test_gmin_reported_by_the_service_matches_the_reference(reference_pkg, current_pkg) -> None:
    # Was also: the 4-stage inverter chain at gmin 1e-5; it now converges at gmin 0 (P3b gmin
    # continuation, tests/test_native_convergence.py).
    for netlist, mode, options in (
        ("V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n", "op", {}),
        ("V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n", "tran", {"t_stop": 1e-3, "t_step": 1e-4}),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = run_request(reference_pkg, netlist, mode, options)
            cur = run_request(current_pkg, netlist, mode, options)
        assert ref.error is None and cur.error is None
        assert cur.response.metadata["gmin"] == ref.response.metadata["gmin"]
