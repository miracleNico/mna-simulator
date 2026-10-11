"""Native Backward-Euler transient (``_core.solve_transient`` and its facades) against the frozen oracle.

* The time grid is bitwise equal to the reference ``_transient_time_points``.
* Every corpus ``.tran`` waveform matches the reference (at the gmin the reference service
  settled on) within the case's tolerance class.
* Linear constant-step runs factor once (the cached-factorization path), the Krylov path
  records the ``solve_linear`` stats entries.
* Progress callback: counts, payload, exceptions.
* The P3 rulings of CPP_MIGRATION_PLAN.md 2.4.1 at the facade level.
* Threads: bitwise identical results for 1 and 4 threads; a small circuit is not slower
  with 4 threads (device evaluation stays serial below the parallel threshold).
"""

from __future__ import annotations

import random
import threading
import time
import warnings

import numpy as np
import pytest

from corpus import iter_analyses, rlc_mesh_netlist
from differential.compare import TOLERANCES, run_request
from mna_reference import solvers as ref_solvers
from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions
from mna_simulation.backends.python_backend import problem_at_gmin
from mna_simulation.core_basic import run_basic_analysis
from mna_simulation.errors import CircuitSimulatorError, NetlistError, SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.linalg import solve_linear
from mna_simulation.netlist import parse_netlist_text

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

core = native.require_core()

RC = "V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\n"
DIODE_RC = "V1 a 0 DC 5\nR1 a b 1k\nD1 b 0 1e-14\nC1 b 0 1n\n"
BJT_STAGE = """
VCC vcc 0 DC 12
VIN in 0 SIN 0.01 1k
CIN in b 1u
RB1 vcc b 100k
RB2 b 0 20k
RC vcc c 4.7k
RE e 0 1k
Q1 c b e QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
"""


def _diode_rc_fan(count: int) -> str:
    lines = ["V1 vdd 0 SIN 1 1e5"]
    for i in range(count):
        lines.append(f"R{i} vdd n{i} {1000 + i}")
        lines.append(f"D{i} n{i} 0 1e-14")
        lines.append(f"C{i} n{i} 0 1n")
    return "\n".join(lines) + "\n"


def _source_arrays(problem):
    rows, programs = solvers._time_sources(problem.b_time_str)
    return rows, programs, np.ascontiguousarray(problem.b_dc.reshape(-1))


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------


def _grid_pairs() -> list[tuple[float, float]]:
    rng = random.Random("time-grid-pairs")
    pairs = [
        (1.0, 0.1),
        (1e-3, 3e-4),
        (1e-3, 5e-3),
        (1e-3 + 1e-13, 1e-4),
        (1e-3 + 1e-11, 1e-4),
        (1e-3 - 1e-13, 1e-4),
        (1e-3 - 1e-11, 1e-4),
        (6e-9, 0.25e-9),
        (10e-3, 5e-6),
        (40e-9, 1e-9),
        (0.0, 1e-3),
        (2e-3, 1e-3),
        (1e-12, 1e-13),
        (7.0, 0.7),
        (5e-6, 0.5e-6),
    ]
    while len(pairs) < 50:
        t_step = 10.0 ** rng.uniform(-12, 0)
        pairs.append((t_step * rng.uniform(0.5, 400.0), t_step))
    return pairs


@pytest.mark.parametrize(("t_stop", "t_step"), _grid_pairs())
def test_time_grid_is_bitwise_equal_to_the_reference(t_stop: float, t_step: float) -> None:
    current = solvers._transient_time_points(t_stop, t_step)
    expected = ref_solvers._transient_time_points(t_stop, t_step)
    assert current.dtype == np.float64 and current.shape == expected.shape
    assert current.tobytes() == np.asarray(expected, dtype=np.float64).tobytes()


@pytest.mark.parametrize(
    ("t_stop", "t_step", "message"),
    [
        (-1.0, 1e-3, "t_stop must be non-negative."),
        (1e-3, 0.0, "t_step must be positive."),
        (float("nan"), 1e-3, "t_stop must be finite."),
        (float("inf"), 1e-3, "t_stop must be finite."),
        (1e-3, float("inf"), "t_step must be finite."),
        (1e-3, float("nan"), "t_step must be finite."),
    ],
)
def test_time_grid_errors_are_tran_error(t_stop: float, t_step: float, message: str) -> None:
    with pytest.raises(NetlistError) as exc:
        solvers._transient_time_points(t_stop, t_step)
    assert exc.value.code == "TRAN_ERROR" and str(exc.value) == message


# ---------------------------------------------------------------------------
# Corpus waveform parity
# ---------------------------------------------------------------------------


def _tran_analyses():
    params = []
    for case, _index, mode, options in iter_analyses():
        if mode != "tran" or options.get("use_mor") or case.expects_error is not None:
            continue
        params.append(pytest.param(case, options, id=f"{case.name}-{len(params)}"))
    return params


TRAN_ANALYSES = _tran_analyses()


def _analysis_options(options: dict) -> AnalysisOptions:
    return AnalysisOptions(
        mode=AnalysisMode.TRAN,
        tran_stop=options["t_stop"],
        tran_step=options["t_step"],
        max_iter=options.get("max_iter", 10000),
        use_krylov=bool(options.get("use_krylov", False)),
        krylov_rank=options.get("krylov_rank", "auto"),
        krylov_method=options.get("krylov_method", "auto"),
    )


def test_the_corpus_has_transient_cases() -> None:
    assert len(TRAN_ANALYSES) >= 30


@pytest.mark.parametrize(("case", "options"), TRAN_ANALYSES)
def test_transient_waveform_matches_the_reference(case, options, reference_pkg) -> None:
    reference = run_request(reference_pkg, case.netlist_text, "tran", options)
    assert reference.error is None, reference.error
    gmin = float(reference.response.metadata["gmin"])
    circuit = parse_netlist_text(case.netlist_text)
    problem = problem_at_gmin(build_mna_problem(circuit), circuit, gmin)
    result = run_basic_analysis(problem, _analysis_options(options))
    ref_waveform = reference.response.waveform
    if "output_max_points" in options:
        # The service decimates the response; compare the kept samples.
        keep = np.searchsorted(result.waveform.time, np.asarray(ref_waveform["time"], dtype=float))
        values, times = result.waveform.values[:, keep], result.waveform.time[keep]
    else:
        values, times = result.waveform.values, result.waveform.time
    np.testing.assert_array_equal(times, np.asarray(ref_waveform["time"], dtype=float))
    rtol, atol = TOLERANCES[case.tolerance_for("tran", options)]
    np.testing.assert_allclose(values, np.asarray(ref_waveform["values"], dtype=float), rtol=rtol, atol=atol)
    assert result.metadata["op_used"] == reference.response.metadata["op_used"]


# ---------------------------------------------------------------------------
# Binding behaviour
# ---------------------------------------------------------------------------


def test_linear_constant_step_transient_factors_once() -> None:
    problem = build_mna_problem(parse_netlist_text(rlc_mesh_netlist(6, 6)))
    rows, programs, b_dc = _source_arrays(problem)
    x0 = np.zeros(problem.native.n)
    grid = solvers._transient_time_points(40e-9, 1e-9)
    values, info = core.solve_transient(problem.native, x0, grid, b_dc, rows, programs)
    assert values.shape == (problem.native.n, len(grid))
    # t_k - t_{k-1} of the accumulated grid changes in its last bits a few times; every
    # step uses its exact h (as the reference does), so the run factors once per run of
    # bitwise-equal steps, not once per step.
    steps = np.diff(grid)
    runs = 1 + int(np.count_nonzero(steps[1:] != steps[:-1]))
    assert 1 < runs < 10
    assert info["factorizations"] == runs and info["newton_iterations"] == 0
    # The cached factorization gives what a fresh solve per step gives.
    G, C = problem.G, problem.C
    x_prev = x0.reshape(-1, 1)
    for step in range(1, 6):
        h = float(grid[step] - grid[step - 1])
        b_now = solvers.compile_time_source_func(problem.b_time_str)(float(grid[step])) + problem.b_dc
        x_prev = np.linalg.solve(G + C / h, (C / h) @ x_prev + b_now)
        np.testing.assert_allclose(values[:, step], x_prev.reshape(-1), rtol=1e-9, atol=1e-15)
    uneven, uneven_info = core.solve_transient(
        problem.native, x0, solvers._transient_time_points(1.1, 0.25), b_dc, rows, programs
    )
    assert uneven.shape[1] == 6 and uneven_info["factorizations"] == 2


def test_progress_callback_counts_and_payload() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE_RC))
    rows, programs, b_dc = _source_arrays(problem)
    x0 = np.zeros(problem.native.n)
    grid = solvers._transient_time_points(1e-6, 1e-7)  # 10 steps
    for every, expected in ((0, []), (1, list(range(1, 11))), (3, [3, 6, 9, 10]), (25, [10])):
        calls: list[tuple[int, float, np.ndarray]] = []
        values, info = core.solve_transient(
            problem.native,
            x0,
            grid,
            b_dc,
            rows,
            programs,
            progress=lambda step, t, x: calls.append((step, t, x)),
            progress_every=every,
        )
        assert [step for step, _t, _x in calls] == expected
        assert info["progress_calls"] == len(expected)
        for step, t, x in calls:
            assert t == grid[step]
            np.testing.assert_array_equal(x, values[:, step])


def test_progress_callback_exceptions_propagate() -> None:
    problem = build_mna_problem(parse_netlist_text(RC))
    rows, programs, b_dc = _source_arrays(problem)

    def fail(step: int, t: float, x: np.ndarray) -> None:
        if step == 2:
            raise RuntimeError("stop streaming")

    with pytest.raises(RuntimeError, match="stop streaming"):
        core.solve_transient(
            problem.native, np.zeros(3), solvers._transient_time_points(1e-3, 1e-4), b_dc, rows, programs,
            progress=fail, progress_every=1,
        )
    with pytest.raises(TypeError):
        core.solve_transient(
            problem.native, np.zeros(3), np.array([0.0, 1.0]), b_dc, rows, programs, progress=42, progress_every=1
        )


def test_solve_transient_facade_streams_progress() -> None:
    problem = build_mna_problem(parse_netlist_text(BJT_STAGE))
    steps: list[int] = []
    waveform = solvers.solve_transient(
        problem, 2e-3, 1e-5, max_iter=200, progress=lambda step, t, x: steps.append(step), progress_every=50
    )
    assert waveform.values.shape[1] == 201
    assert steps == [50, 100, 150, 200]


def test_bad_binding_inputs_are_netlist_fatal() -> None:
    problem = build_mna_problem(parse_netlist_text(RC))
    rows, programs, b_dc = _source_arrays(problem)
    with pytest.raises(core.CoreError) as exc:
        core.solve_transient(problem.native, np.zeros(2), np.array([0.0, 1.0]), b_dc, rows, programs)
    assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError):
        core.solve_transient(problem.native, np.zeros(3), np.array([0.0, 0.0]), b_dc, rows, programs)
    with pytest.raises(core.CoreError):
        core.solve_transient(problem.native, np.zeros(3), np.array([0.0, 1.0]), b_dc, np.array([7], dtype=np.int64), [])


def test_krylov_transient_stats_equal_the_solve_linear_entries() -> None:
    problem = build_mna_problem(parse_netlist_text(rlc_mesh_netlist(8, 8)))
    stats: list[dict] = []
    x0 = np.zeros((problem.native.n, 1))
    waveform = solvers.solve_transient(
        problem, 3e-9, 1e-9, init_cond=x0, use_krylov=True, krylov_rank="auto", krylov_stats=stats
    )
    assert len(stats) == 3
    h = 1e-9
    operator = problem.G_sparse + problem.C_sparse / h
    b_now = solvers.compile_time_source_func(problem.b_time_str)(1e-9) + problem.b_dc
    expected_stats: list[dict] = []
    expected = solve_linear(
        operator,
        (problem.C_sparse / h) @ x0 + b_now,
        use_krylov=True,
        krylov_rank="auto",
        krylov_stats=expected_stats,
        x0=x0,
    )
    assert stats[0] == expected_stats[0]
    assert stats[0]["operator_storage"] == "csr"
    np.testing.assert_array_equal(waveform.values[:, 1], expected.reshape(-1))


# ---------------------------------------------------------------------------
# P3 rulings (CPP_MIGRATION_PLAN.md 2.4.1) at the facade level
# ---------------------------------------------------------------------------


def test_zero_t_stop_returns_the_initial_sample() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE_RC))
    waveform = solvers.solve_transient(problem, 0.0, 1e-7, max_iter=200)
    assert waveform.time.tolist() == [0.0]
    np.testing.assert_allclose(waveform.values[:, 0], [5.0, 0.6698509496871571, -0.004330149050312843], rtol=1e-6)
    result = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=0.0, tran_step=1e-7, max_iter=200))
    assert result.waveform.time.tolist() == [0.0]
    level1 = build_mna_problem(parse_netlist_text(BJT_STAGE))
    op_only = run_basic_analysis(level1, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=0.0, max_iter=200))
    assert op_only.metadata["op_used"] is True
    np.testing.assert_array_equal(op_only.waveform.values[:, 0], op_only.metadata["operating_point"]["values"])


def test_singular_and_non_finite_steps_are_nr_convergence() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 -1u\n"))
    with pytest.raises(SolverConvergenceError) as exc:
        solvers.solve_transient(problem, 5e-3, 1e-3)
    assert exc.value.code == "NR_CONVERGENCE"
    assert str(exc.value) == "Linear system is singular at t=1.00e-03 s (G + C/h matrix): Singular matrix"
    nan_source = build_mna_problem(parse_netlist_text("V1 a 0 FUNC 0 nan\nR1 a 0 1k\n"))
    with pytest.raises(SolverConvergenceError) as nan_exc:
        solvers.solve_transient(nan_source, 2e-3, 1e-3, init_cond=np.zeros((2, 1)))
    assert nan_exc.value.code == "NR_CONVERGENCE" and str(nan_exc.value) == "non-finite solution at t=1.00e-03 s"


def test_init_condition_on_transient() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE_RC))
    default = solvers.solve_transient(problem, 1e-6, 2e-7, max_iter=200)
    np.random.seed(5)
    random_start = solvers.solve_transient(problem, 1e-6, 2e-7, max_iter=200, init_cond="random")
    np.testing.assert_allclose(random_start.values, default.values, rtol=1e-6, atol=1e-12)
    with pytest.raises(NetlistError) as exc:
        solvers.solve_transient(problem, 1e-6, 2e-7, init_cond="garbage")
    assert str(exc.value) == "Unsupported init condition 'garbage'."
    with pytest.raises(NetlistError) as length_exc:
        solvers.solve_transient(problem, 1e-6, 2e-7, init_cond=np.zeros(2))
    assert str(length_exc.value) == "init_condition has 2 values, expected 3 (one per unknown)."
    diagnostics: list[str] = []
    linear = build_mna_problem(parse_netlist_text(RC))
    solvers.solve_transient(linear, 2e-3, 1e-3, init_cond="DEFAULT", diagnostics=diagnostics)
    assert len(diagnostics) == 1 and "ignored" in diagnostics[0]


def test_num_threads_reaches_the_native_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []
    real_op, real_tran = core.solve_op, core.solve_transient

    def op(*args, **kwargs):
        seen.append(("op", args[7]))
        return real_op(*args, **kwargs)

    def tran(*args, **kwargs):
        seen.append(("tran", args[10]))
        return real_tran(*args, **kwargs)

    monkeypatch.setattr(core, "solve_op", op)
    monkeypatch.setattr(core, "solve_transient", tran)
    problem = build_mna_problem(parse_netlist_text(BJT_STAGE))
    run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=1e-4, tran_step=1e-5, num_threads=3))
    run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.OP, num_threads="auto"))
    assert seen == [("op", 3), ("tran", 3), ("op", 1)]


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "options"), [p for p in TRAN_ANALYSES if p.values[1].get("max_iter")][:8])
def test_transient_is_bitwise_identical_for_one_and_four_threads(case, options) -> None:
    problem = build_mna_problem(parse_netlist_text(case.netlist_text))
    outputs = []
    for threads in (1, 4):
        analysis = _analysis_options(options)
        analysis.num_threads = threads
        try:
            outputs.append(run_basic_analysis(problem, analysis).waveform.values.tobytes())
        except CircuitSimulatorError as exc:
            outputs.append((type(exc).__name__, exc.code, str(exc)))
    assert outputs[0] == outputs[1]


def test_large_transient_is_bitwise_identical_with_parallel_device_evaluation() -> None:
    count = int(core.PARALLEL_DEVICE_MIN_COUNT) + 10
    problem = build_mna_problem(parse_netlist_text(_diode_rc_fan(count)))
    rows, programs, b_dc = _source_arrays(problem)
    x0 = np.zeros(problem.native.n)
    grid = solvers._transient_time_points(4e-6, 1e-6)
    serial, serial_info = core.solve_transient(problem.native, x0, grid, b_dc, rows, programs, max_iter=200, num_threads=1)
    parallel, parallel_info = core.solve_transient(
        problem.native, x0, grid, b_dc, rows, programs, max_iter=200, num_threads=4
    )
    assert serial_info["newton_iterations"] == parallel_info["newton_iterations"] > 0
    assert serial.tobytes() == parallel.tobytes()


def test_small_transient_is_not_slower_with_four_threads() -> None:
    # Below PARALLEL_DEVICE_MIN_COUNT devices the device evaluation stays serial, so a
    # thread request never pays thread start-up per Newton evaluation (it used to cost ~10x).
    problem = build_mna_problem(parse_netlist_text(BJT_STAGE))

    def run(threads: int) -> tuple[float, bytes]:
        options = AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=4e-3, tran_step=5e-6, max_iter=200, num_threads=threads)
        run_basic_analysis(problem, options)
        best = float("inf")
        payload = b""
        for _ in range(5):
            started = time.perf_counter()
            result = run_basic_analysis(problem, options)
            best = min(best, time.perf_counter() - started)
            payload = result.waveform.values.tobytes()
        return best, payload

    serial_time, serial = run(1)
    parallel_time, parallel = run(4)
    assert serial == parallel
    assert parallel_time <= 1.5 * serial_time + 0.005, (serial_time, parallel_time)


def test_solve_transient_releases_the_gil() -> None:
    problem = build_mna_problem(parse_netlist_text(BJT_STAGE))
    rows, programs, b_dc = _source_arrays(problem)
    x0 = np.ascontiguousarray(solvers.solve_operating_point(problem, problem.b_dc, max_iter=200).reshape(-1))
    grid = solvers._transient_time_points(20e-3, 5e-6)
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
        before = ticks
        core.solve_transient(problem.native, x0, grid, b_dc, rows, programs, max_iter=200)
        during = ticks - before
    finally:
        stop.set()
        worker.join()
    assert during > 1000


def test_the_oracle_still_treats_zero_t_stop_as_unset(reference_pkg) -> None:
    # The oracle keeps the old behaviour (0 means unset); no corpus case depends on it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = run_request(reference_pkg, RC, "tran", {"t_stop": 0.0, "t_step": 0.05})
    assert ref.error is None and len(ref.response.waveform["time"]) == 3
