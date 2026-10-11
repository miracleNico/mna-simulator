"""Native AC sweep (``_core.ac_sweep`` behind ``solvers.solve_ac``) vs the frozen reference.

* Every corpus ``ac`` analysis on the full sweep (the MOR sweeps have their own path and are
  covered by the differential suite): frequencies bit-identical, magnitudes and the AC operating
  point within the case tolerance (linear 1e-9 relative, Level-1/nonlinear 1e-6), and 1 vs 4
  threads bitwise identical. The 1035-unknown RLC mesh is compared too (slow).
* ``points`` 1 and 2, and ``f_start = 0`` (a DC point since 2.4.1) against the reference
  ``solve_ac`` called with the same literal values.
* Thread determinism 1 / 2 / 4 (bitwise) on a 10x10 RLC mesh and the three-stage BJT
  amplifier, direct and Krylov.
* Singular systems: ``SolverConvergenceError`` ``AC_ERROR`` with the reference message, and the
  lowest singular frequency reported for every thread count.
* Krylov option: the per-frequency ``krylov_stats`` entries are the ones the pre-native loop
  (``solve_linear`` on the dense system) recorded, so every ``krylov_*`` metadata field keeps
  its meaning; the contract fields match the oracle.
* Binding: shapes, dtypes, input validation, error codes, GIL release.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from corpus import BJT_THREE_STAGE, iter_analyses, rlc_mesh_netlist
from differential.compare import TOLERANCES, run_request
from mna_reference import solvers as ref_solvers
from mna_reference.mna_builder import build_mna_problem as reference_build
from mna_reference.netlist import parse_netlist_text as reference_parse

from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions
from mna_simulation.core_basic import _result_metadata, build_and_run_basic, run_basic_analysis
from mna_simulation.errors import SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.netlist import parse_netlist_text

core = native.require_core()

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

RC_NET = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n"
# Node c is connected through capacitors only: singular at 0 Hz, regular elsewhere.
DC_SINGULAR_NET = "V1 a 0 AC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n"
# A resistor island with no path to ground: singular at every frequency.
ISLAND_NET = "V1 a 0 AC 1\nR1 a 0 1k\nR2 x y 1k\n"
CE_NET = (
    "VCC vcc 0 DC 12\nVIN in 0 AC 1\nCIN in b 10u\nRB1 vcc b 100k\nRB2 b 0 22k\n"
    "RC vcc c 4.7k\nRE e 0 1k\nCE e 0 100u\nQ1 c b e QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5\n"
)


def ac_mesh(rows: int, cols: int) -> str:
    """The workload RLC mesh driven by an AC current source (its STEP source has no AC part)."""

    netlist = rlc_mesh_netlist(rows, cols, ".tran 80n 10n")
    assert netlist.startswith("IIN n_0_0 0 STEP 1m 20n\n")
    return netlist.replace("IIN n_0_0 0 STEP 1m 20n", "IIN n_0_0 0 AC 1m", 1)


def _full_sweep_cases() -> list:
    params = []
    for case, index, mode, options in iter_analyses():
        if mode != "ac" or options.get("use_mor") or case.expects_error:
            continue
        marks = [pytest.mark.slow] if "slow" in case.tags else []
        params.append(pytest.param(case, options, id=f"{case.name}-{index}", marks=marks))
    return params


FULL_SWEEP_CASES = _full_sweep_cases()


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _assert_bitwise(a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape
    assert np.array_equal(_bits(a), _bits(b))


def _assert_close(current, reference, rtol: float, atol: float) -> None:
    cur = np.asarray(current, dtype=float)
    ref = np.asarray(reference, dtype=float)
    assert cur.shape == ref.shape
    bad = ~(np.abs(cur - ref) <= atol + rtol * np.abs(ref))
    assert not bad.any(), (np.argwhere(bad)[:5].tolist(), cur[bad][:5], ref[bad][:5])


def _problems(netlist: str):
    return build_mna_problem(parse_netlist_text(netlist)), reference_build(reference_parse(netlist))


# ---------------------------------------------------------------------------
# Parity with the oracle
# ---------------------------------------------------------------------------


def test_every_full_sweep_corpus_case_is_covered() -> None:
    names = {param.values[0].name for param in FULL_SWEEP_CASES}
    assert len(FULL_SWEEP_CASES) >= 30
    assert {"rlc_mesh_3x3_test", "bjt_three_stage_test", "rand_mos_chain_s1", "rand_diode_s0"} <= names


@pytest.mark.parametrize(("case", "options"), FULL_SWEEP_CASES)
def test_corpus_ac_matches_the_reference(case, options, reference_pkg, current_pkg) -> None:
    rtol, atol = TOLERANCES[case.tolerance_for("ac", options)]
    ref = run_request(reference_pkg, case.netlist_text, "ac", options)
    cur = run_request(current_pkg, case.netlist_text, "ac", {**options, "num_threads": 1})
    assert ref.ok and cur.ok, (ref.error, cur.error)
    ref_spectrum, cur_spectrum = ref.response.spectrum, cur.response.spectrum
    assert cur_spectrum["labels"] == ref_spectrum["labels"]
    _assert_bitwise(np.asarray(cur_spectrum["frequencies"]), np.asarray(ref_spectrum["frequencies"]))
    _assert_close(cur_spectrum["magnitudes"], ref_spectrum["magnitudes"], rtol, atol)
    for key in ("op_used", "linearized_device_count", "gmin"):
        assert cur.response.metadata.get(key) == ref.response.metadata.get(key), key
    if "operating_point" in ref.response.metadata:
        _assert_close(
            cur.response.metadata["operating_point"]["values"],
            ref.response.metadata["operating_point"]["values"],
            rtol,
            atol,
        )

    threaded = run_request(current_pkg, case.netlist_text, "ac", {**options, "num_threads": 4})
    _assert_bitwise(np.asarray(threaded.response.spectrum["magnitudes"]), np.asarray(cur_spectrum["magnitudes"]))


@pytest.mark.slow
def test_rlc_mesh_1035_unknowns_matches_the_reference() -> None:
    problem, ref_problem = _problems(ac_mesh(23, 23))
    assert problem.native.n == 1035
    ref = ref_solvers.solve_ac(ref_problem, 1e6, 1e9, 6)
    for threads in (1, 4):
        cur = solvers.solve_ac(problem, 1e6, 1e9, 6, num_threads=threads)
        _assert_bitwise(cur.frequencies, ref.frequencies)
        _assert_close(cur.magnitudes, ref.magnitudes, *TOLERANCES["linear"])


@pytest.mark.parametrize("netlist", [RC_NET, CE_NET, ac_mesh(3, 3)], ids=["rc", "ce_level1", "rlc_3x3"])
@pytest.mark.parametrize("sweep", [(1e3, 1e6, 1), (1e3, 1e6, 2), (0.0, 1e6, 5), (0.0, 0.0, 2)], ids=["p1", "p2", "f0", "dc_only"])
def test_points_and_zero_start_match_the_reference_solve_ac(netlist: str, sweep) -> None:
    problem, ref_problem = _problems(netlist)
    nonlinear = bool(problem.level1_bjt_devices)
    rtol, atol = TOLERANCES["nonlinear" if nonlinear else "linear"]
    ref = ref_solvers.solve_ac(ref_problem, *sweep)
    cur = solvers.solve_ac(problem, *sweep, num_threads=2)
    _assert_bitwise(cur.frequencies, ref.frequencies)
    assert cur.magnitudes.shape == ref.magnitudes.shape == (problem.native.n, sweep[2])
    _assert_close(cur.magnitudes, ref.magnitudes, rtol, atol)
    assert cur.metadata.keys() == ref.metadata.keys()
    assert cur.metadata["op_used"] is ref.metadata["op_used"] is nonlinear


def test_zero_start_through_run_basic_analysis_is_a_dc_point() -> None:
    problem, ref_problem = _problems(RC_NET)
    result = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.AC, ac_start=0.0, ac_stop=1e3, ac_points=3))
    ref = ref_solvers.solve_ac(ref_problem, 0.0, 1e3, 3)
    assert result.spectrum.frequencies.tolist() == [0.0, 500.0, 1000.0]
    _assert_close(result.spectrum.magnitudes, ref.magnitudes, *TOLERANCES["linear"])
    assert result.spectrum.magnitudes[1, 0] == pytest.approx(1.0, rel=1e-15)


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_krylov", [False, True], ids=["direct", "krylov"])
@pytest.mark.parametrize("netlist", [ac_mesh(10, 10), BJT_THREE_STAGE], ids=["rlc_10x10", "bjt_3stage"])
def test_sweep_is_bitwise_identical_for_1_2_4_threads(netlist: str, use_krylov: bool) -> None:
    problem = build_mna_problem(parse_netlist_text(netlist))
    runs = []
    for threads in (1, 2, 4):
        stats: list[dict[str, object]] = []
        spectrum = solvers.solve_ac(problem, 10.0, 1e8, 40, use_krylov=use_krylov, krylov_stats=stats, num_threads=threads)
        runs.append((spectrum, stats))
    base, base_stats = runs[0]
    for spectrum, stats in runs[1:]:
        _assert_bitwise(spectrum.magnitudes, base.magnitudes)
        assert spectrum.metadata == base.metadata
        assert [{k: v for k, v in s.items() if k != "threads_used"} for s in stats] == [
            {k: v for k, v in s.items() if k != "threads_used"} for s in base_stats
        ]


def test_native_sweep_reports_the_threads_it_used() -> None:
    problem = build_mna_problem(parse_netlist_text(ac_mesh(4, 4)))
    b_ac = np.ascontiguousarray(problem.b_ac.reshape(-1), dtype=complex)
    frequencies = np.linspace(1e6, 1e9, 9)
    expected = {1: 1, 2: 2, 4: min(4, core.effective_threads(4))}
    serial, _ = core.ac_sweep(problem.native, None, b_ac, frequencies, 0.0, None, 1)
    for threads, used in expected.items():
        magnitudes, info = core.ac_sweep(problem.native, None, b_ac, frequencies, 0.0, None, threads)
        assert info["threads_used"] == used and info["krylov"] is None
        _assert_bitwise(magnitudes, serial)
    # points < 2 * threads stays serial.
    _, info = core.ac_sweep(problem.native, None, b_ac, frequencies[:3], 0.0, None, 4)
    assert info["threads_used"] == 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _reference_basic_error(netlist: str, **options) -> BaseException:
    from mna_reference.api.contracts import AnalysisMode as RefMode
    from mna_reference.api.contracts import AnalysisOptions as RefOptions
    from mna_reference.core_basic import build_and_run_basic as ref_build_and_run_basic

    with pytest.raises(Exception) as exc:
        ref_build_and_run_basic(reference_parse(netlist), RefOptions(mode=RefMode.AC, **options))
    return exc.value


@pytest.mark.parametrize(
    ("netlist", "sweep", "frequency"),
    [
        (DC_SINGULAR_NET, (-70.0, 130.0, 21), "0.000e+00"),  # only f[7] == 0 is singular
        (ISLAND_NET, (5.0, 50.0, 16), "5.000e+00"),  # every frequency is singular
    ],
    ids=["dc_only", "island"],
)
def test_singular_system_raises_ac_error_at_the_first_singular_frequency(netlist: str, sweep, frequency: str) -> None:
    message = f"AC solve failed at frequency {frequency} Hz."
    start, stop, points = sweep
    reference = _reference_basic_error(netlist, ac_start=start, ac_stop=stop, ac_points=points)
    assert (type(reference).__name__, reference.code, str(reference)) == ("SolverConvergenceError", "AC_ERROR", message)
    for threads in (1, 2, 4):
        with pytest.raises(SolverConvergenceError) as exc:
            build_and_run_basic(
                parse_netlist_text(netlist),
                AnalysisOptions(mode=AnalysisMode.AC, ac_start=start, ac_stop=stop, ac_points=points, num_threads=threads),
            )
        assert type(exc.value) is SolverConvergenceError
        assert (exc.value.code, str(exc.value)) == ("AC_ERROR", message)


def test_nonpositive_points_raise_ac_error_before_any_work() -> None:
    problem = build_mna_problem(parse_netlist_text(RC_NET))
    for points in (0, -2):
        with pytest.raises(SolverConvergenceError) as exc:
            solvers.solve_ac(problem, 1.0, 10.0, points)
        assert (exc.value.code, str(exc.value)) == ("AC_ERROR", f"AC sweep needs at least one frequency point (points = {points}).")
    b_ac = np.ascontiguousarray(problem.b_ac.reshape(-1), dtype=complex)
    with pytest.raises(core.CoreError) as core_exc:
        core.ac_sweep(problem.native, None, b_ac, np.zeros(0), 0.0, None, 1)
    assert core_exc.value.code == "AC_ERROR"


# ---------------------------------------------------------------------------
# Krylov option
# ---------------------------------------------------------------------------


def _pre_native_krylov_stats(problem, sweep, **krylov) -> list[dict[str, object]]:
    """The stats the pre-native ``solve_ac`` loop recorded: ``solve_linear`` on the dense system."""

    stats: list[dict[str, object]] = []
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    G = problem.G
    if problem.level1_mos_devices or problem.level1_bjt_devices or solvers.is_nonlinear(solver_f_str):
        x = solvers.solve_dc_nr(
            problem.G,
            solver_f_str,
            problem.b_dc,
            use_krylov=True,
            krylov_stats=stats,
            level1_mos_devices=problem.level1_mos_devices,
            level1_bjt_devices=problem.level1_bjt_devices,
            **krylov,
        )
        _, jac = solvers.compile_nl_functions(
            solver_f_str, G.shape[0], level1_mos_devices=problem.level1_mos_devices, level1_bjt_devices=problem.level1_bjt_devices
        )
        G = G + np.asarray(jac(*x.flatten()), dtype=float)
    for frequency in np.linspace(*sweep):
        solvers.solve_linear(G + 1j * (2 * np.pi * frequency) * problem.C, problem.b_ac, use_krylov=True, krylov_stats=stats, **krylov)
    return stats


def _comparable(entry: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in entry.items() if key not in ("threads_used", "relative_residual")}


@pytest.mark.parametrize(
    ("netlist", "krylov"),
    [
        (ac_mesh(3, 3), {}),  # n < 64: legacy kernels, dense storage
        (ac_mesh(6, 6), {"krylov_rank": 40}),  # n >= 64: SciPy-semantics kernels
        (CE_NET, {"krylov_method": "gmres"}),  # Level-1 operating point first
        (ac_mesh(3, 3), {"krylov_rank": 1, "krylov_max_iter": 1}),  # direct fallback
    ],
    ids=["rlc_3x3", "rlc_6x6", "ce_level1", "fallback"],
)
def test_krylov_stats_are_the_entries_solve_linear_recorded(netlist: str, krylov: dict[str, object]) -> None:
    problem = build_mna_problem(parse_netlist_text(netlist))
    sweep = (1e5, 1e9, 7)
    stats: list[dict[str, object]] = []
    spectrum = solvers.solve_ac(problem, *sweep, use_krylov=True, krylov_stats=stats, num_threads=2, **krylov)
    expected = _pre_native_krylov_stats(problem, sweep, **krylov)
    assert len(stats) == len(expected) >= 7
    for got, want in zip(stats, expected):
        assert list(got) == list(want)
        assert _comparable(got) == _comparable(want)
        assert got["relative_residual"] == pytest.approx(want["relative_residual"], rel=1e-9, abs=1e-300)
    if krylov.get("krylov_max_iter") == 1:
        assert any(entry["used_direct_fallback"] for entry in stats)
    # The krylov_* result metadata is derived from these entries and keeps its meaning.
    options = AnalysisOptions(mode=AnalysisMode.AC, use_krylov=True, **krylov)
    got_meta = _result_metadata(problem, options, stats)
    want_meta = _result_metadata(problem, options, expected)
    assert {k: v for k, v in got_meta.items() if k.startswith("krylov_")} == {
        k: v for k, v in want_meta.items() if k.startswith("krylov_")
    }
    # The Krylov answer agrees with the direct sweep to the Krylov tolerance.
    direct = solvers.solve_ac(problem, *sweep)
    _assert_close(spectrum.magnitudes, direct.magnitudes, 1e-6, 1e-9 * float(np.max(direct.magnitudes)))


def test_krylov_contract_metadata_matches_the_oracle(reference_pkg, current_pkg) -> None:
    netlist = ac_mesh(3, 3)
    options = {"f_start": 1e6, "f_stop": 1e9, "points": 9, "use_krylov": True}
    ref = run_request(reference_pkg, netlist, "ac", options)
    cur = run_request(current_pkg, netlist, "ac", options)
    for key in ("krylov_method", "krylov_matrix_kind", "krylov_converged", "krylov_solve_count", "krylov_methods", "krylov_used_direct_fallback"):
        assert cur.response.metadata[key] == ref.response.metadata[key], key
    _assert_close(cur.response.spectrum["magnitudes"], ref.response.spectrum["magnitudes"], *TOLERANCES["nonlinear"])


def test_unknown_krylov_method_is_a_value_error() -> None:
    problem = build_mna_problem(parse_netlist_text(RC_NET))
    with pytest.raises(ValueError, match="Unknown Krylov method 'bicgstab'"):
        solvers.solve_ac(problem, 1.0, 10.0, 3, use_krylov=True, krylov_method="bicgstab")
    b_ac = np.ascontiguousarray(problem.b_ac.reshape(-1), dtype=complex)
    with pytest.raises(ValueError, match="Unknown Krylov method"):
        core.ac_sweep(problem.native, None, b_ac, np.ones(3), 0.0, {"use_krylov": True, "krylov_method": "bicgstab"}, 1)


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def test_binding_shapes_and_input_validation() -> None:
    problem = build_mna_problem(parse_netlist_text(CE_NET))
    n = problem.native.n
    b_ac = np.ascontiguousarray(problem.b_ac.reshape(-1), dtype=complex)
    x_op = np.zeros(n)
    magnitudes, info = core.ac_sweep(problem.native, x_op, b_ac, np.linspace(1.0, 1e3, 4), 0.0, {}, 1)
    assert magnitudes.dtype == np.float64 and magnitudes.shape == (n, 4) and magnitudes.flags["C_CONTIGUOUS"]
    assert set(info) == {"threads_used", "krylov"}

    def code_of(*args) -> str | None:
        with pytest.raises(core.CoreError) as exc:
            core.ac_sweep(*args)
        return exc.value.code

    f = np.ones(2)
    assert code_of(problem.native, x_op[:-1], b_ac, f, 0.0, None, 1) == "NETLIST_FATAL"
    assert code_of(problem.native, x_op, b_ac[:-1], f, 0.0, None, 1) == "NETLIST_FATAL"
    assert code_of(problem.native, x_op, b_ac, f, 0.0, {"use_krylov": True, "krylov_rank": 0}, 1) == "NETLIST_FATAL"
    assert code_of(problem.native, x_op, b_ac, f, 0.0, {"tolerance": 1.0}, 1) == "NETLIST_FATAL"
    assert code_of(problem.native, x_op, b_ac, f, 0.0, [1, 2], 1) == "NETLIST_FATAL"


def test_linearized_g_is_g_plus_device_jacobian() -> None:
    problem = build_mna_problem(parse_netlist_text(CE_NET))
    x = solvers.solve_dc_nr(
        problem.G, problem.solver_f_str, problem.b_dc, level1_bjt_devices=problem.level1_bjt_devices
    ).reshape(-1)
    col_ptr, row_idx = problem.native.pattern()
    n = problem.native.n
    values = core.linearized_g(problem.native, np.ascontiguousarray(x), 0.0)
    dense = np.zeros((n, n))
    for col in range(n):
        for slot in range(col_ptr[col], col_ptr[col + 1]):
            dense[row_idx[slot], col] = values[slot]
    jac = core.device_jacobian_dense(problem.native, np.ascontiguousarray(x))
    np.testing.assert_array_equal(dense, problem.G + jac)
    np.testing.assert_array_equal(core.linearized_g(problem.native, None, 1e-3), problem.native.g_effective(1e-3))


def test_sweep_releases_the_gil() -> None:
    problem = build_mna_problem(parse_netlist_text(ac_mesh(23, 23)))
    b_ac = np.ascontiguousarray(problem.b_ac.reshape(-1), dtype=complex)
    frequencies = np.linspace(1e6, 1e9, 200)
    started = threading.Event()
    done = threading.Event()

    def worker() -> None:
        started.set()
        core.ac_sweep(problem.native, None, b_ac, frequencies, 0.0, None, 1)
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
    time.sleep(0.005)
    ticks = 0
    while not done.is_set():
        ticks += 1
    thread.join()
    # With the GIL held for the whole sweep this loop could not run until the sweep ended.
    assert ticks > 1000
