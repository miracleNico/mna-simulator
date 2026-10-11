"""P3b convergence improvements (CPP_MIGRATION_PLAN.md 2.4.1: "improve, not parity").

* Junction voltage-step limiting (SPICE pnjlim) with the natural monotonicity test: a diode
  across an ideal source converges within ``max_iter``; circuits the reference iteration
  solves (the 1 ohm series versions) still match the oracle and are bitwise unchanged.
* gmin continuation inside ``.op``: CMOS inverter chains of 4, 6 and 8 stages converge at gmin 0.
* Node-to-ground gmin during gmin stepping: the floating island converges at the first gmin step,
  and the exposed ``G`` carries the node stamps.
* Level-1 BJT exponent: linear continuation beyond |v/vt| = 80, continuous in value and slope.
* Damped HB Newton: iteration counts on the mixer with and without damping, the restart after a
  breakdown (two-diode DC string) and the sample-wise KCL of its solution.
* Thread determinism (1 vs 4 threads, bitwise) of ``.op``, ``.tran`` and ``.hb`` on these paths,
  including parallel device evaluation (more than PARALLEL_DEVICE_MIN_COUNT devices).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from corpus import get_case
from differential.compare import TOLERANCES, run_request
from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, AnalysisRequest
from mna_simulation.api.service import SimulationService
from mna_simulation.backends.python_backend import PythonBackend
from mna_simulation.core_basic import build_and_run_basic
from mna_simulation.errors import SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.netlist import parse_netlist_text

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

core = native.require_core()
VT_BJT = 0.02585


def _svc(netlist: str, mode: str, **options):
    return SimulationService().run_request(AnalysisRequest(netlist_text=netlist, mode=AnalysisMode(mode), options=options))


def _op_native(netlist: str, gmin: float = 0.0, max_iter: int = 200, **kwargs):
    problem = build_mna_problem(parse_netlist_text(netlist))
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    return problem, core.solve_op(problem.native, b, None, gmin, max_iter, **kwargs)


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


def _bitwise(a, b) -> bool:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


# ---------------------------------------------------------------------------
# Junction voltage-step limiting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("volts", [1.0, 5.0])
def test_pnjlim_diode_across_an_ideal_source_converges_within_max_iter(volts: float) -> None:
    netlist = f"V1 a 0 DC {volts}\nD1 a 0 1e-14\n"
    problem, (x, info) = _op_native(netlist)
    assert info["iterations"] <= 200
    assert info["junction_switches"] == 1 and info["limited_steps"] > 0
    assert not info["continuation_used"]
    assert x[0] == volts
    assert x[1] == pytest.approx(-1e-14 * math.expm1(volts / 0.025), rel=1e-12)
    # The reference iteration alone creeps and does not converge.
    with pytest.raises(core.CoreError) as exc:
        core.solve_op(problem.native, np.ascontiguousarray(problem.b_dc.reshape(-1)), None, 0.0, 200,
                      junction_limiting=False, gmin_continuation=False)
    assert exc.value.code == "NR_CONVERGENCE"
    # Through the service it converges at the first gmin step it tries: gmin 0.
    response = _svc(netlist, "op", max_iter=200)
    assert response.metadata["gmin"] == 0.0
    assert response.dc_solution == pytest.approx(list(x), rel=1e-12)


@pytest.mark.parametrize("volts", ["5", "100", "1000"])
def test_one_ohm_series_versions_match_the_oracle_and_keep_the_reference_iteration(volts: str, reference_pkg) -> None:
    netlist = f"V1 a 0 DC {volts}\nR1 a b 1\nD1 b 0 1e-14\n"
    reference = run_request(reference_pkg, netlist, "op", {"max_iter": 200})
    current = run_request("mna_simulation", netlist, "op", {"max_iter": 200})
    assert reference.error is None and current.error is None
    rtol, atol = TOLERANCES["nonlinear"]
    np.testing.assert_allclose(current.response.dc_solution, reference.response.dc_solution, rtol=rtol, atol=atol)
    assert current.response.metadata["gmin"] == reference.response.metadata["gmin"] == 0.0
    # The reference iteration solves it, so the junction strategy never engages: bitwise unchanged.
    _, (x_on, info_on) = _op_native(netlist)
    _, (x_off, info_off) = _op_native(netlist, junction_limiting=False, gmin_continuation=False)
    assert _bitwise(x_on, x_off) and info_on["iterations"] == info_off["iterations"]
    assert info_on["junction_switches"] == 0 and info_on["limited_steps"] == 0


def test_transient_newton_steps_use_the_junction_strategy_after_a_stall() -> None:
    # A 0 -> 1 V step across a diode (with R1 to keep a linear path): the Newton solve of the step
    # creeps like the clamped .op and the junction strategy finishes it.
    netlist = "V1 a 0 FUNC 0 Piecewise((0, t < 0.5e-9), (1, True))\nR1 a 0 1k\nD1 a 0 1e-14\n"
    problem = build_mna_problem(parse_netlist_text(netlist))
    rows, programs = solvers._time_sources(problem.b_time_str)
    b_dc = np.ascontiguousarray(problem.b_dc.reshape(-1))
    grid = core.transient_time_points(3e-9, 1e-9)
    x0 = np.zeros(problem.native.n)
    values, info = core.solve_transient(problem.native, x0, grid, b_dc, rows, programs, 0.0, 200)
    assert info["junction_switches"] >= 1
    assert values[0, -1] == 1.0
    assert values[1, -1] == pytest.approx(-(1e-3 + 1e-14 * math.expm1(40.0)), rel=1e-12)
    with pytest.raises(core.CoreError):
        core.solve_transient(problem.native, x0, grid, b_dc, rows, programs, 0.0, 200, junction_limiting=False)


# ---------------------------------------------------------------------------
# gmin continuation and node-to-ground gmin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stages", [4, 6, 8])
def test_cmos_chains_converge_at_gmin_0(stages: int) -> None:
    netlist = _inverter_chain(stages)
    problem, (x, info) = _op_native(netlist)
    assert info["continuation_used"] and info["continuation_levels"] > 0
    assert info["iterations"] <= 200
    labels = problem.metadata["labels"]
    for stage in range(1, stages + 1):
        level = 1.2 if stage % 2 else 0.0
        assert x[labels.index(f"V(o{stage})")] == pytest.approx(level, abs=1e-6)
    response = _svc(netlist, "op", max_iter=200)
    assert response.metadata["gmin"] == 0.0
    np.testing.assert_allclose(response.dc_solution, x, rtol=0.0, atol=1e-12)
    with pytest.raises(core.CoreError) as exc:
        core.solve_op(problem.native, np.ascontiguousarray(problem.b_dc.reshape(-1)), None, 0.0, 200,
                      gmin_continuation=False)
    assert exc.value.code == "NR_CONVERGENCE"
    assert exc.value.message == "Newton step could not reduce residual at iteration 1."


def test_floating_island_converges_at_the_first_gmin_step_and_g_carries_the_node_stamps() -> None:
    netlist = "V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\n"
    with pytest.raises(SolverConvergenceError):
        build_and_run_basic(parse_netlist_text(netlist), AnalysisOptions(mode=AnalysisMode.OP))
    response = _svc(netlist, "op")
    assert response.metadata["gmin"] == 1e-12
    assert response.labels == ["V(a)", "V(x)", "V(y)", "I(V1)"]
    assert response.dc_solution == pytest.approx([1.0, 0.0, 0.0, -(1e-3 + 1e-12)], rel=1e-12)
    G = np.asarray(response.matrices["G"], dtype=float)
    # Node-to-ground gmin on every node row (a, x, y), never on the branch row I(V1).
    assert G[0, 0] == pytest.approx(1e-3 + 1e-12, rel=1e-15)
    assert G[1, 1] == pytest.approx(1e-3 + 1e-12, rel=1e-15) and G[2, 2] == G[1, 1]
    assert G[3, 3] == 0.0


# ---------------------------------------------------------------------------
# BJT exponent beyond the clip
# ---------------------------------------------------------------------------


def test_bjt_exponent_extrapolation_is_continuous_at_the_clip_point() -> None:
    device = {"type": "QNPN", "is": 1e-15, "bf": 150, "br": 3, "vaf": 100, "var": 25}
    clip = 80 * VT_BJT
    for vb in (clip, 5.0 - clip):  # vbe at +80 vt, then vbc = vb - vc at -80 vt
        below_c, below_d = solvers._level1_bjt_currents_and_derivatives(device, 5.0, math.nextafter(vb, -10.0), 0.0)
        above_c, above_d = solvers._level1_bjt_currents_and_derivatives(device, 5.0, math.nextafter(vb, 10.0), 0.0)
        for i in range(3):
            assert above_c[i] == pytest.approx(below_c[i], rel=1e-12, abs=1e-12 * abs(below_c[0]))
            for j in range(3):
                scale = max(abs(v) for row in below_d for v in row)
                assert above_d[i][j] == pytest.approx(below_d[i][j], rel=1e-12, abs=1e-12 * scale)
    # Beyond the clip the current is linear in vbe with the clip-point slope as its derivative.
    c1, d1 = solvers._level1_bjt_currents_and_derivatives(device, 5.0, 3.0, 0.0)
    c2, _ = solvers._level1_bjt_currents_and_derivatives(device, 5.0, 3.5, 0.0)
    assert (c2[0] - c1[0]) / 0.5 == pytest.approx(d1[0][1], rel=1e-9)
    _, d_clip = solvers._level1_bjt_currents_and_derivatives(device, 5.0, clip, 0.0)
    assert d1[0][1] == pytest.approx(d_clip[0][1], rel=1e-12)


# ---------------------------------------------------------------------------
# Damped HB Newton
# ---------------------------------------------------------------------------


def _mixer_hb(damping: str, num_threads: int = 1, tol: float = 1e-6):
    case = get_case("hb_ghz_mixer")
    problem = build_mna_problem(parse_netlist_text(case.netlist_text))
    rows, programs = solvers._hb_time_sources(problem.b_time_str)
    b = np.ascontiguousarray(problem.b_dc.reshape(-1))
    return core.hb_solve(problem.native, 2 * np.pi * 5e6, 205, b, rows, programs, max_iter=200, tol=tol,
                         damping=damping, num_threads=num_threads)


def test_hb_mixer_iteration_counts_with_and_without_damping() -> None:
    x_off, off = _mixer_hb("off")
    x_default, default = _mixer_hb("on_failure")
    x_always, always = _mixer_hb("always")
    print(f"mixer Newton iterations: undamped {off['newton_iterations']}, damped {always['newton_iterations']}")
    # Default: the reference iteration converges, damping never engages (bitwise the reference).
    assert _bitwise(x_default, x_off) and default["damping_restarts"] == 0 and default["limited_steps"] == 0
    # Damped from the start: at most 2x the iterations (in practice ~3.5x fewer), same solution
    # within the HB class once both are converged tightly. The env node is an LC tank resonant at
    # the 5 MHz beat (|Z| ~ 4.3 kOhm), so at tol 1e-6 an iterate with ||F|| ~ 1e-7 can be ~1e-3 V
    # off and two converged runs need not agree beyond that.
    assert always["newton_iterations"] <= 2 * off["newton_iterations"]
    assert always["newton_iterations"] < off["newton_iterations"]
    assert always["limited_steps"] > 0
    x_tight_off, _ = _mixer_hb("off", tol=1e-12)
    x_tight_always, _ = _mixer_hb("always", tol=1e-12)
    np.testing.assert_allclose(x_tight_always, x_tight_off, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError):
        _mixer_hb("sometimes")


def test_hb_two_diode_string_converges_after_a_damped_restart() -> None:
    case = get_case("rand_diode_s2")
    options = dict(next(opts for mode, opts in case.analyses if mode == "hb"))
    backend = PythonBackend()
    circuit = backend.parse_text(case.netlist_text)
    result = backend.run(circuit, backend.build_options(circuit, overrides=options, mode=AnalysisMode.HB))
    stats = result.metadata["hb_solver"]
    assert result.metadata["gmin"] == 0.0
    assert stats["damping_restarts"] == 1
    # No capacitors: HB is collocation, so KCL holds at every sample of the solution.
    problem = build_mna_problem(circuit)
    harmonics = int(result.metadata["harmonics"])
    K = 2 * harmonics + 1
    omega0 = 2 * np.pi * float(result.metadata["base_frequency_hz"])
    _, gamma_inv = core.hb_fourier_matrices(harmonics, K)
    x_bar = np.asarray(result.harmonic_state, dtype=float).reshape(-1, K)
    rows, programs = solvers._time_sources(problem.b_time_str)
    b_dc = np.ascontiguousarray(problem.b_dc.reshape(-1))
    G = problem.native.to_dense("G")
    worst = 0.0
    for k, t in enumerate(core.hb_time_points(omega0, K)):
        x = np.ascontiguousarray(x_bar @ gamma_inv[k])
        b = core.time_source_vector(problem.native, b_dc, rows, programs, float(t))
        residual = G @ x + core.device_f(problem.native, x, 0.0).reshape(-1) - b
        worst = max(worst, float(np.max(np.abs(residual))))
    assert worst < 1e-9
    # Its DC component is the operating point shifted by the small AC drive only.
    dc = _svc(case.netlist_text, "op", max_iter=200).dc_solution
    np.testing.assert_allclose(x_bar[:, 0], dc, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Thread determinism on the new paths (1 vs 4 threads, bitwise)
# ---------------------------------------------------------------------------

PARALLEL_DIODES = "".join(f"D{k} a 0 1e-14\n" for k in range(2100))


@pytest.mark.parametrize(
    ("netlist", "mode", "options"),
    [
        # Junction strategy with parallel device evaluation (2100 diodes > PARALLEL_DEVICE_MIN_COUNT).
        ("V1 a 0 DC 1\n" + PARALLEL_DIODES, "op", {"max_iter": 200}),
        ("V1 a 0 FUNC 0 Piecewise((0, t < 0.5e-9), (1, True))\nR1 a 0 1k\n" + PARALLEL_DIODES, "tran",
         {"t_stop": 3e-9, "t_step": 1e-9, "max_iter": 200}),
        # gmin continuation.
        (_inverter_chain(8), "op", {"max_iter": 200}),
        # Node-to-ground gmin at the first gmin step.
        ("V1 a 0 DC 1\nR1 a b 1k\nD1 b 0 1e-14\nR2 x y 1k\n", "op", {"max_iter": 200}),
        # Damped HB restart (the two-diode string).
        (get_case("rand_diode_s2").netlist_text, "hb", {"harmonics": 6, "max_iter": 200}),
    ],
    ids=["op_junction_parallel", "tran_junction_parallel", "op_continuation", "op_node_gmin", "hb_damped_restart"],
)
def test_thread_determinism_on_the_p3b_paths(netlist: str, mode: str, options: dict) -> None:
    results = []
    for threads in (1, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = _svc(netlist, mode, **options, num_threads=threads)
        assert response.metadata["threads_used"] == threads
        if mode == "op":
            results.append(np.asarray(response.dc_solution, dtype=float))
        elif mode == "tran":
            results.append(np.asarray(response.waveform["values"], dtype=float))
        else:
            results.append(np.asarray(response.spectrum["magnitudes"], dtype=float))
        assert response.metadata["gmin"] in (0.0, 1e-12)
    assert _bitwise(results[0], results[1])


def test_thread_determinism_of_the_damped_hb_mixer() -> None:
    x1, s1 = _mixer_hb("always", num_threads=1)
    x4, s4 = _mixer_hb("always", num_threads=4)
    assert _bitwise(x1, x4) and s1["newton_iterations"] == s4["newton_iterations"]
