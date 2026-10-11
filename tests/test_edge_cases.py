"""Behaviour specification of the CURRENT pure-Python numerical core.

This module is the acceptance spec for the C++ port: every test pins what the
core does today on an edge case (the exception class and ``.code``, or the
numbers). Nothing here says the behaviour is *right*; ``tests/EDGE_CASES.md``
lists each case with a "looks intentional?" verdict so the coordinator can
decide which behaviours the new core must preserve and which it should define
differently. When a behaviour is changed on purpose, update the test, the
table, and (if the oracle is affected) ``tests/reference/README.md``. Tests
commented "Decided in CPP_MIGRATION_PLAN.md 2.4.1" pin a ruling that has been
applied; the table keeps the previous behaviour under "Was:".

Conventions used below:

* ``_basic(...)`` runs ``core_basic.build_and_run_basic`` at gmin=0 (no gmin
  stepping), ``_svc(...)`` runs the full ``SimulationService`` path, which
  retries over ``DEFAULT_GMIN_STEPS`` and reports the gmin that succeeded.
* Linear results are asserted to ~1e-12 relative, nonlinear ones to 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, AnalysisRequest, CircuitIR
from mna_simulation.api.service import SimulationService
from mna_simulation.core_basic import _mos_level1_region, build_and_run_basic
from mna_simulation.errors import NetlistError, SolverConvergenceError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.netlist import parse_directive, parse_netlist_text
from mna_simulation.solvers import (
    _level1_bjt_currents_and_derivatives,
    _level1_mos_current_and_derivatives,
    _transient_time_points,
    compile_nl_functions,
    solve_transient,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

LIN = dict(rel=1e-12, abs=1e-15)
NL = dict(rel=1e-6, abs=1e-12)


def _svc(netlist: str, mode: str | None, **options):
    return SimulationService().run_request(
        AnalysisRequest(netlist_text=netlist, mode=AnalysisMode(mode) if mode else None, options=options)
    )


def _basic(netlist: str, mode: AnalysisMode, **options):
    return build_and_run_basic(parse_netlist_text(netlist), AnalysisOptions(mode=mode, **options))


def _dc(result) -> list[float]:
    return np.asarray(result.dc_solution, dtype=float).flatten().tolist()


def _assert_error(exc_info, cls: type[BaseException], code: str | None, message: str | None = None) -> None:
    assert type(exc_info.value) is cls
    assert getattr(exc_info.value, "code", None) == code
    if message is not None:
        assert str(exc_info.value) == message


# ---------------------------------------------------------------------------
# Empty and single-node circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   \n\n", "* only a comment\n.op\n", ".tran 1m\n.end\n"])
def test_empty_netlist_raises_netlist_fatal(text: str) -> None:
    with pytest.raises(NetlistError) as parse_exc:
        parse_netlist_text(text)
    _assert_error(parse_exc, NetlistError, "NETLIST_FATAL", "No components entered in netlist.")
    with pytest.raises(NetlistError) as svc_exc:
        _svc(text, "op")
    _assert_error(svc_exc, NetlistError, "NETLIST_FATAL", "No components entered in netlist.")


def test_gnd_only_netlist_is_accepted_and_returns_empty_ok_result() -> None:
    response = _svc("GND vss\n", "op")
    assert response.status == "ok"
    assert response.labels == []
    assert response.dc_solution == []
    assert response.metadata["num_nodes"] == 0 and response.metadata["num_branches"] == 0
    assert response.matrices == {"G": [], "C": [], "f_str": [], "b_dc": [], "b_ac": [], "b_time_str": []}


def test_empty_circuit_ir_builds_a_zero_size_problem() -> None:
    op = build_and_run_basic(CircuitIR(components=[]), AnalysisOptions(mode=AnalysisMode.OP))
    assert op.labels == [] and np.asarray(op.dc_solution).shape == (0, 1)
    tran = build_and_run_basic(
        CircuitIR(components=[]), AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=1e-3, tran_step=1e-4)
    )
    assert np.asarray(tran.waveform.values).shape == (0, 11)


def test_single_node_current_source_injects_into_node1() -> None:
    # SPICE convention would give V(n1) = -1 V (current flows n+ -> n- through the source).
    result = _basic("I1 n1 0 DC 1m\nR1 n1 0 1k\n", AnalysisMode.OP)
    assert result.labels == ["V(n1)"]
    assert _dc(result) == pytest.approx([1.0], **LIN)


def test_single_node_voltage_source_only_and_resistor_only() -> None:
    assert _dc(_basic("V1 n1 0 DC 5\n", AnalysisMode.OP)) == pytest.approx([5.0, 0.0], **LIN)
    assert _dc(_basic("R1 n1 0 1k\n", AnalysisMode.OP)) == [0.0]


def test_single_node_capacitor_only_singular_at_gmin0_solved_by_gmin_stepping() -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _basic("C1 n1 0 1u\n", AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "Linear system is singular (G matrix): Singular matrix")
    response = _svc("C1 n1 0 1u\n", "op")
    assert response.metadata["gmin"] == 1e-12
    assert response.dc_solution == [0.0]


# ---------------------------------------------------------------------------
# Singular G: floating nodes, source loops, inductor loops
# ---------------------------------------------------------------------------

FLOATING_ISLAND = "V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\n"
CAP_ONLY_NODE = "V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n"
SINGULAR_MESSAGE = "Linear system is singular (G matrix): Singular matrix"


def test_floating_island_is_singular_at_gmin0() -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(FLOATING_ISLAND, AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


def test_floating_island_fails_every_gmin_step_in_the_service() -> None:
    # gmin is only stamped across C, D and Level-1 BJT/MOS terminals, never on bare nodes.
    with pytest.raises(SolverConvergenceError) as exc:
        _svc(FLOATING_ISLAND, "op")
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


def test_capacitor_only_node_is_singular_at_gmin0() -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(CAP_ONLY_NODE, AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


def test_capacitor_only_node_is_solved_by_gmin_stepping_in_the_service() -> None:
    response = _svc(CAP_ONLY_NODE, "op")
    assert response.metadata["gmin"] == 1e-12
    assert response.labels == ["V(a)", "V(b)", "V(c)", "I(V1)"]
    v_a, v_b, v_c, i_v1 = response.dc_solution
    # gmin=1e-12 S across each capacitor: a 1k + 1T + 1T divider.
    assert [v_a, v_b, v_c] == pytest.approx([1.0, 0.9999999995, 0.49999999975], rel=1e-12)
    assert i_v1 == pytest.approx(-5e-13, rel=1e-6)  # ill-conditioned: only ~7 digits agree with the analytic value
    tran = _svc(CAP_ONLY_NODE, "tran", t_stop=1e-3, t_step=1e-4)
    assert tran.metadata["gmin"] == 1e-12


@pytest.mark.parametrize(
    "netlist",
    [
        "V1 a 0 DC 1\nV2 b 0 DC 2\nV3 a b DC 0.5\nR1 a 0 1k\n",  # inconsistent KVL loop
        "V1 a 0 DC 1\nV2 b 0 DC 2\nV3 b a DC 1\nR1 a 0 1k\n",  # consistent KVL loop: still singular
    ],
    ids=["inconsistent", "consistent"],
)
def test_voltage_source_loop_passes_the_parser_and_is_singular(netlist: str) -> None:
    parse_netlist_text(netlist)
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(netlist, AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)
    with pytest.raises(SolverConvergenceError) as svc_exc:
        _svc(netlist, "op")
    _assert_error(svc_exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


@pytest.mark.parametrize(
    "netlist",
    [
        "I1 a 0 DC 1m\nL1 a b 1m\nL2 b 0 1m\nL3 a 0 1m\n",  # inductor-only loop
        "I1 a 0 DC 1m\nL1 a 0 1m\nL2 a 0 2m\n",  # parallel inductors (not rejected like parallel V sources)
    ],
    ids=["loop", "parallel"],
)
def test_inductor_only_loop_is_singular_at_dc_for_op_and_tran(netlist: str) -> None:
    for mode, options in (("op", {}), ("tran", {"t_stop": 1e-3, "t_step": 1e-4})):
        with pytest.raises(SolverConvergenceError) as exc:
            _svc(netlist, mode, **options)
        _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


# ---------------------------------------------------------------------------
# Zero and negative component values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "0.0", "0k", "1e-999"])
def test_zero_resistor_raises_zero_division_error(value: str) -> None:
    netlist = f"V1 a 0 DC 1\nR1 a b {value}\nR2 b 0 1k\n"
    for mode in ("op", "show_matrix"):
        with pytest.raises(ZeroDivisionError) as exc:
            _svc(netlist, mode)
        _assert_error(exc, ZeroDivisionError, None, "float division by zero")


def test_zero_capacitor_is_silently_skipped() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 0\n"))
    assert not problem.C.any()
    assert problem.metadata["C_nnz"] == 0
    # A node reached only through C=0 is therefore floating.
    with pytest.raises(SolverConvergenceError) as exc:
        _basic("V1 a 0 DC 1\nR1 a b 1k\nC1 b c 0\n", AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", SINGULAR_MESSAGE)


def test_zero_inductor_keeps_its_branch_and_acts_as_a_short() -> None:
    netlist = "V1 a 0 DC 1\nR1 a b 1k\nL1 b 0 0\n"
    problem = build_mna_problem(parse_netlist_text(netlist))
    assert problem.metadata["labels"] == ["V(a)", "V(b)", "I(V1)", "I(L1)"]
    assert not problem.C.any()
    assert _dc(_basic(netlist, AnalysisMode.OP)) == pytest.approx([1.0, 0.0, -1e-3, 1e-3], **LIN)
    tran = _basic(netlist, AnalysisMode.TRAN, tran_stop=1e-3, tran_step=1e-4)
    assert len(tran.waveform.time) == 11


def test_negative_resistor_is_accepted() -> None:
    result = _basic("V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 -2k\n", AnalysisMode.OP)
    assert _dc(result) == pytest.approx([1.0, 2.0, 1e-3], **LIN)


def test_negative_capacitor_and_inductor_are_accepted_in_transient() -> None:
    cap = _basic("V1 a 0 SIN 1 1k\nR1 a b 1k\nC1 b 0 -1u\n", AnalysisMode.TRAN, tran_stop=3e-4, tran_step=1e-4)
    np.testing.assert_allclose(
        np.asarray(cap.waveform.values)[1],
        [0.0, -0.06530947247694148, -0.17823902678495207, -0.3037163093494082],
        rtol=1e-12,
    )
    # With L = -1 mH and h = 1 us the Backward-Euler matrix G + C/h is exactly singular
    # (det 0 in exact arithmetic). The Python oracle returned values from a 1e-16 LAPACK
    # pivot; the native KLU solve reports the singularity. Decided in CPP_MIGRATION_PLAN.md
    # 2.4.1 ("singular linear transient step"): NR_CONVERGENCE with the step time (was a raw
    # LinAlgError). Which step hits the exact zero pivot depends on the rounding of C/h.
    with pytest.raises(SolverConvergenceError) as exc:
        _basic("V1 a 0 DC 1\nR1 a b 1k\nL1 b 0 -1m\n", AnalysisMode.TRAN, tran_stop=3e-6, tran_step=1e-6)
    assert exc.value.code == "NR_CONVERGENCE"
    assert str(exc.value).startswith("Linear system is singular at t=")
    assert str(exc.value).endswith(" s (G + C/h matrix): Singular matrix")


def test_singular_linear_transient_step_raises_nr_convergence_with_the_step_time() -> None:
    # G + C/h is exactly singular for R=1k, C=-1u, h=1m. Decided in CPP_MIGRATION_PLAN.md
    # 2.4.1: NR_CONVERGENCE naming the step time (was: a raw numpy LinAlgError "Singular
    # matrix", the only unwrapped linear solve).
    netlist = "V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 -1u\n"
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(netlist, AnalysisMode.TRAN, tran_stop=5e-3, tran_step=1e-3)
    _assert_error(
        exc,
        SolverConvergenceError,
        "NR_CONVERGENCE",
        "Linear system is singular at t=1.00e-03 s (G + C/h matrix): Singular matrix",
    )
    # NR_CONVERGENCE is numerical, so gmin stepping still applies: gmin=1e-12 across C1
    # makes the step regular.
    assert _svc(netlist, "tran", t_stop=5e-3, t_step=1e-3).metadata["gmin"] == 1e-12


# ---------------------------------------------------------------------------
# Parser-level conflicts, duplicate names, ground aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "netlist",
    [
        "V1 a 0 DC 1\nV2 a 0 DC 2\nR1 a 0 1k\n",
        "V1 a 0 DC 1\nV2 0 a DC 2\nR1 a 0 1k\n",
        "V1 a 0 DC 1\nE1 a 0 b 0 2\nR1 a b 1k\nR2 b 0 1k\n",
    ],
    ids=["same_polarity", "reversed", "v_and_vcvs"],
)
def test_parallel_voltage_sources_are_rejected(netlist: str) -> None:
    with pytest.raises(NetlistError) as exc:
        _svc(netlist, "op")
    _assert_error(
        exc,
        NetlistError,
        "NETLIST_FATAL",
        "Voltage source parallel conflict: Two or more voltage sources are in parallel.",
    )


def test_voltage_source_with_both_terminals_on_ground_is_rejected() -> None:
    with pytest.raises(NetlistError) as exc:
        parse_netlist_text("GND vss\nGND g2\nV1 vss g2 DC 1\nR1 a vss 1k\nR2 a 0 1k\n")
    _assert_error(exc, NetlistError, "NETLIST_FATAL", "Voltage source has both terminals on ground.")
    with pytest.raises(NetlistError) as same_node:
        parse_netlist_text("V1 a a DC 1\nR1 a 0 1k\n")
    assert same_node.value.code == "COMPONENT_ERROR"


@pytest.mark.parametrize(
    ("netlist", "name"),
    [
        ("V1 a 0 DC 1\nV1 b 0 DC 2\nR1 a b 1k\n", "V1"),
        ("V1 a 0 DC 1\nL1 a b 1m\nL1 b 0 1m\n", "L1"),
        ("V1 a 0 DC 1\nV1:L a b 1m\nR1 b 0 1k\n", "V1"),
    ],
    ids=["V", "L", "V_vs_explicit_L"],
)
def test_duplicate_branch_component_names_are_rejected_at_build_time(netlist: str, name: str) -> None:
    parse_netlist_text(netlist)  # the parser itself accepts them
    with pytest.raises(NetlistError) as exc:
        _svc(netlist, "op")
    _assert_error(
        exc,
        NetlistError,
        "NETLIST_FATAL",
        f"Duplicate component name '{name}'. Use unique names or rename via '<name>:<TYPE>'.",
    )


def test_duplicate_non_branch_component_names_are_silently_accepted() -> None:
    response = _svc("V1 a 0 DC 1\nR1 a b 1k\nR1 b 0 1k\n", "op")
    assert response.dc_solution == pytest.approx([1.0, 0.5, -5e-4], **LIN)
    stacked = _svc("V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\nC1 b 0 1u\n", "show_matrix")
    assert stacked.matrices["C"][1][1] == pytest.approx(2e-6, **LIN)
    # Names are case-sensitive identifiers: R1 and r1 are distinct.
    assert _svc("V1 a 0 DC 1\nR1 a b 1k\nr1 b 0 1k\n", "op").dc_solution == pytest.approx([1.0, 0.5, -5e-4], **LIN)


def test_gnd_aliases() -> None:
    two = _svc("GND vss\nGND gnd2\nV1 a vss DC 1\nR1 a gnd2 1k\n", "op")
    assert two.labels == ["V(a)", "I(V1)"] and two.dc_solution == pytest.approx([1.0, -1e-3], **LIN)
    after_use = _svc("V1 a vss DC 1\nR1 a vss 1k\nGND vss\n", "op")
    assert after_use.labels == ["V(a)", "I(V1)"]
    lower = _svc("gnd vss\nV1 a vss DC 1\nR1 a vss 1k\n", "op")
    assert lower.labels == ["V(a)", "I(V1)"]
    control = _svc("GND vss\nV1 a 0 DC 1\nR1 a b 1k\nR2 b 0 1k\nG1 c vss b vss 1m\nR3 c 0 1k\n", "op")
    assert control.labels == ["V(a)", "V(b)", "V(c)", "I(V1)"]
    assert control.dc_solution == pytest.approx([1.0, 0.5, -0.5, -5e-4], **LIN)
    noop = _svc("GND 0\nGND1 0\nV1 a 0 DC 1\nR1 a 0 1k\n", "op")
    assert noop.labels == ["V(a)", "I(V1)"]
    for bad, count in (("GND\n", 0), ("GND a b\n", 2)):
        with pytest.raises(NetlistError) as exc:
            _svc(bad + "V1 a 0 DC 1\nR1 a 0 1k\n", "op")
        _assert_error(exc, NetlistError, "COMPONENT_ERROR", f"'GND' GND statement expects '<node>' (got {count} arg(s)).")


# ---------------------------------------------------------------------------
# NaN / Inf in netlists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["nan", "NaN", "inf"])
def test_nan_and_inf_literals_are_rejected_by_the_value_pattern(token: str) -> None:
    with pytest.raises(NetlistError) as exc:
        parse_netlist_text(f"V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 {token}\n")
    _assert_error(exc, NetlistError, "NETLIST_FATAL", f"'R2' has invalid value format '{token}'.")
    with pytest.raises(NetlistError) as source_exc:
        parse_netlist_text(f"V1 a 0 DC {token}\nR1 a 0 1k\n")
    _assert_error(source_exc, NetlistError, "NETLIST_FATAL", f"'V1' has invalid value '{token}'.")


def test_overflowing_literals_parse_to_infinity() -> None:
    # 1e999 matches the number pattern and float() turns it into inf.
    open_resistor = _svc("V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 1e999\n", "op")
    assert open_resistor.dc_solution == pytest.approx([1.0, 1.0, 0.0], **LIN)
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: a non-finite solution is NR_CONVERGENCE (was:
    # status "ok" with [nan, -inf]). The code is numerical, so every gmin step is tried first.
    with pytest.raises(SolverConvergenceError) as exc:
        _svc("V1 a 0 DC 1e999\nR1 a 0 1k\n", "op")
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "non-finite solution")
    with pytest.raises(TypeError):
        _svc("V1 a 0 SIN 1 1e999\nR1 a 0 1k\n", "tran", t_stop=2e-3, t_step=1e-3)


def test_func_expression_nan_raises_non_finite_solution() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: NR_CONVERGENCE (was: status "ok" with an
    # all-NaN waveform). The initial operating point with b_dc + b_time(0) is already NaN.
    with pytest.raises(SolverConvergenceError) as exc:
        _svc("V1 a 0 FUNC 0 nan\nR1 a 0 1k\n", "tran", t_stop=2e-3, t_step=1e-3)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "non-finite solution")
    # From a given finite state the first step is the first non-finite solution.
    with pytest.raises(SolverConvergenceError) as step_exc:
        _basic(
            "V1 a 0 FUNC 0 nan\nR1 a 0 1k\n",
            AnalysisMode.TRAN,
            tran_stop=2e-3,
            tran_step=1e-3,
            init_condition=np.zeros((2, 1)),
        )
    _assert_error(step_exc, SolverConvergenceError, "NR_CONVERGENCE", "non-finite solution at t=1.00e-03 s")


# ---------------------------------------------------------------------------
# Magnitude spread
# ---------------------------------------------------------------------------

SPREAD = "V1 a 0 DC 1\nR1 a b 1G\nR2 b 0 1m\nC1 b 0 1f\nL1 a c 1n\nR3 c 0 1e-15\nI1 d 0 DC 1e9\nR4 d 0 1e-15\n"


def test_magnitude_spread_1e_minus15_to_1e9_in_one_circuit() -> None:
    result = _basic(SPREAD, AnalysisMode.OP)
    assert result.labels == ["V(a)", "V(b)", "V(c)", "V(d)", "I(V1)", "I(L1)"]
    expected = [1.0, 1e-3 / (1e9 + 1e-3), 1.0, 1e-6, -(1e15 + 1.0 / (1e9 + 1e-3)), 1e15]
    assert _dc(result) == pytest.approx(expected, rel=1e-12)
    tran = _basic(SPREAD, AnalysisMode.TRAN, tran_stop=1e-12, tran_step=1e-13)
    assert len(tran.waveform.time) == 11 and np.isfinite(tran.waveform.values).all()


# ---------------------------------------------------------------------------
# Diode exponential overflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("volts", ["5", "20"])
def test_diode_clamped_by_an_ideal_source_never_converges(volts: str) -> None:
    # 5 V: exp(200) is finite; 20 V: exp(800) overflows to inf. Damped Newton
    # rejects non-finite trial residuals and creeps ~Vt per iteration either way.
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(f"V1 a 0 DC {volts}\nD1 a 0 1e-14\n", AnalysisMode.OP, max_iter=200)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "Solver did not converge after 200 iterations.")


def test_diode_expression_overflows_to_inf() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 20\nD1 a 0 1e-14\n"))
    f_func, j_func = compile_nl_functions(problem.solver_f_str, problem.G.shape[0])
    assert np.isinf(np.asarray(f_func(20.0, 0.0))[0, 0])
    assert np.isinf(np.asarray(j_func(20.0, 0.0))[0, 0])


@pytest.mark.parametrize(("volts", "v_diode"), [("5", 0.8415334423073746), ("100", 0.9208027701065821), ("1000", 0.9785741881898568)])
def test_diode_huge_forward_bias_through_a_resistor_converges(volts: str, v_diode: float) -> None:
    result = _basic(f"V1 a 0 DC {volts}\nR1 a b 1\nD1 b 0 1e-14\n", AnalysisMode.OP, max_iter=200)
    v_a, v_b, i_v1 = _dc(result)
    assert v_a == pytest.approx(float(volts), rel=1e-12)
    assert v_b == pytest.approx(v_diode, **NL)
    # Device equation holds at the returned point.
    assert (v_a - v_b) / 1.0 == pytest.approx(1e-14 * (math.exp(v_b / 0.025) - 1.0), rel=1e-6)
    assert i_v1 == pytest.approx(-(v_a - v_b), rel=1e-9)


# ---------------------------------------------------------------------------
# Level-1 BJT exponent clipping
# ---------------------------------------------------------------------------

BJT = {"type": "QNPN", "is": 1e-15, "bf": 150, "br": 3, "vaf": 100, "var": 25}
VT_BJT = 0.02585


def test_level1_bjt_exponent_is_clipped_at_80_vt() -> None:
    at_clip, d_clip = _level1_bjt_currents_and_derivatives(BJT, 5.0, 80 * VT_BJT, 0.0)
    for vb in (5.0, 6.0):
        currents, derivatives = _level1_bjt_currents_and_derivatives(BJT, 5.0, vb, 0.0)
        assert [float(c) for c in currents] == pytest.approx([float(c) for c in at_clip], rel=1e-12)
        assert float(derivatives[0][1]) == pytest.approx(float(d_clip[0][1]), rel=1e-12)
    ic = float(at_clip[0])
    assert ic == pytest.approx(5.779125996966741e19, rel=1e-12)
    # The current is flat beyond the clip but the reported dIc/dVb is not zero. Kept for now
    # (CPP_MIGRATION_PLAN.md 2.4.1): a zero slope made the HB Newton Jacobian singular when an
    # iterate crossed the clip; P3 replaces the hard clip by junction limiting with linear
    # extrapolation, which is consistent and nonsingular.
    assert float(d_clip[0][1]) == pytest.approx(2.2356386835461277e21, rel=1e-12)
    flat_a, _ = _level1_bjt_currents_and_derivatives(BJT, 5.0, 5.0, 0.0)
    flat_b, _ = _level1_bjt_currents_and_derivatives(BJT, 5.0, 5.001, 0.0)
    assert float(flat_a[0]) == float(flat_b[0])


def test_level1_bjt_negative_clip_gives_reverse_saturation_currents() -> None:
    currents, _ = _level1_bjt_currents_and_derivatives(BJT, 5.0, -5.0, 0.0)
    assert [float(c) for c in currents] == pytest.approx(
        [-2.4304635761589396e-16, -2.0662251655629145e-16, 4.496688741721854e-16], rel=1e-12
    )


# ---------------------------------------------------------------------------
# Level-1 MOS region boundaries
# ---------------------------------------------------------------------------

NMOS = {"type": "NMOS", "beta": 1e-3, "vth": 0.4, "lambda": 0.02}
PMOS = {"type": "PMOS", "beta": 1e-3, "vth": 0.4, "lambda": 0.02}


def _compiled_drain_current(mos_type: str, vd: float, vg: float, vs: float, vth: str = "0.4") -> float:
    """Same route as ``tests/test_physical_sram._level1_current`` (compiled f_func)."""

    problem = build_mna_problem(
        parse_netlist_text(
            f"M1 d g s {mos_type} LEVEL1 1m {vth} 0.02 2f 1f\nVd d 0 DC 0\nVg g 0 DC 0\nVs s 0 DC 0\n.op\n"
        )
    )
    f_func, _ = compile_nl_functions(
        problem.solver_f_str, problem.G.shape[0], level1_mos_devices=problem.level1_mos_devices
    )
    labels = problem.metadata["labels"]
    x = np.zeros(problem.G.shape[0])
    x[labels.index("V(d)")], x[labels.index("V(g)")], x[labels.index("V(s)")] = vd, vg, vs
    return float(np.asarray(f_func(*x)).reshape(-1)[labels.index("V(d)")])


def test_level1_mos_exactly_at_threshold_is_off() -> None:
    assert _level1_mos_current_and_derivatives(NMOS, 1.0, 0.4, 0.0) == (0.0, 0.0, 0.0, 0.0)
    assert _compiled_drain_current("NMOS", 1.0, 0.4, 0.0) == 0.0
    assert _mos_level1_region("NMOS", 1.0, 0.4, 0.0, 0.4) == "off"
    just_above = _level1_mos_current_and_derivatives(NMOS, 1.0, 0.4 + 1e-12, 0.0)
    assert just_above[0] > 0.0


def test_level1_mos_triode_saturation_boundary_uses_saturation_and_is_c1_continuous() -> None:
    at = _level1_mos_current_and_derivatives(NMOS, 0.8, 1.2, 0.0)  # vds == vgs - vth == 0.8
    assert at == pytest.approx((3.2512e-4, 6.4e-6, 8.128e-4, -8.192e-4), rel=1e-12)
    assert _mos_level1_region("NMOS", 0.8, 1.2, 0.0, 0.4) == "saturation"
    below = _level1_mos_current_and_derivatives(NMOS, 0.8 - 1e-9, 1.2, 0.0)
    assert below == pytest.approx(at, rel=1e-6)
    assert _compiled_drain_current("NMOS", 0.8, 1.2, 0.0) == pytest.approx(3.2512e-4, rel=1e-12)


def test_level1_mos_reverse_mode_swaps_source_and_drain() -> None:
    reverse = _level1_mos_current_and_derivatives(NMOS, 0.0, 1.2, 0.1)
    assert reverse == pytest.approx((-7.515e-5, 8.031e-4, -1.002e-4, -7.029e-4), rel=1e-12)
    assert _mos_level1_region("NMOS", 0.0, 1.2, 0.1, 0.4) == "reverse"
    assert _compiled_drain_current("NMOS", 0.0, 1.2, 0.1) == pytest.approx(-7.515e-5, rel=1e-12)
    # vd == vs takes the forward branch: zero current, non-zero output conductance.
    assert _level1_mos_current_and_derivatives(NMOS, 0.3, 1.2, 0.3) == pytest.approx((0.0, 5e-4, 0.0, -5e-4), rel=1e-12)


def test_level1_pmos_polarity() -> None:
    pmos = _level1_mos_current_and_derivatives(PMOS, 0.2, 0.0, 1.2)
    assert pmos == pytest.approx((-3.264e-4, 6.4e-6, 8.16e-4, -8.224e-4), rel=1e-12)
    assert _compiled_drain_current("PMOS", 0.2, 0.0, 1.2) == pytest.approx(-3.264e-4, rel=1e-12)
    assert _mos_level1_region("PMOS", 0.2, 0.0, 1.2, 0.4) == "saturation"


def test_level1_mos_negative_vth_is_taken_as_absolute_value() -> None:
    problem = build_mna_problem(
        parse_netlist_text("M1 d g s NMOS LEVEL1 1m -0.4 0.02 2f 1f\nVd d 0 DC 0\nVg g 0 DC 0\nVs s 0 DC 0\n")
    )
    assert problem.level1_mos_devices[0]["vth"] == 0.4
    assert _compiled_drain_current("NMOS", 1.0, 0.4, 0.0, vth="-0.4") == 0.0


def _inverter_chain(stages: int, keepers: bool = False) -> str:
    lines = ["VDD vdd 0 DC 1.2", "VIN in 0 DC 0"]
    previous = "in"
    for stage in range(1, stages + 1):
        lines += [
            f"MP{stage} o{stage} {previous} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
            f"MN{stage} o{stage} {previous} 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
        ]
        if keepers:
            lines.append(f"RK{stage} o{stage} {'vdd' if stage % 2 else '0'} 10Meg")
        previous = f"o{stage}"
    return "\n".join(lines) + "\n"


def test_cmos_inverter_chain_dc_convergence_depends_on_depth() -> None:
    three = _svc(_inverter_chain(3), "op", max_iter=200)
    assert three.metadata["gmin"] == 0.0
    assert three.dc_solution[:5] == pytest.approx([0.0, 1.2, 0.0, 1.2, 1.2], abs=1e-8)  # stops at ~2e-9 V error
    # Four stages only converge at gmin=1e-5, and the gmin shunt visibly moves the rails.
    four = _svc(_inverter_chain(4), "op", max_iter=200)
    assert four.metadata["gmin"] == 1e-5
    assert four.dc_solution[1:5] == pytest.approx([1.179054, 0.008481, 1.178832, 0.008484], abs=1e-6)
    # Six stages fail at every gmin step.
    with pytest.raises(SolverConvergenceError) as exc:
        _svc(_inverter_chain(6), "op", max_iter=200)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "Newton step could not reduce residual at iteration 1.")
    # 10 Meg keepers toward the logic levels make the same chain converge at gmin=0.
    kept = _svc(_inverter_chain(6, keepers=True), "op", max_iter=200)
    assert kept.metadata["gmin"] == 0.0


# ---------------------------------------------------------------------------
# Transient time grid
# ---------------------------------------------------------------------------

RC = "V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\n"


def test_transient_time_points_with_zero_t_stop() -> None:
    np.testing.assert_array_equal(_transient_time_points(0.0, 1e-3), [0.0])


def test_solve_transient_with_zero_t_stop_returns_only_the_initial_point() -> None:
    waveform = solve_transient(build_mna_problem(parse_netlist_text(RC)), t_stop=0.0, t_step=1e-4)
    np.testing.assert_array_equal(waveform.time, [0.0])
    assert np.asarray(waveform.values).tolist() == [[1.0], [1.0], [0.0]]


@pytest.mark.parametrize(
    ("t_stop", "t_step", "message"),
    [
        (-1.0, 1e-3, "t_stop must be non-negative."),
        (1e-3, 0.0, "t_step must be positive."),
        (1e-3, -1e-4, "t_step must be positive."),
    ],
)
def test_transient_time_points_reject_bad_arguments_with_tran_error(t_stop, t_step, message) -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: TRAN_ERROR is an input error, so NetlistError (was base CSE).
    with pytest.raises(NetlistError) as exc:
        _transient_time_points(t_stop, t_step)
    _assert_error(exc, NetlistError, "TRAN_ERROR", message)


def test_negative_t_stop_through_core_and_service_raises_tran_error() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: NetlistError (was base CSE); the service still re-raises it.
    with pytest.raises(NetlistError) as core_exc:
        _basic(RC, AnalysisMode.TRAN, tran_stop=-1e-3, tran_step=1e-4)
    _assert_error(core_exc, NetlistError, "TRAN_ERROR", "t_stop must be non-negative.")
    with pytest.raises(NetlistError) as svc_exc:
        _svc(RC, "tran", t_stop=-1e-3, t_step=1e-4)
    _assert_error(svc_exc, NetlistError, "TRAN_ERROR", "t_stop must be non-negative.")


def test_t_step_larger_than_t_stop_gives_a_single_step() -> None:
    np.testing.assert_array_equal(_transient_time_points(1e-3, 5e-3), [0.0, 1e-3])


def test_t_step_not_dividing_t_stop_ends_exactly_at_t_stop() -> None:
    assert _transient_time_points(1e-3, 3e-4).tolist() == [0.0, 3e-4, 6e-4, 9e-4, 1e-3]


def test_time_grid_is_built_by_repeated_addition() -> None:
    assert _transient_time_points(1.0, 0.1).tolist() == [
        0.0,
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6,
        0.7,
        0.7999999999999999,
        0.8999999999999999,
        1.0,
    ]


def test_time_grid_snapping_tolerance_is_1e_minus_9_relative() -> None:
    # Overshoot inside max(|t_stop|, |t_step|) * 1e-9: snapped, 11 points.
    inside = _transient_time_points(1e-3 + 1e-13, 1e-4)
    assert len(inside) == 11 and inside[-1] == 1e-3 + 1e-13
    # Overshoot outside the tolerance: an extra ~1e-11 s step is appended.
    outside = _transient_time_points(1e-3 + 1e-11, 1e-4)
    assert len(outside) == 12 and outside[-1] == 1e-3 + 1e-11
    assert outside[-1] - outside[-2] == pytest.approx(1e-11, rel=1e-3)
    # Undershoot (inside or outside the tolerance): the last regular step is shortened.
    assert len(_transient_time_points(1e-3 - 1e-13, 1e-4)) == 11
    assert len(_transient_time_points(1e-3 - 1e-11, 1e-4)) == 11


def test_core_basic_treats_only_none_as_unset_tran_stop_and_tran_step() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: only None means "unset" (was: ``options.tran_stop
    # or 0.1`` / ``options.tran_step or 1e-5``, so 0 ran to 0.1 s and t_step=0 used 1e-5).
    zero_stop = _basic(RC, AnalysisMode.TRAN, tran_stop=0.0, tran_step=0.01)
    assert zero_stop.waveform.time.tolist() == [0.0]
    assert np.asarray(zero_stop.waveform.values).tolist() == [[1.0], [1.0], [0.0]]  # the initial operating point
    unset_stop = _basic(RC, AnalysisMode.TRAN, tran_step=0.05)
    assert len(unset_stop.waveform.time) == 3 and unset_stop.waveform.time[-1] == 0.1
    unset_step = _basic(RC, AnalysisMode.TRAN, tran_stop=1e-4)
    assert len(unset_step.waveform.time) == 11  # t_step defaults to 1e-5
    with pytest.raises(NetlistError) as exc:
        _basic(RC, AnalysisMode.TRAN, tran_stop=1e-4, tran_step=0.0)
    _assert_error(exc, NetlistError, "TRAN_ERROR", "t_step must be positive.")
    # TRAN_ERROR is structural: the service raises it at once (no gmin retries).
    with pytest.raises(NetlistError) as svc_exc:
        _svc(RC, "tran", t_stop=1e-4, t_step=0.0)
    _assert_error(svc_exc, NetlistError, "TRAN_ERROR", "t_step must be positive.")
    assert _svc(RC, "tran", t_stop=0.0, t_step=1e-5).waveform["time"] == [0.0]


def test_tran_directive_defaults() -> None:
    assert parse_directive(".tran 0").params == {"t_stop": 0.0, "t_step": 1e-5}
    assert parse_directive(".tran 1m").params == {"t_stop": 1e-3, "t_step": 1e-5}
    with pytest.raises(NetlistError) as exc:
        parse_directive(".tran")
    _assert_error(exc, NetlistError, "NETLIST_FATAL", "Use .tran <time> (<step_time>).")


def test_service_parses_string_time_options() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: metric-suffix strings are parsed like .tran (was TypeError).
    response = _svc(RC, "tran", t_stop="1m", t_step="0.25m")
    assert response.waveform["time"] == pytest.approx([0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3], rel=1e-15)
    assert _svc(RC, "tran", t_stop=" 2e-3 ", t_step=1e-3).waveform["time"] == [0.0, 1e-3, 2e-3]
    with pytest.raises(ValueError) as exc:
        _svc(RC, "tran", t_stop="1 ms", t_step="1u")
    assert type(exc.value) is ValueError and str(exc.value) == "t_stop must be a number (metric suffixes allowed)"


# ---------------------------------------------------------------------------
# Non-convergence paths
# ---------------------------------------------------------------------------

DIODE = "V1 a 0 DC 5\nR1 a b 1k\nD1 b 0 1e-14\n"


@pytest.mark.parametrize("max_iter", [1, 2])
def test_nr_convergence_after_max_iter(max_iter: int) -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(DIODE, AnalysisMode.OP, max_iter=max_iter)
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", f"Solver did not converge after {max_iter} iterations.")
    with pytest.raises(SolverConvergenceError) as svc_exc:
        _svc(DIODE, "op", max_iter=max_iter)
    assert svc_exc.value.code == "NR_CONVERGENCE"


@pytest.mark.parametrize(
    "netlist",
    [DIODE + "R2 x y 1k\n", "VDD d 0 DC 1\nM1 d g 0 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"],
    ids=["diode_plus_floating_island", "mos_floating_gate"],
)
def test_jacobian_singular_is_reachable(netlist: str) -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(netlist, AnalysisMode.OP)
    _assert_error(exc, SolverConvergenceError, "JACOBIAN_SINGULAR", "Jacobian is singular at iteration 0.")
    with pytest.raises(SolverConvergenceError) as svc_exc:
        _svc(netlist, "op")
    _assert_error(svc_exc, SolverConvergenceError, "JACOBIAN_SINGULAR", "Jacobian is singular at iteration 0.")


def test_transient_newton_failure_reports_the_time_point() -> None:
    problem = build_mna_problem(parse_netlist_text(DIODE + "C1 b 0 1n\n"))
    with pytest.raises(SolverConvergenceError) as exc:
        _basic(
            DIODE + "C1 b 0 1n\n",
            AnalysisMode.TRAN,
            tran_stop=1e-6,
            tran_step=1e-7,
            max_iter=1,
            init_condition=np.zeros((problem.G.shape[0], 1)),
        )
    _assert_error(exc, SolverConvergenceError, "NR_CONVERGENCE", "NR failed to converge at t=1.00e-07 s")


def test_harmonic_balance_newton_failure() -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _svc("V1 a 0 SIN 0.5 1k\nR1 a b 1k\nD1 b 0 1e-14\n", "hb", harmonics=4, max_iter=1)
    _assert_error(exc, SolverConvergenceError, "HB_ERROR", "Harmonic balance solver did not converge within limit.")


def test_ac_operating_point_honours_max_iter() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: the AC operating point receives max_iter, v_tol and
    # f_tol (was: ignored, so .ac with max_iter=1 succeeded where .op with the same options failed).
    netlist = "V1 a 0 DC 5\nVs s 0 SIN 0.1 1k\nR1 a b 1k\nR2 s b 1k\nD1 b 0 1e-14\n"
    with pytest.raises(SolverConvergenceError) as op_exc:
        _svc(netlist, "op", max_iter=1)
    with pytest.raises(SolverConvergenceError) as ac_exc:
        _svc(netlist, "ac", f_start=10, f_stop=100, points=2, max_iter=1)
    _assert_error(ac_exc, SolverConvergenceError, "NR_CONVERGENCE", str(op_exc.value))
    response = _svc(netlist, "ac", f_start=10, f_stop=100, points=2, max_iter=200)
    assert response.status == "ok" and response.metadata["op_used"] is True
    # The AC operating point is the .op solution for the same Newton options (both use b_dc):
    # a huge f_tol stops at the zero start, a loose v_tol after the first damped steps.
    for newton in ({"max_iter": 200}, {"v_tol": 0.5}, {"f_tol": 1e3}):
        op = _dc(_basic(netlist, AnalysisMode.OP, **newton))
        ac = _basic(netlist, AnalysisMode.AC, ac_start=10, ac_stop=100, ac_points=2, **newton)
        assert ac.metadata["operating_point"]["values"] == op
    assert _dc(_basic(netlist, AnalysisMode.OP, f_tol=1e3)) == [0.0] * 5


# ---------------------------------------------------------------------------
# init_condition
# ---------------------------------------------------------------------------


def test_init_condition_default_random_and_array_reach_the_same_point() -> None:
    reference = _dc(_basic(DIODE, AnalysisMode.OP, init_condition="DEFAULT"))
    assert reference == pytest.approx([5.0, 0.6698509496871571, -0.004330149050312843], **NL)
    np.random.seed(1234)  # "RANDOM" draws from the global NumPy RNG
    assert _dc(_basic(DIODE, AnalysisMode.OP, init_condition="RANDOM")) == pytest.approx(reference, **NL)
    start = np.array([[5.0], [0.7], [-0.004]])
    assert _dc(_basic(DIODE, AnalysisMode.OP, init_condition=start)) == pytest.approx(reference, **NL)
    np.testing.assert_array_equal(start, [[5.0], [0.7], [-0.004]])  # not modified in place


@pytest.mark.parametrize("value", ["garbage", "", "DEFAULTS"])
def test_init_condition_unknown_string_raises_on_nonlinear_op(value: str) -> None:
    with pytest.raises(NetlistError) as exc:
        _basic(DIODE, AnalysisMode.OP, init_condition=value)
    _assert_error(exc, NetlistError, "NETLIST_FATAL", f"Unsupported init condition '{value}'.")


@pytest.mark.parametrize("value", ["default", " Default ", "random", "RANDOM"])
def test_init_condition_strings_are_case_insensitive(value: str) -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1 (was: NETLIST_FATAL for "default" / "random").
    np.random.seed(7)
    assert _dc(_basic(DIODE, AnalysisMode.OP, init_condition=value)) == pytest.approx(
        [5.0, 0.6698509496871571, -0.004330149050312843], **NL
    )


def test_init_condition_strings_on_linear_and_nonlinear_paths() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: strings set the Newton starting point of every
    # nonlinear .op/.tran (the initial operating point of .tran) and are ignored with a
    # diagnostic on linear paths. Was: ignored silently on linear .op and on every .tran.
    linear = _basic("V1 a 0 DC 5\nR1 a 0 1k\n", AnalysisMode.OP, init_condition="garbage")
    assert _dc(linear) == pytest.approx([5.0, -5e-3], **LIN)
    assert linear.diagnostics == [
        "init_condition 'garbage' ignored: the circuit is linear, its operating point needs no starting point."
    ]
    assert _basic("V1 a 0 DC 5\nR1 a 0 1k\n", AnalysisMode.OP).diagnostics == []
    linear_tran = _basic(RC, AnalysisMode.TRAN, tran_stop=2e-3, tran_step=1e-3, init_condition="RANDOM")
    assert len(linear_tran.diagnostics) == 1 and "ignored" in linear_tran.diagnostics[0]
    with pytest.raises(NetlistError) as exc:
        _basic(DIODE + "C1 b 0 1n\n", AnalysisMode.TRAN, tran_stop=1e-6, tran_step=2e-7, init_condition="garbage")
    _assert_error(exc, NetlistError, "NETLIST_FATAL", "Unsupported init condition 'garbage'.")
    default = _basic(DIODE + "C1 b 0 1n\n", AnalysisMode.TRAN, tran_stop=1e-6, tran_step=2e-7, init_condition="DEFAULT")
    np.random.seed(3)
    random = _basic(DIODE + "C1 b 0 1n\n", AnalysisMode.TRAN, tran_stop=1e-6, tran_step=2e-7, init_condition="random")
    assert len(default.waveform.time) == 6 and default.diagnostics == []
    np.testing.assert_allclose(random.waveform.values, default.waveform.values, rtol=1e-6, atol=1e-12)


def test_init_condition_array_sets_the_transient_initial_state() -> None:
    result = _basic(
        RC, AnalysisMode.TRAN, tran_stop=2e-3, tran_step=1e-3, init_condition=np.array([[1.0], [0.0], [0.0]])
    )
    assert result.metadata["op_used"] is False
    np.testing.assert_allclose(
        result.waveform.values, [[1.0, 1.0, 1.0], [0.0, 0.5, 0.75], [0.0, -5e-4, -2.5e-4]], rtol=1e-12, atol=1e-15
    )


def test_init_condition_with_wrong_length_raises_netlist_fatal() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1 (was: TypeError from the lambdified function on
    # nonlinear .op, ignored on linear .op).
    message = "init_condition has 2 values, expected 3 (one per unknown)."
    for netlist, mode, options in (
        (DIODE, AnalysisMode.OP, {}),
        ("V1 a 0 DC 5\nR1 a b 1k\nR2 b 0 1k\n", AnalysisMode.OP, {}),
        (RC, AnalysisMode.TRAN, {"tran_stop": 2e-3, "tran_step": 1e-3}),
    ):
        with pytest.raises(NetlistError) as exc:
            _basic(netlist, mode, init_condition=np.array([1.0, 2.0]), **options)
        _assert_error(exc, NetlistError, "NETLIST_FATAL", message)


def test_init_condition_json_list_through_the_service() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: build_options turns a numeric list into a float column
    # array (was AttributeError), and DEFAULT/RANDOM are case-insensitive through the service.
    reference = _dc(_basic(DIODE, AnalysisMode.OP, init_condition="DEFAULT"))
    assert _svc(DIODE, "op", init_condition=[5.0, 0.7, -0.004]).dc_solution == pytest.approx(reference, **NL)
    tran = _svc(RC, "tran", t_stop=2e-3, t_step=1e-3, init_condition=[1.0, 0.0, 0.0])
    assert tran.metadata["op_used"] is False
    np.testing.assert_allclose(
        tran.waveform["values"], [[1.0, 1.0, 1.0], [0.0, 0.5, 0.75], [0.0, -5e-4, -2.5e-4]], rtol=1e-12, atol=1e-15
    )
    linear = _svc("V1 a 0 DC 5\nR1 a 0 1k\n", "op", init_condition=[123.0, 456.0])
    assert linear.dc_solution == pytest.approx([5.0, -5e-3], **LIN)
    assert _svc(DIODE, "op", init_condition=" default ").dc_solution == pytest.approx(reference, **NL)
    # Length validation (P3 ruling, decided): NETLIST_FATAL, raised at once (was: TypeError
    # after every gmin step).
    with pytest.raises(NetlistError) as length_exc:
        _svc(DIODE, "op", init_condition=[1.0, 2.0])
    _assert_error(length_exc, NetlistError, "NETLIST_FATAL", "init_condition has 2 values, expected 3 (one per unknown).")
    with pytest.raises(ValueError) as exc:
        _svc("V1 a 0 DC 5\nR1 a 0 1k\n", "op", init_condition="garbage")
    assert type(exc.value) is ValueError
    assert str(exc.value) == "init_condition must be 'DEFAULT', 'RANDOM', or a numeric array"


# ---------------------------------------------------------------------------
# probe_nodes and output decimation
# ---------------------------------------------------------------------------


def test_unknown_probe_nodes_are_ignored_without_mor() -> None:
    op = _svc("V1 a 0 DC 1\nR1 a 0 1k\n", "op", probe_nodes=["V(nope)", "zzz"])
    assert op.status == "ok" and op.labels == ["V(a)", "I(V1)"]
    tran = _svc("V1 a 0 SIN 1 1k\nR1 a 0 1k\n", "tran", t_stop=1e-3, t_step=1e-4, probe_nodes="V(nope), x")
    assert tran.waveform["labels"] == ["V(a)", "I(V1)"]


def test_unknown_probe_nodes_are_rejected_with_mor() -> None:
    with pytest.raises(NetlistError) as exc:
        _svc(
            "V1 a 0 AC 1\nR1 a b 1k\nR2 b 0 1k\n",
            "ac",
            f_start=1,
            f_stop=10,
            points=2,
            use_mor=True,
            mor_output_nodes=["V(b)"],
            probe_nodes=["V(nope)"],
        )
    _assert_error(
        exc, NetlistError, "MOR_CONFIG", "MOR output label 'V(nope)' is not present in simulation labels (V(a), V(b), I(V1))."
    )


DECIMATION_NET = "V1 n1 0 SIN 1 1k\nR1 n1 0 1k\n"


@pytest.mark.parametrize(
    ("max_points", "returned", "metadata_max"),
    [
        (0, 11, None),
        (None, 11, None),
        (-5, 11, None),
        (1, 2, 1),  # returned_points exceeds max_points: the last sample is always appended
        (2, 3, 2),
        (10, 6, 10),  # stride ceil(11/10) = 2 halves the output although 10 points were allowed
        ("7", 6, 7),
        (7.9, 6, 7),  # int() truncation
        (11, 11, None),
        (12, 11, None),
        (50, 11, None),
    ],
)
def test_output_decimation_boundaries(max_points, returned: int, metadata_max) -> None:
    response = _svc(DECIMATION_NET, "tran", t_stop=1e-3, t_step=1e-4, output_max_points=max_points)
    time = response.waveform["time"]
    assert len(time) == returned
    assert len(response.waveform["values"][0]) == returned
    assert time[0] == 0.0 and time[-1] == pytest.approx(1e-3, rel=1e-15)
    if metadata_max is None:
        assert "output_decimation" not in response.metadata
    else:
        assert response.metadata["output_decimation"] == {
            "original_points": 11,
            "returned_points": returned,
            "max_points": metadata_max,
        }


def test_output_decimation_applies_to_hb_windows_but_not_to_op() -> None:
    op = _svc(DECIMATION_NET, "op", output_max_points=1)
    assert "output_decimation" not in op.metadata
    hb = _svc("V1 a 0 SIN 1 1k\nR1 a 0 1k\n.hb 3 1m\n", "hb", output_max_points=5)
    assert len(hb.waveform["time"]) == 6
    assert hb.metadata["output_decimation"] == {"original_points": 2000, "returned_points": 6, "max_points": 5}


# ---------------------------------------------------------------------------
# Krylov options
# ---------------------------------------------------------------------------

KRYLOV_NET = "V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 2k\nG1 b 0 a 0 1m\n"


def test_krylov_non_convergence_falls_back_to_the_direct_solve() -> None:
    response = _svc(KRYLOV_NET, "op", use_krylov=True, krylov_max_iter=1, krylov_rank=1)
    assert response.dc_solution == pytest.approx([1.0, 0.0, -1e-3], **LIN)
    md = response.metadata
    assert md["krylov_converged"] is False
    assert md["krylov_used_direct_fallback"] is True
    assert md["krylov_method"] == "arnoldi_gmres"
    assert md["krylov_matrix_kind"] == "general"
    assert (md["krylov_iterations"], md["krylov_solve_count"], md["krylov_resolved_rank"]) == (1, 1, 1)


@pytest.mark.parametrize(
    ("rank", "message"),
    [
        (0, "krylov_rank must be 'auto' or a positive integer."),
        (-3, "krylov_rank must be 'auto' or a positive integer."),
        # Decided in CPP_MIGRATION_PLAN.md 2.4.1: strict integers (was int()'s message; 2.7 was truncated).
        ("abc", "krylov_rank must be an integer"),
        ("", "krylov_rank must be an integer"),
        (2.7, "krylov_rank must be an integer"),
        ("2.5", "krylov_rank must be an integer"),
    ],
)
def test_invalid_krylov_rank_raises_plain_value_error(rank, message: str) -> None:
    with pytest.raises(ValueError) as exc:
        _svc(KRYLOV_NET, "op", use_krylov=True, krylov_rank=rank)
    assert type(exc.value) is ValueError and str(exc.value) == message


@pytest.mark.parametrize(
    ("rank", "resolved", "mode"),
    # 2.7 is now rejected (see above); integral floats and strings are accepted (2.4.1).
    [(2.0, 2, "manual"), ("1", 1, "manual"), (None, 2, "auto"), ("AUTO", 2, "auto"), ("  auto ", 2, "auto")],
)
def test_krylov_rank_coercion(rank, resolved: int, mode: str) -> None:
    md = _svc(KRYLOV_NET, "op", use_krylov=True, krylov_rank=rank).metadata
    assert (md["krylov_resolved_rank"], md["krylov_rank_mode"]) == (resolved, mode)


def test_krylov_restart_alone_sets_a_manual_rank() -> None:
    md = _svc(KRYLOV_NET, "op", use_krylov=True, krylov_restart=5).metadata
    assert (md["krylov_resolved_rank"], md["krylov_rank_mode"]) == (5, "manual")


@pytest.mark.parametrize("use_krylov", [True, False])
@pytest.mark.parametrize("method", ["foo", "hermitian"])
def test_invalid_krylov_method_raises_even_when_krylov_is_off(method: str, use_krylov: bool) -> None:
    with pytest.raises(ValueError) as exc:
        _svc(KRYLOV_NET, "op", use_krylov=use_krylov, krylov_method=method)
    assert str(exc.value) == f"Unknown Krylov method '{method}'."


@pytest.mark.parametrize(
    ("method", "used", "requested"),
    [("GMRES", "arnoldi_gmres", "arnoldi_gmres"), ("minres", "conjugate_residual", "conjugate_residual"), (None, "arnoldi_gmres", "auto"), ("", "arnoldi_gmres", "auto")],
)
def test_krylov_method_aliases(method, used: str, requested: str) -> None:
    md = _svc(KRYLOV_NET, "op", use_krylov=True, krylov_method=method).metadata
    assert (md["krylov_method"], md["krylov_requested_method"]) == (used, requested)


def test_use_krylov_string_false_keeps_krylov_off() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: strict booleans (was bool("false") is True, Krylov ON).
    assert "linear_solver" not in _svc(KRYLOV_NET, "op", use_krylov="false").metadata
    assert "linear_solver" not in _svc(KRYLOV_NET, "op", krylov="no").metadata
    assert _svc(KRYLOV_NET, "op", use_krylov="True").metadata["linear_solver"] == "krylov"
    with pytest.raises(ValueError) as exc:
        _svc(KRYLOV_NET, "op", use_krylov="maybe")
    assert type(exc.value) is ValueError and str(exc.value) == "use_krylov must be a boolean"


# ---------------------------------------------------------------------------
# MOR options
# ---------------------------------------------------------------------------

MOR_AC = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n"
MOR_AC_OPTS = dict(f_start=1, f_stop=1e3, points=3, use_mor=True)


def test_mor_with_an_invalid_output_node() -> None:
    with pytest.raises(NetlistError) as exc:
        _svc(MOR_AC, "ac", mor_output_nodes=["V(nope)"], **MOR_AC_OPTS)
    _assert_error(
        exc, NetlistError, "MOR_CONFIG", "MOR output label 'V(nope)' is not present in simulation labels (V(in), V(out), I(V1))."
    )


def test_mor_without_output_nodes() -> None:
    with pytest.raises(NetlistError) as exc:
        _svc(MOR_AC, "ac", **MOR_AC_OPTS)
    _assert_error(exc, NetlistError, "MOR_CONFIG", "MOR requires at least one output node, e.g. V(out) or I(L1).")


def test_mor_with_probe_nodes_not_a_subset_of_outputs() -> None:
    with pytest.raises(NetlistError) as exc:
        _svc(MOR_AC, "ac", mor_output_nodes=["V(out)"], probe_nodes=["V(in)"], **MOR_AC_OPTS)
    _assert_error(
        exc,
        NetlistError,
        "MOR_CONFIG",
        "Display nodes must be a subset of MOR output nodes when MOR is enabled ('V(in)' is not in ['V(out)']).",
    )
    unchecked = _svc(MOR_AC, "ac", mor_output_nodes=["V(out)"], probe_nodes=["V(in)"], mor_validate=False, **MOR_AC_OPTS)
    assert unchecked.labels == ["V(out)"] and unchecked.metadata["mor_used"] is True


def test_mor_output_labels_are_normalised_and_deduplicated() -> None:
    response = _svc(MOR_AC, "ac", mor_output_nodes="out, V(out), I(V1)", **MOR_AC_OPTS)
    assert response.labels == ["V(out)", "I(V1)"]


def test_mor_method_and_mode_mismatches() -> None:
    with pytest.raises(NetlistError) as tpwl_ac:
        _svc(MOR_AC, "ac", mor_method="tpwl", mor_output_nodes=["V(out)"], **MOR_AC_OPTS)
    _assert_error(tpwl_ac, NetlistError, "MOR_CONFIG", "tpwl MOR is transient-only; use auto or linear_krylov for .ac.")
    with pytest.raises(NetlistError) as linear_nl:
        _svc(
            "I1 in 0 STEP 1m 1u\nR1 in out 100\nD1 out 0 1e-12\nC1 out 0 1n\n",
            "tran",
            t_stop=5e-6,
            t_step=1e-6,
            use_mor=True,
            mor_method="linear",
            mor_output_nodes=["V(out)"],
        )
    _assert_error(
        linear_nl, NetlistError, "MOR_CONFIG", "linear_krylov MOR cannot be used for nonlinear transient; choose auto or tpwl."
    )


def test_invalid_mor_method_and_order_raise_plain_value_error() -> None:
    with pytest.raises(ValueError) as method_exc:
        _svc(MOR_AC, "ac", mor_method="xyz", mor_output_nodes=["V(out)"], **MOR_AC_OPTS)
    assert type(method_exc.value) is ValueError and str(method_exc.value) == "Unknown MOR method 'xyz'."
    with pytest.raises(ValueError) as order_exc:
        _svc(MOR_AC, "ac", mor_order=0, mor_output_nodes=["V(out)"], **MOR_AC_OPTS)
    assert str(order_exc.value) == "mor_order must be 'auto' or a positive integer."


def test_mor_on_show_matrix_reports_not_used() -> None:
    md = _svc(MOR_AC, "show_matrix", use_mor=True).metadata
    assert md["mor_used"] is False
    assert md["mor_fallback_reason"] == ".show_matrix requires the full MNA system."


# ---------------------------------------------------------------------------
# Harmonic balance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("netlist", ["V1 a 0 DC 1\nR1 a 0 1k\n", "V1 a 0 SIN 1 0\nR1 a 0 1k\n"], ids=["dc_only", "sin_0Hz"])
def test_hb_without_a_usable_sinusoidal_source(netlist: str) -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _svc(netlist, "hb", harmonics=3)
    _assert_error(exc, SolverConvergenceError, "HB_ERROR", "No sinusoidal sources found.")


def test_hb_source_frequency_below_the_gcd_resolution() -> None:
    with pytest.raises(SolverConvergenceError) as exc:
        _svc("V1 a 0 SIN 1 1e-6\nR1 a 0 1k\n", "hb", harmonics=3)
    _assert_error(exc, SolverConvergenceError, "HB_ERROR", "Could not determine HB base frequency.")


def test_hb_harmonics_smaller_than_needed_are_expanded_to_needed_plus_4() -> None:
    netlist = "V1 a 0 SIN 1 1k\nV2 a b SIN 1 3k\nR1 b 0 1k\n"
    expanded = _svc(netlist, "hb", harmonics=2)
    assert expanded.metadata["harmonics"] == 7
    assert expanded.metadata["base_frequency_hz"] == 1000.0000000000001  # gcd units * 1e-5 resolution
    assert len(expanded.spectrum["frequencies"]) == 8
    assert _svc(netlist, "hb", harmonics=3).metadata["harmonics"] == 3


def test_hb_harmonics_unset_means_8_and_non_positive_is_an_error() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: only None means unset. Was: 0 became 8 (`or 8`)
    # and a negative count was silently expanded to needed + 4.
    netlist = "V1 a 0 SIN 1 1k\nR1 a 0 1k\n"
    assert _svc(netlist, "hb").metadata["harmonics"] == 8
    assert _svc(netlist + ".hb\n", "hb").metadata["harmonics"] == 16  # the directive default differs
    for harmonics in (0, -3):
        with pytest.raises(SolverConvergenceError) as exc:
            _svc(netlist, "hb", harmonics=harmonics)
        _assert_error(exc, SolverConvergenceError, "HB_ERROR", f"harmonics must be a positive integer, got {harmonics}.")
    with pytest.raises(SolverConvergenceError) as directive_exc:
        _svc(netlist + ".hb 0\n", "hb")
    _assert_error(directive_exc, SolverConvergenceError, "HB_ERROR", "harmonics must be a positive integer, got 0.")


def test_hb_directive_parameters() -> None:
    assert parse_directive(".hb").params == {"harmonics": 16}
    assert parse_directive(".hb 5").params == {"harmonics": 5}
    assert parse_directive(".hb 1m").params == {"harmonics": 16, "time_window": 1e-3}
    assert parse_directive(".hb 5 1m").params == {"harmonics": 5, "time_window": 1e-3}
    assert parse_directive(".hb 2.5 1m").params == {"harmonics": 2, "time_window": 1e-3}


# ---------------------------------------------------------------------------
# AC sweep
# ---------------------------------------------------------------------------

AC_NET = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n"


def _rc_mag(f: float) -> float:
    return 1.0 / math.sqrt(1.0 + (2 * math.pi * f * 1e3 * 1e-6) ** 2)


def test_ac_single_point() -> None:
    response = _svc(AC_NET, "ac", f_start=100, f_stop=1e3, points=1)
    assert response.spectrum["frequencies"] == [100.0]
    assert response.spectrum["magnitudes"][1] == pytest.approx([_rc_mag(100.0)], rel=1e-12)


def test_ac_zero_start_and_stop_are_dc_points_and_nonpositive_points_raise() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: only None means unset (was ``ac_start or 1.0``,
    # ``ac_stop or 1e6``, ``ac_points or 100``). f = 0 is a DC point; points <= 0 is AC_ERROR.
    dc = _svc(AC_NET, "ac", f_start=0, f_stop=1e3, points=3)
    assert dc.spectrum["frequencies"] == [0.0, 500.0, 1000.0]
    assert dc.spectrum["magnitudes"][1] == pytest.approx([1.0, _rc_mag(500.0), _rc_mag(1e3)], rel=1e-12)
    assert dc.spectrum["magnitudes"][2][0] == 0.0  # no DC current through the capacitor
    assert _svc(AC_NET, "ac", f_start=0.0, f_stop=0, points=2).spectrum["frequencies"] == [0.0, 0.0]
    for points in (0, -1):
        message = f"AC sweep needs at least one frequency point (points = {points})."
        with pytest.raises(SolverConvergenceError) as svc_exc:
            _svc(AC_NET, "ac", f_start=10, f_stop=1e3, points=points)
        _assert_error(svc_exc, SolverConvergenceError, "AC_ERROR", message)
        with pytest.raises(SolverConvergenceError) as mor_exc:
            _svc(AC_NET, "ac", f_start=10, f_stop=1e3, points=points, use_mor=True, mor_output_nodes=["V(out)"])
        _assert_error(mor_exc, SolverConvergenceError, "AC_ERROR", message)
    unset = _basic(AC_NET, AnalysisMode.AC).spectrum.frequencies
    assert (len(unset), unset[0], unset[-1]) == (100, 1.0, 1e6)
    assert parse_directive(".ac 0 1k 1").params == {"f_start": 0.0, "f_stop": 1000.0, "points": 1}
    assert _svc(AC_NET + ".ac 0 1k 1\n", None).spectrum["frequencies"] == [0.0]


def test_ac_sweep_is_linear_and_accepts_descending_and_negative_frequencies() -> None:
    assert _svc(AC_NET, "ac", f_start=1e3, f_stop=10, points=3).spectrum["frequencies"] == [1000.0, 505.0, 10.0]
    negative = _svc(AC_NET, "ac", f_start=-10, f_stop=10, points=3)
    assert negative.spectrum["frequencies"] == [-10.0, 0.0, 10.0]
    assert negative.spectrum["magnitudes"][1] == pytest.approx([_rc_mag(10.0), 1.0, _rc_mag(10.0)], rel=1e-12)


def test_ac_points_must_be_an_integer() -> None:
    # Decided in CPP_MIGRATION_PLAN.md 2.4.1: integral strings/floats are coerced, the rest is a
    # plain ValueError (was TypeError from np.linspace for both 2.5 and "3").
    for points in ("3", 3.0):
        assert _svc(AC_NET, "ac", f_start=1, f_stop=10, points=points).spectrum["frequencies"] == [1.0, 5.5, 10.0]
    for points in (2.5, "2.5"):
        with pytest.raises(ValueError) as exc:
            _svc(AC_NET, "ac", f_start=1, f_stop=10, points=points)
        assert type(exc.value) is ValueError and str(exc.value) == "points must be an integer"


def test_ac_without_an_ac_source_returns_zero_magnitudes() -> None:
    response = _svc("V1 in 0 DC 1\nR1 in out 1k\nC1 out 0 1u\n", "ac", f_start=1, f_stop=10, points=2)
    assert response.spectrum["magnitudes"] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


# ---------------------------------------------------------------------------
# .show_matrix and mode selection
# ---------------------------------------------------------------------------


def test_show_matrix_mode_request_payload() -> None:
    response = _svc("V1 n1 0 DC 5\nR1 n1 0 1k\nD1 n1 0 1e-14\nV2 n2 0 SIN 1 1k\nR2 n2 0 1k\n", "show_matrix")
    assert response.mode == AnalysisMode.SHOW_MATRIX and response.status == "ok"
    assert response.dc_solution is None and response.waveform is None and response.spectrum is None
    assert sorted(response.matrices) == ["C", "G", "b_ac", "b_dc", "b_time_str", "f_str"]
    assert response.matrices["f_str"] == [["(1e-14 * (np.exp((x[0] - 0) / 0.025) - 1))"], ["0"], ["0"], ["0"]]
    assert response.matrices["b_time_str"] == [["0"], ["0"], ["0"], ["(1.0 * np.sin(2 * np.pi * 1000.0 * t))"]]
    # One complex entry turns every b_ac entry into a {real, imag} dict.
    assert response.matrices["b_ac"][0] == [{"real": 0.0, "imag": 0.0}]
    assert response.matrices["b_ac"][3][0]["imag"] == -1.0
    assert response.matrices["b_ac"][3][0]["real"] == pytest.approx(0.0, abs=1e-15)


def test_show_matrix_is_not_a_netlist_directive() -> None:
    with pytest.raises(NetlistError) as exc:
        _svc("V1 n1 0 DC 5\nR1 n1 0 1k\n.show_matrix\n", None)
    _assert_error(exc, NetlistError, "NETLIST_FATAL", "Unsupported directive '.show_matrix'.")


def test_mode_selection_rules() -> None:
    net = "V1 n1 0 DC 5\nR1 n1 0 1k\n"
    assert _svc(net, None).mode == AnalysisMode.SHOW_MATRIX  # no directive, no mode
    assert _svc(net + ".op\n.tran 1m 0.5m\n", None).mode == AnalysisMode.TRAN  # last directive wins
    overridden = _svc(net + ".tran 1m 0.5m\n", "op")
    assert overridden.mode == AnalysisMode.OP and overridden.dc_solution == pytest.approx([5.0, -5e-3], **LIN)
    dyn = _svc(net + ".tran 1m 0.5m\n", "dyn")
    assert dyn.mode == AnalysisMode.TRAN and len(dyn.waveform["time"]) == 3


# ---------------------------------------------------------------------------
# Source semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "t0_value"),
    [("FUNC 0 1.5", 1.5), ("STEP 2 0", 2.0), ("COS 2 1k", 2.0)],
)
def test_op_ignores_time_dependent_sources_even_when_nonzero_at_t0(source: str, t0_value: float) -> None:
    netlist = f"V1 a 0 {source}\nR1 a 0 1k\n"
    assert _svc(netlist, "op").dc_solution == [0.0, 0.0]
    tran = _svc(netlist, "tran", t_stop=1e-3, t_step=1e-3)
    assert tran.waveform["values"][0][0] == pytest.approx(t0_value, **LIN)


def test_controlled_source_sign_conventions() -> None:
    vccs = _svc("V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 1k\nG1 c 0 b 0 1m\nR3 c 0 1k\n", "op")
    assert vccs.dc_solution == pytest.approx([1.0, 0.5, -0.5, -5e-4], **LIN)
    cccs = _svc("V1 a 0 DC 1\nR1 a 0 1k\nF1 c 0 V1 2\nR3 c 0 1k\n", "op")
    assert cccs.dc_solution == pytest.approx([1.0, 2.0, -1e-3], **LIN)
    ccvs = _svc("V1 a 0 DC 1\nR1 a 0 1k\nH1 c 0 V1 100\nR3 c 0 1k\n", "op")
    assert ccvs.dc_solution == pytest.approx([1.0, -0.1, -1e-3, 1e-4], **LIN)
    vcvs = _svc("V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 1k\nE1 c 0 b 0 3\nR3 c 0 1k\n", "op")
    assert vcvs.dc_solution == pytest.approx([1.0, 0.5, 1.5, -5e-4, -1.5e-3], **LIN)
