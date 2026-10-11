"""Declarative circuit corpus for differential (reference vs. current) testing.

Every :class:`Case` is a netlist plus the analyses that apply to it. The
differential harness (``tests/differential/test_reference_parity.py``) runs
each ``(case, analysis)`` pair through the frozen oracle ``mna_reference`` and
the live ``mna_simulation`` package with identical ``AnalysisRequest``
payloads and compares the responses with the case's tolerance class.

Contents
--------
* ``existing``: every netlist embedded in the pre-existing test modules
  (``test_netlist_extensions``, ``test_service``, ``test_krylov_solvers``,
  ``test_mor``, ``test_harmonic_balance``, ``test_parser_and_builder``,
  ``test_physical_sram``), copied verbatim apart from leading indentation
  (the parser strips every line). Netlists that appear in several tests are
  listed once with all their sources in the comment above the case.
* ``workload``: the four ``benchmarks/workload_netlists.py`` workloads. The
  SRAM and RLC-mesh generators are frozen copies (see ``sram_netlist`` and
  ``rlc_mesh_netlist`` below) so the corpus does not move when the benchmark
  scripts change.
* ``random``: seeded generators (deterministic for a given seed; seeds are
  hashed through ``random.Random(str)``, which is independent of
  ``PYTHONHASHSEED``).

Analysis parameters are always explicit so that runs stay small: transient at
most ~60 steps, AC at most 40 points, HB at most 8 harmonics (the GHz mixer
is the only exception: its two tones force H=205), and nonlinear runs use
``max_iter=200``.

Deliberate behaviour changes (``CPP_MIGRATION_PLAN.md`` 2.4.1 rulings that make
the live package differ from the oracle) are recorded per analysis mode as a
:class:`Deviation`: the oracle must still behave as recorded on the case
netlist, and the live result is checked against the oracle's result on an
equivalent netlist (the change made explicit) or against an analytic answer.

Tolerance classes (see ``tests/differential/compare.py``): ``linear``,
``nonlinear``, ``hb``, ``mor``. A case declares its default class;
:meth:`Case.tolerance_for` upgrades a single analysis to ``mor`` when it
enables MOR, to ``hb`` for harmonic balance and to at least ``nonlinear`` when
it uses the iterative Krylov solver (iterative solutions only agree to the
Krylov tolerance, not to direct-solve precision).
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

TOLERANCE_CLASSES = ("linear", "nonlinear", "hb", "mor")
_TOLERANCE_RANK = {"linear": 0, "nonlinear": 1, "hb": 2, "mor": 3}
VALID_MODES = ("show_matrix", "op", "tran", "ac", "hb")

NL_MAX_ITER = 200


@dataclass(frozen=True)
class Deviation:
    """A deliberate behaviour change of the live package for one analysis mode (a 2.4.1 ruling).

    The oracle still runs the case netlist and must end as recorded: ``reference_error`` is
    the exception class name it raises (None: it succeeds). The live package must succeed
    (the P3b changes only make circuits converge or change a gmin-defined answer), report
    ``gmin`` in its metadata when given, and match:

    * ``equivalent_netlist`` run through the oracle: the same circuit with the change made
      explicit (for example the gmin shunts written as resistors), compared like a normal
      parity run except for ``metadata["gmin"]`` (and ``matrices`` unless
      ``compare_matrices``);
    * ``expected_dc``: an analytic operating point;
    * nothing more when neither is given (only success is required).

    ``tolerance`` overrides the case's class with a class name or an ``(rtol, atol)`` pair;
    ``ruling`` names the decision (``CPP_MIGRATION_PLAN.md`` 2.4.1) and the reason.
    """

    ruling: str
    reference_error: str | None = None
    equivalent_netlist: str | None = None
    gmin: float | None = None
    expected_dc: tuple[float, ...] | None = None
    tolerance: str | tuple[float, float] | None = None
    compare_matrices: bool = True


@dataclass(frozen=True)
class Case:
    """One corpus circuit and the analyses that apply to it.

    ``expects_error`` is the exception class the frozen oracle raises. When an
    applied ``CPP_MIGRATION_PLAN.md`` 2.4.1 ruling changes the error of the live
    package, ``current_error`` is ``(class name, code)`` of the new error; the
    oracle is not edited, so the harness then checks each side against its own
    expectation instead of comparing the two errors.
    """

    name: str
    netlist_text: str
    analyses: list[tuple[str, dict[str, Any]]]
    tolerance: str = "linear"
    expects_error: str | None = None
    tags: tuple[str, ...] = ()
    current_error: tuple[str, str | None] | None = None
    #: (mode, Deviation) pairs: analyses whose live result deliberately differs from the oracle.
    deviations: tuple[tuple[str, Deviation], ...] = ()

    def __post_init__(self) -> None:
        if self.tolerance not in TOLERANCE_CLASSES:
            raise ValueError(f"{self.name}: unknown tolerance class {self.tolerance!r}")
        if not self.analyses:
            raise ValueError(f"{self.name}: a case needs at least one analysis")
        for mode, options in self.analyses:
            if mode not in VALID_MODES:
                raise ValueError(f"{self.name}: unknown analysis mode {mode!r}")
            if not isinstance(options, dict):
                raise TypeError(f"{self.name}: analysis options must be a dict")
        modes = {mode for mode, _ in self.analyses}
        for mode, deviation in self.deviations:
            if mode not in modes or not isinstance(deviation, Deviation):
                raise ValueError(f"{self.name}: deviation for {mode!r} does not match an analysis")

    def deviation_for(self, mode: str) -> Deviation | None:
        """The recorded deliberate change of ``mode``, if any."""

        for deviation_mode, deviation in self.deviations:
            if deviation_mode == mode:
                return deviation
        return None

    def tolerance_for(self, mode: str, options: dict[str, Any]) -> str:
        """Return the tolerance class for one analysis of this case."""

        cls = self.tolerance
        if options.get("use_krylov"):
            cls = _max_class(cls, "nonlinear")
        if mode == "hb":
            cls = _max_class(cls, "hb")
        if options.get("use_mor"):
            cls = "mor"
        return cls


def _max_class(a: str, b: str) -> str:
    return a if _TOLERANCE_RANK[a] >= _TOLERANCE_RANK[b] else b


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def show(**options: Any) -> tuple[str, dict[str, Any]]:
    return ("show_matrix", dict(options))


def op(**options: Any) -> tuple[str, dict[str, Any]]:
    return ("op", dict(options))


def op_nl(**options: Any) -> tuple[str, dict[str, Any]]:
    return ("op", {"max_iter": NL_MAX_ITER, **options})


def tran(t_stop: float, t_step: float, **options: Any) -> tuple[str, dict[str, Any]]:
    return ("tran", {"t_stop": t_stop, "t_step": t_step, **options})


def tran_nl(t_stop: float, t_step: float, **options: Any) -> tuple[str, dict[str, Any]]:
    return tran(t_stop, t_step, max_iter=NL_MAX_ITER, **options)


def ac(f_start: float, f_stop: float, points: int, **options: Any) -> tuple[str, dict[str, Any]]:
    return ("ac", {"f_start": f_start, "f_stop": f_stop, "points": points, **options})


def ac_nl(f_start: float, f_stop: float, points: int, **options: Any) -> tuple[str, dict[str, Any]]:
    return ac(f_start, f_stop, points, max_iter=NL_MAX_ITER, **options)


def hb(harmonics: int, **options: Any) -> tuple[str, dict[str, Any]]:
    return ("hb", {"harmonics": harmonics, **options})


def hb_nl(harmonics: int, **options: Any) -> tuple[str, dict[str, Any]]:
    return hb(harmonics, max_iter=NL_MAX_ITER, **options)


def mor(outputs: list[str], probes: list[str] | None = None, **options: Any) -> dict[str, Any]:
    """Option fragment enabling MOR with display nodes equal to the outputs."""

    return {
        "use_mor": True,
        "mor_output_nodes": list(outputs),
        "probe_nodes": list(outputs if probes is None else probes),
        **options,
    }


def _fmt(value: float) -> str:
    """Netlist-safe 4-significant-digit number (no metric suffix)."""

    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Frozen copies of the workload generators
# ---------------------------------------------------------------------------


def sram_netlist(rows: int = 10, cols: int = 10, step: str = "0.25n", selected_row: int = 3, selected_col: int = 4) -> str:
    """Verbatim copy of ``benchmarks/workload_netlists.sram_netlist``.

    It is also identical to ``tests/test_physical_sram._sram_netlist``.
    """

    v_high = "1.2"
    lines = [
        "GND gnd",
        f"VDD vdd gnd DC {v_high}",
        f"VPCH pch gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 3.5e-9), (0, t < 4.5e-9), ({v_high}, True))",
        f"VWR wr gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 2.5e-9), (0, True))",
        f"VWRB wr_b gnd FUNC 0 Piecewise(({v_high}, t < 1e-9), (0, t < 2.5e-9), ({v_high}, True))",
    ]
    for row in range(rows):
        if row == selected_row:
            lines.append(
                f"VWL{row} wl_{row} gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 2.5e-9), "
                f"(0, t < 4.5e-9), ({v_high}, t < 6e-9), (0, True))"
            )
        else:
            lines.append(f"VWL{row} wl_{row} gnd DC 0")
    for col in range(cols):
        lines.extend(
            [
                f"MPCHBL{col} bl_{col} pch vdd PMOS LEVEL1 3m 0.4 0.02 2f 1f",
                f"MPCHBLB{col} blb_{col} pch vdd PMOS LEVEL1 3m 0.4 0.02 2f 1f",
                f"CBL{col} bl_{col} gnd 20f",
                f"CBLB{col} blb_{col} gnd 20f",
            ]
        )
    lines.extend(
        [
            f"MWRBL bl_{selected_col} wr_b vdd PMOS LEVEL1 8m 0.4 0.02 2f 1f",
            f"MWRBLB blb_{selected_col} wr gnd NMOS LEVEL1 8m 0.4 0.02 2f 1f",
        ]
    )
    for row in range(rows):
        for col in range(cols):
            q = f"q_{row}_{col}"
            qb = f"qb_{row}_{col}"
            q_starts_high = False if (row == selected_row and col == selected_col) else (row + col) % 2 == 0
            lines.extend(
                [
                    f"MPQ{row}_{col} {q} {qb} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
                    f"MPQB{row}_{col} {qb} {q} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
                    f"MNQ{row}_{col} {q} {qb} gnd NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
                    f"MNQB{row}_{col} {qb} {q} gnd NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
                    f"MAXQ{row}_{col} {q} wl_{row} bl_{col} NMOS LEVEL1 2.5m 0.4 0.02 2f 1f",
                    f"MAXQB{row}_{col} {qb} wl_{row} blb_{col} NMOS LEVEL1 2.5m 0.4 0.02 2f 1f",
                    f"CQ{row}_{col} {q} gnd 4f",
                    f"CQB{row}_{col} {qb} gnd 4f",
                    f"RINITQ{row}_{col} {q} {'vdd' if q_starts_high else 'gnd'} 200Meg",
                    f"RINITQB{row}_{col} {qb} {'gnd' if q_starts_high else 'vdd'} 200Meg",
                ]
            )
    lines.extend([f".tran 6n {step}", ".end"])
    return "\n".join(lines)


def rlc_mesh_netlist(rows: int = 23, cols: int = 23, tran_line: str = ".tran 200n 10n") -> str:
    """Verbatim copy of ``benchmarks/workload_netlists.rlc_mesh_netlist``.

    With ``rows == cols`` and ``tran_line`` set accordingly it reproduces
    ``_large_rlc_mesh_netlist`` / ``_large_rlc_rect_netlist`` from
    ``tests/test_krylov_solvers.py`` and ``_rlc_mesh_netlist`` from
    ``tests/test_mor.py``.
    """

    lines = ["IIN n_0_0 0 STEP 1m 20n"]
    for row in range(rows):
        for col in range(cols):
            node = f"n_{row}_{col}"
            lines.append(f"C_{row}_{col} {node} 0 200p")
            lines.append(f"RG_{row}_{col} {node} 0 50Meg")
            if col + 1 < cols:
                lines.append(f"R_H_{row}_{col} {node} n_{row}_{col + 1} 25")
            if row + 1 < rows:
                lines.append(f"L_V_{row}_{col} {node} n_{row + 1}_{col} 20n")
    lines.append(f"RLOAD n_{rows - 1}_{cols - 1} 0 50")
    lines.append(tran_line)
    lines.append(".end")
    return "\n".join(lines)


BJT_THREE_STAGE = """
VCC vcc 0 DC 15
Vin vin 0 SIN 0.001 1k
Cin vin inp 1u
Cref inn 0 1u
RbpT vcc inp 150k
RbpB inp 0 20k
RbnT vcc inn 150k
RbnB inn 0 20k
Qd1 c1 inp etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Qd2 dout inn etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rc1 vcc c1 5.6k
Rc2 vcc dout 5.6k
Rtail etail 0 2.2k
Cd dout midin 1u
RmT vcc midin 150k
RmB midin 0 20k
Qm mout midin me QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rcm vcc mout 12k
Rem me 0 1.5k
Cmo mout fbase 1u
RfT vcc fbase 68k
RfB fbase 0 75k
Qo vcc fbase out QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Reo out 0 1.2k
Rload out 0 10k
.tran 10m 5u
.end
"""

HB_CLOSE_TONE_MIXER = """
IBIAS mix 0 DC 50u
I1 mix 0 SIN 40u 1G
I2 mix 0 SIN 40u 1.005G
RLOAD mix 0 10k
D1 mix 0 1e-12
GENV env 0 mix 0 1m
LENV env 0 1u
CENV env 0 1.013n
RQ env 0 4.3k
.hb 205
.end
"""


# ---------------------------------------------------------------------------
# Existing-test netlists (verbatim)
# ---------------------------------------------------------------------------


def _level1_mos_probe_netlist(mos_type: str, beta: str = "1m") -> str:
    """``tests/test_physical_sram._level1_current`` netlist."""

    return f"""
M1 d g s {mos_type} LEVEL1 {beta} 0.4 0.02 2f 1f
Vd d 0 DC 0
Vg g 0 DC 0
Vs s 0 DC 0
.op
.end
"""


CE_LEVEL1_SERVICE = """
VCC vcc 0 DC 15
Vin in 0 SIN 0.005 1k
Cin in base 1u
RbTop vcc base 150k
RbBot base 0 20k
Rc vcc collector 5.6k
Re emitter 0 680
Q1 collector base emitter QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Cout collector out 4.7u
Rload out 0 10k
.ac 1 10000 5
.end
"""

THREE_STAGE_TEST = """
VCC vcc 0 DC 15
Vin vin 0 SIN 0.001 1k
Cin vin inp 1u
Cref inn 0 1u
RbpT vcc inp 150k
RbpB inp 0 20k
RbnT vcc inn 150k
RbnB inn 0 20k
Qd1 c1 inp etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Qd2 dout inn etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rc1 vcc c1 5.6k
Rc2 vcc dout 5.6k
Rtail etail 0 2.2k
Cd dout midin 1u
RmT vcc midin 150k
RmB midin 0 20k
Qm mout midin me QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rcm vcc mout 12k
Rem me 0 1.5k
Cmo mout fbase 1u
RfT vcc fbase 68k
RfB fbase 0 75k
Qo vcc fbase out QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Reo out 0 1.2k
Rload out 0 10k
.tran 10m 5u
.end
"""

FUNC_WHITESPACE_EXPR = "Piecewise((0, t < 1e-3), (sin(t) + cos(t) + Max(t, 0), True))"
VPCH_EXPR = "Piecewise((0, t < 1e-9), (1.2, t < 3.5e-9), (0, t < 4.5e-9), (1.2, True))"


def _existing_cases() -> list[Case]:
    E = ("existing",)
    cases: list[Case] = [
        # test_netlist_extensions::test_simple_resistor_op_gives_expected_voltage_and_current,
        # ::test_branch_indices_assigned_after_node_indices,
        # test_service::test_service_returns_matrix_payload / ::test_service_returns_dc_solution,
        # test_parser_and_builder::test_parse_op_netlist_and_select_analysis.
        Case(
            "simple_resistor_op",
            """
V1 n1 0 DC 5
R1 n1 0 1k
.op
.end
""",
            [show(), op(), op(use_krylov=True, krylov_rank="auto")],
            tags=E + ("dc",),
        ),
        # test_netlist_extensions::test_resistor_divider_op_matches_hand_calculation
        Case(
            "resistor_divider_op",
            """
V1 n1 0 DC 12
R1 n1 mid 1k
R2 mid 0 2k
.op
.end
""",
            [op()],
            tags=E + ("dc",),
        ),
        # test_netlist_extensions::test_arbitrary_node_names_resolve_correctly
        Case(
            "arbitrary_node_names",
            """
V1 vcc 0 DC 9
R1 vcc out_node 1k
R2 out_node 0 1k
.op
.end
""",
            [op()],
            tags=E + ("dc", "parser"),
        ),
        # test_netlist_extensions::test_decoupled_name_type_resistor
        Case(
            "decoupled_name_type",
            """
supply:V n1 0 DC 5
load:R n1 0 1k
.op
.end
""",
            [op(), show()],
            tags=E + ("dc", "parser"),
        ),
        # test_netlist_extensions::test_decoupled_name_handles_underscores_and_long_identifiers
        Case(
            "decoupled_name_underscores",
            """
bias_supply:V vcc 0 DC 12
feedback_resistor:R vcc out 4.7k
load_resistor:R out 0 470
.op
.end
""",
            [op()],
            tags=E + ("dc", "parser"),
        ),
        # test_netlist_extensions::test_invalid_explicit_type_is_rejected
        Case(
            "invalid_explicit_type",
            """
mything:NOPE n1 0 5
.op
.end
""",
            [op()],
            expects_error="NetlistError",
            tags=E + ("error", "parser"),
        ),
        # test_netlist_extensions::test_gnd_statement_aliases_node_to_zero
        Case(
            "gnd_statement_alias",
            """
GND vss
V1 in vss DC 5
R1 in vss 1k
.op
.end
""",
            [op(), show()],
            tags=E + ("dc", "parser", "gnd"),
        ),
        # test_netlist_extensions::test_named_ground_via_explicit_gnd_component
        Case(
            "gnd_named_component",
            """
GND1 ground_rail
V1 vcc ground_rail DC 9
R1 vcc ground_rail 1k
.op
.end
""",
            [op()],
            tags=E + ("dc", "parser", "gnd"),
        ),
        # test_netlist_extensions::test_redundant_gnd_zero_is_a_noop
        Case(
            "gnd_zero_noop",
            """
GND 0
V1 n1 0 DC 1
R1 n1 0 100
.op
.end
""",
            [op()],
            tags=E + ("dc", "parser", "gnd"),
        ),
        # test_netlist_extensions::test_gnd_aliasing_catches_v_source_shorted_to_ground
        Case(
            "gnd_alias_vsource_shorted",
            """
GND vss
V1 vss 0 DC 5
R1 vss 0 1k
.op
.end
""",
            [op()],
            expects_error="NetlistError",
            tags=E + ("error", "parser", "gnd"),
        ),
        # test_netlist_extensions::test_solver_handles_zero_diagonal_for_capacitor_input
        Case(
            "zero_diagonal_capacitor_input",
            """
V1 in 0 DC 0
C1 in mid 1u
R1 mid 0 1k
.op
.end
""",
            [op(), tran(1e-3, 50e-6), show()],
            tags=E + ("dc", "tran"),
        ),
        # test_netlist_extensions::test_capacitor_input_transient_runs_without_singular_g
        Case(
            "capacitor_input_transient",
            """
V1 in 0 SIN 1 100
C1 in mid 1u
R1 mid 0 1k
.tran 20m 100u
.end
""",
            [tran(20e-3, 0.5e-3), ac(10, 1e4, 20), hb(3)],
            tags=E + ("tran", "ac", "hb"),
        ),
        # test_netlist_extensions::test_hybrid_pi_small_signal_amp_runs_without_singular_g
        Case(
            "hybrid_pi_vccs_amp",
            """
V1 in 0 SIN 0.05 1k
Vcc vcc 0 DC 12
Cin in base 10u
Rpi base emitter 2.5k
Re emitter 0 470
Rc vcc collector 4.7k
G1 collector emitter base emitter 0.04
.tran 5m 50u
.end
""",
            [tran(5e-3, 100e-6), op(), ac(10, 1e5, 20)],
            tags=E + ("tran", "ac", "controlled"),
        ),
        # test_netlist_extensions::test_plus_15v_ce_demo_bias_has_midrail_collector_and_gain
        Case(
            "ce_level1_15v_demo",
            """
VCC vcc 0 DC 15
Vin vin 0 SIN 0.02 1k
Cin vin base 1u
RbTop vcc base 220k
RbBot base 0 15k
Rc vcc collector 12k
Re emitter 0 330
Q1 collector base emitter QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Cout collector out 4.7u
Rload out 0 10k
.tran 10m 5u
.end
""",
            [tran_nl(2e-3, 50e-6), op_nl(), ac_nl(10, 1e5, 20)],
            tolerance="nonlinear",
            tags=E + ("bjt", "tran", "ac"),
        ),
        # test_netlist_extensions::test_ac_coupled_three_stage_bjt_demo_biases_each_stage_and_amplifies
        # (the workload copy BJT_THREE_STAGE is exercised separately below).
        Case(
            "bjt_three_stage_test",
            THREE_STAGE_TEST,
            [op_nl(), ac_nl(10, 1e5, 20)],
            tolerance="nonlinear",
            tags=E + ("bjt", "ac"),
        ),
        # test_netlist_extensions::test_qnpn_small_signal_device_matches_hybrid_pi_shape (expanded)
        Case(
            "hybrid_pi_expanded",
            """
V1 in 0 SIN 0.05 1k
Rpi base emitter 2.5k
Re emitter 0 470
Rc collector 0 4.7k
Ro collector emitter 100k
Cpi base emitter 8p
Cmu base collector 3p
G1 collector emitter base emitter 40m
Cin in base 10u
.tran 5m 50u
.end
""",
            [tran(5e-3, 100e-6), ac(10, 1e6, 20)],
            tags=E + ("tran", "ac", "controlled"),
        ),
        # test_netlist_extensions::test_qnpn_small_signal_device_matches_hybrid_pi_shape (compact)
        Case(
            "hybrid_pi_compact_qnpn",
            """
V1 in 0 SIN 0.05 1k
Re emitter 0 470
Rc collector 0 4.7k
Q1 collector base emitter QNPN 40m 2.5k
Cin in base 10u
.tran 5m 50u
.end
""",
            [tran(5e-3, 100e-6), ac(10, 1e6, 20), op()],
            tags=E + ("tran", "ac", "bjt_small_signal"),
        ),
        # test_netlist_extensions::test_qnpn_extended_small_signal_model_stamps_ro_and_caps
        Case(
            "qnpn_extended_small_signal_only",
            """
Q1 collector base emitter QNPN 40m 2.5k 100k 8p 3p 1p
.ac 1 1e9 4
.end
""",
            [show()],
            tags=E + ("matrix", "bjt_small_signal"),
        ),
        # test_netlist_extensions::test_level1_bjt_parser_registers_nonlinear_device_and_junction_caps
        Case(
            "level1_bjt_only",
            """
Q1 collector base emitter QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
.op
.end
""",
            [show()],
            tags=E + ("matrix", "bjt"),
        ),
        # test_netlist_extensions::test_mos_small_signal_model_is_backend_owned_with_caps
        Case(
            "mos_small_signal_only",
            """
M1 drain gate source NMOS 5m 50k 5p 1p 0 0 0
.ac 1 1e9 4
.end
""",
            [show()],
            tags=E + ("matrix", "mos_small_signal"),
        ),
        # test_netlist_extensions::test_duplicate_branch_name_raises
        Case(
            "duplicate_branch_name",
            """
V1 n1 0 DC 5
V1 n2 0 DC 3
R1 n1 n2 1k
.op
.end
""",
            [op()],
            expects_error="NetlistError",
            tags=E + ("error", "parser"),
        ),
        # test_service::test_service_reports_transistor_operating_points (BJT small-signal)
        Case(
            "service_bjt_small_signal_op",
            """
VCC c 0 DC 5
R1 c out 1k
Q1 out b 0 QNPN 40m 2.5k 100k 8p 3p 0 0 0
VB b 0 DC 0.01
.op
.end
""",
            [op(), ac(10, 1e8, 20)],
            tags=E + ("dc", "bjt_small_signal"),
        ),
        # test_service::test_service_reports_transistor_operating_points (MOS Level-1 inverter)
        Case(
            "service_mos_inverter_op",
            """
VDD vdd 0 DC 1.2
MP out in vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f
MN out in 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f
VIN in 0 DC 0
.op
.end
""",
            [op_nl(), tran_nl(1e-9, 0.1e-9)],
            tolerance="nonlinear",
            tags=E + ("mos",),
        ),
        # test_service::test_service_uses_level1_bjt_op_for_ac_and_tran_metadata
        Case(
            "service_ce_level1_ac_tran",
            CE_LEVEL1_SERVICE,
            [ac_nl(1, 10000, 5), tran_nl(20e-6, 5e-6), op_nl(), hb_nl(4)],
            tolerance="nonlinear",
            tags=E + ("bjt", "ac", "tran", "hb"),
        ),
        # test_service::test_service_decimates_large_waveform_output_when_requested
        # (1001-point grid shrunk to 51 points; max_points shrunk to keep decimation active).
        Case(
            "service_decimation",
            """
V1 n1 0 SIN 1 1k
R1 n1 0 1k
.tran 1m 1u
.end
""",
            [tran(1e-3, 20e-6, output_max_points=10), tran(1e-3, 20e-6, output_max_points=51), tran(1e-3, 20e-6)],
            tags=E + ("tran", "decimation"),
        ),
        # test_krylov_solvers::test_backend_accepts_auto_and_arbitrary_manual_krylov_rank,
        # ::test_service_accepts_krylov_option_and_reports_policy, ::test_service_reports_arbitrary_manual_krylov_rank,
        # ::test_service_allows_manual_krylov_method_override
        Case(
            "krylov_current_source_op",
            """
I1 n1 0 DC 1m
R1 n1 0 1k
.op
.end
""",
            [
                op(),
                op(use_krylov=True, krylov_rank="auto"),
                op(use_krylov=True, krylov_rank=137),
                op(use_krylov=True, krylov_rank="auto", krylov_method="arnoldi_gmres"),
                op(use_krylov=True, krylov_method="cr"),
            ],
            tags=E + ("dc", "krylov"),
        ),
        # test_krylov_solvers::test_transient_time_grid_ends_at_requested_stop
        Case(
            "step_current_time_grid",
            """
I1 n1 0 STEP 1m 20n
R1 n1 0 1k
.tran 200n 10n
.end
""",
            [tran(200e-9, 10e-9, probe_nodes=["V(n1)"]), tran(200e-9, 30e-9)],
            tags=E + ("tran",),
        ),
        # test_krylov_solvers::test_sparse_mna_storage_matches_dense_matrices (_large_rlc_mesh_netlist(size=3))
        Case(
            "rlc_mesh_3x3_test",
            rlc_mesh_netlist(3, 3, ".tran 80n 10n"),
            [show(), tran(80e-9, 2e-9), tran(80e-9, 2e-9, use_krylov=True), ac(1e6, 1e9, 20)],
            tags=E + ("rlc", "tran", "krylov"),
        ),
        # test_krylov_solvers::test_large_rlc_mesh_energy_converges_when_time_step_is_halved (size=6)
        Case(
            "rlc_mesh_6x6_energy",
            rlc_mesh_netlist(6, 6, ".tran 200n 5n"),
            [tran(200e-9, 10e-9), tran(200e-9, 5e-9)],
            tags=E + ("rlc", "tran"),
        ),
        # test_mor::test_backend_accepts_mor_options
        Case(
            "mor_backend_options_op",
            """
I1 out 0 DC 1m
R1 out 0 1k
.op
.end
""",
            [op(), op(**mor(["V(out)"], mor_method="linear", mor_order=37))],
            tags=E + ("dc", "mor"),
        ),
        # test_mor::test_linear_ac_mor_matches_full_selected_output
        Case(
            "rc_ladder_ac_mor",
            """
V1 in 0 AC 1
R1 in n1 100
C1 n1 0 1n
R2 n1 out 200
C2 out 0 2n
.ac 1 1000000 8
.end
""",
            [ac(1, 1e6, 8), ac(1, 1e6, 8, **mor(["V(out)"]))],
            tags=E + ("ac", "mor"),
        ),
        # test_mor::test_linear_transient_mor_matches_full_selected_output,
        # ::test_linear_transient_requested_tpwl_routes_to_linear_krylov_mor
        Case(
            "rc_step_tran_mor",
            """
I1 in 0 STEP 1m 1u
R1 in out 100
C1 out 0 1n
R2 out 0 1k
.tran 5u 0.5u
.end
""",
            [
                tran(5e-6, 0.5e-6),
                tran(5e-6, 0.5e-6, **mor(["V(out)"])),
                tran(5e-6, 0.5e-6, **mor(["V(out)"], mor_method="tpwl", mor_order=4)),
            ],
            tags=E + ("tran", "mor"),
        ),
        # test_mor::test_linear_mor_auto_validates_large_sparse_rlc_outputs (500 steps shrunk to 50)
        Case(
            "rlc_mesh_8x8_mor",
            rlc_mesh_netlist(8, 8, ".tran 500n 1n"),
            [tran(50e-9, 1e-9, **mor(["V(n_0_0)", "V(n_4_4)"], mor_method="auto", mor_order="auto"))],
            tags=E + ("rlc", "tran", "mor"),
        ),
        # test_mor::test_ac_mor_uses_level1_transistor_operating_point_linearization
        Case(
            "ce_level1_ac_mor",
            """
VCC vcc 0 DC 15
Vin in 0 AC 0.01
Cin in base 1u
RbTop vcc base 220k
RbBot base 0 15k
Rc vcc collector 12k
Re emitter 0 330
Q1 collector base emitter QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Cout collector out 4.7u
Rload out 0 10k
.ac 1 10000 5
.end
""",
            [ac_nl(1, 10000, 5), ac_nl(1, 10000, 5, **mor(["V(out)"]))],
            tolerance="nonlinear",
            tags=E + ("bjt", "ac", "mor"),
        ),
        # test_mor::test_nonlinear_transient_tpwl_trains_and_reuses_cache
        Case(
            "diode_tpwl",
            """
I1 in 0 STEP 1m 1u
R1 in out 100
D1 out 0 1e-12
C1 out 0 1n
.tran 5u 0.5u
.end
""",
            [tran_nl(5e-6, 0.5e-6), tran_nl(5e-6, 0.5e-6, **mor(["V(out)"], mor_method="tpwl"))],
            tolerance="nonlinear",
            tags=E + ("diode", "tran", "mor"),
        ),
        # test_mor::test_service_rejects_display_nodes_outside_mor_outputs (plain run)
        Case(
            "divider_ac",
            """
V1 in 0 AC 1
R1 in out 1k
R2 out 0 1k
.ac 1 1000 3
.end
""",
            [ac(1, 1000, 3), ac(1, 1000, 3, **mor(["V(out)"]))],
            tags=E + ("ac", "mor"),
        ),
        # test_mor::test_service_rejects_display_nodes_outside_mor_outputs (error run)
        Case(
            "divider_ac_mor_probe_not_subset",
            """
V1 in 0 AC 1
R1 in out 1k
R2 out 0 1k
.ac 1 1000 3
.end
""",
            [ac(1, 1000, 3, **mor(["V(out)"], probes=["V(in)"]))],
            tolerance="mor",
            expects_error="NetlistError",
            tags=E + ("error", "mor"),
        ),
        # test_mor::test_op_and_hb_report_mor_not_used (.op part)
        Case(
            "op_mor_not_used",
            """
V1 out 0 DC 1
R1 out 0 1k
.op
.end
""",
            [op(use_mor=True, mor_output_nodes=["V(out)"])],
            tags=E + ("dc", "mor"),
        ),
        # test_mor::test_op_and_hb_report_mor_not_used (.hb part)
        Case(
            "hb_mor_not_used",
            """
V1 out 0 SIN 1 1000
R1 out 0 1k
.hb 3
.end
""",
            [hb(3), hb(3, use_mor=True, mor_output_nodes=["V(out)"])],
            tags=E + ("hb", "mor"),
        ),
        # test_harmonic_balance::test_hb_uses_gcd_fundamental_for_multitone_sources
        Case(
            "hb_multitone_gcd",
            """
V1 in 0 SIN 1 1k
V2 in mid SIN 0.5 1.5k
R1 mid 0 1k
.hb 3
.end
""",
            [hb(3), hb(3, time_window=4e-3), tran(4e-3, 0.1e-3), ac(100, 2000, 20)],
            tags=E + ("hb", "tran", "ac"),
        ),
        # test_harmonic_balance::test_hb_expands_harmonic_count_to_cover_highest_source
        Case(
            "hb_multitone_expand",
            """
V1 in 0 SIN 1 1k
V2 in mid SIN 0.5 1.5k
R1 mid 0 1k
.hb 1
.end
""",
            [hb(1), ("hb", {})],
            tags=E + ("hb",),
        ),
        # test_harmonic_balance::test_hb_ghz_close_tone_diode_mixer_resolves_beat_without_long_transient
        # (identical to the HB_CLOSE_TONE_MIXER workload, which therefore is not repeated).
        Case(
            "hb_ghz_mixer",
            """
IBIAS mix 0 DC 50u
I1 mix 0 SIN 40u 1G
I2 mix 0 SIN 40u 1.005G
RLOAD mix 0 10k
D1 mix 0 1e-12
GENV env 0 mix 0 1m
LENV env 0 1u
CENV env 0 1.013n
RQ env 0 4.3k
.hb 205
.end
""",
            [hb_nl(205)],
            tolerance="nonlinear",
            tags=E + ("workload", "hb", "diode", "slow"),
        ),
        # test_harmonic_balance::test_hb_ghz_close_tone_demo_reconstructs_steady_state_beat_window
        Case(
            "hb_ghz_mixer_window",
            """
IBIAS mix 0 DC 50u
I1 mix 0 SIN 40u 1G
I2 mix 0 SIN 40u 1.005G
RLOAD mix 0 10k
D1 mix 0 1e-12
GENV env 0 mix 0 1m
LENV env 0 1u
CENV env 0 1.013n
RQ env 0 4.3k
.hb 205 200n
.end
""",
            [hb_nl(205, time_window=200e-9)],
            tolerance="nonlinear",
            tags=E + ("hb", "diode", "slow"),
        ),
        # test_harmonic_balance::test_hb_requires_a_sinusoidal_source
        Case(
            "hb_no_sinusoid",
            """
V1 in 0 DC 1
R1 in 0 1k
.hb 3
.end
""",
            [hb(3)],
            tolerance="hb",
            expects_error="SolverConvergenceError",
            tags=E + ("error", "hb"),
        ),
        # test_parser_and_builder::test_builder_creates_branch_metadata_for_voltage_sources
        Case(
            "branch_metadata_vsource",
            """
V1 n1 0 DC 1
R1 n1 0 1k
.op
.end
""",
            [op(), show()],
            tags=E + ("dc",),
        ),
        # test_parser_and_builder::test_builder_supports_vccs_cccs_and_ccvs
        Case(
            "vccs_cccs_ccvs",
            """
V1 n1 0 DC 1
V2 n3 n4 DC 0
G1 n2 0 n1 0 2
F1 n2 0 V1 3
H1 n3 0 V1 4
R1 n2 0 1k
R2 n4 0 1k
.op
.end
""",
            [op(), show(), op(use_krylov=True)],
            tags=E + ("dc", "controlled"),
        ),
        # test_parser_and_builder::test_builder_populates_ac_rhs_for_small_signal_sources
        Case(
            "ac_source_rhs",
            """
V1 n1 0 AC 1
R1 n1 0 1k
.ac 1 10 5
.end
""",
            [ac(1, 10, 5), show()],
            tags=E + ("ac",),
        ),
        # test_physical_sram::test_func_source_accepts_whitespace_expression_and_schematic_roundtrip
        Case(
            "func_whitespace_expression",
            f"""
V1 n 0 FUNC 0 {FUNC_WHITESPACE_EXPR}
R1 n 0 1k
.tran 2m 1m
.end
""",
            [tran(2e-3, 1e-3), tran(2e-3, 0.05e-3)],
            tags=E + ("tran", "func"),
        ),
        # test_physical_sram::test_vpch_precharge_func_expression_is_preserved_exactly
        Case(
            "vpch_precharge_func",
            f"""
VPCH pch gnd FUNC 0 {VPCH_EXPR}
Rload pch gnd 1Meg
GND gnd
.tran 6n 1n
.end
""",
            [tran(6e-9, 1e-9), tran(6e-9, 0.25e-9)],
            tags=E + ("tran", "func", "gnd"),
        ),
        # test_physical_sram::_level1_current (NMOS / PMOS)
        Case(
            "level1_mos_probe_nmos",
            _level1_mos_probe_netlist("NMOS"),
            [op_nl(), show()],
            tolerance="nonlinear",
            tags=E + ("mos",),
        ),
        Case(
            "level1_mos_probe_pmos",
            _level1_mos_probe_netlist("PMOS"),
            [op_nl()],
            tolerance="nonlinear",
            tags=E + ("mos",),
        ),
        # test_physical_sram::test_level1_inverter_transient_converges
        Case(
            "level1_inverter_func_tran",
            """
VDD vdd 0 DC 1.2
VIN in 0 FUNC 0 Piecewise((0, t < 1e-9), (1.2, True))
MP out in vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f
MN out in 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f
Cout out 0 5f
.tran 2n 0.5n
.end
""",
            [tran_nl(2e-9, 0.5e-9), tran_nl(2e-9, 0.05e-9), op_nl()],
            tolerance="nonlinear",
            tags=E + ("mos", "tran", "func"),
        ),
        # test_physical_sram::test_single_6t_cell_writes_holds_and_reads_differential_bitlines
        Case(
            "sram_1x1",
            sram_netlist(rows=1, cols=1, step="0.5n", selected_row=0, selected_col=0),
            [tran_nl(6e-9, 0.5e-9), op_nl()],
            tolerance="nonlinear",
            tags=E + ("mos", "sram", "tran"),
        ),
        # test_physical_sram::test_full_10x10_sram_transient_runs_selected_cell
        Case(
            "sram_10x10_test",
            sram_netlist(rows=10, cols=10, step="1n"),
            [tran_nl(6e-9, 1e-9)],
            tolerance="nonlinear",
            tags=E + ("mos", "sram", "tran", "slow"),
        ),
    ]
    return cases


def _workload_cases() -> list[Case]:
    W = ("workload",)
    return [
        # benchmarks/workload_netlists.sram_netlist() with its default .tran 6n 0.25n (24 steps).
        Case(
            "workload_sram_10x10",
            sram_netlist(),
            [tran_nl(6e-9, 0.25e-9)],
            tolerance="nonlinear",
            tags=W + ("mos", "sram", "tran", "slow"),
        ),
        # benchmarks/workload_netlists.rlc_mesh_netlist() (23x23, 1035 unknowns), default and Krylov paths.
        Case(
            "workload_rlc_mesh_23x23",
            rlc_mesh_netlist(),
            [
                tran(200e-9, 10e-9),
                tran(
                    200e-9,
                    10e-9,
                    use_krylov=True,
                    krylov_rank="auto",
                    probe_nodes=["V(n_0_0)", "V(n_11_11)", "I(L_V_10_11)"],
                ),
            ],
            tags=W + ("rlc", "tran", "krylov", "slow"),
        ),
        # benchmarks/workload_netlists.BJT_THREE_STAGE (2000-step .tran shrunk to 40 steps).
        Case(
            "workload_bjt_three_stage",
            BJT_THREE_STAGE,
            [tran_nl(2e-3, 50e-6), op_nl()],
            tolerance="nonlinear",
            tags=W + ("bjt", "tran"),
        ),
        # benchmarks/workload_netlists.HB_CLOSE_TONE_MIXER is byte-for-byte the netlist of the
        # existing case "hb_ghz_mixer"; it is exercised there (a second 205-harmonic run would only
        # cost time). Its text is kept above as HB_CLOSE_TONE_MIXER for reference.
    ]


# ---------------------------------------------------------------------------
# Seeded random generators
# ---------------------------------------------------------------------------


def random_rlc_ladder(seed: int) -> Case:
    """SIN-driven RC/RLC ladder with 3-11 nodes plus a DC bias current."""

    rng = random.Random(f"rlc_ladder:{seed}")
    sections = rng.randint(2, 5)
    base_f = rng.choice([1e3, 2e3, 5e3, 1e4])
    amp = rng.uniform(0.5, 2.0)
    lines = [f"V1 in 0 SIN {_fmt(amp)} {_fmt(base_f)}"]
    prev = "in"
    for index in range(1, sections + 1):
        node = f"n{index}"
        if rng.random() < 0.5:
            mid = f"m{index}"
            lines.append(f"R{index} {prev} {mid} {_fmt(rng.uniform(10, 1000))}")
            lines.append(f"L{index} {mid} {node} {_fmt(rng.uniform(1e-4, 1e-2))}")
        else:
            lines.append(f"R{index} {prev} {node} {_fmt(rng.uniform(10, 1000))}")
        lines.append(f"C{index} {node} 0 {_fmt(rng.uniform(1e-8, 1e-6))}")
        if rng.random() < 0.3:
            lines.append(f"RSH{index} {node} 0 {_fmt(rng.uniform(1e3, 1e5))}")
        prev = node
    lines.append(f"RL {prev} 0 {_fmt(rng.uniform(100, 10e3))}")
    lines.append(f"IB {prev} 0 DC {_fmt(rng.uniform(1e-4, 1e-3))}")
    lines.append(".end")
    t_stop = 2.0 / base_f
    analyses = [op(), tran(t_stop, t_stop / 40), ac(base_f / 10, base_f * 100, 30), hb(3)]
    if seed % 2 == 0:
        analyses.append(ac(base_f / 10, base_f * 100, 30, **mor([f"V({prev})"])))
    else:
        analyses.append(tran(t_stop, t_stop / 40, use_krylov=True))
    return Case(f"rand_rlc_ladder_s{seed}", "\n".join(lines), analyses, tags=("random", "rlc"))


def random_rlc_mesh(seed: int) -> Case:
    """Small RLC mesh (4-12 nodes) driven by STEP, AC and DC current sources."""

    rng = random.Random(f"rlc_mesh:{seed}")
    rows, cols = rng.choice([(2, 2), (2, 3), (3, 3), (2, 4), (3, 4)])
    amp = rng.uniform(1e-4, 1e-2)
    last = f"n_{rows - 1}_{cols - 1}"
    lines = [
        f"IIN n_0_0 0 STEP {_fmt(amp)} {_fmt(rng.uniform(2e-9, 10e-9))}",
        f"IAC n_0_0 0 AC {_fmt(amp)}",
        f"IDC {last} 0 DC {_fmt(rng.uniform(1e-4, 1e-3))}",
    ]
    for row in range(rows):
        for col in range(cols):
            node = f"n_{row}_{col}"
            lines.append(f"C_{row}_{col} {node} 0 {_fmt(rng.uniform(1e-12, 1e-10))}")
            lines.append(f"RG_{row}_{col} {node} 0 {_fmt(rng.uniform(1e6, 1e8))}")
            if col + 1 < cols:
                lines.append(f"R_H_{row}_{col} {node} n_{row}_{col + 1} {_fmt(rng.uniform(5, 100))}")
            if row + 1 < rows:
                lines.append(f"L_V_{row}_{col} {node} n_{row + 1}_{col} {_fmt(rng.uniform(1e-9, 1e-7))}")
    lines.append(f"RLOAD {last} 0 {_fmt(rng.uniform(20, 200))}")
    lines.append(".end")
    analyses = [op(), tran(40e-9, 1e-9), ac(1e6, 1e9, 30)]
    variant = seed % 3
    if variant == 0:
        analyses.append(tran(40e-9, 1e-9, use_krylov=True))
    elif variant == 1:
        analyses.append(ac(1e6, 1e9, 30, **mor([f"V({last})"])))
    else:
        analyses.append(tran(40e-9, 1e-9, **mor([f"V({last})", "V(n_0_0)"])))
    return Case(f"rand_rlc_mesh_s{seed}", "\n".join(lines), analyses, tags=("random", "rlc"))


def random_resistive_controlled(seed: int) -> Case:
    """Connected resistor network with V/I sources and all four controlled sources."""

    rng = random.Random(f"resistive_controlled:{seed}")
    count = rng.randint(3, 8)
    nodes = [f"a{index}" for index in range(count)]
    edges: set[tuple[str, str]] = set()
    lines: list[str] = []

    def add_resistor(n1: str, n2: str) -> None:
        key = tuple(sorted((n1, n2)))
        if n1 == n2 or key in edges:
            return
        edges.add(key)  # type: ignore[arg-type]
        lines.append(f"R{len(edges)} {n1} {n2} {_fmt(rng.uniform(100, 10e3))}")

    for index in range(1, count):
        add_resistor(nodes[index], nodes[rng.randrange(index)])
    for _ in range(rng.randint(0, count)):
        add_resistor(rng.choice(nodes), rng.choice(nodes))
    for node in rng.sample(nodes, k=rng.randint(1, min(3, count))):
        add_resistor(node, "0")

    v_nodes = rng.sample(nodes, k=rng.randint(1, min(2, count - 1)))
    for index, node in enumerate(v_nodes, start=1):
        lines.append(f"V{index} {node} 0 DC {_fmt(rng.uniform(-10, 10))}")
    for index in range(1, rng.randint(1, 2) + 1):
        n1 = rng.choice(nodes)
        n2 = rng.choice([node for node in nodes if node != n1] + ["0"])
        lines.append(f"I{index} {n1} {n2} DC {_fmt(rng.uniform(-5e-3, 5e-3))}")

    c1, c2 = rng.sample(nodes, k=2)
    explicit = seed % 2 == 1  # alternate between bare and SPICE-style / name:TYPE spellings
    vcvs = f"E1 eout 0 VCVS {c1} {c2}" if explicit else f"E1 eout 0 {c1} {c2}"
    lines.append(f"{vcvs} {_fmt(rng.uniform(0.5, 5))}")
    lines.append(f"RE eout 0 {_fmt(rng.uniform(100, 10e3))}")
    g_out = rng.choice(nodes)
    g1, g2 = rng.sample(nodes + ["0"], k=2)
    vccs_name = "trans:VCCS" if explicit else "G1"
    lines.append(f"{vccs_name} {g_out} 0 {g1} {g2} {_fmt(rng.uniform(1e-5, 1e-3))}")
    f_out = rng.choice(nodes)
    lines.append(f"F1 {f_out} 0 {'CCCS ' if explicit else ''}V1 {_fmt(rng.uniform(0.1, 2))}")
    lines.append(f"H1 hout 0 V1 {_fmt(rng.uniform(10, 1000))}")
    lines.append(f"RH hout 0 {_fmt(rng.uniform(100, 10e3))}")
    lines.append(".op")
    lines.append(".end")
    # Full-restart GMRES (rank >= n): with the default auto rank (ceil(n/2)) the
    # restarted iteration on these ill-conditioned 11x11 systems stalls exactly at
    # the tolerance and the converged flag depends on rounding luck between
    # implementations (seed 3 flipped when the native solver landed).
    analyses = [op(), show(), op(use_krylov=True, krylov_rank=64)]
    return Case(f"rand_resistive_ctrl_s{seed}", "\n".join(lines), analyses, tags=("random", "dc", "controlled"))


def random_mos_inverter_chain(seed: int) -> Case:
    """Level-1 CMOS inverter chain.

    Even seeds: digital chain of 1-6 stages driven by a FUNC step (``.op``
    sees VIN=0). Every output carries a 10 Meg keeper resistor toward its
    t=0 logic level (like the SRAM generator's ``RINIT*`` resistors): without
    it the current Newton solver needs gmin=1e-5 or fails outright on chains of
    three or more stages (see ``tests/EDGE_CASES.md``), which would make the
    case gmin-fragile.

    Odd seeds: analog chain of 1-2 stages, input biased in the high-gain
    region through a resistor with a small SIN injected through a coupling
    capacitor, so the linearized ``.ac`` is meaningful.
    """

    rng = random.Random(f"mos_chain:{seed}")
    digital = seed % 2 == 0
    stages = rng.randint(1, 6) if digital else rng.randint(1, 2)
    vdd = rng.choice([1.0, 1.2, 1.8])
    vth = rng.uniform(0.3, 0.5)
    lam = rng.uniform(0.01, 0.05)
    lines = [f"VDD vdd 0 DC {_fmt(vdd)}"]
    if digital:
        lines.append(f"VIN in 0 FUNC 0 Piecewise((0, t < 1e-9), ({_fmt(vdd)}, True))")
    else:
        lines.append(f"VBIAS vb 0 DC {_fmt(vdd * rng.uniform(0.3, 0.7))}")
        lines.append(f"RIN vb in {_fmt(rng.uniform(5e3, 50e3))}")
        lines.append(f"VAC acs 0 SIN 0.01 {_fmt(rng.choice([1e7, 2e7, 5e7]))}")
        lines.append(f"CAC acs in {_fmt(rng.uniform(1e-13, 1e-12))}")
    prev = "in"
    for stage in range(1, stages + 1):
        out = f"o{stage}"
        beta_n = rng.uniform(0.5e-3, 3e-3)
        beta_p = beta_n * rng.uniform(0.3, 0.8)
        lines.append(f"MP{stage} {out} {prev} vdd PMOS LEVEL1 {_fmt(beta_p)} {_fmt(vth)} {_fmt(lam)} 2f 1f")
        lines.append(f"MN{stage} {out} {prev} 0 NMOS LEVEL1 {_fmt(beta_n)} {_fmt(vth)} {_fmt(lam)} 2f 1f")
        lines.append(f"CL{stage} {out} 0 {_fmt(rng.uniform(2e-15, 20e-15))}")
        if digital:
            lines.append(f"RK{stage} {out} {'vdd' if stage % 2 == 1 else '0'} 10Meg")
        prev = out
    lines.append(".end")
    if digital:
        analyses = [op_nl(), tran_nl(3e-9, 0.1e-9)]
    else:
        analyses = [op_nl(), tran_nl(200e-9, 5e-9), ac_nl(1e5, 1e9, 25)]
    return Case(f"rand_mos_chain_s{seed}", "\n".join(lines), analyses, tolerance="nonlinear", tags=("random", "mos"))


def random_bjt_ce_stage(seed: int) -> Case:
    """Level-1 BJT common-emitter stage, divider biased; seed 2 uses a QPNP."""

    rng = random.Random(f"bjt_ce:{seed}")
    vcc = rng.choice([9.0, 12.0, 15.0])
    vb = rng.uniform(1.5, 2.5)
    rb2 = rng.uniform(10e3, 22e3)
    rb1 = rb2 * (vcc / vb - 1.0)
    ie = rng.uniform(0.5e-3, 2e-3)
    re = (vb - 0.7) / ie
    rc = rng.uniform(0.3, 0.45) * vcc / ie
    freq = rng.choice([1e3, 2e3, 5e3])
    amp = rng.uniform(1e-3, 10e-3)
    bf = rng.choice([80, 100, 150, 200])
    pnp = seed == 2
    lines = [f"VCC vcc 0 DC {_fmt(vcc)}", f"Vin vin 0 SIN {_fmt(amp)} {_fmt(freq)}", "Cin vin base 1u"]
    if pnp:
        # Mirror image: emitter resistor to VCC, collector resistor to ground.
        lines += [
            f"RbTop vcc base {_fmt(rb2)}",
            f"RbBot base 0 {_fmt(rb1)}",
            f"Re vcc emitter {_fmt(re)}",
            f"Rc collector 0 {_fmt(rc)}",
            f"Q1 collector base emitter QPNP LEVEL1 1e-15 {bf} 3 100 25 4p 2p 50 0.5 5",
        ]
    else:
        lines += [
            f"RbTop vcc base {_fmt(rb1)}",
            f"RbBot base 0 {_fmt(rb2)}",
            f"Rc vcc collector {_fmt(rc)}",
            f"Re emitter 0 {_fmt(re)}",
            f"Q1 collector base emitter QNPN LEVEL1 1e-15 {bf} 3 100 25 4p 2p 50 0.5 5",
        ]
    if rng.random() < 0.5:
        lines.append(f"Ce emitter {'vcc' if pnp else '0'} 10u")
    lines += ["Cout collector out 4.7u", f"Rload out 0 {_fmt(rng.uniform(5e3, 50e3))}", ".end"]
    t_stop = 2.0 / freq
    analyses = [op_nl(), tran_nl(t_stop, t_stop / 40), ac_nl(10, 1e6, 25), hb_nl(4)]
    return Case(f"rand_bjt_ce_s{seed}", "\n".join(lines), analyses, tolerance="nonlinear", tags=("random", "bjt"))


def random_linear_sources(seed: int) -> Case:
    """Linear RC/RL/RLC net driven by SIN/COS/STEP/FUNC sources."""

    rng = random.Random(f"linear_sources:{seed}")
    base_f = rng.choice([500.0, 1e3, 2e3, 5e3])
    period = 1.0 / base_f
    variant = seed % 4
    lines: list[str] = []
    if variant == 0:
        lines += [
            f"V1 a 0 SIN {_fmt(rng.uniform(0.5, 2))} {_fmt(base_f)}",
            f"V2 b 0 COS {_fmt(rng.uniform(0.1, 1))} {_fmt(2 * base_f)}",
            f"I1 c 0 STEP {_fmt(rng.uniform(1e-4, 1e-3))} {_fmt(period / 3)}",
        ]
    elif variant == 1:
        lines += [
            f"V1 a 0 COS {_fmt(rng.uniform(0.5, 2))} {_fmt(base_f)}",
            f"V2 b 0 STEP {_fmt(rng.uniform(0.5, 1.5))} {_fmt(base_f)} {rng.choice(['50%', '0.3', '25%'])}",
            f"I1 c 0 SIN {_fmt(rng.uniform(1e-4, 1e-3))} {_fmt(3 * base_f)}",
        ]
    elif variant == 2:
        lines += [
            f"V1 a 0 SIN {_fmt(rng.uniform(0.5, 2))} {_fmt(base_f)}",
            f"V2 b 0 FUNC 0 {_fmt(rng.uniform(0.2, 1))}*exp(-t/{_fmt(period / 2)}) + 0.1",
            f"I1 c 0 FUNC {_fmt(period)} Piecewise(({_fmt(rng.uniform(1e-4, 1e-3))}, t < {_fmt(period / 2)}), (0, True))",
        ]
    else:
        lines += [
            f"V1 a 0 SIN {_fmt(rng.uniform(0.5, 2))} {_fmt(base_f)}",
            f"V2 b 0 FUNC {_fmt(period)} {_fmt(rng.uniform(0.5, 1))}*t/{_fmt(period)}",
            f"I1 c 0 STEP {_fmt(rng.uniform(1e-4, 1e-3))} {_fmt(base_f)} 75%",
        ]
    tau = period / rng.uniform(2, 10)
    r1 = rng.uniform(100, 5e3)
    lines += [
        f"R1 a c {_fmt(r1)}",
        f"C1 c 0 {_fmt(tau / r1)}",
        f"R2 b d {_fmt(rng.uniform(100, 5e3))}",
        f"L1 d c {_fmt(rng.uniform(1e-3, 50e-3))}",
        f"R3 c 0 {_fmt(rng.uniform(1e3, 20e3))}",
        ".end",
    ]
    t_stop = 2.0 * period
    analyses = [op(), tran(t_stop, t_stop / 50), ac(base_f / 10, base_f * 10, 30), hb(6), hb(6, time_window=period)]
    return Case(f"rand_linear_sources_s{seed}", "\n".join(lines), analyses, tags=("random", "tran", "func", "hb"))


def random_diode_net(seed: int) -> Case:
    """Diode/resistor nets: rectifier, clipper, biased diode string."""

    rng = random.Random(f"diode_net:{seed}")
    variant = seed % 3
    freq = rng.choice([1e3, 2e3, 5e3])
    amp = rng.uniform(0.3, 0.8)
    is_ = rng.choice(["1e-14", "1e-12", "5e-15"])
    lines: list[str]
    if variant == 0:
        lines = [
            f"V1 in 0 SIN {_fmt(amp)} {_fmt(freq)}",
            f"VB bias 0 DC {_fmt(rng.uniform(0.2, 0.5))}",
            f"RB bias a {_fmt(rng.uniform(1e3, 10e3))}",
            f"R1 in a {_fmt(rng.uniform(100, 2e3))}",
            f"D1 a 0 {is_}",
            f"C1 a 0 {_fmt(rng.uniform(1e-9, 1e-7))}",
        ]
    elif variant == 1:
        lines = [
            f"V1 in 0 SIN {_fmt(amp)} {_fmt(freq)}",
            f"R1 in out {_fmt(rng.uniform(200, 2e3))}",
            f"D1 out 0 {is_}",
            f"D2 0 out {is_}",
            f"RL out 0 {_fmt(rng.uniform(1e3, 20e3))}",
            f"IB out 0 DC {_fmt(rng.uniform(-1e-4, 1e-4))}",
        ]
    else:
        lines = [
            f"V1 top 0 DC {_fmt(rng.uniform(2, 5))}",
            f"R1 top a {_fmt(rng.uniform(500, 5e3))}",
            f"D1 a b {is_}",
            f"D2 b 0 {is_}",
            f"R2 a 0 {_fmt(rng.uniform(5e3, 50e3))}",
            f"I1 b 0 SIN {_fmt(rng.uniform(1e-5, 1e-4))} {_fmt(freq)}",
        ]
    lines.append(".end")
    t_stop = 2.0 / freq
    analyses = [op_nl(), tran_nl(t_stop, t_stop / 50), ac_nl(freq / 10, freq * 100, 25), hb_nl(6)]
    deviations: tuple[tuple[str, Deviation], ...] = ()
    if variant == 2:
        # The oracle's HB Newton is undamped and starts from x=0; it cannot climb the two-diode
        # DC string (HB_ERROR "Singular HB Jacobian."). The live HB restarts with the damped
        # Newton after that breakdown and converges (P3b, CPP_MIGRATION_PLAN.md 2.4.1 / 2.6).
        deviations = (
            (
                "hb",
                Deviation(
                    ruling="P3b damped HB Newton: restart with backtracking and junction limiting after a breakdown",
                    reference_error="SolverConvergenceError",
                ),
            ),
        )
    return Case(
        f"rand_diode_s{seed}",
        "\n".join(lines),
        analyses,
        tolerance="nonlinear",
        tags=("random", "diode"),
        deviations=deviations,
    )


RANDOM_GENERATORS = {
    "rlc_ladder": (random_rlc_ladder, (0, 1, 2, 3)),
    "rlc_mesh": (random_rlc_mesh, (0, 1, 2)),
    "resistive_controlled": (random_resistive_controlled, (0, 1, 2, 3, 4)),
    "mos_inverter_chain": (random_mos_inverter_chain, (0, 1, 2, 3, 14)),
    "bjt_ce_stage": (random_bjt_ce_stage, (0, 1, 2)),
    "linear_sources": (random_linear_sources, (0, 1, 2, 3)),
    "diode_net": (random_diode_net, (0, 1, 2, 3)),
}


def _random_cases() -> list[Case]:
    cases: list[Case] = []
    for generator, seeds in RANDOM_GENERATORS.values():
        cases.extend(generator(seed) for seed in seeds)
    return cases


# ---------------------------------------------------------------------------
# Extra error / gmin cases (behaviour parity for failure paths)
# ---------------------------------------------------------------------------


def _error_and_gmin_cases() -> list[Case]:
    X = ("extra",)
    return [
        Case(
            "parallel_vsources",
            "V1 a 0 DC 1\nV2 a 0 DC 2\nR1 a 0 1k\n.op\n",
            [op()],
            expects_error="NetlistError",
            tags=X + ("error", "parser"),
        ),
        Case(
            "current_source_series_conflict",
            "V1 a 0 DC 1\nR1 a 0 1k\nI1 a b DC 1m\nI2 b 0 DC 1m\n.op\n",
            [op()],
            expects_error="NetlistError",
            tags=X + ("error", "parser"),
        ),
        Case(
            "unsupported_directive",
            "V1 a 0 DC 1\nR1 a 0 1k\n.noise\n",
            [op()],
            expects_error="NetlistError",
            tags=X + ("error", "parser"),
        ),
        # 2.4.1 ruling "R = 0": the live builder raises NETLIST_FATAL "'R1' has zero
        # resistance" (once, no gmin retries); the oracle keeps ZeroDivisionError.
        Case(
            "zero_resistor",
            "V1 a 0 DC 1\nR1 a b 0\nR2 b 0 1k\n.op\n",
            [op(), show()],
            expects_error="ZeroDivisionError",
            tags=X + ("error",),
            current_error=("NetlistError", "NETLIST_FATAL"),
        ),
        # The oracle fails at every gmin step (gmin is stamped only across C, D and transistor
        # terminals). P3b node-to-ground gmin regularizes the island at the first gmin step; the
        # equivalent netlist writes those 1e-12 S shunts (a, x, y) as resistors.
        Case(
            "floating_resistor_island",
            "V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\n.op\n",
            [op()],
            expects_error="SolverConvergenceError",
            tags=X + ("error", "singular", "gmin"),
            deviations=(
                (
                    "op",
                    Deviation(
                        ruling="P3b node-to-ground gmin during gmin stepping",
                        reference_error="SolverConvergenceError",
                        equivalent_netlist="V1 a 0 DC 1\nR1 a 0 1k\nR2 x y 1k\nRGA a 0 1e12\nRGX x 0 1e12\nRGY y 0 1e12\n.op\n",
                        gmin=1e-12,
                    ),
                ),
            ),
        ),
        Case(
            "unknown_controlling_branch",
            "V1 a 0 DC 1\nR1 a 0 1k\nF1 a 0 VX 2\n.op\n",
            [op()],
            expects_error="NetlistError",
            tags=X + ("error",),
        ),
        Case(
            "nonconvergent_diode_max_iter_1",
            "V1 a 0 DC 5\nR1 a b 1k\nD1 b 0 1e-14\n.op\n",
            [op(max_iter=1)],
            tolerance="nonlinear",
            expects_error="SolverConvergenceError",
            tags=X + ("error", "diode"),
        ),
        # Capacitor-only node: singular at gmin=0, solved once gmin stepping stamps gmin across C.
        # Its DC voltage is defined by gmin alone: the oracle's 1e-12 S across each capacitor make
        # V(c) = V(b) / 2; P3b adds 1e-12 S from every node to ground, so V(c) = V(b) / 3. The
        # equivalent netlist writes all of the live gmin conductances as resistors.
        Case(
            "capacitor_only_node_gmin",
            "V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n.op\n",
            [op(), tran(1e-3, 50e-6)],
            tags=X + ("dc", "gmin"),
            deviations=tuple(
                (
                    mode,
                    Deviation(
                        ruling="P3b node-to-ground gmin: a gmin-defined operating point moves",
                        equivalent_netlist=(
                            "V1 a 0 DC 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\nRC1 b c 1e12\nRC2 c 0 1e12\n"
                            "RNA a 0 1e12\nRNB b 0 1e12\nRNC c 0 1e12\n.op\n"
                        ),
                        gmin=1e-12,
                    ),
                )
                for mode in ("op", "tran")
            ),
        ),
        # A diode across an ideal source: the oracle's damped Newton creeps ~vt per iteration
        # and never converges; the P3b junction strategy (pnjlim + natural monotonicity test)
        # does, at gmin 0, to the exact answer.
        Case(
            "diode_ideal_source",
            "V1 a 0 DC 1\nD1 a 0 1e-14\n.op\n",
            [op_nl()],
            tolerance="nonlinear",
            expects_error="SolverConvergenceError",
            tags=X + ("dc", "diode", "convergence"),
            deviations=(
                (
                    "op",
                    Deviation(
                        ruling="P3b junction voltage-step limiting",
                        reference_error="SolverConvergenceError",
                        expected_dc=(1.0, -1e-14 * math.expm1(1.0 / 0.025)),
                        gmin=0.0,
                    ),
                ),
            ),
        ),
        # Six Level-1 CMOS inverters without keepers: the oracle fails at every gmin step; the
        # P3b gmin continuation converges at gmin 0. The keeper version (which the oracle solves)
        # has the same operating point: the keepers carry no current at the rails. The keeperless
        # solve stops at ||F|| < f_tol = 1e-9 A, which leaves up to ~5e-7 V on the rails (MOS
        # on-conductance ~1e-3 S), hence 1e-6 absolute (volts, and the device vds/vgs) + 1e-6 relative.
        Case(
            "cmos_chain6_no_keepers",
            "VDD vdd 0 DC 1.2\nVIN in 0 DC 0\n"
            + "".join(
                f"MP{k} o{k} {'in' if k == 1 else f'o{k - 1}'} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f\n"
                f"MN{k} o{k} {'in' if k == 1 else f'o{k - 1}'} 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f\n"
                for k in range(1, 7)
            )
            + ".op\n",
            [op_nl()],
            tolerance="nonlinear",
            expects_error="SolverConvergenceError",
            tags=X + ("dc", "mos", "convergence"),
            deviations=(
                (
                    "op",
                    Deviation(
                        ruling="P3b gmin continuation inside .op",
                        reference_error="SolverConvergenceError",
                        equivalent_netlist="VDD vdd 0 DC 1.2\nVIN in 0 DC 0\n"
                        + "".join(
                            f"MP{k} o{k} {'in' if k == 1 else f'o{k - 1}'} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f\n"
                            f"MN{k} o{k} {'in' if k == 1 else f'o{k - 1}'} 0 NMOS LEVEL1 1.8m 0.4 0.02 2f 1f\n"
                            f"RK{k} o{k} {'vdd' if k % 2 else '0'} 10Meg\n"
                            for k in range(1, 7)
                        )
                        + ".op\n",
                        gmin=0.0,
                        tolerance=(1e-6, 1e-6),
                        compare_matrices=False,
                    ),
                ),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_cases() -> list[Case]:
    cases = _existing_cases() + _workload_cases() + _random_cases() + _error_and_gmin_cases()
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate corpus case names: {duplicates}")
    return cases


CASES: list[Case] = _build_cases()


def iter_cases(tags: str | Iterable[str] | None = None, exclude: str | Iterable[str] | None = None) -> Iterator[Case]:
    """Yield corpus cases.

    ``tags``: keep cases carrying **any** of these tags (``None`` keeps all).
    ``exclude``: drop cases carrying any of these tags.
    """

    wanted = {tags} if isinstance(tags, str) else set(tags or ())
    unwanted = {exclude} if isinstance(exclude, str) else set(exclude or ())
    for case in CASES:
        case_tags = set(case.tags)
        if wanted and not (case_tags & wanted):
            continue
        if unwanted and (case_tags & unwanted):
            continue
        yield case


def iter_analyses(
    tags: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
) -> Iterator[tuple[Case, int, str, dict[str, Any]]]:
    """Yield ``(case, index, mode, options)`` with a private copy of the options."""

    for case in iter_cases(tags, exclude):
        for index, (mode, options) in enumerate(case.analyses):
            yield case, index, mode, copy.deepcopy(options)


def get_case(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)
