"""SPICE-dialect front end (``HIERARCHY_AND_MODELS_PLAN.md`` Part B, phase M1).

Covers the lexer (continuations, comments, case, value suffixes), ``.model`` cards and
the ``ComponentRecord`` data contract of section 3.2, ``.subckt``/``X`` flattening,
``.param`` and ``{...}`` expressions, ``.include``/``.lib``, the SPICE source forms and
their waveforms, the diagnostics/errors of the unsupported constructs (section 3.4),
and that the netlist dialect itself is unaffected. Devices built from cards
(``metadata["model"]`` in ``spice``/``gummel_poon``/``spice_level1``) are only parsed
here; end-to-end runs use macromodels of passives and controlled sources.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from corpus import iter_cases
from mna_reference.netlist import parse_netlist_text as reference_parse

from mna_simulation import cli
from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest, ComponentRecord
from mna_simulation.api.service import SimulationService
from mna_simulation.device_models import source_expression
from mna_simulation.errors import NetlistError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.expressions import compile_expression
from mna_simulation.netlist import parse_netlist_file, parse_netlist_text
from mna_simulation.spice import model_cards
from mna_simulation.spice.expressions import ExpressionError, UnknownParameter, evaluate_text
from mna_simulation.spice.frontend import FileIncludeResolver
from mna_simulation.spice.values import normalize_value

D1N4148 = ".model D1N4148 D(IS=2.52n RS=.568 N=1.752 CJO=4p M=.4 TT=20n BV=100 IBV=0.1u)"
Q2N2222 = (
    ".model Q2N2222 NPN(IS=14.34f BF=255.9 NF=1 VAF=74.03 IKF=.2847 ISE=14.34f NE=1.307 BR=6.092 NR=1 RB=10 RC=1 "
    "CJC=7.306p MJC=.3416 VJC=.75 FC=.5 CJE=22.01p MJE=.377 VJE=.75 TR=46.91n TF=411.1p ITF=.6 VTF=1.7 XTF=3)"
)
# A Level-3 power MOSFET card in the vendor style (continuation lines, upper case, E notation).
BS170 = """.MODEL BS170 NMOS (LEVEL=3 RS=0.5 RD=0.5 VTO=1.824 KP=.1233 GAMMA=0 PHI=.6
+ LAMBDA=0 CBD=35p CGSO=1.2n CGDO=0.25n THETA=0 ETA=0 VMAX=0 KAPPA=0.2 TOX=1e-7)"""

Q2N2222_PARAMS = {
    "is": "14.34f",
    "bf": "255.9",
    "nf": "1",
    "vaf": "74.03",
    "ikf": ".2847",
    "ise": "14.34f",
    "ne": "1.307",
    "br": "6.092",
    "nr": "1",
    "rb": "10",
    "rc": "1",
    "cjc": "7.306p",
    "mjc": ".3416",
    "vjc": ".75",
    "fc": ".5",
    "cje": "22.01p",
    "mje": ".377",
    "vje": ".75",
    "tr": "46.91n",
    "tf": "411.1p",
    "itf": ".6",
    "vtf": "1.7",
    "xtf": "3",
}


def _fields(records) -> list[dict]:
    """Field dicts of records (the frozen oracle's record class is a distinct type)."""

    return [dataclasses.asdict(record) for record in records]


def _error(text: str, **kwargs) -> NetlistError:
    with pytest.raises(NetlistError) as exc:
        parse_netlist_text(text, **kwargs)
    return exc.value


def _by_name(circuit) -> dict[str, ComponentRecord]:
    return {record.name: record for record in circuit.components}


def _op(text: str, **kwargs) -> dict[str, float]:
    response = SimulationService().run_request(AnalysisRequest(netlist_text=text, mode=AnalysisMode.OP), **kwargs)
    assert response.status == "ok"
    return dict(zip(response.labels, response.dc_solution))


# ---------------------------------------------------------------------------
# Lexing
# ---------------------------------------------------------------------------


def test_continuation_lines_and_comment_lines_between_them() -> None:
    circuit = parse_netlist_text(
        "* a card split over lines, comments in between\n"
        ".model DX D(IS=1n\n"
        "* comment inside the card\n"
        "+ N=2\n"
        "\n"
        "+ RS=1)\n"
        "V1 a 0\n"
        "+ DC 5\n"
        "R1 a k 1k\n"
        "D1 k 0 DX\n"
    )
    records = _by_name(circuit)
    assert records["V1"].subtype == "DC" and records["V1"].value == "5"
    assert records["D1"].metadata == {"is": "1n", "n": "2", "rs": "1", "model": "spice", "model_name": "DX"}


def test_continuation_without_a_line_to_continue_is_an_error() -> None:
    exc = _error("+ R1 a 0 1k\nV1 a 0 1\n")
    assert (exc.code, exc.message) == ("NETLIST_FATAL", "line 1: continuation line '+' has no line to continue")


def test_inline_semicolon_and_dollar_comments() -> None:
    circuit = parse_netlist_text("V1 a 0 DC 5 ; supply\nR1 a b 1k $ load\nR2 b 0 1k;tail\n.op $ analysis\n")
    assert [(r.name, r.value) for r in circuit.components] == [("V1", "5"), ("R1", "1k"), ("R2", "1k")]
    assert [d.mode for d in circuit.directives] == [AnalysisMode.OP]
    # '$' glued to a token is not a comment.
    exc = _error("V1 a 0 1 ; x\nR1 a 0 1k$ohm\n")
    assert exc.message == "line 2: 'R1' has invalid value format '1k$ohm'."


def test_case_insensitive_keywords_models_keys_and_nodes() -> None:
    circuit = parse_netlist_text(
        ".MODEL d1n4148 d(Is=2.52n N=1.752)\n"
        ".Subckt Stage IN OUT\n"
        "R1 in Out 1k\n"
        ".ENDS stage\n"
        "V1 Vin 0 1\n"
        "x1 vin VOUT STAGE\n"
        "D1 VOUT 0 D1N4148\n"
        "R2 vout 0 1k\n"
    )
    records = _by_name(circuit)
    # 'Vin'/'vin' and 'VOUT'/'vout' are one net each, spelled as at their first occurrence.
    assert (records["x1.R1"].node1, records["x1.R1"].node2) == ("Vin", "VOUT")
    assert records["R2"].node1 == "VOUT" and records["D1"].node1 == "VOUT"
    assert records["D1"].metadata["is"] == "2.52n" and records["D1"].metadata["model_name"] == "d1n4148"
    assert _op("V1 Vin 0 1\nR1 vin OUT 1k\nR2 out 0 1k\n") == pytest.approx(
        {"V(OUT)": 0.5, "V(Vin)": 1.0, "I(V1)": -5e-4}
    )


def test_controlling_source_names_are_case_insensitive() -> None:
    circuit = parse_netlist_text("V1 a 0 DC 1\nR1 a 0 1k\nF1 b 0 v1 2\nR2 b 0 1k\n")
    assert _by_name(circuit)["F1"].ctrl_source == "V1"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("2.2K", "2.2K"),
        ("1e-9", "1e-9"),
        ("10kOhm", "10k"),
        ("5V", "5"),
        ("1uF", "1u"),
        ("100nS", "100n"),
        ("10mA", "10m"),
        ("1kHz", "1k"),
        ("1MEG", "1MEG"),
        ("1meg", "1Meg"),
        ("4.7megohm", "4.7Meg"),
        ("2T", "2000000000000.0"),
        ("10mil", repr(10 * 25.4e-6)),
        ("1e3k", "1000000.0"),
        ("1x", "1x"),  # not a unit word: stays an invalid literal
        ("D1N4148", "D1N4148"),
    ],
)
def test_value_suffixes_and_unit_letters(token: str, expected: str) -> None:
    assert normalize_value(token) == expected


def test_unit_letters_in_netlist_values_and_directives() -> None:
    circuit = parse_netlist_text("V1 a 0 DC 5V\nR1 a b 10kOhm\nC1 b 0 1uF\nL1 b c 100uH\nR2 c 0 1MEGohm\n.tran 10ms 1us\n")
    assert [r.value for r in circuit.components] == ["5", "10k", "1u", "100u", "1Meg"]
    assert circuit.directives[0].params == {"t_stop": pytest.approx(10e-3), "t_step": pytest.approx(1e-6)}


def test_key_value_spacing_and_parenthesised_or_bare_cards_are_equivalent() -> None:
    cards = [
        ".model NM NMOS(LEVEL=1 VTO=0.7 KP=50u)",
        ".model NM NMOS (LEVEL = 1, VTO = 0.7, KP = 50u)",
        ".model NM NMOS LEVEL=1 VTO=0.7 KP=50u",
    ]
    lines = ["M1 d g 0 0 NM L=2u W=10u", "M1 d g 0 0 NM L = 2u W= 10u", "M1 d g 0 0 NM L =2u W =10u"]
    seen = []
    for card, line in zip(cards, lines):
        record = _by_name(parse_netlist_text(f"{card}\nV1 d 0 1\nV2 g 0 1\n{line}\n"))["M1"]
        seen.append(record.metadata)
    assert seen[0] == seen[1] == seen[2]
    assert seen[0] == {
        "level": "1",
        "vto": "0.7",
        "kp": "50u",
        "model": "spice_level1",
        "model_name": "NM",
        "l": "2u",
        "w": "10u",
    }


def test_no_title_line() -> None:
    # The first line is parsed like every other line (comment a deck's title out with '*').
    exc = _error("My amplifier deck\nV1 a 0 1\nR1 a 0 1k\n")
    assert exc.code == "COMPONENT_ERROR" and "'My'" in exc.message
    assert len(parse_netlist_text("* My amplifier deck\nV1 a 0 1\nR1 a 0 1k\n").components) == 2


# ---------------------------------------------------------------------------
# .model cards and the record contract
# ---------------------------------------------------------------------------


def test_diode_card_record_follows_the_contract() -> None:
    circuit = parse_netlist_text(f"{D1N4148}\nV1 a 0 5\nR1 a k 1k\nD1 k 0 D1N4148\nD2 k 0 D1N4148 2 M=3\n")
    records = _by_name(circuit)
    card = {"is": "2.52n", "rs": ".568", "n": "1.752", "cjo": "4p", "m": ".4", "tt": "20n", "bv": "100", "ibv": "0.1u"}
    assert records["D1"] == ComponentRecord(
        name="D1",
        type="D",
        node1="k",
        node2="0",
        value="2.52n",  # compatibility: value holds is
        metadata={**card, "model": "spice", "model_name": "D1N4148"},  # no defaults filled in
    )
    # Instance area 2; a diode has no instance multiplier (its 'm' is the card's grading coefficient).
    assert records["D2"].metadata == {**card, "model": "spice", "model_name": "D1N4148", "area": "2"}
    assert circuit.diagnostics == [
        "model D1N4148: parameters m, tt are recorded but not modelled yet",
        "line 5: 'D2': instance multiplier M= dropped (a diode scales with its area factor; the card's M is the "
        "grading coefficient)",
    ]


def test_infinite_card_parameters_pass_through() -> None:
    record = _by_name(parse_netlist_text(".model QX NPN(IS=1f VAF=inf IKF=INF)\nV1 c 0 1\nQ1 c c 0 QX\n"))["Q1"]
    assert (record.metadata["vaf"], record.metadata["ikf"]) == ("inf", "inf")


def test_diode_card_without_is_uses_the_spice_default_in_value() -> None:
    record = _by_name(parse_netlist_text(".model DX D(N=2)\nV1 a 0 1\nD1 a 0 DX AREA=3\n"))["D1"]
    assert record.value == "1e-14"
    assert record.metadata == {"n": "2", "model": "spice", "model_name": "DX", "area": "3"}


def test_bjt_card_record_follows_the_contract() -> None:
    circuit = parse_netlist_text(
        f"{Q2N2222}\n.model QP PNP(IS=1f BF=80)\nVCC vcc 0 5\nRB vcc b 100k\nRC vcc c 1k\n"
        "Q1 c b 0 Q2N2222\nQ2 c b e sub Q2N2222 2 M=4\nQ3 c b e QP\nRE e 0 1\nRS sub 0 1\n"
    )
    records = _by_name(circuit)
    assert records["Q1"] == ComponentRecord(
        name="Q1",
        type="QNPN",
        node1="c",
        node2="0",
        value="14.34f",
        subtype="QNPN",
        ctrl_node1="b",
        value2="255.9",
        value3="6.092",
        metadata={**Q2N2222_PARAMS, "model": "gummel_poon", "model_name": "Q2N2222"},
    )
    q2 = records["Q2"]
    assert (q2.node1, q2.ctrl_node1, q2.node2, q2.ctrl_node2) == ("c", "b", "e", "sub")
    assert (q2.metadata["area"], q2.metadata["m"]) == ("2", "4")  # a BJT's M= is the instance multiplier
    q3 = records["Q3"]
    assert (q3.type, q3.subtype, q3.value, q3.value2, q3.value3) == ("QPNP", "QPNP", "1f", "80", "1")
    assert circuit.diagnostics == [
        "model Q2N2222: parameters fc, itf, mjc, mje, tf, tr, vjc, vje, vtf, xtf are recorded but not modelled yet",
        "line 7: 'Q2': substrate node 'sub' is not modelled",
    ]


def test_mos_level3_card_three_and_four_terminal_instances() -> None:
    circuit = parse_netlist_text(
        f"{BS170}\nVDD vdd 0 10\nR1 vdd d 1k\nVG g 0 5\nM1 d g 0 BS170 W=10u L=2u M=2 AD=1p OFF\nM2 d g s b BS170\n"
        "R2 s 0 1\nR3 b 0 1\n"
    )
    records = _by_name(circuit)
    card = {
        "level": "3",
        "rs": "0.5",
        "rd": "0.5",
        "vto": "1.824",
        "kp": ".1233",
        "gamma": "0",
        "phi": ".6",
        "lambda": "0",
        "cbd": "35p",
        "cgso": "1.2n",
        "cgdo": "0.25n",
        "theta": "0",
        "eta": "0",
        "vmax": "0",
        "kappa": "0.2",
        "tox": "1e-7",
    }
    assert records["M1"] == ComponentRecord(
        name="M1",
        type="NMOS",
        node1="d",
        node2="0",
        value=None,
        subtype="NMOS",
        ctrl_node1="g",
        ctrl_node2=None,  # three terminals: bulk = source
        metadata={**card, "model": "spice_level1", "model_name": "BS170", "w": "10u", "l": "2u", "m": "2", "ad": "1p"},
    )
    assert records["M2"].ctrl_node2 == "b"
    assert records["M2"].metadata == {**card, "model": "spice_level1", "model_name": "BS170"}
    assert circuit.diagnostics == [
        "model BS170: LEVEL=3 card mapped to the Level-1 MOS equations",
        "model BS170: parameters cbd, eta, kappa, rd, rs, theta, tox, vmax are recorded but not modelled yet",
        "line 6: 'M1': instance parameters off are ignored",
    ]


def test_mos_levels_beyond_3_are_errors_naming_the_level() -> None:
    exc = _error(".model N1 NMOS(LEVEL=49 VTH0=0.4)\nV1 d 0 1\nM1 d d 0 0 N1\n")
    assert exc.code == "NETLIST_FATAL"
    assert exc.message.startswith("line 3: model 'N1': MOS LEVEL=49 is not supported")
    level2 = parse_netlist_text(".model N2 NMOS(LEVEL=2 VTO=1)\nV1 d 0 1\nM1 d d 0 0 N2\n")
    assert level2.diagnostics == ["model N2: LEVEL=2 card mapped to the Level-1 MOS equations"]


def test_unknown_card_parameters_are_copied_and_reported_once_per_card() -> None:
    circuit = parse_netlist_text(
        ".model 1N4148 D(Is=2.52n Rs=.568 N=1.752 Iave=200m Vpk=75 mfg=OnSemi type=silicon)\n"
        "V1 a 0 1\nD1 a b 1N4148\nD2 b 0 1N4148\n"
    )
    assert _by_name(circuit)["D1"].metadata["mfg"] == "OnSemi"
    assert circuit.diagnostics == ["model 1N4148: parameters iave, mfg, type, vpk are not used by the diode equations"]


def test_passive_cards_flatten_to_plain_values() -> None:
    circuit = parse_netlist_text(
        ".model RMOD R(R=2k TC1=1e-3)\n"
        ".model RMUL RES(R=1.5 TC2=1e-6)\n"
        ".model CMOD C(C=10p)\n"
        "V1 a 0 1\n"
        "R1 a b RMOD\n"
        "R2 b c 1k RMOD\n"
        "R3 c d 1k RMUL\n"
        "R4 d e 1k M=2\n"
        "C1 e 0 CMOD M=3\n"
        "R5 e 0 1k TC1=0.01\n"
    )
    records = _by_name(circuit)
    assert [(records[n].value, records[n].metadata) for n in ("R1", "R2", "R3", "R4", "C1")] == [
        ("2k", {}),  # an R card gives the value the instance omits
        ("1k", {}),  # the instance value wins
        ("1500.0", {}),  # a RES card multiplies it
        ("500.0", {}),  # M=2 parallel resistors
        ("3e-11", {}),  # M=3 parallel capacitors
    ]
    assert circuit.diagnostics == [
        "model RMOD: temperature coefficients tc1 are ignored",
        "model RMUL: temperature coefficients tc2 are ignored",
        "line 10: 'R5': temperature coefficients tc1 are ignored",
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("V1 a 0 1\nD1 a 0 D1N4148\n", "line 2: model 'D1N4148' is not defined (referenced by 'D1')"),
        ("V1 a 0 1\nQ1 a a2 0 QX\nR1 a2 0 1k\n", "line 2: model 'QX' is not defined (referenced by 'Q1')"),
        ("V1 a 0 1\nM1 a a 0 0 NMODX\n", "line 2: model 'NMODX' is not defined (referenced by 'M1')"),
        (".model QX NPN(IS=1f)\nV1 a 0 1\nD1 a 0 QX\n", "line 3: 'D1' needs a diode model, but model 'QX' has type NPN"),
        (
            ".model VD VDMOS(VTO=2)\nV1 a 0 1\nM1 a a 0 VD\n",
            "line 3: model 'VD' has type VDMOS, which is not supported (referenced by 'M1')",
        ),
        (".model DX D(IS=abc)\nV1 a 0 1\nD1 a 0 DX\n", "line 1: model 'DX': parameter 'is' has invalid value 'abc'"),
        (".model DX D(IS=1n)\n.model dx D(IS=2n)\nV1 a 0 1\nD1 a 0 DX\n", "line 2: model dx is defined twice (first at line 1)"),
    ],
    ids=["undefined_diode", "undefined_bjt", "undefined_mos", "wrong_type", "unsupported_type", "bad_value", "duplicate"],
)
def test_model_card_errors(text: str, message: str) -> None:
    exc = _error(text)
    assert (exc.code, exc.message) == ("NETLIST_FATAL", message)


def test_unreferenced_cards_of_unsupported_types_are_a_diagnostic() -> None:
    circuit = parse_netlist_text(".model SW1 SW(RON=1)\n.model J1 NJF(VTO=-2)\n.model DX D\nV1 a 0 1\nR1 a 0 1k\n")
    assert circuit.diagnostics == ["models of unsupported types were not used: J1 (NJF), SW1 (SW)"]


def test_dialect_device_lines_keep_their_meaning() -> None:
    text = (
        "VCC vcc 0 DC 5\nR1 vcc a 1k\nD1 a 0 1e-15\nD2 a 0\n"
        "Q1 vcc a e QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5\nQ2 vcc a e 10m 2.5k\nRE e 0 1k\n"
        "M1 vcc a s NMOS LEVEL1 1m 0.4 0.02 2f 1f\nM2 vcc a s 1m 50k 2p 1p\nRS s 0 1k\n"
    )
    assert _fields(parse_netlist_text(text).components) == _fields(reference_parse(text).components)


def test_supported_key_lists_match_the_plan_tables() -> None:
    assert model_cards.SUPPORTED_KEYS["D"] == {"is", "n", "rs", "bv", "ibv", "cjo", "vj", "m", "fc", "tt", "eg", "xti", "tnom"}
    assert model_cards.MODELLED_KEYS["D"] == {"is", "n", "rs", "bv", "ibv", "cjo"}
    assert model_cards.SUPPORTED_KEYS["NPN"] == model_cards.SUPPORTED_KEYS["PNP"]
    assert {"is", "bf", "nf", "br", "nr", "vaf", "var", "ikf", "ikr", "ise", "ne", "isc", "nc", "rb", "re", "rc"} <= (
        model_cards.MODELLED_KEYS["NPN"]
    )
    assert {"rbm", "irb", "tf", "tr", "xtf", "vtf", "itf", "ptf", "xtb", "eg", "xti", "tnom"} <= model_cards.SUPPORTED_KEYS["NPN"]
    assert {"vto", "kp", "lambda", "gamma", "phi", "cgso", "cgdo", "cgbo", "level"} <= model_cards.MODELLED_KEYS["NMOS"]
    assert {"tox", "nsub", "uo", "vmax", "theta", "eta", "kappa", "rd", "rs", "cbd", "cbs"} <= model_cards.SUPPORTED_KEYS["NMOS"]
    assert model_cards.MOS_INSTANCE_KEYS == ("w", "l", "m", "ad", "as", "pd", "ps", "nrd", "nrs")


# ---------------------------------------------------------------------------
# .subckt / X flattening
# ---------------------------------------------------------------------------


NESTED = """
.global vcc
.subckt INNER a b
R1 a mid 1k
R2 mid b 2k
RV mid vcc 10k
.ends INNER
.subckt OUTER in out
X1 in n INNER
X2 n out INNER
R3 n 0 5k
.ends
VCC vcc 0 DC 10
V1 src 0 DC 1
X1 src load OUTER
RL load 0 1k
.op
"""


def test_nested_flattening_names_nets_ports_ground_and_globals() -> None:
    circuit = parse_netlist_text(NESTED)
    assert [(r.name, r.node1, r.node2) for r in circuit.components] == [
        ("VCC", "vcc", "0"),
        ("V1", "src", "0"),
        ("X1.X1.R1", "src", "X1.X1.mid"),
        ("X1.X1.R2", "X1.X1.mid", "X1.n"),
        ("X1.X1.RV", "X1.X1.mid", "vcc"),
        ("X1.X2.R1", "X1.n", "X1.X2.mid"),
        ("X1.X2.R2", "X1.X2.mid", "load"),
        ("X1.X2.RV", "X1.X2.mid", "vcc"),
        ("X1.R3", "X1.n", "0"),
        ("RL", "load", "0"),
    ]
    assert circuit.diagnostics == []


def test_flattened_op_matches_the_hand_flattened_netlist() -> None:
    hand = """
VCC vcc 0 DC 10
V1 src 0 DC 1
R11 src X1.X1.mid 1k
R12 X1.X1.mid X1.n 2k
R1V X1.X1.mid vcc 10k
R21 X1.n X1.X2.mid 1k
R22 X1.X2.mid load 2k
R2V X1.X2.mid vcc 10k
R3 X1.n 0 5k
RL load 0 1k
.op
"""
    flattened = _op(NESTED)
    expected = _op(hand)
    nodes = [label for label in expected if label.startswith("V(")]
    assert [flattened[label] for label in nodes] == pytest.approx([expected[label] for label in nodes], rel=1e-12)
    assert "V(X1.X1.mid)" in flattened


def test_local_definitions_and_controlling_sources_inside_subcircuits() -> None:
    circuit = parse_netlist_text(
        ".subckt SENSE in out\n"
        ".subckt SHUNT a b\nR1 a b 1\n.ends\n"
        "Vs in m 0\n"
        "X1 m out SHUNT\n"
        "H1 x 0 Vs 100\n"
        "Rx x 0 1k\n"
        ".ends\n"
        "V1 a 0 1\nX1 a b SENSE\nRL b 0 9\n"
    )
    records = _by_name(circuit)
    assert records["X1.H1"].ctrl_source == "X1.Vs"
    assert records["X1.X1.R1"].node1 == "X1.m"
    exc = _error(".subckt A a b\n.subckt LOCAL a b\nR1 a b 1\n.ends\nR1 a b 1\n.ends\nV1 a 0 1\nX1 a 0 LOCAL\n")
    assert exc.message == "line 8: subcircuit 'LOCAL' is not defined (instance 'X1')"


def test_model_cards_inside_subcircuits_follow_the_instance_parameters() -> None:
    circuit = parse_netlist_text(
        ".subckt CLAMP a PARAMS: isat=1f\n.model DLOC D(IS={isat*2} N=1)\nD1 a 0 DLOC\n.ends\n"
        "V1 n 0 1\nX1 n CLAMP\nX2 n CLAMP isat=5f\n"
    )
    records = _by_name(circuit)
    # Suffixed literals scale like parse_value: 5f is 5 * 1e-15.
    assert records["X1.D1"].metadata["is"] == repr(1 * 1e-15 * 2)
    assert records["X2.D1"].metadata["is"] == repr(5 * 1e-15 * 2)
    assert records["X1.D1"].value == records["X1.D1"].metadata["is"]
    assert circuit.diagnostics == []


def test_component_names_stay_unique_across_the_flattened_design() -> None:
    text = ".subckt CELL a b\nV1 a m 0\nR1 m b 1k\n.ends\nV0 in 0 1\nX1 in 0 CELL\nX2 in 0 CELL\n.op\n"
    problem = build_mna_problem(parse_netlist_text(text))
    assert problem.index_map.branch_names == ["V0", "X1.V1", "X2.V1"]
    assert problem.metadata["labels"][-3:] == ["I(V0)", "I(X1.V1)", "I(X2.V1)"]


def test_tied_ports_drop_a_shorted_resistor_with_a_diagnostic() -> None:
    circuit = parse_netlist_text(".subckt AMP inp inn out\nRIN inp inn 2meg\nR2 inp out 1k\n.ends\nV1 a 0 1\nX1 a a b AMP\nRL b 0 1k\n")
    assert [r.name for r in circuit.components] == ["V1", "X1.R2", "RL"]
    assert circuit.diagnostics == [
        "line 2 (subcircuit AMP, instance X1): 'X1.RIN' dropped: both terminals are on net 'a'"
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("V1 a 0 1\nX1 a 0 NOPE\nR1 a 0 1k\n", "line 2: subcircuit 'NOPE' is not defined (instance 'X1')"),
        (
            ".subckt S a b\nR1 a b 1k\n.ends\nV1 a 0 1\nX1 a S\nR1 a 0 1k\n",
            "line 5: instance 'X1' connects 1 node(s) but subcircuit S has 2 port(s) (a b)",
        ),
        (
            ".subckt S a b\nX1 a b S\n.ends\nV1 a 0 1\nX1 a 0 S\n",
            "line 2 (subcircuit S, instance X1): subcircuit S instantiates itself (S -> S)",
        ),
        (
            ".subckt A a b\nX1 a b B\n.ends\n.subckt B a b\nX1 a b A\n.ends\nV1 a 0 1\nX1 a 0 A\n",
            "line 5 (subcircuit B, instance X1.X1): subcircuit A instantiates itself (A -> B -> A)",
        ),
        (
            ".subckt S a b\nR1 a b 1k\n.ends\n.subckt s a b\nR1 a b 2k\n.ends\nV1 a 0 1\nX1 a 0 S\n",
            "line 4: subcircuit s is defined twice (first at line 1)",
        ),
        (".subckt S a b\nR1 a b 1k\nV1 a 0 1\n", "line 1: .subckt S has no .ends"),
        (".subckt S a b\nR1 a b 1k\n.ends T\nV1 a 0 1\n", "line 3: '.ends T' closes .subckt S"),
        ("V1 a 0 1\nR1 a 0 1k\n.ends\n", "line 3: .ends without .subckt"),
        (
            ".subckt S a b\nR1 a b 1k\n.ends\nV1 a 0 1\nX1 a 0 S\nx1 a 0 S\n",
            "line 6: duplicate subcircuit instance name 'x1'",
        ),
        (
            ".subckt S a b PARAMS: r=1k\nR1 a b {r}\n.ends\nV1 a 0 1\nX1 a 0 S q=2\n",
            "line 5: subcircuit S has no parameter 'q' (instance 'X1')",
        ),
        (".subckt S a b\nR1 a b 1k\n.tran 1m\n.ends\nV1 a 0 1\nX1 a 0 S\n", "line 3: '.tran' is not allowed inside .subckt S"),
        (".subckt S a a\nR1 a 0 1k\n.ends\nV1 a 0 1\n", "line 1: subcircuit S lists port 'a' twice"),
    ],
    ids=[
        "undefined",
        "port_count",
        "recursion",
        "indirect_recursion",
        "duplicate_definition",
        "unclosed",
        "ends_mismatch",
        "ends_alone",
        "duplicate_instance",
        "unknown_parameter",
        "analysis_inside",
        "duplicate_port",
    ],
)
def test_subcircuit_errors(text: str, message: str) -> None:
    exc = _error(text)
    assert (exc.code, exc.message) == ("NETLIST_FATAL", message)


def test_element_errors_inside_subcircuits_name_line_and_instance() -> None:
    exc = _error(".subckt S a b\nR1 a b\n.ends\nV1 a 0 1\nX7 a 0 S\n")
    assert (exc.code, exc.message) == (
        "NETLIST_FATAL",
        "line 2 (subcircuit S, instance X7): 'R1' expected 4 parts, got 3",
    )


# ---------------------------------------------------------------------------
# .param and {...}
# ---------------------------------------------------------------------------


def test_param_scopes_overrides_and_expressions() -> None:
    circuit = parse_netlist_text(
        ".param rload={2*rbase} rbase=1k gain=10\n"  # forward reference
        ".subckt AMP in out PARAMS: g=2 rin={rbase*100}\n"
        ".param rfb={g*rin}\n"
        "RIN in 0 {rin}\n"
        "E1 out 0 in 0 {g*gain}\n"
        "RF in out {rfb}\n"
        ".ends\n"
        "V1 a 0 1\n"
        "X1 a b AMP\n"
        "X2 b c AMP g=3 rin={rload}\n"
        "X3 c d AMP PARAMS: g={gain/5}\n"
        "RL d 0 {rload}\n"
    )
    values = {r.name: r.value for r in circuit.components}
    assert values["X1.RIN"] == "100000.0" and values["X1.E1"] == "20.0" and values["X1.RF"] == "200000.0"
    assert values["X2.RIN"] == "2000.0" and values["X2.E1"] == "30.0" and values["X2.RF"] == "6000.0"
    assert values["X3.E1"] == "20.0" and values["RL"] == "2000.0"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+2*3", 7.0),
        ("(1+2)*3", 9.0),
        ("-2**2", -4.0),
        ("2^3^2", 512.0),
        ("10k/2", 5000.0),
        ("1meg*2", 2e6),
        ("sqrt(16)+abs(-2)", 6.0),
        ("exp(0)+ln(e)+log(1)+log10(1000)", 5.0),
        ("min(3, 1, 2)+max(4, 5)", 6.0),
        ("pow(2, 10)", 1024.0),
        ("int(-2.7)+floor(2.7)+ceil(2.2)", 3.0),
        ("if(3 > 2, 10, 20)", 10.0),
        ("if(1 == 2, 1/0, 5)", 5.0),  # the branch not taken is not evaluated
        ("2 <= 1 ? 7 : 8", 8.0),
        ("pi", math.pi),
        ("10uF*1k", 10e-3),
    ],
)
def test_expression_evaluator(expression: str, expected: float) -> None:
    def no_parameters(name: str) -> float:
        raise UnknownParameter(name)

    assert evaluate_text(expression, no_parameters) == pytest.approx(expected)


def test_parameters_shadow_the_constants() -> None:
    assert evaluate_text("pi*2", {"pi": 3.0}.__getitem__) == 6.0
    assert _by_name(parse_netlist_text(".param e=2\nV1 a 0 {e*2}\nR1 a 0 {pi}\n"))["V1"].value == "4.0"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (".param a={b*2}\nV1 a 0 {a}\nR1 a 0 1k\n", "line 1: parameter 'a' = {b*2}: parameter 'b' is not defined"),
        ("V1 a 0 {foo+1}\nR1 a 0 1k\n", "line 1: {foo+1}: parameter 'foo' is not defined"),
        ("V1 a 0 {2*(3+}\nR1 a 0 1k\n", "line 1: {2*(3+}: unexpected end of expression"),
        ("V1 a 0 {1/0}\nR1 a 0 1k\n", "line 1: {1/0}: division by zero"),
        ("V1 a 0 {sqrt(-1)}\nR1 a 0 1k\n", "line 1: {sqrt(-1)}: math domain error in sqrt()"),
        ("V1 a 0 {foo(1)}\nR1 a 0 1k\n", "line 1: {foo(1)}: unknown function 'foo'"),
        (".param a={a+1}\nV1 x 0 {a}\nR1 x 0 1k\n", "line 1: parameter 'a' depends on itself"),
        ("V1 a 0 {__import__('os')}\nR1 a 0 1k\n", "line 1: {__import__('os')}: unexpected character '''"),
        (".param a=1 a=2\nV1 x 0 {a}\nR1 x 0 1k\n", "line 1: parameter 'a' is defined twice"),
        (".func f(x) {x*2}\nV1 a 0 1\nR1 a 0 1k\n", "line 1: .func: user-defined functions are not supported (use .param or write the expression inline)"),
    ],
    ids=["unresolved_param", "unresolved_brace", "syntax", "division", "domain", "function", "self", "no_eval", "duplicate", "func"],
)
def test_parameter_errors(text: str, message: str) -> None:
    exc = _error(text)
    assert (exc.code, exc.message) == ("NETLIST_FATAL", message)


def test_expression_errors_are_expression_errors() -> None:
    def lookup(name: str) -> float:
        raise UnknownParameter(name)

    with pytest.raises(ExpressionError, match="parameter 'x' is not defined"):
        evaluate_text("x + 1", lookup)
    with pytest.raises(ExpressionError, match="invalid number '2x'"):
        evaluate_text("2x", lookup)


# ---------------------------------------------------------------------------
# .include / .lib
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_include_and_lib_sections_resolve_relative_to_the_including_file(tmp_path: Path) -> None:
    _write(tmp_path / "models" / "parts.lib", f"* vendor library\n{D1N4148}\n.include \"sub/amp.inc\"\n")
    _write(tmp_path / "models" / "sub" / "amp.inc", ".subckt BUF in out\nE1 out 0 in 0 1\n.ends\n")
    _write(
        tmp_path / "models" / "corners.lib",
        ".lib TT\n.param rcorner=1k\n.lib 'corners.lib' COMMON\n.endl TT\n"
        ".lib FF\n.param rcorner=800\n.lib 'corners.lib' COMMON\n.endl\n"
        ".lib COMMON\n.model RC1 R(R={rcorner})\n.endl COMMON\n",
    )
    deck = _write(
        tmp_path / "deck" / "top.cir",
        '.include "../models/parts.lib"\n.lib "../models/corners.lib" FF\nV1 a 0 5\nX1 a b BUF\nR1 b k RC1\nD1 k 0 D1N4148\n.op\n',
    )
    circuit = parse_netlist_file(deck)
    records = _by_name(circuit)
    assert records["R1"].value == "800.0"
    assert records["X1.E1"].node1 == "b"
    assert records["D1"].metadata["model_name"] == "D1N4148"


def test_lib_search_directories_and_missing_files(tmp_path: Path) -> None:
    _write(tmp_path / "libs" / "divider.lib", ".subckt DIV in out\nR1 in out 1k\nR2 out 0 1k\n.ends\n")
    deck = _write(tmp_path / "deck.cir", '.include "divider.lib"\nV1 a 0 2\nX1 a b DIV\n.op\n')
    with pytest.raises(NetlistError) as exc:
        parse_netlist_file(deck)
    assert exc.value.message == f"line 1: file 'divider.lib' not found (searched {tmp_path})"
    circuit = parse_netlist_file(deck, search_dirs=[tmp_path / "libs"])
    assert [r.name for r in circuit.components] == ["V1", "X1.R1", "X1.R2"]
    resolver = FileIncludeResolver(search_dirs=[tmp_path / "libs"])
    assert len(parse_netlist_text("V1 a 0 2\nX1 a b DIV\n.include divider.lib\n", include_resolver=resolver).components) == 3


def test_recursive_include_and_missing_section_are_errors(tmp_path: Path) -> None:
    _write(tmp_path / "a.lib", '.include "b.lib"\n')
    _write(tmp_path / "b.lib", '.include "a.lib"\n')
    deck = _write(tmp_path / "deck.cir", '.include "a.lib"\nV1 a 0 1\nR1 a 0 1k\n')
    with pytest.raises(NetlistError) as exc:
        parse_netlist_file(deck)
    assert exc.value.message == 'line 1 of b.lib: recursive .include "a.lib" (the file is already being read)'
    _write(tmp_path / "c.lib", ".lib TT\n.endl\n")
    deck = _write(tmp_path / "deck2.cir", '.lib "c.lib" SS\nV1 a 0 1\nR1 a 0 1k\n')
    with pytest.raises(NetlistError) as exc:
        parse_netlist_file(deck)
    assert exc.value.message == "line 1: library section 'SS' not found in 'c.lib'"


def test_include_from_netlist_text_asks_for_the_library_to_be_pasted() -> None:
    exc = _error('.include "models.lib"\nV1 a 0 1\nR1 a 0 1k\n')
    assert exc.code == "NETLIST_FATAL"
    assert exc.message == (
        'line 1: .include "models.lib" cannot be read from netlist text: paste the contents of models.lib into the '
        "netlist (a netlist file run from the CLI reads it; --lib adds search directories)"
    )
    exc = _error('.lib "models.lib" TT\nV1 a 0 1\nR1 a 0 1k\n')
    assert exc.message.startswith('line 1: .lib "models.lib" TT cannot be read from netlist text')


def test_pasted_library_sections_are_skipped_with_a_diagnostic() -> None:
    circuit = parse_netlist_text(".lib TT\n.model DX D(IS=1n)\n.endl\nV1 a 0 1\nR1 a 0 1k\n")
    assert circuit.diagnostics == ["line 1: library section 'TT' skipped (a section is read with .lib \"<file>\" TT)"]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


SOURCES = """
V1 a 0 5
V2 b 0 DC 5 AC 1 90
V3 c 0 AC 2 45
V4 d 0 SIN(0 1 1k)
V5 e 0 SIN 1 1k
V6 f 0 PULSE(0 5 1u 1u 1u 5u 10u)
V7 g 0 PWL(0 0 1u 5 2u 5 3u 0) R=1u TD=1u
V8 h 0 DC 1 EXP(0 1 1u 1u 5u 1u)
I1 i 0 PULSE 0 1m 0
R1 a 0 1k
R2 b 0 1k
R3 c 0 1k
R4 d 0 1k
R5 e 0 1k
R6 f 0 1k
R7 g 0 1k
R8 h 0 1k
R9 i 0 1k
"""


def test_source_records_follow_the_contract() -> None:
    records = _by_name(parse_netlist_text(SOURCES))
    fields = {
        name: (r.subtype, r.value, r.value2, r.metadata) for name, r in records.items() if r.type in {"V", "I"}
    }
    assert fields == {
        "V1": ("DC", "5", None, {}),
        "V2": ("DC", "5", None, {"ac_mag": "1", "ac_phase": "90"}),
        "V3": ("AC", "2", None, {"ac_mag": "2", "ac_phase": "45"}),
        "V4": ("SIN", None, "1k", {"sin": ["0", "1", "1k", "0", "0", "0"]}),  # value2: frequency (HB)
        "V5": ("SIN", "1", "1k", {}),  # the dialect's two-token SIN amp freq
        "V6": ("PULSE", None, None, {"pulse": ["0", "5", "1u", "1u", "1u", "5u", "10u"]}),
        "V7": ("PWL", None, None, {"points": ["0", "0", "1u", "5", "2u", "5", "3u", "0"], "repeat": "1u", "td": "1u"}),
        "V8": ("EXP", "1", None, {"exp": ["0", "1", "1u", "1u", "5u", "1u"]}),
        "I1": ("PULSE", None, None, {"pulse": ["0", "1m", "0", "0", "0", "0", "0"]}),
    }


def test_transient_defaults_come_from_the_tran_directive() -> None:
    circuit = parse_netlist_text(SOURCES + ".tran 20u 0.5u\n")
    records = _by_name(circuit)
    t_stop = repr(circuit.directives[0].params["t_stop"])
    t_step = repr(circuit.directives[0].params["t_step"])
    # Omitted (or zero) tr/tf take the step, pw/per the stop time (SPICE rules).
    assert records["I1"].metadata["pulse"] == ["0", "1m", "0", t_step, t_step, t_stop, t_stop]
    assert records["V4"].value2 == "1k" and records["V4"].metadata["sin"][2] == "1k"
    records = _by_name(parse_netlist_text("V1 a 0 SIN(0 1)\nR1 a 0 1k\n.tran 1m 1u\n"))
    assert records["V1"].metadata["sin"][2] == "1000.0" and records["V1"].value2 == "1000.0"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("V1 a 0 PWL FILE=wave.txt", "line 1: 'V1': PWL FILE= sources are not supported (write the points inline)"),
        ("V1 a 0 PWL(0 0 1u)", "line 1: 'V1': PWL needs time-value pairs"),
        ("V1 a 0 PWL(0 0 2u 1 1u 0)", "line 1: 'V1': PWL times must not decrease"),
        ("V1 a 0 SFFM(0 1 1k 5 100)", "line 1: 'V1': SFFM sources are not supported"),
        ("V1 a 0 PULSE(0)", "line 1: 'V1': PULSE takes 2 to 7 values, got 1"),
        ("V1 a 0 DC 1 DC 2", "line 1: 'V1': the DC value is given twice"),
    ],
)
def test_source_errors(line: str, message: str) -> None:
    exc = _error(f"{line}\nR1 a 0 1k\n")
    assert (exc.code, exc.message) == ("NETLIST_FATAL", message)


def test_sine_is_an_alias_of_spice_sin() -> None:
    record = parse_netlist_text("V1 a 0 SINE(0 1 1k)\nR1 a 0 1k\n").components[0]
    assert (record.subtype, record.metadata) == ("SIN", {"sin": ["0", "1", "1k", "0", "0", "0"]})


def test_invalid_source_subtype_keeps_the_dialect_message() -> None:
    exc = _error("V1 a 0 XX 1\nR1 a 0 1k\n")
    assert (exc.code, exc.message) == ("COMPONENT_ERROR", "'V1' has invalid subtype 'XX'.")


def _waveform(line: str):
    record = parse_netlist_text(f"{line}\nR1 a 0 1k\n").components[0]
    program = compile_expression(source_expression(record), time_dependent=True, n_unknowns=0)
    return lambda t: program.evaluate(t=t)


def test_pulse_waveform_has_linear_edges_and_repeats() -> None:
    wave = _waveform("V1 a 0 PULSE(1 5 2u 1u 2u 3u 10u)")
    expected = {0.0: 1, 1e-6: 1, 2e-6: 1, 2.5e-6: 3, 3e-6: 5, 5e-6: 5, 6e-6: 5, 7e-6: 3, 8e-6: 1, 9.9e-6: 1, 12.5e-6: 3, 15.5e-6: 5}
    for t, value in expected.items():
        assert wave(t) == pytest.approx(value, abs=1e-9), t
    ideal = _waveform("V1 a 0 PULSE(0 1 1u)")  # no .tran: ideal edge, never ends, single pulse
    assert [ideal(t) for t in (0.0, 0.999e-6, 1e-6, 1.0)] == [0.0, 0.0, 1.0, 1.0]


def test_pwl_waveform_interpolates_linearly_with_delay_and_repeat() -> None:
    wave = _waveform("V1 a 0 PWL(0 0 1u 4 3u 0)")
    for t, value in {0.0: 0, 0.5e-6: 2, 1e-6: 4, 2e-6: 2, 3e-6: 0, 9e-6: 0}.items():
        assert wave(t) == pytest.approx(value, abs=1e-9), t
    repeated = _waveform("V1 a 0 PWL(0 0 1u 4 3u 0) R=0 TD=1u")
    for t, value in {0.5e-6: 0, 1.5e-6: 2, 2e-6: 4, 3e-6: 2, 4.5e-6: 2, 5e-6: 4}.items():
        assert repeated(t) == pytest.approx(value, abs=1e-9), t
    step = _waveform("V1 a 0 PWL(0 0 1u 0 1u 5 2u 5)")  # equal times: a vertical step
    assert [step(t) for t in (0.5e-6, 0.999e-6, 1e-6, 1.5e-6)] == pytest.approx([0, 0, 5, 5])


def test_exp_and_sin_waveforms() -> None:
    exp_wave = _waveform("V1 a 0 EXP(0 1 1u 1u 5u 2u)")
    assert exp_wave(0.5e-6) == pytest.approx(0.0)
    assert exp_wave(2e-6) == pytest.approx(1 - math.exp(-1))
    t = 7e-6
    assert exp_wave(t) == pytest.approx((1 - math.exp(-6)) - (1 - math.exp(-1)))
    sin_wave = _waveform("V1 a 0 SIN(1 2 1k 1m 100 30)")
    assert sin_wave(0.5e-3) == pytest.approx(1 + 2 * math.sin(math.radians(30)))
    t = 1.3e-3
    assert sin_wave(t) == pytest.approx(1 + 2 * math.exp(-100 * 0.3e-3) * math.sin(2 * math.pi * 1e3 * 0.3e-3 + math.radians(30)))


def test_op_uses_dc_ac_uses_the_phasor_and_tran_the_waveform() -> None:
    text = "V1 a 0 DC 2 AC 1 90 SIN(0 1 10k)\nR1 a 0 1k\nI1 0 b DC 1m PULSE(0 2m 10u 1u 1u 20u 50u)\nR2 b 0 1k\n"
    problem = build_mna_problem(parse_netlist_text(text))
    branch = problem.index_map.branch_to_index["V1"]
    assert problem.b_dc[branch, 0] == 2.0
    assert problem.b_ac[branch, 0] == pytest.approx(1j)
    service = SimulationService()
    op = service.run_request(AnalysisRequest(netlist_text=text + ".op\n"))
    assert dict(zip(op.labels, op.dc_solution)) == pytest.approx({"V(a)": 2.0, "V(b)": -1.0, "I(V1)": -2e-3})
    tran = service.run_request(AnalysisRequest(netlist_text=text + ".tran 40u 0.5u\n"))
    labels = tran.waveform["labels"]
    time = np.asarray(tran.waveform["time"])
    va = np.asarray(tran.waveform["values"][labels.index("V(a)")])
    vb = np.asarray(tran.waveform["values"][labels.index("V(b)")])
    assert va == pytest.approx(np.sin(2 * np.pi * 10e3 * time), abs=1e-9)  # the DC value is not added
    assert vb[time.searchsorted(5e-6)] == pytest.approx(0.0, abs=1e-12)
    assert vb[time.searchsorted(20e-6)] == pytest.approx(-2.0, abs=1e-9)  # current into the first node (0)


def test_ac_phase_reaches_the_ac_analysis() -> None:
    service = SimulationService()
    response = service.run_request(AnalysisRequest(netlist_text="V1 a 0 AC 1 90\nR1 a b 1k\nR2 b 0 1k\n.ac 1k 1k 1\n"))
    labels = response.spectrum["labels"]
    assert response.spectrum["magnitudes"][labels.index("V(b)")] == pytest.approx([0.5])
    problem = build_mna_problem(parse_netlist_text("V1 a 0 AC 1 90\nR1 a b 1k\nR2 b 0 1k\n"))
    assert problem.b_ac[problem.index_map.branch_to_index["V1"], 0] == pytest.approx(1j)


# ---------------------------------------------------------------------------
# Unsupported constructs: diagnostics and errors
# ---------------------------------------------------------------------------


def test_harmless_directives_are_diagnostics() -> None:
    circuit = parse_netlist_text(
        "V1 a 0 1\nR1 a 0 1k\n.options reltol=1e-4\n.temp 50\n.nodeset V(a)=1\n.ic V(a)=0\n.probe\n"
        ".print tran V(a)\n.plot tran V(a)\n.backanno\n.step param r 1k 2k 1k\n.meas tran x avg V(a)\n"
        ".four 1k V(a)\n.noise V(a) V1 dec 10 1 1k\n.tf V(a) V1\n.dc V1 0 5 1\n.op\n"
    )
    assert circuit.diagnostics == [
        "line 3: '.options' ignored (simulator options are not supported)",
        "line 4: '.temp' ignored (the circuit is simulated at 27 C)",
        "line 5: '.nodeset' ignored (node sets are not supported)",
        "line 6: '.ic' ignored (initial conditions are not supported yet)",
        "line 7: '.probe' ignored (outputs are selected by the client)",
        "line 8: '.print' ignored (outputs are selected by the client)",
        "line 9: '.plot' ignored (outputs are selected by the client)",
        "line 10: '.backanno' ignored (back-annotation is not supported)",
        "line 11: '.step' ignored (parameter sweeps are not supported)",
        "line 12: '.meas' ignored (measurements are not supported)",
        "line 13: '.four' ignored (Fourier analysis is not supported)",
        "line 14: '.noise' ignored (noise analysis is not supported)",
        "line 15: '.tf' ignored (transfer-function analysis is not supported)",
        "line 16: '.dc' sweeps are not supported: the circuit runs as .op at its source values",
    ]
    assert [d.mode for d in circuit.directives] == [AnalysisMode.OP, AnalysisMode.OP]


def test_directive_diagnostics_are_the_same_through_the_front_end() -> None:
    plain = parse_netlist_text("V1 a 0 1\nR1 a 0 1k\n.options reltol=1e-4\n.op\n")
    spice = parse_netlist_text("V1 a 0 1 ; comment forces the front end\nR1 a 0 1k\n.options reltol=1e-4\n.op\n")
    assert plain.diagnostics == spice.diagnostics == ["line 3: '.options' ignored (simulator options are not supported)"]
    assert _fields(plain.components) == _fields(spice.components)


def test_control_blocks_are_skipped() -> None:
    circuit = parse_netlist_text("V1 a 0 1\nR1 a 0 1k\n.control\nrun\nplot v(a)\n.endc\n.op\n")
    assert [r.name for r in circuit.components] == ["V1", "R1"]
    assert circuit.diagnostics == ["line 3: '.control' block ignored (simulator commands are not supported)"]


def test_bare_noise_stays_an_unsupported_directive() -> None:
    exc = _error("V1 a 0 DC 1\nR1 a 0 1k\n.noise\n")
    assert (exc.code, exc.message) == ("NETLIST_FATAL", "Unsupported directive '.noise'.")


def test_spice_analysis_forms() -> None:
    circuit = parse_netlist_text("V1 a 0 AC 1\nR1 a 0 1k\n.ac dec 10 1 1meg\n.tran 1u 1m 0 2u uic\n.tran 1u 1m\n")
    ac, spice_tran, dialect_tran = circuit.directives
    assert ac.params == {"f_start": 1.0, "f_stop": 1e6, "points": 61}
    assert spice_tran.params == {"t_stop": 1e-3, "t_step": 1e-6}
    assert dialect_tran.params == {"t_stop": 1e-6, "t_step": 1e-3}
    assert circuit.diagnostics == [
        "line 3: '.ac dec 10 1 1meg' runs as a linear sweep from 1 Hz to 1e+06 Hz with 61 points (logarithmic sweeps are not supported)",
        "line 4: '.tran ... uic' ignored (initial conditions are not supported yet)",
        "line 4: .tran maximum step 2u ignored (the step is fixed)",
        "line 5: '.tran 1u 1m' is read as .tran <t_stop> <t_step> (t_stop=1u, t_step=1m); SPICE writes the step first",
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("V1 a 0 1\nE1 b 0 TABLE {V(a)} = (0,0) (1,2)\nR1 b 0 1k\n", "line 2: 'E1': TABLE controlled sources are not supported"),
        ("V1 a 0 1\nG1 b 0 LAPLACE {V(a)} = {1/(1+s)}\nR1 b 0 1k\n", "line 2: 'G1': LAPLACE controlled sources are not supported"),
        ("V1 a 0 1\nE1 b 0 VALUE={V(a)*2}\nR1 b 0 1k\n", "line 2: 'E1': VALUE controlled sources are not supported"),
        ("V1 a 0 1\nE1 b 0 POLY(1) a 0 0 1\nR1 b 0 1k\n", "line 2: 'E1': POLY controlled sources are not supported"),
        ("V1 a 0 1\nB1 b 0 V={V(a)*2}\nR1 b 0 1k\n", "line 2: 'B1': behavioural B sources with V=/I= expressions are not supported"),
        ("V1 a 0 1\nS1 a 0 c 0 SMOD\nR1 a 0 1k\n", "line 2: 'S1': voltage-controlled switches (S elements) are not supported"),
        ("V1 a 0 1\nW1 a 0 V1 WMOD\nR1 a 0 1k\n", "line 2: 'W1': current-controlled switches (W elements) are not supported"),
        ("V1 a 0 1\nT1 a 0 b 0 Z0=50 TD=1n\nR1 b 0 50\n", "line 2: 'T1': lossless transmission lines (T elements) are not supported"),
        ("V1 a 0 1\nO1 a 0 b 0 LMOD\nR1 b 0 50\n", "line 2: 'O1': lossy transmission lines (O elements) are not supported"),
        ("V1 a 0 1\nU1 a b 0 UMOD L=1m\nR1 b 0 50\n", "line 2: 'U1': uniform RC lines (U elements) are not supported"),
        ("V1 a 0 1\nL1 a 0 1m\nL2 b 0 1m\nK1 L1 L2 0.9\nR1 b 0 50\n", "line 4: 'K1': coupled inductors (K elements) are not supported"),
        ("V1 a 0 1\nJ1 a b 0 JMOD\nR1 b 0 50\n", "line 2: 'J1': JFETs (J elements) are not supported"),
        ("V1 a 0 1\nZ1 a b 0 ZMOD\nR1 b 0 50\n", "line 2: 'Z1': MESFETs (Z elements) are not supported"),
        (".if (a == 1)\nV1 a 0 1\n.endif\nR1 a 0 1k\n", "line 1: .if: conditional netlists (.if/.elseif/.else/.endif) are not supported"),
    ],
    ids=["TABLE", "LAPLACE", "VALUE", "POLY", "B_V", "S", "W", "T", "O", "U", "K", "J", "Z", "if"],
)
def test_answer_changing_constructs_are_errors(text: str, message: str) -> None:
    exc = _error(text)
    assert (exc.code, exc.message) == ("NETLIST_FATAL", message)


def test_short_lines_with_unsupported_letters_keep_the_dialect_error() -> None:
    exc = _error("V1 a 0 DC 1\nZfoo a 0 1k\n")
    assert (exc.code, exc.message) == (
        "COMPONENT_ERROR",
        "Cannot infer type from 'Zfoo'. Use '<name>:<TYPE>' to declare the type explicitly.",
    )


def test_current_sources_in_subcircuits_get_a_polarity_diagnostic() -> None:
    circuit = parse_netlist_text(".subckt BIAS out\nIB out 0 10u\nRB out 0 1k\n.ends\nX1 n BIAS\nX2 m BIAS\n")
    assert circuit.diagnostics == [
        "subcircuit BIAS: current source(s) IB drive their current into the first node (this simulator's "
        "convention, the opposite of SPICE); swap the nodes of a model written for SPICE"
    ]


# ---------------------------------------------------------------------------
# Diagnostics plumbing, macromodel end to end, CLI
# ---------------------------------------------------------------------------


def test_service_response_carries_the_parser_diagnostics() -> None:
    response = SimulationService().run_request(
        AnalysisRequest(netlist_text="V1 a 0 1\nR1 a 0 1k\n.options gmin=1e-12\n.op\n")
    )
    assert response.diagnostics[0] == "line 3: '.options' ignored (simulator options are not supported)"


OPAMP_MACRO = """
* single-pole op-amp macromodel and a non-inverting stage built on it
.param gain=2 rg=10k
.subckt OPAMP inp inn vout PARAMS: aol=100k fp=10 rout=75
.param rp=1meg cp={1/(2*pi*fp*rp)}
RIN inp inn 2meg
G1 0 n1 inp inn {aol/rp}
RP n1 0 {rp}
CP n1 0 {cp}
E2 n2 0 n1 0 1
RO n2 vout {rout}
.ends OPAMP
.subckt NONINV in out PARAMS: g=2
X1 in fb out OPAMP aol=200k
RF out fb {(g-1)*rg}
RG fb 0 {rg}
.ends NONINV
V1 in1 0 DC 0.5 PULSE(0 1 1u 1u 1u 10u 20u)
V2 in2 0 DC 0.25 PWL(0 0 10u 1 20u 0)
X1 in1 out1 NONINV g={gain}
X2 in2 out2 NONINV g=3
RL1 out1 0 10k
RL2 out2 0 10k
"""

OPAMP_HAND = """
V1 in1 0 DC 0.5 PULSE(0 1 1u 1u 1u 10u 20u)
V2 in2 0 DC 0.25 PWL(0 0 10u 1 20u 0)
R111 in1 X1.fb 2meg
G111 0 X1.X1.n1 in1 X1.fb 0.2
R112 X1.X1.n1 0 1meg
C111 X1.X1.n1 0 {cp}
E111 X1.X1.n2 0 X1.X1.n1 0 1
R113 X1.X1.n2 out1 75
R11 out1 X1.fb 10k
R12 X1.fb 0 10k
R211 in2 X2.fb 2meg
G211 0 X2.X1.n1 in2 X2.fb 0.2
R212 X2.X1.n1 0 1meg
C211 X2.X1.n1 0 {cp}
E211 X2.X1.n2 0 X2.X1.n1 0 1
R213 X2.X1.n2 out2 75
R21 out2 X2.fb 20k
R22 X2.fb 0 10k
RL1 out1 0 10k
RL2 out2 0 10k
""".replace("{cp}", repr(1 / (2 * math.pi * 10 * 1e6)))


def _node_values(response) -> dict[str, object]:
    return {label: value for label, value in zip(response.labels, response.dc_solution) if label.startswith("V(")}


def test_opamp_macromodel_op_matches_the_hand_flattened_netlist() -> None:
    service = SimulationService()
    flattened = service.run_request(AnalysisRequest(netlist_text=OPAMP_MACRO + ".op\n"))
    hand = service.run_request(AnalysisRequest(netlist_text=OPAMP_HAND + ".op\n"))
    flat_nodes, hand_nodes = _node_values(flattened), _node_values(hand)
    assert sorted(flat_nodes) == sorted(hand_nodes)
    assert [flat_nodes[k] for k in sorted(flat_nodes)] == pytest.approx([hand_nodes[k] for k in sorted(hand_nodes)], rel=1e-12)
    # Non-inverting gains 2 and 3 on the DC values (finite open-loop gain 200k).
    assert flat_nodes["V(out1)"] == pytest.approx(1.0, rel=1e-4)
    assert flat_nodes["V(out2)"] == pytest.approx(0.75, rel=1e-4)


def test_opamp_macromodel_tran_matches_the_hand_flattened_netlist() -> None:
    service = SimulationService()
    flattened = service.run_request(AnalysisRequest(netlist_text=OPAMP_MACRO + ".tran 30u 0.25u\n"))
    hand = service.run_request(AnalysisRequest(netlist_text=OPAMP_HAND + ".tran 30u 0.25u\n"))
    for label in ("V(out1)", "V(out2)", "V(X1.X1.n1)"):
        flat = np.asarray(flattened.waveform["values"][flattened.waveform["labels"].index(label)])
        ref = np.asarray(hand.waveform["values"][hand.waveform["labels"].index(label)])
        assert flat == pytest.approx(ref, rel=1e-9, abs=1e-12), label


def test_cli_run_reads_includes_through_lib_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "vendor" / "divider.lib", "* vendor divider\n.subckt DIV in out\nR1 in out 1k\nR2 out 0 3k\n.ends DIV\n")
    deck = _write(tmp_path / "work" / "top.cir", '.include "divider.lib"\nV1 a 0 DC 4\nX1 a b DIV\n.op\n')
    assert cli.main(["run", str(deck), "--lib", str(tmp_path / "vendor")]) == 0
    output = capsys.readouterr().out
    assert "Status: ok" in output
    assert "[4.0, 3.0, -0.001]" in output
    assert cli.main(["run", str(deck)]) == 1
    assert "file 'divider.lib' not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Model-card devices end to end (parser records -> phase M2 equations)
# ---------------------------------------------------------------------------


VT_CARD = 8.617333262e-5 * 300.15


def _record_op(records: list[ComponentRecord]) -> dict[str, float]:
    from mna_simulation.api.contracts import AnalysisOptions, CircuitIR
    from mna_simulation.core_basic import build_and_run_basic

    result = build_and_run_basic(CircuitIR(components=records), AnalysisOptions(mode=AnalysisMode.OP))
    return dict(zip(result.labels, np.asarray(result.dc_solution, dtype=float).reshape(-1)))


def test_parsed_1n4148_card_runs_through_the_service() -> None:
    response = SimulationService().run_request(
        AnalysisRequest(netlist_text=f"{D1N4148}\nV1 in 0 5\nR1 in a 1kOhm\nD1 a 0 D1N4148\n.op\n")
    )
    values = dict(zip(response.labels, response.dc_solution))
    current = (values["V(in)"] - values["V(a)"]) / 1000.0
    n, is_, rs = 1.752, 2.52e-9, 0.568
    assert values["V(a)"] == pytest.approx(n * VT_CARD * math.log(current / is_ + 1.0) + current * rs, abs=1e-8)
    assert values["V(D1#a)"] == pytest.approx(values["V(a)"] - current * rs, abs=1e-9)  # rs internal node
    assert response.diagnostics == ["model D1N4148: parameters m, tt are recorded but not modelled yet"]


def test_parsed_cards_give_the_same_op_as_contract_records() -> None:
    text = (
        f"{Q2N2222}\n{BS170}\n"
        "VCC vcc 0 DC 12\nRB1 vcc b 47k\nRB2 b 0 10k\nRC vcc c 2.2k\nRE e 0 470\nQ1 c b e Q2N2222\n"
        "RD vcc d 10k\nVG g 0 DC 3\nM1 d g 0 BS170 W=100u L=2u\n.op\n"
    )
    parsed = _op(text)
    card_q = {**Q2N2222_PARAMS, "model": "gummel_poon", "model_name": "Q2N2222"}
    card_m = {
        "model": "spice_level1",
        "model_name": "BS170",
        "level": "3",
        "vto": "1.824",
        "kp": ".1233",
        "gamma": "0",
        "phi": ".6",
        "lambda": "0",
        "cgso": "1.2n",
        "cgdo": "0.25n",
        "w": "100u",
        "l": "2u",
    }
    records = [
        ComponentRecord("VCC", "V", "vcc", "0", "12", "DC"),
        ComponentRecord("RB1", "R", "vcc", "b", "47k"),
        ComponentRecord("RB2", "R", "b", "0", "10k"),
        ComponentRecord("RC", "R", "vcc", "c", "2.2k"),
        ComponentRecord("RE", "R", "e", "0", "470"),
        ComponentRecord("Q1", "QNPN", "c", "e", "14.34f", "QNPN", "b", None, None, "255.9", "6.092", card_q),
        ComponentRecord("RD", "R", "vcc", "d", "10k"),
        ComponentRecord("VG", "V", "g", "0", "3", "DC"),
        ComponentRecord("M1", "NMOS", "d", "0", None, "NMOS", "g", None, None, None, None, card_m),
    ]
    expected = _record_op(records)
    assert sorted(parsed) == sorted(expected)
    assert [parsed[k] for k in sorted(parsed)] == pytest.approx([expected[k] for k in sorted(expected)], rel=1e-12, abs=1e-15)
    assert 0.5 < parsed["V(b)"] - parsed["V(e)"] < 0.8 and parsed["V(c)"] < 12.0  # the BJT conducts
    assert parsed["V(d)"] < 12.0  # the MOSFET conducts


def test_vendor_style_macromodel_with_a_local_diode_card_simulates() -> None:
    # A clamp macromodel: its own .model card, a parameterised series resistor, ports bound to the parent.
    text = (
        ".subckt CLAMP in out PARAMS: rser=100\n"
        ".model DCL D(IS=1n N=1.5 RS=2)\n"
        "R1 in out {rser}\n"
        "D1 out 0 DCL\n"
        ".ends CLAMP\n"
        "V1 src 0 DC 3\n"
        "X1 src node CLAMP rser=1k\n"
        "RL node 0 100k\n"
        ".op\n"
    )
    values = _op(text)
    assert "V(X1.D1#a)" in values
    diode_current = (values["V(src)"] - values["V(node)"]) / 1e3 - values["V(node)"] / 100e3
    expected = 1.5 * VT_CARD * math.log(diode_current / 1e-9 + 1.0) + diode_current * 2.0
    assert values["V(node)"] == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# The netlist dialect is unaffected
# ---------------------------------------------------------------------------


CORPUS_SAMPLE = [case for case in iter_cases() if case.expects_error is None][::7]


@pytest.mark.parametrize("case", CORPUS_SAMPLE, ids=[case.name for case in CORPUS_SAMPLE])
def test_corpus_netlists_parse_exactly_as_before(case) -> None:
    current = parse_netlist_text(case.netlist_text)
    reference = reference_parse(case.netlist_text)
    assert _fields(current.components) == _fields(reference.components)
    assert [type(record) for record in current.components] == [ComponentRecord] * len(current.components)
    assert [(d.mode.value, d.params, d.raw_line) for d in current.directives] == [
        (d.mode.value, d.params, d.raw_line) for d in reference.directives
    ]
    assert current.diagnostics == []
