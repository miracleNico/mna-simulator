"""Model-card devices (phase M2) in full circuits: builder, every analysis, operating points.

The circuits are built from ``ComponentRecord``s with the card parameters as text in
``metadata`` (the contract of ``HIERARCHY_AND_MODELS_PLAN.md`` 3.2), so they do not depend on
the SPICE-dialect parser. Device-level accuracy is covered by ``tests/device_reference``;
here the solutions are checked against the device equations (diode law with series
resistance, Gummel-Poon beta at the solved bias, body-effect threshold shift) and every
analysis (``.op``, ``.ac``, ``.tran``, ``.hb``, TPWL and linear MOR) runs on them.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from device_reference import VT, GummelPoonCard, breakdown_voltage, gummel_poon, mos_card, threshold
from mna_simulation import native
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, CircuitIR, ComponentRecord
from mna_simulation.backends.python_backend import PythonBackend
from mna_simulation.card_devices import CARD_VT, parse_card_value
from mna_simulation.core_advanced import build_and_run_advanced
from mna_simulation.core_basic import build_and_run_basic
from mna_simulation.errors import CircuitSimulatorError
from mna_simulation.mna_builder import build_mna_problem

core = native.require_core()

D1N4148 = {"model": "spice", "model_name": "D1N4148", "is": "2.52n", "n": "1.752", "rs": "0.568", "bv": "100", "ibv": "0.1u", "cjo": "4p"}
Q2N2222 = {
    "model": "gummel_poon",
    "model_name": "Q2N2222",
    "is": "14.34f",
    "bf": "255.9",
    "nf": "1",
    "br": "6.092",
    "nr": "1",
    "vaf": "74.03",
    "ikf": ".2847",
    "ise": "14.34f",
    "ne": "1.307",
    "isc": "0",
    "nc": "2",
    "ikr": "0",
    "rb": "10",
    "rc": "1",
    "cje": "22.01p",
    "cjc": "7.306p",
    "vje": ".75",
    "mje": ".377",
    "tf": "411.1p",
}
Q2N2907 = {
    "model": "gummel_poon",
    "model_name": "Q2N2907",
    "is": "650.6E-18",
    "bf": "231.7",
    "nf": "1",
    "br": "3.563",
    "nr": "1",
    "vaf": "115.7",
    "ikf": "1.079",
    "ise": "54.81f",
    "ne": "1.829",
    "isc": "0",
    "nc": "2",
    "rb": "10",
    "rc": ".715",
    "cje": "19.82p",
    "cjc": "14.76p",
}
NMOS_CARD = {
    "model": "spice_level1",
    "model_name": "NCARD",
    "level": "1",
    "vto": "0.8",
    "kp": "110u",
    "gamma": "0.45",
    "phi": "0.7",
    "lambda": "0.02",
    "cgso": "0.2n",
    "cgdo": "0.2n",
    "cgbo": "0.1n",
}
PMOS_CARD = {
    "model": "spice_level1",
    "model_name": "PCARD",
    "level": "3",
    "vto": "-0.9",
    "kp": "40u",
    "gamma": "0.5",
    "phi": "0.7",
    "lambda": "0.03",
    "cgso": "0.2n",
    "cgdo": "0.2n",
}


def R(name: str, a: str, b: str, value: str) -> ComponentRecord:
    return ComponentRecord(name=name, type="R", node1=a, node2=b, value=value)


def C(name: str, a: str, b: str, value: str) -> ComponentRecord:
    return ComponentRecord(name=name, type="C", node1=a, node2=b, value=value)


def V(name: str, a: str, b: str, value: str, subtype: str = "DC", freq: str | None = None) -> ComponentRecord:
    return ComponentRecord(name=name, type="V", node1=a, node2=b, value=value, subtype=subtype, value2=freq)


def D(name: str, anode: str, cathode: str, card: dict, **instance: str) -> ComponentRecord:
    return ComponentRecord(name=name, type="D", node1=anode, node2=cathode, value=card.get("is"), metadata={**card, **instance})


def Q(name: str, kind: str, c: str, b: str, e: str, card: dict, substrate: str | None = None, **instance: str) -> ComponentRecord:
    return ComponentRecord(
        name=name, type=kind, node1=c, node2=e, value=None, ctrl_node1=b, ctrl_node2=substrate, metadata={**card, **instance}
    )


def M(name: str, kind: str, d: str, g: str, s: str, card: dict, bulk: str | None = None, **instance: str) -> ComponentRecord:
    return ComponentRecord(
        name=name, type=kind, node1=d, node2=s, value=None, ctrl_node1=g, ctrl_node2=bulk, metadata={**card, **instance}
    )


def op(components: list[ComponentRecord], **options) -> tuple[dict[str, float], object]:
    result = build_and_run_basic(CircuitIR(components=components), AnalysisOptions(mode=AnalysisMode.OP, **options))
    values = dict(zip(result.labels, np.asarray(result.dc_solution, dtype=float).reshape(-1)))
    return values, result


def gp_card(card: dict, area: float = 1.0) -> GummelPoonCard:
    def value(key: str, default: float) -> float:
        raw = card.get(key)
        number = default if raw is None else parse_card_value(raw)
        return math.inf if key in {"vaf", "var", "ikf", "ikr"} and number == 0.0 else number

    return GummelPoonCard(
        is_=value("is", 1e-16) * area,
        bf=value("bf", 100.0),
        nf=value("nf", 1.0),
        br=value("br", 1.0),
        nr=value("nr", 1.0),
        vaf=value("vaf", math.inf),
        var=value("var", math.inf),
        ikf=value("ikf", math.inf) * area,
        ikr=value("ikr", math.inf) * area,
        ise=value("ise", 0.0) * area,
        ne=value("ne", 1.5),
        isc=value("isc", 0.0) * area,
        nc=value("nc", 2.0),
    )


# ---- builder: internal nodes, defaults, area -----------------------------------------------


def test_card_vt_is_kt_over_q_at_27_celsius() -> None:
    assert CARD_VT == VT == core.card_vt
    assert abs(CARD_VT - 0.025865) < 1e-5


def test_internal_nodes_follow_the_sorted_external_nodes_and_precede_branches() -> None:
    components = [
        V("V1", "vcc", "0", "5"),
        R("R1", "vcc", "a", "1k"),
        D("D1", "a", "0", D1N4148),
        Q("Q1", "QNPN", "c", "b", "0", Q2N2222),
        R("RB", "vcc", "b", "100k"),
        R("RC", "vcc", "c", "1k"),
        D("D2", "a", "0", {**D1N4148, "rs": "0"}),
    ]
    problem = build_mna_problem(CircuitIR(components=components))
    names = problem.index_map.node_names
    assert names == ["a", "b", "c", "vcc", "D1#a", "Q1#b", "Q1#c"]
    assert problem.index_map.branch_names == ["V1"]
    assert problem.index_map.branch_to_index["V1"] == len(names)
    assert problem.metadata["num_nodes"] == 7
    assert problem.metadata["labels"] == [f"V({n})" for n in names] + ["I(V1)"]
    assert problem.native.node_count == 7
    summary = {entry["name"]: entry for entry in problem.metadata["card_devices"]}
    assert summary["D1"]["internal_nodes"] == ["D1#a"]
    assert summary["D2"]["internal_nodes"] == []  # RS = 0: no internal node
    assert summary["Q1"]["internal_nodes"] == ["Q1#b", "Q1#c"]  # RE = 0
    json.dumps(problem.metadata, allow_nan=False)
    assert problem.native.card_device_counts == {"spice_diode": 2, "gummel_poon": 1, "mos4": 0}
    assert problem.native.device_counts == {"diode": 0, "mos": 0, "bjt": 0, "behavioral": 0}
    # f_str carries readable placeholders; solver_f_str has no rows for native-table devices.
    f_str = [str(value) for value in problem.f_str.reshape(-1)]
    assert "I_spice_diode(D1)" in f_str[names.index("D1#a")]
    assert "Ic_gummel_poon(Q1)" in f_str[names.index("Q1#c")]
    assert all(str(value) == "0" for value in problem.solver_f_str.reshape(-1))
    # The 0.568 ohm series resistance sits between the anode and D1#a.
    a, ia = names.index("a"), names.index("D1#a")
    assert problem.G[a, ia] == pytest.approx(-1.0 / 0.568, rel=1e-15)
    # CJO and CJE / CJC are constant capacitors at the intrinsic junctions.
    assert problem.C[ia, ia] == pytest.approx(4e-12, rel=1e-15)
    qb, qc = names.index("Q1#b"), names.index("Q1#c")
    assert problem.C[qb, qc] == pytest.approx(-7.306e-12, rel=1e-15)


def test_internal_node_collisions_and_bad_parameters_are_netlist_errors() -> None:
    with pytest.raises(CircuitSimulatorError, match="collides"):
        build_mna_problem(CircuitIR(components=[V("V1", "D1#a", "0", "1"), D("D1", "D1#a", "0", D1N4148), R("R1", "D1#a", "0", "1k")]))
    with pytest.raises(CircuitSimulatorError, match="is = -1"):
        build_mna_problem(CircuitIR(components=[V("V1", "a", "0", "1"), D("D1", "a", "0", {**D1N4148, "is": "-1"})]))
    with pytest.raises(CircuitSimulatorError, match="LEVEL 8"):
        build_mna_problem(CircuitIR(components=[V("V1", "a", "0", "1"), M("M1", "NMOS", "a", "a", "0", {**NMOS_CARD, "level": "8"})]))
    with pytest.raises(CircuitSimulatorError, match="cannot read parameter bf"):
        build_mna_problem(CircuitIR(components=[V("V1", "a", "0", "1"), Q("Q1", "QNPN", "a", "a", "0", {**Q2N2222, "bf": "lots"})]))


def test_card_values_accept_spice_suffixes_and_units() -> None:
    assert parse_card_value("2.52n") == 2.52 * 1e-9
    assert parse_card_value("100MEG") == 100e6
    assert parse_card_value("4pF") == 4e-12
    assert parse_card_value("10mil") == pytest.approx(254e-6)
    assert parse_card_value("1.5e-3") == 1.5e-3
    assert parse_card_value("2T") == 2e12
    assert parse_card_value("inf") == math.inf
    assert parse_card_value(3) == 3.0


def test_defaults_apply_to_missing_keys() -> None:
    problem = build_mna_problem(
        CircuitIR(
            components=[
                V("V1", "a", "0", "1"),
                D("D1", "a", "0", {"model": "spice"}),
                Q("Q1", "QNPN", "a", "a", "0", {"model": "gummel_poon"}),
                M("M1", "NMOS", "a", "a", "0", {"model": "spice_level1"}),
            ]
        )
    )
    junctions = problem.native.junctions()
    assert junctions["is"][0] == 1e-14 and junctions["vt"][0] == CARD_VT  # diode IS, N = 1
    assert junctions["is"][1] == 1e-16  # BJT IS
    assert problem.native.spice_diode_breakdown_voltages()[0] == math.inf
    # MOS defaults: KP 2e-5, W = L = 100u: beta = 2e-5; VTO = 0.
    x = np.zeros(problem.native.n)
    x[0] = 1.0
    points = core.operating_points(problem.native, x)
    assert points["mos4"]["ids"][0] == pytest.approx(0.5 * 2e-5 * 1.0**2, rel=1e-15)


# ---- diode ---------------------------------------------------------------------------------


@pytest.mark.parametrize("supply", ["0.8", "1", "5", "20"])
def test_1n4148_with_series_resistor_obeys_the_diode_law(supply: str) -> None:
    values, result = op([V("V1", "in", "0", supply), R("R1", "in", "a", "1k"), D("D1", "a", "0", D1N4148)])
    current = (values["V(in)"] - values["V(a)"]) / 1000.0
    n, is_, rs = 1.752, 2.52e-9, 0.568
    expected = n * VT * math.log(current / is_ + 1.0) + current * rs
    assert values["V(a)"] == pytest.approx(expected, abs=1e-8)
    assert values["V(D1#a)"] == pytest.approx(values["V(a)"] - current * rs, abs=1e-9)
    point = result.metadata["device_operating_points"][0]
    assert point["name"] == "D1" and point["region"] == "forward" and point["model"] == "spice"
    assert point["id"] == pytest.approx(current, rel=1e-6)
    json.dumps(result.metadata["device_operating_points"], allow_nan=False)


@pytest.mark.parametrize("volts", ["1", "5", "20"])
def test_spice_diode_across_an_ideal_source_converges_through_junction_limiting(volts: str) -> None:
    # The junction is registered for pnjlim (vt = N vt, IS) like the ideal diode; with the
    # exponent continued linearly beyond 80 N vt even 20 V gives a finite current.
    card = {**D1N4148, "rs": "0"}
    problem = build_mna_problem(CircuitIR(components=[V("V1", "a", "0", volts), D("D1", "a", "0", card)]))
    junctions = problem.native.junctions()
    assert list(junctions["anode"]) == [0] and junctions["vt"][0] == 1.752 * VT and junctions["is"][0] == 2.52e-9
    values, _ = op([V("V1", "a", "0", volts), D("D1", "a", "0", card)], max_iter=200)
    expected, _ = spice_diode_reference(float(volts))
    assert values["V(a)"] == float(volts)
    assert -values["I(V1)"] == pytest.approx(expected, rel=1e-9)


def spice_diode_reference(v: float) -> tuple[float, float]:
    from device_reference import spice_diode

    current, conductance = spice_diode(v, 2.52e-9, 1.752, breakdown_voltage(2.52e-9, 100.0, 1e-7))
    return float(current), float(conductance)


def test_gummel_poon_base_driven_by_an_ideal_source_converges() -> None:
    components = [V("VB", "b", "0", "1.2"), V("VCC", "vcc", "0", "5"), R("RC", "vcc", "c", "10"), Q("Q1", "QNPN", "c", "b", "0", {**Q2N2222, "rb": "0", "rc": "0"})]
    problem = build_mna_problem(CircuitIR(components=components))
    junctions = problem.native.junctions()
    assert len(junctions["anode"]) == 2 and set(junctions["vt"]) == {VT}
    values, result = op(components, max_iter=200)
    point = result.metadata["device_operating_points"][0]
    assert point["region"] == "saturation"
    currents, _ = gummel_poon(values["V(c)"], 1.2, 0.0, gp_card({**Q2N2222, "rb": "0", "rc": "0"}))
    assert point["ic"] == pytest.approx(currents[0], rel=1e-9)
    assert point["ic"] == pytest.approx((5.0 - values["V(c)"]) / 10.0, rel=1e-6)


def test_diode_area_scales_is_and_rs() -> None:
    values, _ = op([V("V1", "in", "0", "5"), R("R1", "in", "a", "1k"), D("D1", "a", "0", D1N4148, area="2")])
    current = (values["V(in)"] - values["V(a)"]) / 1000.0
    expected = 1.752 * VT * math.log(current / (2 * 2.52e-9) + 1.0) + current * 0.568 / 2
    assert values["V(a)"] == pytest.approx(expected, abs=1e-8)


def test_diode_m_is_the_grading_coefficient_not_a_multiplier() -> None:
    plain, _ = op([V("V1", "in", "0", "5"), R("R1", "in", "a", "1k"), D("D1", "a", "0", D1N4148)])
    graded, _ = op([V("V1", "in", "0", "5"), R("R1", "in", "a", "1k"), D("D1", "a", "0", {**D1N4148, "m": "0.333"})])
    assert graded["V(a)"] == plain["V(a)"]


def test_zener_breakdown_follows_the_shifted_breakdown_branch() -> None:
    card = {"model": "spice", "model_name": "DZ5V1", "is": "1e-14", "n": "1", "bv": "5.1", "ibv": "5m"}
    values, result = op([V("V1", "in", "0", "-12"), R("R1", "in", "k", "1k"), D("DZ", "k", "0", card)])
    v = values["V(k)"]
    current = (values["V(in)"] - v) / 1000.0  # negative: reverse current through the diode
    bv_eff = breakdown_voltage(1e-14, 5.1, 5e-3)
    assert current == pytest.approx(-1e-14 * math.exp(-(bv_eff + v) / VT), rel=1e-6)
    assert -5.3 < v < -5.0
    assert result.metadata["device_operating_points"][0]["region"] == "breakdown"


# ---- BJT -----------------------------------------------------------------------------------


def _ce_stage(input_source: ComponentRecord | None = None) -> list[ComponentRecord]:
    components = [
        V("VCC", "vcc", "0", "10"),
        R("RB1", "vcc", "b", "47k"),
        R("RB2", "b", "0", "10k"),
        R("RC", "vcc", "c", "2.2k"),
        R("RE", "e", "0", "470"),
        C("CE", "e", "0", "100u"),
        Q("Q1", "QNPN", "c", "b", "e", Q2N2222),
    ]
    if input_source is not None:
        components += [input_source, C("CIN", "in", "b", "10u")]
    return components


def test_2n2222_common_emitter_bias_matches_the_gummel_poon_beta() -> None:
    values, result = op(_ce_stage())
    point = next(p for p in result.metadata["device_operating_points"] if p["name"] == "Q1")
    assert point["region"] == "forward_active"
    assert point["model"] == "gummel_poon" and point["model_name"] == "Q2N2222"
    assert point["internal_nodes"] == ["Q1#b", "Q1#c"]
    ic, ib = point["ic"], point["ib"]
    assert 1e-3 < ic < 3e-3
    # Kirchhoff on the external resistors.
    assert ic == pytest.approx((values["V(vcc)"] - values["V(c)"]) / 2.2e3, rel=1e-6)
    # Beta at the solved intrinsic bias from the independent reference.
    vc, vb, ve = values["V(Q1#c)"], values["V(Q1#b)"], values["V(e)"]
    currents, _ = gummel_poon(vc, vb, ve, gp_card(Q2N2222))
    assert ic / ib == pytest.approx(currents[0] / currents[1], rel=1e-9)
    assert ic / ib < 255.9  # leakage and high injection pull beta below BF
    json.dumps(result.metadata["device_operating_points"], allow_nan=False)


def test_2n2222_common_emitter_ac_gain_and_transient() -> None:
    ac_source = V("VIN", "in", "0", "1", subtype="AC")
    ir = CircuitIR(components=_ce_stage(ac_source))
    ac = build_and_run_basic(ir, AnalysisOptions(mode=AnalysisMode.AC, ac_start=1e3, ac_stop=1e4, ac_points=3))
    assert ac.metadata["op_used"] is True and ac.metadata["linearized_device_count"] == 1
    out = ac.spectrum.labels.index("V(c)")
    gain = np.abs(ac.spectrum.magnitudes[out])
    assert np.all(np.isfinite(gain))
    assert np.all(gain > 20.0) and np.all(gain < 2000.0)  # gm * RC with a bypassed emitter
    assert ac.metadata["device_operating_points"][0]["region"] == "forward_active"

    sine = V("VIN", "in", "0", "5m", subtype="SIN", freq="1k")
    tran = build_and_run_basic(
        CircuitIR(components=_ce_stage(sine)), AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=2e-3, tran_step=1e-5)
    )
    assert tran.metadata["op_used"] is True
    vc = tran.waveform.values[tran.waveform.labels.index("V(c)")]
    assert np.all(np.isfinite(vc))
    assert 0.05 < float(vc[100:].max() - vc[100:].min()) < 9.0  # amplified, not clipped to the rails


def test_pnp_current_mirror() -> None:
    components = [
        V("VCC", "vcc", "0", "5"),
        Q("Q1", "QPNP", "ref", "ref", "vcc", Q2N2907),
        R("RREF", "ref", "0", "4.3k"),
        Q("Q2", "QPNP", "out", "ref", "vcc", Q2N2907),
        R("RL", "out", "0", "2k"),
    ]
    values, result = op(components)
    points = {p["name"]: p for p in result.metadata["device_operating_points"]}
    i_ref = values["V(ref)"] / 4.3e3
    i_out = values["V(out)"] / 2e3
    assert i_ref == pytest.approx(1e-3, rel=0.1)
    assert i_out == pytest.approx(i_ref, rel=0.05)  # base currents and the Early effect only
    assert points["Q2"]["region"] == "forward_active"
    assert points["Q2"]["ic"] < 0.0  # PNP collector current flows out of the device
    # The mirror ratio equals the reference model's ratio at the solved bias.
    card = gp_card(Q2N2907)
    c1, _ = gummel_poon(values["V(Q1#c)"], values["V(Q1#b)"], values["V(vcc)"], card, -1)
    c2, _ = gummel_poon(values["V(Q2#c)"], values["V(Q2#b)"], values["V(vcc)"], card, -1)
    assert points["Q2"]["ic"] / points["Q1"]["ic"] == pytest.approx(c2[0] / c1[0], rel=1e-9)


def test_substrate_node_is_ignored() -> None:
    components = _ce_stage() + [V("VSUB", "sub", "0", "-5")]
    components[-2] = Q("Q1", "QNPN", "c", "b", "e", Q2N2222, substrate="sub")
    reference, _ = op(_ce_stage())
    values, _ = op(components)
    assert values["V(c)"] == pytest.approx(reference["V(c)"], abs=1e-12)
    floating = _ce_stage()
    floating[-1] = Q("Q1", "QNPN", "c", "b", "e", Q2N2222, substrate="lonely")
    problem = build_mna_problem(CircuitIR(components=floating))
    assert "lonely" not in problem.index_map.node_names


# ---- MOS -----------------------------------------------------------------------------------


def _nmos_load(vg: str, vbulk: str | None) -> list[ComponentRecord]:
    components = [
        V("VDD", "vdd", "0", "5"),
        V("VG", "g", "0", vg),
        R("RD", "vdd", "d", "10k"),
        M("M1", "NMOS", "d", "g", "0", NMOS_CARD, bulk="b" if vbulk is not None else None, w="20u", l="2u"),
    ]
    if vbulk is not None:
        components.append(V("VB", "b", "0", vbulk))
    return components


def test_nmos_inverter_body_effect_threshold_shift_matches_the_formula() -> None:
    beta = 110e-6 * 20 / 2
    gamma, phi, lam = 0.45, 0.7, 0.02

    def extracted_vth(vbulk: str | None) -> float:
        values, result = op(_nmos_load("1.6", vbulk))
        ids = (5.0 - values["V(d)"]) / 10e3
        point = result.metadata["device_operating_points"][0]
        assert point["region"] == "saturation"
        assert point["ids"] == pytest.approx(ids, rel=1e-6)
        vds = values["V(d)"]
        return 1.6 - math.sqrt(2.0 * ids / (beta * (1.0 + lam * vds)))

    vth0 = extracted_vth(None)  # three-node instance: bulk on the source
    vth_body = extracted_vth("-2")
    assert vth0 == pytest.approx(0.8, abs=1e-7)
    expected_shift = gamma * (math.sqrt(phi + 2.0) - math.sqrt(phi))
    assert vth_body - vth0 == pytest.approx(expected_shift, abs=1e-7)
    values, result = op(_nmos_load("1.6", "-2"))
    point = result.metadata["device_operating_points"][0]
    assert point["vth"] == pytest.approx(0.8 + expected_shift, rel=1e-12)
    assert point["gmbs"] > 0.0 and point["vbs"] == pytest.approx(-2.0)
    # Inverter transfer: high input pulls the output low.
    low, _ = op(_nmos_load("5", "-2"))
    off, off_result = op(_nmos_load("0", "-2"))
    assert low["V(d)"] < 0.5 and off["V(d)"] == pytest.approx(5.0, abs=1e-6)
    assert off_result.metadata["device_operating_points"][0]["region"] == "off"


def _cmos_inverter(source: ComponentRecord) -> list[ComponentRecord]:
    return [
        V("VDD", "vdd", "0", "5"),
        source,
        M("MP", "PMOS", "out", "in", "vdd", PMOS_CARD, bulk="vdd", w="40u", l="2u"),
        M("MN", "NMOS", "out", "in", "0", NMOS_CARD, bulk="0", w="10u", l="2u"),
        C("CL", "out", "0", "50f"),
    ]


def test_cmos_inverter_transient_swings_rail_to_rail() -> None:
    components = _cmos_inverter(V("VIN", "in", "mid", "2.5", subtype="SIN", freq="1meg")) + [V("VMID", "mid", "0", "2.5")]
    result = build_and_run_basic(CircuitIR(components=components), AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=2e-6, tran_step=2e-9))
    out = result.waveform.values[result.waveform.labels.index("V(out)")]
    vin = result.waveform.values[result.waveform.labels.index("V(in)")]
    assert np.all(np.isfinite(out))
    assert out.max() > 4.9 and out.min() < 0.1
    high_in = vin > 4.0
    low_in = vin < 1.0
    assert np.all(out[high_in] < 0.5) and np.all(out[low_in] > 4.5)  # inverted
    regions = {p["name"]: p["region"] for p in result.metadata["device_operating_points"]}
    assert set(regions) == {"MP", "MN"}


def test_pmos_three_node_instance_puts_the_bulk_on_the_source() -> None:
    components = [
        V("VDD", "vdd", "0", "5"),
        V("VG", "g", "0", "3"),
        M("MP", "PMOS", "d", "g", "vdd", PMOS_CARD, w="40u", l="2u"),
        R("RL", "d", "0", "2k"),
    ]
    values, result = op(components)
    point = result.metadata["device_operating_points"][0]
    beta = 40e-6 * 40 / 2
    vsd = 5.0 - values["V(d)"]
    expected = 0.5 * beta * (2.0 - 0.9) ** 2 * (1.0 + 0.03 * vsd)  # |vgs| = 2, vth = |VTO|, no body bias
    assert values["V(d)"] / 2e3 == pytest.approx(expected, rel=1e-7)
    assert point["region"] == "saturation" and point["ids"] < 0.0


# ---- harmonic balance, MOR, service path, threads -----------------------------------------


def test_diode_rectifier_harmonic_balance() -> None:
    components = [
        V("VS", "in", "0", "2", subtype="SIN", freq="1k"),
        R("RS", "in", "a", "100"),
        D("D1", "a", "out", D1N4148),
        R("RL", "out", "0", "1k"),
    ]
    result = build_and_run_advanced(CircuitIR(components=components), AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=12, hb_time_window=1e-3))
    assert result.status == "ok"
    out = result.waveform.values[result.waveform.labels.index("V(out)")]
    assert np.all(np.isfinite(out))
    assert out.max() == pytest.approx(2.0 - 0.6, abs=0.25)  # peak minus a diode drop
    assert out.min() > -0.05  # rectified: reverse current is tiny


def test_bjt_stage_harmonic_balance_runs() -> None:
    sine = V("VIN", "in", "0", "5m", subtype="SIN", freq="10k")
    result = build_and_run_advanced(CircuitIR(components=_ce_stage(sine)), AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=4, hb_time_window=1e-4))
    vc = result.waveform.values[result.waveform.labels.index("V(c)")]
    assert np.all(np.isfinite(vc)) and float(vc.max() - vc.min()) > 0.05


def test_tpwl_and_linear_mor_run_on_card_devices() -> None:
    sine = V("VIN", "in", "0", "5m", subtype="SIN", freq="1k")
    tran = build_and_run_basic(
        CircuitIR(components=_ce_stage(sine)),
        AnalysisOptions(
            mode=AnalysisMode.TRAN, tran_stop=1e-3, tran_step=1e-5, use_mor=True, mor_method="tpwl", mor_output_nodes=["V(c)"], mor_order=4
        ),
    )
    assert np.all(np.isfinite(tran.waveform.values))
    ac = build_and_run_basic(
        CircuitIR(components=_ce_stage(V("VIN", "in", "0", "1", subtype="AC"))),
        AnalysisOptions(
            mode=AnalysisMode.AC, ac_start=1e3, ac_stop=1e4, ac_points=3, use_mor=True, mor_method="linear_krylov", mor_output_nodes=["V(c)"]
        ),
    )
    full = build_and_run_basic(
        CircuitIR(components=_ce_stage(V("VIN", "in", "0", "1", subtype="AC"))),
        AnalysisOptions(mode=AnalysisMode.AC, ac_start=1e3, ac_stop=1e4, ac_points=3),
    )
    reduced = np.abs(ac.spectrum.magnitudes[ac.spectrum.labels.index("V(c)")])
    exact = np.abs(full.spectrum.magnitudes[full.spectrum.labels.index("V(c)")])
    assert reduced == pytest.approx(exact, rel=1e-6)


def test_backend_with_gmin_steps_and_json_safe_metadata() -> None:
    backend = PythonBackend()
    options = AnalysisOptions(mode=AnalysisMode.OP, gmin_steps=[0.0, 1e-12, 1e-9])
    result = backend.run(CircuitIR(components=_ce_stage() + [D("D1", "vcc", "x", D1N4148), R("RX", "x", "0", "1k")]), options)
    assert result.status == "ok"
    json.dumps(result.metadata["device_operating_points"], allow_nan=False)
    json.dumps(result.metadata["card_devices"], allow_nan=False)


def test_device_evaluation_is_identical_for_every_thread_count() -> None:
    rng = np.random.default_rng(3)
    n = 400
    count = 900  # 2700 devices: above the parallel threshold of the Newton driver too
    diodes = [{"anode": int(a), "cathode": int(k), "is": 2.52e-9, "n": 1.752, "bv": 100.0, "ibv": 1e-4} for a, k in rng.integers(-1, n, (count, 2))]
    bjts = []
    for c, b, e in rng.integers(-1, n, (count, 3)):
        row = {"type": "QNPN" if rng.random() < 0.5 else "QPNP", "collector": int(c), "base": int(b), "emitter": int(e)}
        row.update({"is": 1e-15, "bf": 200.0, "nf": 1.0, "br": 4.0, "nr": 1.0, "vaf": 70.0, "var": math.inf, "ikf": 0.3, "ikr": math.inf})
        row.update({"ise": 1e-14, "ne": 1.4, "isc": 0.0, "nc": 2.0})
        bjts.append(row)
    moses = []
    for d, g, s, b in rng.integers(-1, n, (count, 4)):
        row = {"type": "NMOS" if rng.random() < 0.5 else "PMOS", "drain": int(d), "gate": int(g), "source": int(s), "bulk": int(b)}
        row.update({"beta": 1e-3, "vto": 0.7, "lambda": 0.02, "gamma": 0.4, "phi": 0.65})
        moses.append(row)
    from mna_simulation.native.circuit import build_circuit, gummel_poon_table, mos4_table, spice_diode_table

    circuit = build_circuit(
        n, spice_diodes=spice_diode_table(diodes), gummel_poon=gummel_poon_table(bjts), mos4=mos4_table(moses), node_count=n
    )
    x = rng.uniform(-1.0, 1.0, n)
    outputs = []
    for threads in (1, 2, 4):
        f = np.empty(n)
        j = np.empty(circuit.nnz)
        core.evaluate_devices(circuit, x, 0.0, threads, f, j)
        outputs.append((f, j))
    for f, j in outputs[1:]:
        assert np.array_equal(f.view(np.uint64), outputs[0][0].view(np.uint64))
        assert np.array_equal(j.view(np.uint64), outputs[0][1].view(np.uint64))


def test_transient_is_identical_for_every_thread_count() -> None:
    sine = V("VIN", "in", "0", "5m", subtype="SIN", freq="1k")
    runs = []
    for threads in (1, 4):
        result = build_and_run_basic(
            CircuitIR(components=_ce_stage(sine)),
            AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=5e-4, tran_step=1e-5, num_threads=threads),
        )
        runs.append(np.asarray(result.waveform.values))
    assert np.array_equal(runs[0], runs[1])


# ---- process parameters -------------------------------------------------------------------


def test_kp_phi_and_gamma_are_derived_from_uo_tox_and_nsub_when_omitted() -> None:
    from mna_simulation.card_devices import CHARGE, EPS_OX, EPS_SI, NI_300K, prepare_card_devices

    card = {"model": "spice_level1", "vto": "0.7", "uo": "550", "tox": "20n", "nsub": "1e16", "w": "10u", "l": "1u"}
    (device,) = prepare_card_devices([M("M1", "NMOS", "d", "g", "0", card)])
    cox = EPS_OX / 20e-9
    kp = 550e-4 * cox
    assert device.params["beta"] == pytest.approx(kp * 10.0, rel=1e-12)
    assert device.params["phi"] == pytest.approx(2.0 * CARD_VT * math.log(1e22 / NI_300K), rel=1e-12)
    assert device.params["gamma"] == pytest.approx(math.sqrt(2.0 * EPS_SI * CHARGE * 1e22) / cox, rel=1e-12)
    # A card that gives KP, PHI or GAMMA keeps its own values.
    explicit = dict(card, kp="1e-4", phi="0.9", gamma="0.3")
    (device,) = prepare_card_devices([M("M1", "NMOS", "d", "g", "0", explicit)])
    assert device.params["beta"] == pytest.approx(1e-3, rel=1e-12)
    assert device.params["phi"] == 0.9 and device.params["gamma"] == 0.3
    # UO without TOX (or TOX without UO) derives nothing: SPICE's default KP stays.
    (device,) = prepare_card_devices([M("M1", "NMOS", "d", "g", "0", {"model": "spice_level1", "uo": "550"})])
    assert device.params["beta"] == pytest.approx(2e-5, rel=1e-12)
    with pytest.raises(CircuitSimulatorError, match="nsub"):
        prepare_card_devices([M("M1", "NMOS", "d", "g", "0", {"model": "spice_level1", "nsub": "1e9", "tox": "20n"})])
