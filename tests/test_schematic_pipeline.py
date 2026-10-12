from __future__ import annotations

import pytest

from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest, SchematicDocument
from mna_simulation.api.service import SimulationService
from mna_simulation.errors import NetlistError
from mna_simulation.schematic_pipeline import _flatten_subcircuits, schematic_to_circuit_ir, schematic_to_netlist


def _pin(component_id: str, pin: str) -> dict[str, str]:
    return {"kind": "component_pin", "component_id": component_id, "pin": pin}


def _simple_op_schematic() -> SchematicDocument:
    return SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "DC", "value": "5"},
            {"id": "r1", "type": "R", "name": "R1", "value": "1k"},
        ],
        wires=[
            {
                "id": "w1",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w2",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "p"},
                "end": {"kind": "component_pin", "component_id": "r1", "pin": "p"},
            },
            {
                "id": "w3",
                "start": {"kind": "component_pin", "component_id": "r1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        analysis={"mode": "op", "params": {}},
    )


def test_schematic_to_netlist_maps_ground_and_analysis() -> None:
    netlist_text, pin_assignment = schematic_to_netlist(_simple_op_schematic())
    assert ".op" in netlist_text
    assert ".end" in netlist_text
    assert "0" in pin_assignment.values()
    assert any(node.startswith("n") for node in pin_assignment.values())


def test_schematic_round_trip_to_circuit_ir() -> None:
    conversion = schematic_to_circuit_ir(_simple_op_schematic())
    assert len(conversion.circuit.components) == 2
    assert conversion.circuit.directives[-1].mode == AnalysisMode.OP
    assert "V1" in conversion.netlist_text
    assert "R1" in conversion.netlist_text


def test_flat_schematic_named_node_markers_become_probeable_net_names() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "FUNC", "value": "0", "value2": "t"},
            {"id": "r1", "type": "R", "name": "R1", "value": "1k"},
        ],
        junctions=[
            {"id": "port_q_3_4"},
        ],
        wires=[
            {
                "id": "w-v-p",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "p"},
                "end": {"kind": "junction", "junction_id": "port_q_3_4"},
            },
            {
                "id": "w-r-p",
                "start": {"kind": "component_pin", "component_id": "r1", "pin": "p"},
                "end": {"kind": "junction", "junction_id": "port_q_3_4"},
            },
            {
                "id": "w-v-g",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-r-g",
                "start": {"kind": "component_pin", "component_id": "r1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        analysis={"mode": "tran", "params": {"t_stop": "6n", "t_step": "1n"}},
    )

    netlist_text, pin_assignment = schematic_to_netlist(schematic)
    assert "V1 q_3_4 0 FUNC 0 t" in netlist_text
    assert "R1 q_3_4 0 1k" in netlist_text
    assert pin_assignment["cp:v1:p"] == "q_3_4"

    response = SimulationService().run_request(
        AnalysisRequest(
            netlist_text="",
            mode=AnalysisMode.TRAN,
            options={"probe_nodes": ["V(q_3_4)"]},
            schematic=schematic,
        )
    )
    assert response.status == "ok"
    assert response.waveform is not None
    assert response.waveform["time"]
    assert "V(q_3_4)" in response.waveform["labels"]


def test_schematic_bjt_defaults_to_level1_in_generated_netlist() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "q1", "type": "QNPN", "name": "Q1", "value": "1e-15", "value2": "150"},
            {"id": "rc", "type": "R", "name": "Rc", "value": "4.7k"},
            {"id": "re", "type": "R", "name": "Re", "value": "470"},
            {"id": "rb", "type": "R", "name": "Rb", "value": "10k"},
        ],
        wires=[
            {
                "id": "w-c",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "c"},
                "end": {"kind": "component_pin", "component_id": "rc", "pin": "p"},
            },
            {
                "id": "w-rc-g",
                "start": {"kind": "component_pin", "component_id": "rc", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-b",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "b"},
                "end": {"kind": "component_pin", "component_id": "rb", "pin": "p"},
            },
            {
                "id": "w-rb-g",
                "start": {"kind": "component_pin", "component_id": "rb", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-e",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "e"},
                "end": {"kind": "component_pin", "component_id": "re", "pin": "p"},
            },
            {
                "id": "w-re-g",
                "start": {"kind": "component_pin", "component_id": "re", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        analysis={"mode": "op", "params": {}},
    )
    netlist_text, _ = schematic_to_netlist(schematic)

    assert "Q1 " in netlist_text
    assert " QNPN LEVEL1 1e-15 150 3" in netlist_text
    assert "VCCS" not in netlist_text


def test_schematic_bjt_can_emit_legacy_small_signal_netlist() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {
                "id": "q1",
                "type": "QNPN",
                "name": "Q1",
                "value": "40m",
                "value2": "2.5k",
                "value3": "100k",
                "metadata": {"model": "small_signal", "cpi": "8p", "cmu": "3p", "ccs": "0", "rb": "0", "re": "0"},
            },
            {"id": "rc", "type": "R", "name": "Rc", "value": "4.7k"},
            {"id": "re", "type": "R", "name": "Re", "value": "470"},
            {"id": "rb", "type": "R", "name": "Rb", "value": "10k"},
        ],
        wires=[
            {
                "id": "w-c",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "c"},
                "end": {"kind": "component_pin", "component_id": "rc", "pin": "p"},
            },
            {
                "id": "w-rc-g",
                "start": {"kind": "component_pin", "component_id": "rc", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-b",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "b"},
                "end": {"kind": "component_pin", "component_id": "rb", "pin": "p"},
            },
            {
                "id": "w-rb-g",
                "start": {"kind": "component_pin", "component_id": "rb", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-e",
                "start": {"kind": "component_pin", "component_id": "q1", "pin": "e"},
                "end": {"kind": "component_pin", "component_id": "re", "pin": "p"},
            },
            {
                "id": "w-re-g",
                "start": {"kind": "component_pin", "component_id": "re", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        analysis={"mode": "op", "params": {}},
    )
    netlist_text, _ = schematic_to_netlist(schematic)

    assert " QNPN 40m 2.5k 100k 8p 3p 0 0 0" in netlist_text
    assert "LEVEL1" not in netlist_text


def test_hierarchy_keeps_top_level_net_names_and_marks_internal_nets() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "DC", "value": "1"},
            {"id": "xamp", "type": "SUBCKT", "name": "Xamp", "subcircuit_id": "amp", "pins": ["in", "out"]},
            {"id": "rload", "type": "R", "name": "Rload", "value": "10k"},
        ],
        wires=[
            {
                "id": "w-v-g",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-in",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "p"},
                "end": {"kind": "component_pin", "component_id": "xamp", "pin": "in"},
            },
            {
                "id": "w-out",
                "start": {"kind": "component_pin", "component_id": "xamp", "pin": "out"},
                "end": {"kind": "component_pin", "component_id": "rload", "pin": "p"},
            },
            {
                "id": "w-r-g",
                "start": {"kind": "component_pin", "component_id": "rload", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        subcircuits={
            "amp": {
                "components": [
                    {"id": "rin", "type": "R", "name": "Rin", "value": "1k"},
                    {"id": "rout", "type": "R", "name": "Rout", "value": "2k"},
                ],
                "junctions": [
                    {"id": "port_in"},
                    {"id": "mid"},
                    {"id": "port_out"},
                ],
                "wires": [
                    {
                        "id": "w-rin-p",
                        "start": {"kind": "junction", "junction_id": "port_in"},
                        "end": {"kind": "component_pin", "component_id": "rin", "pin": "p"},
                    },
                    {
                        "id": "w-rin-n",
                        "start": {"kind": "component_pin", "component_id": "rin", "pin": "n"},
                        "end": {"kind": "junction", "junction_id": "mid"},
                    },
                    {
                        "id": "w-rout-p",
                        "start": {"kind": "junction", "junction_id": "mid"},
                        "end": {"kind": "component_pin", "component_id": "rout", "pin": "p"},
                    },
                    {
                        "id": "w-rout-n",
                        "start": {"kind": "component_pin", "component_id": "rout", "pin": "n"},
                        "end": {"kind": "junction", "junction_id": "port_out"},
                    },
                ],
            }
        },
        analysis={"mode": "op", "params": {}},
    )

    netlist_text, _ = schematic_to_netlist(schematic)

    assert "V1 n1 0 DC 1" in netlist_text
    assert "Rload n2 0 10k" in netlist_text
    assert "R1 n1 x1 1k" in netlist_text
    assert "R2 x1 n2 2k" in netlist_text
    assert " n3 " not in netlist_text


def test_three_level_hierarchy_uses_logical_instance_port_names() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "DC", "value": "1"},
            {"id": "amp-block", "type": "SUBCKT", "name": "AmpBlock", "subcircuit_id": "amp", "pins": ["in", "out"]},
            {"id": "rload", "type": "R", "name": "Rload", "value": "10k"},
        ],
        wires=[
            {
                "id": "w-v-g",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-in",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "p"},
                "end": {"kind": "component_pin", "component_id": "amp-block", "pin": "in"},
            },
            {
                "id": "w-out",
                "start": {"kind": "component_pin", "component_id": "amp-block", "pin": "out"},
                "end": {"kind": "component_pin", "component_id": "rload", "pin": "p"},
            },
            {
                "id": "w-r-g",
                "start": {"kind": "component_pin", "component_id": "rload", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        subcircuits={
            "amp": {
                "components": [
                    {
                        "id": "diff",
                        "type": "SUBCKT",
                        "name": "DiffAmp",
                        "subcircuit_id": "diff-core",
                        "pins": ["in_p", "out"],
                    },
                ],
                "junctions": [{"id": "port_in"}, {"id": "port_out"}],
                "wires": [
                    {
                        "id": "w-amp-in",
                        "start": {"kind": "junction", "junction_id": "port_in"},
                        "end": {"kind": "component_pin", "component_id": "diff", "pin": "in_p"},
                    },
                    {
                        "id": "w-amp-out",
                        "start": {"kind": "component_pin", "component_id": "diff", "pin": "out"},
                        "end": {"kind": "junction", "junction_id": "port_out"},
                    },
                ],
            },
            "diff-core": {
                "components": [
                    {"id": "rdiff", "type": "R", "name": "Rdiff", "value": "1k"},
                ],
                "junctions": [{"id": "port_in_p"}, {"id": "port_out"}],
                "wires": [
                    {
                        "id": "w-core-in",
                        "start": {"kind": "junction", "junction_id": "port_in_p"},
                        "end": {"kind": "component_pin", "component_id": "rdiff", "pin": "p"},
                    },
                    {
                        "id": "w-core-out",
                        "start": {"kind": "component_pin", "component_id": "rdiff", "pin": "n"},
                        "end": {"kind": "junction", "junction_id": "port_out"},
                    },
                ],
            },
        },
        analysis={"mode": "op", "params": {}},
    )

    flattened = _flatten_subcircuits(schematic)
    junction_ids = {junction.id for junction in flattened.junctions}

    assert "AmpBlock.in" in junction_ids
    assert "AmpBlock.DiffAmp.in_p" in junction_ids
    assert "amp-block$port_in" not in junction_ids
    assert "diff$port_in_p" not in junction_ids

    netlist_text, pin_assignment = schematic_to_netlist(schematic)

    assert "V1 n1 0 DC 1" in netlist_text
    assert "Rload n2 0 10k" in netlist_text
    assert "R1 n1 n2 1k" in netlist_text
    assert pin_assignment["cp:AmpBlock.DiffAmp.rdiff:p"] == "n1"
    assert pin_assignment["cp:AmpBlock.DiffAmp.rdiff:n"] == "n2"


def test_hierarchical_sram_cell_flattens_level1_mos_and_probeable_storage_nodes() -> None:
    pins = ["bl", "q", "wl", "blb", "qb", "vdd", "gnd", "init_q", "init_qb"]

    def j(name: str) -> dict[str, str]:
        return {"kind": "junction", "junction_id": name}

    def w(wire_id: str, start: dict[str, str], end: dict[str, str]) -> dict[str, object]:
        return {"id": wire_id, "start": start, "end": end}

    child_components = [
        {"id": "mpq", "type": "PMOS", "name": "MPQ", "value": "0.7m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "mpqb", "type": "PMOS", "name": "MPQB", "value": "0.7m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "mnq", "type": "NMOS", "name": "MNQ", "value": "1.8m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "mnqb", "type": "NMOS", "name": "MNQB", "value": "1.8m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "maxq", "type": "NMOS", "name": "MAXQ", "value": "2.5m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "maxqb", "type": "NMOS", "name": "MAXQB", "value": "2.5m", "value2": "0.4", "value3": "0.02", "metadata": {"model": "level1", "cgs": "2f", "cgd": "1f"}},
        {"id": "cq", "type": "C", "name": "CQ", "value": "4f"},
        {"id": "cqb", "type": "C", "name": "CQB", "value": "4f"},
        {"id": "rinitq", "type": "R", "name": "RINITQ", "value": "200Meg"},
        {"id": "rinitqb", "type": "R", "name": "RINITQB", "value": "200Meg"},
    ]
    child_junctions = [{"id": f"port_{pin}"} for pin in pins]
    child_wires = [
        w("w-mpq-s", _pin("mpq", "s"), j("port_vdd")),
        w("w-mpq-d", _pin("mpq", "d"), j("port_q")),
        w("w-mpq-g", _pin("mpq", "g"), j("port_qb")),
        w("w-mnq-d", _pin("mnq", "d"), j("port_q")),
        w("w-mnq-s", _pin("mnq", "s"), j("port_gnd")),
        w("w-mnq-g", _pin("mnq", "g"), j("port_qb")),
        w("w-mpqb-s", _pin("mpqb", "s"), j("port_vdd")),
        w("w-mpqb-d", _pin("mpqb", "d"), j("port_qb")),
        w("w-mpqb-g", _pin("mpqb", "g"), j("port_q")),
        w("w-mnqb-d", _pin("mnqb", "d"), j("port_qb")),
        w("w-mnqb-s", _pin("mnqb", "s"), j("port_gnd")),
        w("w-mnqb-g", _pin("mnqb", "g"), j("port_q")),
        w("w-maxq-d", _pin("maxq", "d"), j("port_q")),
        w("w-maxq-s", _pin("maxq", "s"), j("port_bl")),
        w("w-maxq-g", _pin("maxq", "g"), j("port_wl")),
        w("w-maxqb-d", _pin("maxqb", "d"), j("port_qb")),
        w("w-maxqb-s", _pin("maxqb", "s"), j("port_blb")),
        w("w-maxqb-g", _pin("maxqb", "g"), j("port_wl")),
        w("w-cq-p", _pin("cq", "p"), j("port_q")),
        w("w-cq-n", _pin("cq", "n"), j("port_gnd")),
        w("w-cqb-p", _pin("cqb", "p"), j("port_qb")),
        w("w-cqb-n", _pin("cqb", "n"), j("port_gnd")),
        w("w-rinitq-p", _pin("rinitq", "p"), j("port_q")),
        w("w-rinitq-n", _pin("rinitq", "n"), j("port_init_q")),
        w("w-rinitqb-p", _pin("rinitqb", "p"), j("port_qb")),
        w("w-rinitqb-n", _pin("rinitqb", "n"), j("port_init_qb")),
    ]

    schematic = SchematicDocument(
        components=[
            {"id": "gnd", "type": "GND", "name": "GND0"},
            {"id": "vdd-src", "type": "V", "name": "VDD", "subtype": "DC", "value": "1.2"},
            {"id": "cell", "type": "SUBCKT", "name": "Cell_0_0", "subcircuit_id": "sram-cell", "pins": pins},
        ],
        junctions=[
            {"id": "port_vdd"},
            {"id": "port_gnd"},
            {"id": "port_bl_0"},
            {"id": "port_blb_0"},
            {"id": "port_wl_0"},
            {"id": "port_q_0_0"},
            {"id": "port_qb_0_0"},
        ],
        wires=[
            w("w-vdd-p", _pin("vdd-src", "p"), j("port_vdd")),
            w("w-vdd-n", _pin("vdd-src", "n"), j("port_gnd")),
            w("w-gnd", _pin("gnd", "g"), j("port_gnd")),
            w("w-cell-vdd", _pin("cell", "vdd"), j("port_vdd")),
            w("w-cell-gnd", _pin("cell", "gnd"), j("port_gnd")),
            w("w-cell-bl", _pin("cell", "bl"), j("port_bl_0")),
            w("w-cell-blb", _pin("cell", "blb"), j("port_blb_0")),
            w("w-cell-wl", _pin("cell", "wl"), j("port_wl_0")),
            w("w-cell-q", _pin("cell", "q"), j("port_q_0_0")),
            w("w-cell-qb", _pin("cell", "qb"), j("port_qb_0_0")),
            w("w-cell-init-q", _pin("cell", "init_q"), j("port_gnd")),
            w("w-cell-init-qb", _pin("cell", "init_qb"), j("port_vdd")),
        ],
        subcircuits={
            "sram-cell": {
                "components": child_components,
                "junctions": child_junctions,
                "wires": child_wires,
            }
        },
        analysis={"mode": "tran", "params": {"t_stop": "6n", "t_step": "1n"}},
    )

    netlist_text, pin_assignment = schematic_to_netlist(schematic)

    assert "SUBCKT" not in netlist_text
    assert "NMOS LEVEL1" in netlist_text
    assert "PMOS LEVEL1" in netlist_text
    assert " q_0_0 " in f" {netlist_text} "
    assert " qb_0_0 " in f" {netlist_text} "
    assert pin_assignment["cp:Cell_0_0.mnq:d"] == "q_0_0"
    assert pin_assignment["cp:Cell_0_0.mnqb:d"] == "qb_0_0"


def test_hierarchy_rejects_child_port_names_that_do_not_match_instance_pins() -> None:
    schematic = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "DC", "value": "1"},
            {
                "id": "diff",
                "type": "SUBCKT",
                "name": "DiffAmp",
                "subcircuit_id": "diff-core",
                "pins": ["in_p", "out"],
            },
            {"id": "rload", "type": "R", "name": "Rload", "value": "10k"},
        ],
        wires=[
            {
                "id": "w-v-g",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
            {
                "id": "w-in",
                "start": {"kind": "component_pin", "component_id": "v1", "pin": "p"},
                "end": {"kind": "component_pin", "component_id": "diff", "pin": "in_p"},
            },
            {
                "id": "w-out",
                "start": {"kind": "component_pin", "component_id": "diff", "pin": "out"},
                "end": {"kind": "component_pin", "component_id": "rload", "pin": "p"},
            },
            {
                "id": "w-r-g",
                "start": {"kind": "component_pin", "component_id": "rload", "pin": "n"},
                "end": {"kind": "component_pin", "component_id": "gnd1", "pin": "g"},
            },
        ],
        subcircuits={
            "diff-core": {
                "components": [{"id": "rdiff", "type": "R", "name": "Rdiff", "value": "1k"}],
                "junctions": [{"id": "port_input"}, {"id": "port_out"}],
                "wires": [],
            },
        },
        analysis={"mode": "op", "params": {}},
    )

    with pytest.raises(NetlistError, match=r"DiffAmp\.in_p"):
        schematic_to_netlist(schematic)


def test_disconnected_pin_raises_error() -> None:
    disconnected = SchematicDocument(
        components=[
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "r1", "type": "R", "name": "R1", "value": "1k"},
        ],
        wires=[],
        analysis={"mode": "op", "params": {}},
    )
    with pytest.raises(NetlistError):
        schematic_to_netlist(disconnected)


def test_service_accepts_schematic_without_netlist_text() -> None:
    service = SimulationService()
    response = service.run_request(
        AnalysisRequest(
            netlist_text="",
            mode=AnalysisMode.OP,
            schematic=_simple_op_schematic(),
        )
    )

    assert response.status == "ok"
    assert response.dc_solution is not None
    assert response.metadata.get("generated_netlist")


# ---------------------------------------------------------------------------
# "Make subcircuit" in the web editor (app/web/src/lib/hierarchy.ts) moves a
# selection into a new child entity and wires a SUBCKT instance in its place.
# tests/fixtures/make_subcircuit_payloads.json holds the payloads the editor
# sends before and after that transform (written by app/web/src/lib/
# hierarchy.test.ts, which fails if the transform output drifts from the file).
# The hierarchical payload must simulate exactly like the flat original.


def _make_subcircuit_cases() -> list[dict]:
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "make_subcircuit_payloads.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["cases"]


def _terminal_nodes(schematic: SchematicDocument) -> dict[tuple[str, str], str]:
    """Net of every component terminal, keyed by the original component id."""
    _, pin_to_node = schematic_to_netlist(schematic, AnalysisMode.OP)
    nodes: dict[tuple[str, str], str] = {}
    for key, node in pin_to_node.items():
        _, component_id, pin = key.split(":", 2)
        # Components inside an instance flatten to "<instance>.<id>"; the
        # transform keeps the ids, so the last segment is the original id.
        nodes[(component_id.rsplit(".", 1)[-1], pin)] = node
    return nodes


def _original_terminals(case: dict) -> list[tuple[str, str]]:
    before = SchematicDocument.model_validate(case["before"])
    return sorted(_terminal_nodes(before))


@pytest.mark.parametrize("case_name", ["mixed", "ce_amplifier"])
def test_make_subcircuit_payload_flattens_to_the_same_circuit(case_name: str) -> None:
    case = next(c for c in _make_subcircuit_cases() if c["name"] == case_name)
    before = SchematicDocument.model_validate(case["before"])
    after = SchematicDocument.model_validate(case["after"])
    assert not any(c.type == "SUBCKT" for c in before.components)
    instances = [c for c in after.components if c.type == "SUBCKT"]
    assert [c.name for c in instances] == [case["instance"]]
    assert instances[0].subcircuit_id in after.subcircuits

    before_nodes = _terminal_nodes(before)
    after_nodes = _terminal_nodes(after)
    # Every terminal of the original circuit still exists ...
    assert set(before_nodes) <= set(after_nodes)
    # ... and two terminals share a net after the transform exactly when they
    # did before (net names may differ, the connectivity may not).
    keys = sorted(before_nodes)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            same_before = before_nodes[left] == before_nodes[right]
            same_after = after_nodes[left] == after_nodes[right]
            assert same_before == same_after, (left, right)
    for key in keys:
        assert (before_nodes[key] == "0") == (after_nodes[key] == "0"), key


@pytest.mark.parametrize("case_name", ["mixed", "ce_amplifier"])
def test_make_subcircuit_payload_has_the_same_operating_point(case_name: str) -> None:
    case = next(c for c in _make_subcircuit_cases() if c["name"] == case_name)
    service = SimulationService()
    voltages: list[dict[tuple[str, str], float]] = []
    for side in ("before", "after"):
        schematic = SchematicDocument.model_validate(case[side])
        response = service.run_request(AnalysisRequest(mode=AnalysisMode.OP, schematic=schematic))
        assert response.status == "ok"
        values = dict(zip(response.labels, response.dc_solution or []))
        nodes = _terminal_nodes(schematic)
        voltages.append(
            {key: 0.0 if node == "0" else values[f"V({node})"] for key, node in nodes.items()}
        )
    for key in _original_terminals(case):
        assert voltages[1][key] == pytest.approx(voltages[0][key], rel=1e-9, abs=1e-12), key


def test_make_subcircuit_payload_has_the_same_transient() -> None:
    case = next(c for c in _make_subcircuit_cases() if c["name"] == "ce_amplifier")
    service = SimulationService()
    traces: list[dict[tuple[str, str], list[float]]] = []
    times: list[list[float]] = []
    for side in ("before", "after"):
        schematic = SchematicDocument.model_validate(case[side])
        response = service.run_request(AnalysisRequest(mode=AnalysisMode.TRAN, schematic=schematic))
        assert response.status == "ok"
        waveform = response.waveform
        assert waveform is not None
        rows = dict(zip(waveform["labels"], waveform["values"]))
        times.append(list(waveform["time"]))
        nodes = _terminal_nodes(schematic)
        traces.append(
            {
                key: [0.0] * len(waveform["time"]) if node == "0" else list(rows[f"V({node})"])
                for key, node in nodes.items()
            }
        )
    assert times[1] == pytest.approx(times[0])
    for key in _original_terminals(case):
        assert traces[1][key] == pytest.approx(traces[0][key], rel=1e-7, abs=1e-9), key
