"""Backend-owned schematic net-formation and conversion pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .api.contracts import (
    AnalysisMode,
    CircuitIR,
    SchematicComponent,
    SchematicDocument,
    SchematicEndpoint,
    SchematicJunction,
    SchematicWire,
)
from .errors import error_handler
from .netlist import parse_netlist_text

PIN_LAYOUT: dict[str, tuple[str, ...]] = {
    "R": ("p", "n"),
    "C": ("p", "n"),
    "L": ("p", "n"),
    "V": ("p", "n"),
    "I": ("p", "n"),
    "D": ("p", "n"),
    "GND": ("g",),
    "VCVS": ("p", "n", "cp", "cn"),
    "VCCS": ("p", "n", "cp", "cn"),
    "CCCS": ("p", "n"),
    "CCVS": ("p", "n"),
    "QNPN": ("c", "b", "e"),
    "QPNP": ("c", "b", "e"),
    "NMOS": ("d", "g", "s"),
    "PMOS": ("d", "g", "s"),
}

DEVICE_NAME_PATTERN = re.compile(r"^(R|C|L|V|I|D|E|G|F|H|Q|M)\w*$", re.IGNORECASE)
PORT_PREFIX = "port_"


@dataclass(slots=True)
class SchematicConversionResult:
    """Result bundle for schematic conversion."""

    circuit: CircuitIR
    netlist_text: str
    net_assignments: dict[str, str]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, key: str) -> None:
        if key not in self.parent:
            self.parent[key] = key
            self.rank[key] = 0

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _endpoint_key(endpoint: SchematicEndpoint) -> str:
    if endpoint.kind == "component_pin":
        return f"cp:{endpoint.component_id}:{endpoint.pin}"
    return f"jn:{endpoint.junction_id}"


def _logical_segment(component: SchematicComponent) -> str:
    raw = (component.name or component.id).strip()
    segment = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return segment or component.id


def _instance_prefix(parent_prefix: str, component: SchematicComponent) -> str:
    return f"{parent_prefix}{_logical_segment(component)}."


def _port_name_from_junction_id(junction_id: str) -> str | None:
    if not junction_id.lower().startswith(PORT_PREFIX):
        return None
    return junction_id[len(PORT_PREFIX) :]


def _safe_net_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return name or "node"


def _port_junction_id(instance_prefix: str, pin: str) -> str:
    return f"{instance_prefix}{pin}"


def _rename_junction_id(junction_id: str, prefix: str) -> str:
    if not prefix:
        return junction_id
    port_name = _port_name_from_junction_id(junction_id)
    if port_name is not None:
        return _port_junction_id(prefix, port_name)
    return f"{prefix}{junction_id}"


def _exported_port_names(schematic: SchematicDocument) -> set[str]:
    ports: set[str] = set()
    for junction in schematic.junctions:
        port_name = _port_name_from_junction_id(junction.id)
        if port_name is not None:
            ports.add(port_name)
    return ports


def _validate_subckt_ports(
    component: SchematicComponent,
    subcircuit: SchematicDocument,
    logical_prefix: str,
) -> None:
    expected = set(component.pins or [])
    actual = _exported_port_names(subcircuit)
    if expected == actual:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts: list[str] = []
    if missing:
        logical = ", ".join(_port_junction_id(logical_prefix, pin) for pin in missing)
        local = ", ".join(f"{PORT_PREFIX}{pin}" for pin in missing)
        parts.append(f"missing child ports {logical} (local junctions: {local})")
    if extra:
        logical = ", ".join(_port_junction_id(logical_prefix, pin) for pin in extra)
        parts.append(f"unexpected child ports {logical}")
    error_handler(
        f"NETLIST_FATAL: SUBCKT instance '{component.name or component.id}' port mismatch: "
        + "; ".join(parts)
    )


def _expected_pins(component_type: str, component: "SchematicComponent | None" = None) -> tuple[str, ...]:
    if component_type == "SUBCKT":
        if component is None or not component.pins:
            error_handler("NETLIST_FATAL: SUBCKT instance requires pins metadata.")
        return tuple(component.pins)  # type: ignore[arg-type,union-attr]
    if component_type not in PIN_LAYOUT:
        error_handler(f"NETLIST_FATAL: Unsupported schematic device type '{component_type}'.")
    return PIN_LAYOUT[component_type]


def _ensure_endpoint_valid(
    endpoint: SchematicEndpoint,
    component_lookup: dict[str, SchematicComponent],
    junction_ids: set[str],
) -> None:
    if endpoint.kind == "junction":
        if endpoint.junction_id not in junction_ids:
            error_handler(f"NETLIST_FATAL: Unknown junction '{endpoint.junction_id}'.")
        return

    assert endpoint.component_id is not None
    if endpoint.component_id not in component_lookup:
        error_handler(f"NETLIST_FATAL: Unknown component id '{endpoint.component_id}' in wire.")

    component = component_lookup[endpoint.component_id]
    expected = set(_expected_pins(component.type, component))
    if endpoint.pin not in expected:
        error_handler(
            f"NETLIST_FATAL: Component '{endpoint.component_id}' pin '{endpoint.pin}' is invalid for type '{component.type}'."
        )


def _build_analysis_line(
    mode: AnalysisMode | None,
    schematic: SchematicDocument,
    option_overrides: dict | None = None,
) -> str | None:
    selected_mode = mode or (schematic.analysis.mode if schematic.analysis else AnalysisMode.OP)
    params: dict[str, object] = {}
    if schematic.analysis:
        params.update(schematic.analysis.params)
    if option_overrides:
        params.update(option_overrides)

    if selected_mode == AnalysisMode.SHOW_MATRIX:
        return None
    if selected_mode == AnalysisMode.OP:
        return ".op"
    if selected_mode == AnalysisMode.TRAN:
        t_stop = params.get("t_stop", "10m")
        t_step = params.get("t_step", "0.1m")
        return f".tran {t_stop} {t_step}"
    if selected_mode == AnalysisMode.AC:
        f_start = params.get("f_start", "1")
        f_stop = params.get("f_stop", "1e6")
        points = params.get("points", "100")
        return f".ac {f_start} {f_stop} {points}"
    if selected_mode == AnalysisMode.HB:
        harmonics = params.get("harmonics", "16")
        if "time_window" in params:
            return f".hb {harmonics} {params['time_window']}"
        return f".hb {harmonics}"
    if selected_mode == AnalysisMode.DYN:
        # .dyn [speed] is executed as a .tran by the backend; the speed value only paces the client stream.
        # We emit a plain .tran directive so existing solvers run unchanged.
        t_stop = params.get("t_stop", "10m")
        t_step = params.get("t_step", "0.1m")
        return f".tran {t_stop} {t_step}"

    error_handler(f"NETLIST_FATAL: Unsupported analysis mode '{selected_mode}'.")


def _auto_name(component: SchematicComponent, counters: dict[str, int], used_names: set[str]) -> str:
    prefix = (
        "Q" if component.type in {"QNPN", "QPNP"}
        else "M" if component.type in {"NMOS", "PMOS"}
        else component.type
    )
    if prefix == "GND":
        return component.name or component.id

    if component.name and DEVICE_NAME_PATTERN.match(component.name) and component.name.upper().startswith(prefix):
        if component.name not in used_names:
            used_names.add(component.name)
            return component.name

    counters[prefix] = counters.get(prefix, 0) + 1
    candidate = f"{prefix}{counters[prefix]}"
    while candidate in used_names:
        counters[prefix] += 1
        candidate = f"{prefix}{counters[prefix]}"
    used_names.add(candidate)
    return candidate


def _build_component_line(
    component: SchematicComponent,
    nodes: dict[str, str],
    name: str,
) -> str:
    node1 = nodes.get("p", "")
    node2 = nodes.get("n", "")
    if component.type in {"R", "C", "L"}:
        return f"{name} {node1} {node2} {component.value}"
    if component.type == "D":
        value = component.value or "1e-15"
        return f"{name} {node1} {node2} {value}"
    if component.type in {"QNPN", "QPNP"}:
        collector = nodes.get("c", "")
        base = nodes.get("b", "")
        emitter = nodes.get("e", "")
        if component.metadata.get("model") == "level1":
            is_ = component.value or "1e-15"
            bf = component.value2 or "150"
            br = component.value3 or "3"
            vaf = component.metadata.get("vaf", "100")
            var = component.metadata.get("var", "25")
            cje = component.metadata.get("cje", "4p")
            cjc = component.metadata.get("cjc", "2p")
            rb = component.metadata.get("rb", "50")
            re = component.metadata.get("re", "0.5")
            rc = component.metadata.get("rc", "5")
            return f"{name} {collector} {base} {emitter} {component.type} LEVEL1 {is_} {bf} {br} {vaf} {var} {cje} {cjc} {rb} {re} {rc}"
        gm = component.value or "40m"
        r_pi = component.value2 or "2.5k"
        r_o = component.value3 or "100k"
        c_pi = component.metadata.get("cpi", "8p")
        c_mu = component.metadata.get("cmu", "3p")
        c_cs = component.metadata.get("ccs", "0")
        rb = component.metadata.get("rb", "0")
        re = component.metadata.get("re", "0")
        return f"{name} {collector} {base} {emitter} {component.type} {gm} {r_pi} {r_o} {c_pi} {c_mu} {c_cs} {rb} {re}"
    if component.type in {"NMOS", "PMOS"}:
        drain = nodes.get("d", "")
        gate = nodes.get("g", "")
        source = nodes.get("s", "")
        if component.metadata.get("model") == "level1":
            beta = component.value or "1m"
            vth = component.value2 or "0.4"
            lambda_ = component.value3 or "0.02"
            c_gs = component.metadata.get("cgs", "2f")
            c_gd = component.metadata.get("cgd", "1f")
            return f"{name} {drain} {gate} {source} {component.type} LEVEL1 {beta} {vth} {lambda_} {c_gs} {c_gd}"
        gm = component.value or "5m"
        r_o = component.value2 or "50k"
        c_gs = component.value3 or "5p"
        c_gd = component.metadata.get("cgd", "1p")
        g_mb = component.metadata.get("gmb", "0")
        c_bs = component.metadata.get("cbs", "0")
        c_bd = component.metadata.get("cbd", "0")
        return f"{name} {drain} {gate} {source} {component.type} {gm} {r_o} {c_gs} {c_gd} {g_mb} {c_bs} {c_bd}"
    if component.type == "VCVS":
        cp = nodes.get("cp", component.ctrl_node1 or "0")
        cn = nodes.get("cn", component.ctrl_node2 or "0")
        gain = component.value or "1"
        emitted_name = name if name.upper().startswith("E") else f"E{name}"
        return f"{emitted_name} {node1} {node2} VCVS {cp} {cn} {gain}"
    if component.type == "VCCS":
        cp = nodes.get("cp", component.ctrl_node1 or "0")
        cn = nodes.get("cn", component.ctrl_node2 or "0")
        gain = component.value or "1"
        emitted_name = name if name.upper().startswith("G") else f"G{name}"
        return f"{emitted_name} {node1} {node2} VCCS {cp} {cn} {gain}"
    if component.type == "CCCS":
        ctrl_src = component.ctrl_source or "V1"
        gain = component.value or "1"
        emitted_name = name if name.upper().startswith("F") else f"F{name}"
        return f"{emitted_name} {node1} {node2} CCCS {ctrl_src} {gain}"
    if component.type == "CCVS":
        ctrl_src = component.ctrl_source or "V1"
        gain = component.value or "1"
        emitted_name = name if name.upper().startswith("H") else f"H{name}"
        return f"{emitted_name} {node1} {node2} CCVS {ctrl_src} {gain}"
    if component.type in {"V", "I"}:
        subtype = (component.subtype or "DC").upper()
        if subtype in {"DC", "AC"}:
            return f"{name} {node1} {node2} {subtype} {component.value}"
        if subtype in {"SIN", "COS"}:
            if not component.value2:
                error_handler(f"NETLIST_FATAL: Source '{name}' subtype '{subtype}' requires value2.")
            return f"{name} {node1} {node2} {subtype} {component.value} {component.value2}"
        if subtype == "STEP":
            if not component.value2:
                error_handler(f"NETLIST_FATAL: Source '{name}' STEP requires value2.")
            if component.value3:
                return f"{name} {node1} {node2} STEP {component.value} {component.value2} {component.value3}"
            return f"{name} {node1} {node2} STEP {component.value} {component.value2}"
        if subtype == "FUNC":
            if not component.value2:
                error_handler(f"NETLIST_FATAL: Source '{name}' FUNC requires value2 expression.")
            period = component.value or "0"
            return f"{name} {node1} {node2} FUNC {period} {component.value2}"
        error_handler(f"NETLIST_FATAL: Unsupported source subtype '{subtype}' for '{name}'.")

    error_handler(f"NETLIST_FATAL: Unsupported schematic component type '{component.type}'.")


def _collect_pin_keys(components: Iterable[SchematicComponent]) -> set[str]:
    pin_keys: set[str] = set()
    for component in components:
        for pin in _expected_pins(component.type, component):
            pin_keys.add(f"cp:{component.id}:{pin}")
    return pin_keys


def _top_level_net_name_hints(schematic: SchematicDocument) -> dict[str, str]:
    """Return flattened endpoint keys that should keep root-canvas n-labels."""

    component_lookup = {component.id: component for component in schematic.components}
    junction_ids = {junction.id for junction in schematic.junctions}
    union_find = _UnionFind()
    ordered_keys: list[str] = []

    def add_key(key: str) -> None:
        union_find.add(key)
        if key not in ordered_keys:
            ordered_keys.append(key)

    for component in schematic.components:
        for pin in _expected_pins(component.type, component):
            add_key(f"cp:{component.id}:{pin}")
    for junction_id in junction_ids:
        add_key(f"jn:{junction_id}")

    connected_pin_keys: set[str] = set()
    for wire in schematic.wires:
        _ensure_endpoint_valid(wire.start, component_lookup, junction_ids)
        _ensure_endpoint_valid(wire.end, component_lookup, junction_ids)
        left_key = _endpoint_key(wire.start)
        right_key = _endpoint_key(wire.end)
        add_key(left_key)
        add_key(right_key)
        union_find.union(left_key, right_key)
        if left_key.startswith("cp:"):
            connected_pin_keys.add(left_key)
        if right_key.startswith("cp:"):
            connected_pin_keys.add(right_key)

    ground_roots: set[str] = set()
    for component in schematic.components:
        if component.type == "GND":
            ground_key = f"cp:{component.id}:g"
            if ground_key in connected_pin_keys:
                ground_roots.add(union_find.find(ground_key))

    root_to_name: dict[str, str] = {}
    for junction_id in junction_ids:
        port_name = _port_name_from_junction_id(junction_id)
        if port_name is None:
            continue
        key = f"jn:{junction_id}"
        if key not in union_find.parent:
            continue
        root = union_find.find(key)
        if root in ground_roots or root in root_to_name:
            continue
        root_to_name[root] = _safe_net_name(port_name)

    counter = 1
    for key in ordered_keys:
        root = union_find.find(key)
        if root in root_to_name or root in ground_roots:
            continue
        root_to_name[root] = f"n{counter}"
        counter += 1

    hints: dict[str, str] = {}
    for key in ordered_keys:
        root = union_find.find(key)
        name = root_to_name.get(root)
        if not name:
            continue
        if key.startswith("cp:"):
            _, component_id, pin = key.split(":", 2)
            component = component_lookup.get(component_id)
            if component and component.type == "SUBCKT":
                hints[f"jn:{_port_junction_id(_instance_prefix('', component), pin)}"] = name
            else:
                hints[key] = name
        else:
            hints[key] = name

    return hints


def _flatten_subcircuits(
    schematic: SchematicDocument,
    instance_prefix: str = "",
) -> SchematicDocument:
    """Recursively inline SUBCKT instances into a single flat SchematicDocument.

    Each instance's inner components/wires/junctions get namespaced by the
    logical instance path. Ports in the sub-schematic are represented locally
    as named junctions in the form ``port_<pinname>`` and flatten to
    ``InstanceName.pinname``. Outer wires touching the SUBCKT instance pin are
    rewritten to terminate on that logical port junction.
    """

    has_sub = any(c.type == "SUBCKT" for c in schematic.components)
    if not has_sub:
        if not instance_prefix:
            return schematic
        # Still need to namespace inner ids for non-root calls.
        renamed_components = [_rename_component(c, instance_prefix) for c in schematic.components]
        renamed_wires = [_rename_wire(w, instance_prefix) for w in schematic.wires]
        renamed_junctions = [
            SchematicJunction(id=_rename_junction_id(j.id, instance_prefix), position=j.position)
            for j in schematic.junctions
        ]
        return SchematicDocument(
            components=renamed_components,
            wires=renamed_wires,
            junctions=renamed_junctions,
            subcircuits=schematic.subcircuits,
        )

    components: list[SchematicComponent] = []
    wires: list[SchematicWire] = []
    junctions: list[SchematicJunction] = list(schematic.junctions)

    # First, namespace the outer schematic components/wires too (if called recursively).
    for c in schematic.components:
        if c.type == "SUBCKT":
            continue
        components.append(_rename_component(c, instance_prefix))
    for w in schematic.wires:
        # Only keep wires whose endpoints are NOT on SUBCKT components — those are rebuilt below.
        if _endpoint_on_subckt(w.start, schematic) or _endpoint_on_subckt(w.end, schematic):
            continue
        wires.append(_rename_wire(w, instance_prefix))
    if instance_prefix:
        junctions = [
            SchematicJunction(id=_rename_junction_id(j.id, instance_prefix), position=j.position)
            for j in junctions
        ]

    wire_counter = [len(wires) + 1]

    def new_wire_id() -> str:
        wid = f"{instance_prefix}auto_w{wire_counter[0]}"
        wire_counter[0] += 1
        return wid

    # For each SUBCKT instance, inline its subcircuit and stitch pins.
    used_child_prefixes: set[str] = set()
    for c in schematic.components:
        if c.type != "SUBCKT":
            continue
        if not c.subcircuit_id or c.subcircuit_id not in schematic.subcircuits:
            error_handler(
                f"NETLIST_FATAL: SUBCKT instance '{c.id}' references missing subcircuit '{c.subcircuit_id}'."
            )
        sub_prefix = _instance_prefix(instance_prefix, c)
        if sub_prefix in used_child_prefixes:
            error_handler(
                f"NETLIST_FATAL: Duplicate SUBCKT logical name '{_logical_segment(c)}' under '{instance_prefix or 'root'}'."
            )
        used_child_prefixes.add(sub_prefix)
        subcircuit = schematic.subcircuits[c.subcircuit_id]
        _validate_subckt_ports(c, subcircuit, sub_prefix)
        inner_flat = _flatten_subcircuits(
            SchematicDocument(
                components=subcircuit.components,
                wires=subcircuit.wires,
                junctions=subcircuit.junctions,
                subcircuits=schematic.subcircuits,
            ),
            instance_prefix=sub_prefix,
        )
        components.extend(inner_flat.components)
        wires.extend(inner_flat.wires)
        junctions.extend(inner_flat.junctions)

        # The SUBCKT block declares its external pins via `pins`; the child
        # schematic must expose exactly matching local junctions named
        # "port_<pinname>". Those child ports flatten to logical junction names
        # such as "DiffAmp.in_p".
        for pin in c.pins or []:
            port_junction_id = _port_junction_id(sub_prefix, pin)
            # Rewrite all outer wires that ended on this SUBCKT instance's pin to the port junction.
            for w in schematic.wires:
                for end in (w.start, w.end):
                    if (
                        end.kind == "component_pin"
                        and end.component_id == c.id
                        and end.pin == pin
                    ):
                        other = w.end if end is w.start else w.start
                        outer_endpoint = _subckt_port_endpoint(other, schematic, instance_prefix)
                        if outer_endpoint is None:
                            outer_endpoint = _rename_endpoint(other, instance_prefix)
                        wires.append(
                            SchematicWire(
                                id=new_wire_id(),
                                start=outer_endpoint,
                                end=SchematicEndpoint(kind="junction", junction_id=port_junction_id),
                            )
                        )

    flattened = SchematicDocument(
        components=components,
        wires=wires,
        junctions=junctions,
        analysis=schematic.analysis if not instance_prefix else None,
        title=schematic.title if not instance_prefix else None,
        subcircuits={},
    )
    return flattened


def _endpoint_on_subckt(endpoint: SchematicEndpoint, schematic: SchematicDocument) -> bool:
    if endpoint.kind != "component_pin":
        return False
    for c in schematic.components:
        if c.id == endpoint.component_id and c.type == "SUBCKT":
            return True
    return False


def _subckt_port_endpoint(
    endpoint: SchematicEndpoint,
    schematic: SchematicDocument,
    instance_prefix: str,
) -> SchematicEndpoint | None:
    if endpoint.kind != "component_pin" or endpoint.component_id is None or endpoint.pin is None:
        return None
    for component in schematic.components:
        if component.id == endpoint.component_id and component.type == "SUBCKT":
            return SchematicEndpoint(
                kind="junction",
                junction_id=_port_junction_id(_instance_prefix(instance_prefix, component), endpoint.pin),
            )
    return None


def _rename_component(c: SchematicComponent, prefix: str) -> SchematicComponent:
    if not prefix:
        return c
    return SchematicComponent(
        id=f"{prefix}{c.id}",
        type=c.type,
        name=(f"{prefix}{c.name}" if c.name else None),
        value=c.value,
        subtype=c.subtype,
        value2=c.value2,
        value3=c.value3,
        position=c.position,
        ctrl_node1=c.ctrl_node1,
        ctrl_node2=c.ctrl_node2,
        ctrl_source=(f"{prefix}{c.ctrl_source}" if c.ctrl_source else None),
        subcircuit_id=c.subcircuit_id,
        pins=list(c.pins) if c.pins else None,
        metadata=dict(c.metadata),
    )


def _rename_endpoint(endpoint: SchematicEndpoint, prefix: str) -> SchematicEndpoint:
    if not prefix:
        return endpoint
    if endpoint.kind == "component_pin":
        return SchematicEndpoint(
            kind="component_pin",
            component_id=f"{prefix}{endpoint.component_id}",
            pin=endpoint.pin,
        )
    return SchematicEndpoint(kind="junction", junction_id=_rename_junction_id(endpoint.junction_id, prefix))


def _rename_wire(w: SchematicWire, prefix: str) -> SchematicWire:
    if not prefix:
        return w
    return SchematicWire(
        id=f"{prefix}{w.id}",
        start=_rename_endpoint(w.start, prefix),
        end=_rename_endpoint(w.end, prefix),
    )


def schematic_to_netlist(
    schematic: SchematicDocument,
    mode_override: AnalysisMode | None = None,
    option_overrides: dict | None = None,
) -> tuple[str, dict[str, str]]:
    """Convert schematic graph data into canonical netlist text."""

    has_hierarchy = any(component.type == "SUBCKT" for component in schematic.components)
    net_name_hints = _top_level_net_name_hints(schematic)

    # Recursively inline subcircuits before running net formation.
    schematic = _flatten_subcircuits(schematic)

    component_lookup = {component.id: component for component in schematic.components}
    junction_ids = {junction.id for junction in schematic.junctions}
    pin_keys = _collect_pin_keys(schematic.components)

    union_find = _UnionFind()
    for key in pin_keys:
        union_find.add(key)
    for junction_id in junction_ids:
        union_find.add(f"jn:{junction_id}")

    connected_pin_keys: set[str] = set()
    for wire in schematic.wires:
        _ensure_endpoint_valid(wire.start, component_lookup, junction_ids)
        _ensure_endpoint_valid(wire.end, component_lookup, junction_ids)
        left_key = _endpoint_key(wire.start)
        right_key = _endpoint_key(wire.end)
        union_find.add(left_key)
        union_find.add(right_key)
        union_find.union(left_key, right_key)
        if left_key.startswith("cp:"):
            connected_pin_keys.add(left_key)
        if right_key.startswith("cp:"):
            connected_pin_keys.add(right_key)

    ground_roots: set[str] = set()
    for component in schematic.components:
        if component.type == "GND":
            ground_key = f"cp:{component.id}:g"
            if ground_key not in connected_pin_keys:
                error_handler(f"NETLIST_FATAL: Ground symbol '{component.id}' is not wired.")
            ground_roots.add(union_find.find(ground_key))

    if not ground_roots:
        error_handler("NETLIST_FATAL: Schematic requires at least one connected GND symbol.")

    root_to_net: dict[str, str] = {}
    used_net_names: set[str] = set()
    for key, net_name in net_name_hints.items():
        if key not in union_find.parent:
            continue
        root = union_find.find(key)
        if root in ground_roots or root in root_to_net:
            continue
        root_to_net[root] = net_name
        used_net_names.add(net_name)

    net_counter = 1
    internal_counter = 1
    for key in pin_keys:
        root = union_find.find(key)
        if root in root_to_net:
            continue
        if root in ground_roots:
            root_to_net[root] = "0"
        elif has_hierarchy:
            root_to_net[root] = f"x{internal_counter}"
            internal_counter += 1
        else:
            while f"n{net_counter}" in used_net_names:
                net_counter += 1
            root_to_net[root] = f"n{net_counter}"
            used_net_names.add(root_to_net[root])
            net_counter += 1

    pin_to_node: dict[str, str] = {}
    for key in pin_keys:
        pin_to_node[key] = root_to_net[union_find.find(key)]

    counters: dict[str, int] = {}
    used_names: set[str] = set()
    lines: list[str] = []

    for component in schematic.components:
        if component.type == "GND":
            continue

        expected = _expected_pins(component.type, component)
        nodes: dict[str, str] = {}
        for pin in expected:
            pin_key = f"cp:{component.id}:{pin}"
            if pin_key not in connected_pin_keys:
                error_handler(f"NETLIST_FATAL: Component '{component.id}' pin '{pin}' is disconnected.")
            nodes[pin] = pin_to_node[pin_key]

        if len(expected) >= 2:
            n1 = nodes[expected[0]]
            n2 = nodes[expected[1]]
            if n1 == n2:
                error_handler(f"NETLIST_FATAL: Component '{component.id}' has both pins on net '{n1}'.")
        name = _auto_name(component, counters, used_names)
        lines.append(_build_component_line(component, nodes, name))

    analysis_line = _build_analysis_line(mode_override, schematic, option_overrides)
    if analysis_line:
        lines.append(analysis_line)
    lines.append(".end")

    return "\n".join(lines), pin_to_node


def schematic_to_circuit_ir(
    schematic: SchematicDocument,
    mode_override: AnalysisMode | None = None,
    option_overrides: dict | None = None,
) -> SchematicConversionResult:
    """Convert schematic graph to validated CircuitIR through canonical netlist text."""

    netlist_text, pin_to_node = schematic_to_netlist(
        schematic=schematic,
        mode_override=mode_override,
        option_overrides=option_overrides,
    )
    circuit = parse_netlist_text(netlist_text)
    return SchematicConversionResult(circuit=circuit, netlist_text=netlist_text, net_assignments=pin_to_node)
