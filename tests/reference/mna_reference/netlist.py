"""Netlist parsing and validation."""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path

from .api.contracts import AnalysisMode, CircuitIR, ComponentRecord, DirectiveRecord
from .errors import NetlistError, error_handler
from .utils import parse_value

V_T = "0.025"
I_S = "1e-15"


@dataclass(slots=True)
class Component:
    """Validated component parsed from a netlist line."""

    name: str
    node1: str | None = None
    node2: str | None = None
    value: str | None = None
    subtype: str | None = None
    ctrl_node1: str | None = None
    ctrl_node2: str | None = None
    ctrl_source: str | None = None
    value2: str | None = None
    value3: str | None = None
    type: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    # Recognized type prefixes.
    # Order matters for `get_type` — list multi-letter prefixes (VCVS, etc.) before
    # their single-letter aliases so we don't classify "VCVS1" as a "V" source.
    TYPE = r"(QNPN|QPNP|NMOS|PMOS|VCVS|VCCS|CCVS|CCCS|GND|R|C|V|L|I|E|F|G|H|D|B|Q|M)"
    NUMBER = r"[+-]?(\d+(\.\d*)?|\.\d+)"
    SCIENTIFIC_NOTATION = r"([eE][+-]?\d+)"
    METRIC_SUFFIXES = r"(u|U|m|M|k|K|Meg|MEG|g|G|n|N|p|P|f|F|%)"
    # Names: any alphanumeric identifier. Optionally suffixed with ":TYPE" to
    # decouple the component name from its type prefix entirely.
    #   R1                — name=R1,   type=R (prefix-inferred)
    #   Rload             — name=Rload type=R (prefix-inferred)
    #   feedback:R        — name=feedback type=R (explicit)
    #   load_resistor:R   — name=load_resistor type=R (explicit)
    NAME_BARE = r"[A-Za-z][A-Za-z0-9_]*"
    NAME_EXPLICIT_TYPE = rf"({NAME_BARE}):({TYPE})"
    COMPONENT_PATTERN = re.compile(rf"^({NAME_BARE}|{NAME_EXPLICIT_TYPE})$", re.IGNORECASE)
    EXPLICIT_TYPE_PATTERN = re.compile(rf"^{NAME_EXPLICIT_TYPE}$", re.IGNORECASE)
    INDEPENDENT_VAL_PATTERN = re.compile(f"^{NUMBER}({SCIENTIFIC_NOTATION}|{METRIC_SUFFIXES})?$")
    DEPENDENT_VAL_PATTERN = re.compile(f"^{NUMBER}({SCIENTIFIC_NOTATION}|{METRIC_SUFFIXES})?$")
    # Node name: anything that's a plain identifier or "0" (ground).
    NODE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
    DUTY_CYCLE = re.compile(r"^(\d+(\.\d*)?|\.\d+)(%)?$")

    @classmethod
    def from_parts(cls, parts: list[str]) -> "Component":
        """Create and validate a component from netlist tokens.

        Supports two name forms:
          1. ``R1`` / ``Rload`` — type inferred from leading letter prefix.
          2. ``feedback:R`` — name and type explicitly decoupled.

        Both produce a ``Component`` with the requested ``type``. The remaining
        tokens are parsed exactly as before for that type.
        """

        if not parts:
            error_handler("NETLIST_FATAL: Empty component definition.")

        name_token = parts[0]
        explicit = cls.EXPLICIT_TYPE_PATTERN.match(name_token)
        if explicit is not None:
            name = explicit.group(1)
            type_ = explicit.group(2).upper()
        else:
            if not cls.COMPONENT_PATTERN.match(name_token):
                error_handler(f"COMPONENT_ERROR: Invalid component name: '{name_token}'")
            name = name_token
            type_ = cls.get_type(name)
            if type_ is None:
                error_handler(
                    f"COMPONENT_ERROR: Cannot infer type from '{name_token}'. "
                    f"Use '<name>:<TYPE>' to declare the type explicitly."
                )

        component = cls(name=name)
        component.type = type_

        if component.type in {"R", "C", "L"}:
            cls._parse_passive(component, parts)
        elif component.type in {"V", "I"}:
            cls._parse_source(component, parts)
        elif component.type in {"VCVS", "VCCS", "CCVS", "CCCS"}:
            cls._parse_dependent(component, parts)
        elif component.type == "D":
            cls._parse_diode(component, parts)
        elif component.type in {"QNPN", "QPNP"}:
            cls._parse_bjt(component, parts)
        elif component.type in {"NMOS", "PMOS"}:
            cls._parse_mos(component, parts)
        elif component.type == "B":
            cls._parse_behavioral(component, parts)
        elif component.type == "GND":
            cls._parse_ground(component, parts)
        else:
            error_handler(f"NETLIST_FATAL: Unsupported component type '{component.type}'.")

        component.validate()
        return component

    @staticmethod
    def get_type(name: str) -> str | None:
        """Infer the normalized type from the device name prefix.

        Multi-letter prefixes (VCVS, VCCS, CCVS, CCCS, GND) are checked first
        so that names like ``GND1`` resolve to type ``GND`` rather than the
        single-letter alias ``G`` (VCCS).
        """

        upper_name = name.upper()
        for prefix, normalized in (
            ("VCVS", "VCVS"),
            ("VCCS", "VCCS"),
            ("CCVS", "CCVS"),
            ("CCCS", "CCCS"),
            ("GND", "GND"),
            ("R", "R"),
            ("C", "C"),
            ("L", "L"),
            ("V", "V"),
            ("I", "I"),
            ("E", "VCVS"),
            ("G", "VCCS"),
            ("F", "CCCS"),
            ("H", "CCVS"),
            ("D", "D"),
            ("B", "B"),
            ("Q", "QNPN"),
            ("M", "NMOS"),
        ):
            if upper_name.startswith(prefix):
                return normalized
        return None

    @classmethod
    def _parse_passive(cls, component: "Component", parts: list[str]) -> None:
        if len(parts) != 4:
            error_handler(f"NETLIST_FATAL: '{component.name}' expected 4 parts, got {len(parts)}")
        component.node1, component.node2, component.value = parts[1], parts[2], parts[3]
        cls._validate_distinct_nodes(component)

    @classmethod
    def _parse_source(cls, component: "Component", parts: list[str]) -> None:
        if len(parts) < 5:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected at least 5 parts, got {len(parts)}")

        component.node1 = parts[1]
        component.node2 = parts[2]
        component.subtype = parts[3].upper()
        cls._validate_distinct_nodes(component)

        if component.subtype == "DC":
            if len(parts) != 5:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 5 parts, got {len(parts)}")
            component.value = parts[4]
        elif component.subtype == "AC":
            if len(parts) != 5:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 5 parts, got {len(parts)}")
            component.value = parts[4]
        elif component.subtype in {"SIN", "COS"}:
            if len(parts) != 6:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 parts, got {len(parts)}")
            component.value = parts[4]
            component.value2 = parts[5]
        elif component.subtype == "STEP":
            if len(parts) not in {6, 7}:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 or 7 parts, got {len(parts)}")
            component.value = parts[4]
            component.value2 = parts[5]
            component.value3 = parts[6] if len(parts) == 7 else None
        elif component.subtype == "FUNC":
            if len(parts) != 6:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 parts, got {len(parts)}")
            component.value = parts[4]
            component.value2 = parts[5].strip()
        else:
            error_handler(
                f"COMPONENT_ERROR: '{component.name}' has invalid subtype '{component.subtype}'."
            )

    @classmethod
    def _parse_dependent(cls, component: "Component", parts: list[str]) -> None:
        # Accept both bare and SPICE-style forms:
        #   E1 n+ n- c+ c- gain               (6 tokens)
        #   E1 n+ n- VCVS c+ c- gain          (7 tokens, with explicit type)
        #   F1 n+ n- ctrl gain                (5 tokens)
        #   F1 n+ n- CCCS ctrl gain           (6 tokens, with explicit type)
        type_token = component.type or ""
        body = parts[1:]
        if body and body[2:3] and body[2].upper() == type_token:
            body = body[:2] + body[3:]

        if component.type in {"VCVS", "VCCS"}:
            expected = 5
        else:
            expected = 4
        if len(body) != expected:
            error_handler(
                f"COMPONENT_ERROR: '{component.name}' expected {expected + 1} or "
                f"{expected + 2} parts, got {len(parts)}"
            )

        component.node1 = body[0]
        component.node2 = body[1]
        cls._validate_distinct_nodes(component)
        if component.type in {"VCVS", "VCCS"}:
            component.ctrl_node1 = body[2]
            component.ctrl_node2 = body[3]
            component.value = body[4]
            if component.ctrl_node1 == component.ctrl_node2:
                error_handler(
                    f"COMPONENT_ERROR: '{component.name}' control nodes must be different."
                )
        else:
            component.ctrl_source = body[2]
            component.value = body[3]

    @classmethod
    def _parse_diode(cls, component: "Component", parts: list[str]) -> None:
        if len(parts) not in {3, 4}:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected 3 or 4 parts, got {len(parts)}")
        component.node1 = parts[1]
        component.node2 = parts[2]
        component.value = parts[3] if len(parts) == 4 else I_S
        cls._validate_distinct_nodes(component)

    @classmethod
    def _parse_bjt(cls, component: "Component", parts: list[str]) -> None:
        # Physical Level-1 BJT:
        #   Q1 collector base emitter QNPN LEVEL1 is bf br vaf var cje cjc rb re rc
        #   Q1 collector base emitter LEVEL1 is bf br vaf var cje cjc rb re rc
        # Small-signal hybrid-pi BJT:
        #   Q1 collector base emitter QNPN gm rpi [ro cpi cmu ccs rb re]
        #   Q1 collector base emitter QNPN SMALLSIG gm rpi [ro cpi cmu ccs rb re]
        #   Q1 collector base emitter gm rpi [ro cpi cmu ccs rb re]       (defaults to QNPN)
        if len(parts) < 5 or len(parts) > 16:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected 5 to 16 parts, got {len(parts)}")
        component.node1 = parts[1]
        component.ctrl_node1 = parts[2]
        component.node2 = parts[3]
        if component.node1 == component.node2:
            error_handler(f"COMPONENT_ERROR: '{component.name}' collector and emitter must be different.")
        token = parts[4].upper() if len(parts) > 4 else ""
        if token in {"QNPN", "QPNP"}:
            bjt_type = parts[4].upper()
            component.type = bjt_type
            component.subtype = bjt_type
            cursor = 5
        else:
            component.subtype = component.type or "QNPN"
            cursor = 4

        model_token = parts[cursor].upper() if len(parts) > cursor else ""
        if model_token == "LEVEL1":
            defaults = ("1e-15", "150", "3", "100", "25", "4p", "2p", "50", "0.5", "5")
            params = parts[cursor + 1 :]
            if len(params) > len(defaults):
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected up to 10 Level-1 BJT params.")
            values = list(params) + list(defaults[len(params) :])
            component.metadata["model"] = "level1"
            component.value = values[0]
            component.value2 = values[1]
            component.value3 = values[2]
            component.metadata["vaf"] = values[3]
            component.metadata["var"] = values[4]
            component.metadata["cje"] = values[5]
            component.metadata["cjc"] = values[6]
            component.metadata["rb"] = values[7]
            component.metadata["re"] = values[8]
            component.metadata["rc"] = values[9]
            return

        if model_token in {"SMALLSIG", "SMALL_SIGNAL", "HYBRID_PI"}:
            cursor += 1
        params = parts[cursor:]
        if len(params) < 2 or len(params) > 8:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected gm/rpi plus up to 6 model params.")
        component.metadata["model"] = "small_signal"
        component.value = params[0]
        component.value2 = params[1]
        component.value3 = params[2] if len(params) >= 3 else "100k"
        component.metadata["cpi"] = params[3] if len(params) >= 4 else "8p"
        component.metadata["cmu"] = params[4] if len(params) >= 5 else "3p"
        component.metadata["ccs"] = params[5] if len(params) >= 6 else "0"
        component.metadata["rb"] = params[6] if len(params) >= 7 else "0"
        component.metadata["re"] = params[7] if len(params) >= 8 else "0"

    @classmethod
    def _parse_mos(cls, component: "Component", parts: list[str]) -> None:
        # Physical Level-1 MOS:
        #   M1 drain gate source NMOS LEVEL1 beta vth lambda cgs cgd
        # Small-signal MOS:
        #   M1 drain gate source NMOS gm ro cgs cgd [gmb cbs cbd]
        #   M1 drain gate source gm ro cgs cgd      (defaults to NMOS)
        if len(parts) < 8 or len(parts) > 13:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected 8 to 13 parts, got {len(parts)}")
        component.node1 = parts[1]
        component.ctrl_node1 = parts[2]
        component.node2 = parts[3]
        if component.node1 == component.node2:
            error_handler(f"COMPONENT_ERROR: '{component.name}' drain and source must be different.")
        if len(parts) == 11 and parts[4].upper() in {"NMOS", "PMOS"} and parts[5].upper() == "LEVEL1":
            mos_type = parts[4].upper()
            component.type = mos_type
            component.subtype = mos_type
            component.metadata["model"] = "level1"
            component.value = parts[6]
            component.value2 = parts[7]
            component.value3 = parts[8]
            component.metadata["cgs"] = parts[9]
            component.metadata["cgd"] = parts[10]
            return
        if len(parts) >= 10 and parts[4].upper() in {"NMOS", "PMOS"} and parts[5].upper() in {"SMALLSIG", "SMALL_SIGNAL"}:
            mos_type = parts[4].upper()
            component.type = mos_type
            component.subtype = mos_type
            params = parts[6:]
        elif len(parts) >= 9 and parts[4].upper() in {"NMOS", "PMOS"}:
            mos_type = parts[4].upper()
            component.type = mos_type
            component.subtype = mos_type
            params = parts[5:]
        elif len(parts) >= 9 and parts[4].upper() in {"SMALLSIG", "SMALL_SIGNAL"}:
            component.subtype = component.type or "NMOS"
            params = parts[5:]
        else:
            component.subtype = component.type or "NMOS"
            params = parts[4:]
        if len(params) < 4 or len(params) > 7:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected gm/ro/cgs/cgd plus up to 3 model params.")
        component.metadata["model"] = "small_signal"
        component.value = params[0]
        component.value2 = params[1]
        component.value3 = params[2]
        component.metadata["cgd"] = params[3]
        component.metadata["gmb"] = params[4] if len(params) >= 5 else "0"
        component.metadata["cbs"] = params[5] if len(params) >= 6 else "0"
        component.metadata["cbd"] = params[6] if len(params) >= 7 else "0"

    @classmethod
    def _parse_behavioral(cls, component: "Component", parts: list[str]) -> None:
        if len(parts) != 4:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected 4 parts, got {len(parts)}")
        component.node1 = parts[1]
        component.node2 = parts[2]
        component.value = parts[3]
        cls._validate_distinct_nodes(component)

    @classmethod
    def _parse_ground(cls, component: "Component", parts: list[str]) -> None:
        """``GND <node>`` — declares ``<node>`` to be the ground reference (alias of '0').

        ``GND<name> <node>`` is also accepted to allow multiple ground stamps.
        ``GND 0`` is allowed and is a no-op.
        """

        if len(parts) != 2:
            error_handler(
                f"COMPONENT_ERROR: '{component.name}' GND statement expects '<node>' "
                f"(got {len(parts) - 1} arg(s))."
            )
        component.node1 = parts[1]
        component.node2 = "0"
        component.value = "0"

    @staticmethod
    def _validate_distinct_nodes(component: "Component") -> None:
        if component.node1 == component.node2:
            error_handler(
                f"COMPONENT_ERROR: '{component.name}' has conflict in nodes: '{component.node1, component.node2}'"
            )

    def validate(self) -> bool:
        """Validate node names and parameter shapes."""

        if self.type is None:
            error_handler(f"NETLIST_FATAL: '{self.name}' has unknown component type.")

        if self.type == "GND":
            if self.node1 is None or not self.NODE_PATTERN.match(self.node1):
                error_handler(
                    f"NETLIST_FATAL: '{self.name}' has invalid GND node name '{self.node1}'."
                )
            return True

        nodes_to_check = [self.node1, self.node2]
        if self.ctrl_node1 is not None:
            nodes_to_check.append(self.ctrl_node1)
        if self.ctrl_node2 is not None:
            nodes_to_check.append(self.ctrl_node2)

        for node in nodes_to_check:
            if node is None or not self.NODE_PATTERN.match(node):
                error_handler(
                    f"NETLIST_FATAL: '{self.name}' has invalid node name '{node}'. "
                    f"Use '0' for ground, or any alphanumeric identifier."
                )

        if self.type in {"V", "I"}:
            if self.subtype not in {"AC", "COS", "SIN", "DC", "FUNC", "STEP"}:
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid subtype '{self.subtype}'.")
            if self.subtype in {"DC", "AC"} and not self.INDEPENDENT_VAL_PATTERN.match(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid value '{self.value}'.")
            if self.subtype in {"SIN", "COS", "STEP"}:
                if not self.DEPENDENT_VAL_PATTERN.match(self.value or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid source value '{self.value}'.")
                if not self.DEPENDENT_VAL_PATTERN.match(self.value2 or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid parameter '{self.value2}'.")
                if self.subtype == "STEP" and self.value3 is not None and not self.DUTY_CYCLE.match(self.value3):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid duty cycle '{self.value3}'.")
            return True

        if self.type in {"R", "C", "L", "D"}:
            if not self.INDEPENDENT_VAL_PATTERN.match(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid value format '{self.value}'.")
            return True

        if self.type in {"QNPN", "QPNP"}:
            if self.metadata.get("model") == "level1":
                values = (
                    ("is", self.value),
                    ("bf", self.value2),
                    ("br", self.value3),
                    ("vaf", self.metadata.get("vaf")),
                    ("var", self.metadata.get("var")),
                    ("cje", self.metadata.get("cje")),
                    ("cjc", self.metadata.get("cjc")),
                    ("rb", self.metadata.get("rb")),
                    ("re", self.metadata.get("re")),
                    ("rc", self.metadata.get("rc")),
                )
            else:
                values = (
                    ("gm", self.value),
                    ("rpi", self.value2),
                    ("ro", self.value3),
                    ("cpi", self.metadata.get("cpi")),
                    ("cmu", self.metadata.get("cmu")),
                    ("ccs", self.metadata.get("ccs")),
                    ("rb", self.metadata.get("rb")),
                    ("re", self.metadata.get("re")),
                )
            for label, value in values:
                if value is not None and not self.INDEPENDENT_VAL_PATTERN.match(value):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid {label} value '{value}'.")
            return True

        if self.type in {"NMOS", "PMOS"}:
            if self.metadata.get("model") == "level1":
                values = (
                    ("beta", self.value),
                    ("vth", self.value2),
                    ("lambda", self.value3),
                    ("cgs", self.metadata.get("cgs")),
                    ("cgd", self.metadata.get("cgd")),
                )
            else:
                values = (
                    ("gm", self.value),
                    ("ro", self.value2),
                    ("cgs", self.value3),
                    ("cgd", self.metadata.get("cgd")),
                    ("gmb", self.metadata.get("gmb")),
                    ("cbs", self.metadata.get("cbs")),
                    ("cbd", self.metadata.get("cbd")),
                )
            for label, value in values:
                if value is None or not self.DEPENDENT_VAL_PATTERN.match(value):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid {label} value '{value}'.")
            return True

        if self.type in {"VCVS", "VCCS", "CCVS", "CCCS"}:
            if self.type in {"VCVS", "VCCS"} and not self.DEPENDENT_VAL_PATTERN.match(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid gain value '{self.value}'.")
            if self.type in {"CCVS", "CCCS"}:
                if not self.COMPONENT_PATTERN.match(self.ctrl_source or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid controlling source '{self.ctrl_source}'.")
                if not self.DEPENDENT_VAL_PATTERN.match(self.value or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid gain value '{self.value}'.")
            return True

        if self.type == "B":
            if not self.value:
                error_handler(f"NETLIST_FATAL: '{self.name}' requires an expression.")
            return True

        error_handler(f"NETLIST_FATAL: '{self.name}'. Could not validate component.")

    def to_record(self) -> ComponentRecord:
        """Convert to the backend-neutral component record."""

        return ComponentRecord(
            name=self.name,
            type=self.type or "",
            node1=self.node1,
            node2=self.node2,
            value=self.value,
            subtype=self.subtype,
            ctrl_node1=self.ctrl_node1,
            ctrl_node2=self.ctrl_node2,
            ctrl_source=self.ctrl_source,
            value2=self.value2,
            value3=self.value3,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class Netlist:
    """Validated netlist container."""

    components: list[Component] = field(default_factory=list)
    voltage_source_node_sets: list[frozenset[str]] = field(default_factory=list)
    current_source_nodes: list[str] = field(default_factory=list)
    other_nodes: list[str] = field(default_factory=list)

    def add_component(self, component: Component) -> None:
        self.components.append(component)
        if component.type == "GND":
            # GND lines participate in node-name discovery but don't drive
            # source-conflict checks.
            self.other_nodes.append(component.node1 or "0")
            return
        if component.type in {"V", "CCVS", "VCVS"}:
            self.voltage_source_node_sets.append(frozenset([component.node1 or "0", component.node2 or "0"]))
            if component.ctrl_node1:
                self.other_nodes.extend([component.ctrl_node1, component.ctrl_node2 or "0"])
        elif component.type in {"I", "VCCS", "CCCS"}:
            self.current_source_nodes.extend([component.node1 or "0", component.node2 or "0"])
            if component.ctrl_node1:
                self.other_nodes.extend([component.ctrl_node1, component.ctrl_node2 or "0"])
        elif component.type in {"QNPN", "QPNP", "NMOS", "PMOS"}:
            self.other_nodes.extend([component.node1 or "0", component.node2 or "0", component.ctrl_node1 or "0"])
        else:
            self.other_nodes.extend([component.node1 or "0", component.node2 or "0"])

    def validate(self) -> bool:
        """Validate netlist-wide topological constraints.

        Ground aliases declared via ``GND <node>`` lines are normalized to "0"
        before topological checks so parallel/series source conflicts that span
        renamed ground rails are still detected.
        """

        ground_aliases: set[str] = {"0"}
        for component in self.components:
            if component.type == "GND" and component.node1:
                ground_aliases.add(component.node1)

        def alias(node: str) -> str:
            return "0" if node in ground_aliases else node

        normalized_v_sets = [
            frozenset(alias(n) for n in s) for s in self.voltage_source_node_sets
        ]
        for s in normalized_v_sets:
            if len(s) == 1 and "0" in s:
                error_handler(
                    "NETLIST_FATAL: Voltage source has both terminals on ground."
                )
        if normalized_v_sets and len(normalized_v_sets) > len(set(normalized_v_sets)):
            error_handler("NETLIST_FATAL: Voltage source parallel conflict: Two or more voltage sources are in parallel.")

        normalized_current_nodes = [alias(n) for n in self.current_source_nodes]
        normalized_other_nodes = {alias(n) for n in self.other_nodes}
        if normalized_current_nodes:
            counts = collections.Counter(normalized_current_nodes)
            junction_nodes = {node for node, count in counts.items() if count > 1}
            conflict_nodes = junction_nodes - junction_nodes.intersection(normalized_other_nodes)
            if conflict_nodes:
                error_handler(
                    f"NETLIST_FATAL: Current source series conflict: Node(s) {conflict_nodes} are the only junctions for series current sources."
                )
        return True

    def to_ir(self, directives: list[DirectiveRecord], source_text: str) -> CircuitIR:
        """Convert to the shared IR."""

        return CircuitIR(
            components=[component.to_record() for component in self.components],
            directives=directives,
            source_text=source_text,
        )


def _split_component_line(line: str) -> list[str]:
    """Split a component line while preserving FUNC expressions as one tail."""

    parts = line.split(maxsplit=5)
    if len(parts) >= 4 and parts[3].upper() == "FUNC":
        return parts
    return line.split()


def parse_directive(line: str) -> DirectiveRecord:
    """Parse a single simulator directive line."""

    parts = line.split()
    command = parts[0].lower()

    if command == ".op" or command == ".dc":
        return DirectiveRecord(mode=AnalysisMode.OP, raw_line=line)

    if command == ".tran":
        if len(parts) not in {2, 3}:
            error_handler("NETLIST_FATAL: Use .tran <time> (<step_time>).")
        params = {"t_stop": parse_value(parts[1]), "t_step": parse_value(parts[2]) if len(parts) == 3 else parse_value("0.01m")}
        return DirectiveRecord(mode=AnalysisMode.TRAN, params=params, raw_line=line)

    if command == ".ac":
        if len(parts) != 4:
            error_handler("NETLIST_FATAL: Use .ac <start_freq> <stop_freq> <points>.")
        params = {
            "f_start": parse_value(parts[1]),
            "f_stop": parse_value(parts[2]),
            "points": int(parse_value(parts[3])),
        }
        return DirectiveRecord(mode=AnalysisMode.AC, params=params, raw_line=line)

    if command == ".hb":
        params: dict[str, float | int] = {"harmonics": 16}
        args = parts[1:]
        if len(args) == 1:
            if re.match(r"^\d+$", args[0]):
                params["harmonics"] = int(args[0])
            else:
                params["time_window"] = parse_value(args[0])
        elif len(args) >= 2:
            params["harmonics"] = int(parse_value(args[0]))
            params["time_window"] = parse_value(args[1])
        return DirectiveRecord(mode=AnalysisMode.HB, params=params, raw_line=line)

    error_handler(f"NETLIST_FATAL: Unsupported directive '{line}'.")


def parse_netlist_text(netlist_text: str) -> CircuitIR:
    """Parse multiline netlist text into a validated IR."""

    netlist = Netlist()
    directives: list[DirectiveRecord] = []
    lines = netlist_text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith("#"):
            continue
        if line.lower() == ".end":
            break
        if line.startswith("."):
            directives.append(parse_directive(line))
            continue

        parts = _split_component_line(line)
        component = Component.from_parts(parts)
        netlist.add_component(component)

    if not netlist.components:
        raise NetlistError("No components entered in netlist.")

    netlist.validate()
    return netlist.to_ir(directives=directives, source_text=netlist_text)


def parse_netlist_file(path: str | Path) -> CircuitIR:
    """Parse a netlist from a `.cir`/`.sp` file."""

    text = Path(path).read_text(encoding="utf-8")
    return parse_netlist_text(text)


def select_analysis(circuit: CircuitIR, fallback: AnalysisMode = AnalysisMode.SHOW_MATRIX) -> DirectiveRecord:
    """Select the final directive, or use a fallback mode."""

    if circuit.directives:
        return circuit.directives[-1]
    return DirectiveRecord(mode=fallback, raw_line="")
