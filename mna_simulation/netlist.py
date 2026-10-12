"""Netlist parsing and validation."""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .api.contracts import AnalysisMode, CircuitIR, ComponentRecord, DirectiveRecord
from .errors import NetlistError, error_handler
from .spice.context import ParseContext
from .spice.model_cards import EQUATION_SETS, MOS_INSTANCE_KEYS, MULTIPLIER_CARD_TYPES, TEMPERATURE_KEYS
from .spice.sources import SourceError, fill_transient_defaults, parse_source_spec
from .spice.values import fine_tokens, format_number, is_value, merge_assignments, normalize_value
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
    # Node name: letters, digits and ``_``, plus ``.`` (flattened subcircuit nets such as
    # ``X1.out``) and ``+``/``-`` (vendor port names such as ``IN+``); "0" is ground.
    NODE_PATTERN = re.compile(r"^[A-Za-z0-9_.+\-]+$")
    DUTY_CYCLE = re.compile(r"^(\d+(\.\d*)?|\.\d+)(%)?$")

    @classmethod
    def from_parts(cls, parts: list[str], context: ParseContext | None = None) -> "Component":
        """Create and validate a component from netlist tokens.

        Supports two name forms:
          1. ``R1`` / ``Rload`` — type inferred from leading letter prefix.
          2. ``feedback:R`` — name and type explicitly decoupled.

        Both produce a ``Component`` with the requested ``type``. The remaining
        tokens are parsed exactly as before for that type; the SPICE forms (model
        references, source functions, instance parameters, unit letters) are read on
        top of that. ``context`` supplies the ``.model`` cards in scope, collects
        diagnostics and names the line in new error messages (a fresh
        :class:`~mna_simulation.spice.context.ParseContext` when omitted).
        """

        if not parts:
            error_handler("NETLIST_FATAL: Empty component definition.")
        if context is None:
            context = ParseContext()

        name, type_ = _name_and_type(parts[0])

        component = cls(name=name)
        component.type = type_

        if component.type in {"R", "C", "L"}:
            cls._parse_passive(component, parts, context)
        elif component.type in {"V", "I"}:
            cls._parse_source(component, parts, context)
        elif component.type in {"VCVS", "VCCS", "CCVS", "CCCS"}:
            cls._parse_dependent(component, parts, context)
        elif component.type == "D":
            cls._parse_diode(component, parts, context)
        elif component.type in {"QNPN", "QPNP"}:
            cls._parse_bjt(component, parts, context)
        elif component.type in {"NMOS", "PMOS"}:
            cls._parse_mos(component, parts, context)
        elif component.type == "B":
            cls._parse_behavioral(component, parts, context)
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
        # Only names starting with V, C or G have a multi-letter prefix; every other
        # prefix is the single first letter (same result as testing the prefixes
        # VCVS, VCCS, CCVS, CCCS, GND, R, C, L, V, I, E, G, F, H, D, B, Q, M in order).
        initial = upper_name[:1]
        if initial == "V":
            if upper_name.startswith("VCVS"):
                return "VCVS"
            if upper_name.startswith("VCCS"):
                return "VCCS"
            return "V"
        if initial == "C":
            if upper_name.startswith("CCVS"):
                return "CCVS"
            if upper_name.startswith("CCCS"):
                return "CCCS"
            return "C"
        if initial == "G":
            if upper_name.startswith("GND"):
                return "GND"
            return "VCCS"
        return _SINGLE_LETTER_TYPES.get(initial)

    @classmethod
    def _parse_passive(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        # R1 a b value                              (the dialect's own form)
        # R1 a b MODEL | R1 a b [value] [MODEL] [k=v ...]   (SPICE: model card, M=, TC1=, R=)
        if len(parts) == 4:
            token: str | None = parts[3]
            if not _independent_value_ok(token):
                if context.has_models and not is_value(token) and context.lookup_model(token) is not None:
                    token = None  # a model card: read below
                else:
                    token = normalize_value(token)
            if token is not None:
                component.node1, component.node2, component.value = parts[1], parts[2], token
                cls._validate_distinct_nodes(component)
                return
        if len(parts) < 4:
            error_handler(f"NETLIST_FATAL: '{component.name}' expected 4 parts, got {len(parts)}")
        component.node1, component.node2 = parts[1], parts[2]
        cls._validate_distinct_nodes(component)
        key = component.type.lower()  # type: ignore[union-attr]  # r / c / l
        value: str | None = None
        card = None
        options: dict[str, str] = {}
        for token in merge_assignments(parts[3:]):
            if "=" in token:
                option, _, option_value = token.partition("=")
                options[option.lower()] = option_value
            elif value is None and card is None and is_value(token):
                value = normalize_value(token)
            elif card is None and not is_value(token):
                card = context.require_model(token, component.name, (component.type,))  # type: ignore[arg-type]
            else:
                error_handler(f"NETLIST_FATAL: '{component.name}' expected 4 parts, got {len(parts)}")
        if value is None and key in options:
            value = normalize_value(options.pop(key))
        multiplier = options.pop("m", None)
        temperature = sorted(k for k in options if k in TEMPERATURE_KEYS or k in {"temp", "dtemp"})
        ignored = sorted(k for k in options if k not in temperature)
        if temperature:
            context.diagnose(
                f"{context.where()}: '{component.name}': temperature coefficients {', '.join(temperature)} are ignored"
            )
        if ignored:
            context.diagnose(f"{context.where()}: '{component.name}': instance parameters {', '.join(ignored)} are ignored")
        number: float | None = None
        if card is not None:
            card_value = card.params.get(key)
            if card.type in MULTIPLIER_CARD_TYPES:
                if value is None:
                    context.error(f"'{component.name}' has no value (model '{card.name}' only scales it)")
                if card_value is not None and parse_value(card_value) != 1.0:
                    number = parse_value(value) * parse_value(card_value)
            elif value is None:
                if card_value is None:
                    context.error(f"'{component.name}' has no value and model '{card.name}' gives no {key.upper()}=")
                value = card_value
        if value is None or not is_value(value):
            context.error(f"'{component.name}' has no value")
        if multiplier is not None:
            factor = parse_value(cls._option_value(multiplier, "M", component.name, context))
            if factor == 0.0:
                context.error(f"'{component.name}' has M=0")
            base = parse_value(value) if number is None else number
            number = base * factor if key == "c" else base / factor
        component.value = value if number is None else format_number(number)

    @staticmethod
    def _option_value(text: str, label: str, name: str, context: ParseContext) -> str:
        value = normalize_value(text)
        if not is_value(value):
            context.error(f"'{name}' has invalid {label} value '{text}'")
        return value

    # Source forms of the dialect itself and their token counts (``V1 a b DC 5``, ...).
    _LEGACY_SOURCE_PARTS = {"DC": (5,), "AC": (5,), "SIN": (6,), "COS": (6,), "STEP": (6, 7), "FUNC": (6,)}

    @classmethod
    def _parse_source(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        keyword = parts[3].upper() if len(parts) > 3 else ""
        legacy_counts = cls._LEGACY_SOURCE_PARTS.get(keyword)
        if (
            len(parts) < 4
            or (len(parts) == 4 and legacy_counts is not None)
            or keyword in {"COS", "STEP", "FUNC"}
            or (
                legacy_counts is not None
                and len(parts) in legacy_counts
                and (keyword != "SIN" or ("(" not in parts[4] and "(" not in parts[5]))
            )
        ):
            cls._parse_legacy_source(component, parts)
            return
        # SPICE forms: a bare DC value, DC/AC combinations, SIN(...), PULSE(...), PWL(...), EXP(...).
        component.node1 = parts[1]
        component.node2 = parts[2]
        cls._validate_distinct_nodes(component)
        try:
            spec = parse_source_spec(fine_tokens(" ".join(parts[3:])))
        except SourceError as exc:
            if exc.invalid_subtype is not None:
                error_handler(f"COMPONENT_ERROR: '{component.name}' has invalid subtype '{exc.invalid_subtype}'.")
            context.error(f"'{component.name}': {exc}")
        component.subtype = spec.subtype
        component.value = spec.value
        component.value2 = spec.value2
        component.metadata = spec.metadata
        if spec.needs_defaults:
            context.needs_defaults = True
        for message in spec.diagnostics:
            context.diagnose(f"{context.where()}: '{component.name}': {message}")

    @classmethod
    def _parse_legacy_source(cls, component: "Component", parts: list[str]) -> None:
        if len(parts) < 5:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected at least 5 parts, got {len(parts)}")

        component.node1 = parts[1]
        component.node2 = parts[2]
        component.subtype = parts[3].upper()
        cls._validate_distinct_nodes(component)

        if component.subtype == "DC":
            if len(parts) != 5:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 5 parts, got {len(parts)}")
            component.value = normalize_value(parts[4])
        elif component.subtype == "AC":
            if len(parts) != 5:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 5 parts, got {len(parts)}")
            component.value = normalize_value(parts[4])
        elif component.subtype in {"SIN", "COS"}:
            if len(parts) != 6:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 parts, got {len(parts)}")
            component.value = normalize_value(parts[4])
            component.value2 = normalize_value(parts[5])
        elif component.subtype == "STEP":
            if len(parts) not in {6, 7}:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 or 7 parts, got {len(parts)}")
            component.value = normalize_value(parts[4])
            component.value2 = normalize_value(parts[5])
            component.value3 = parts[6] if len(parts) == 7 else None
        elif component.subtype == "FUNC":
            if len(parts) != 6:
                error_handler(f"COMPONENT_ERROR: '{component.name}' expected 6 parts, got {len(parts)}")
            component.value = normalize_value(parts[4])
            component.value2 = parts[5].strip()
        else:
            error_handler(
                f"COMPONENT_ERROR: '{component.name}' has invalid subtype '{component.subtype}'."
            )

    @classmethod
    def _parse_dependent(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        # Accept both bare and SPICE-style forms:
        #   E1 n+ n- c+ c- gain               (6 tokens)
        #   E1 n+ n- VCVS c+ c- gain          (7 tokens, with explicit type)
        #   F1 n+ n- ctrl gain                (5 tokens)
        #   F1 n+ n- CCCS ctrl gain           (6 tokens, with explicit type)
        # Behavioural forms (TABLE, LAPLACE, VALUE, POLY, ...) are rejected by name.
        check_behavioural_element(parts, context)
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
            component.value = normalize_value(body[4])
            if component.ctrl_node1 == component.ctrl_node2:
                error_handler(
                    f"COMPONENT_ERROR: '{component.name}' control nodes must be different."
                )
        else:
            component.ctrl_source = body[2]
            context.ctrl_refs = True  # resolved case-insensitively once every name is known
            component.value = normalize_value(body[3])

    @classmethod
    def _parse_diode(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        #   D1 anode cathode [Is]                       (ideal Shockley diode of the dialect)
        #   D1 anode cathode MODEL [area] [AREA=] [M=]  (SPICE diode card)
        if len(parts) >= 4 and not _independent_value_ok(parts[3]) and not is_value(parts[3]):
            card = context.require_model(parts[3], component.name, ("D",))
            component.node1 = parts[1]
            component.node2 = parts[2]
            cls._validate_distinct_nodes(component)
            area, _ = cls._instance_area(component.name, parts[4:], context, multiplier=False)
            component.value = card.params.get("is", "1e-14")
            component.metadata = cls._card_metadata(card, area)
            return
        if len(parts) not in {3, 4}:
            error_handler(f"COMPONENT_ERROR: '{component.name}' expected 3 or 4 parts, got {len(parts)}")
        component.node1 = parts[1]
        component.node2 = parts[2]
        component.value = normalize_value(parts[3]) if len(parts) == 4 else I_S
        cls._validate_distinct_nodes(component)

    @staticmethod
    def _card_metadata(card, area: str | None = None) -> dict[str, str]:  # noqa: ANN001
        """``metadata`` of a device using ``card``: every card parameter plus the common keys."""

        metadata = dict(card.params)
        metadata["model"] = EQUATION_SETS[card.canonical_type]
        metadata["model_name"] = card.name
        if area is not None:
            metadata["area"] = area
        return metadata

    @classmethod
    def _instance_area(
        cls, name: str, tokens: list[str], context: ParseContext, multiplier: bool
    ) -> tuple[str | None, str | None]:
        """``(area, m)`` of a diode/BJT instance tail (``[area] [AREA=a] [M=m] [OFF] [IC=..]``).

        A BJT's ``M=`` is the instance multiplier (``metadata["m"]``). A diode has no
        instance multiplier besides the area (its ``metadata["m"]`` is the card's grading
        coefficient, ruling M2): ``M=`` on a diode line is dropped with a diagnostic.
        """

        area: str | None = None
        count: str | None = None
        ignored: list[str] = []
        for token in merge_assignments(tokens):
            if "=" in token:
                key, _, text = token.partition("=")
                key = key.lower()
                if key == "area":
                    area = cls._option_value(text, "area", name, context)
                elif key == "m" and multiplier:
                    count = cls._option_value(text, "M", name, context)
                elif key == "m":
                    context.diagnose(
                        f"{context.where()}: '{name}': instance multiplier M= dropped (a diode scales with its area "
                        f"factor; the card's M is the grading coefficient)"
                    )
                else:
                    ignored.append(key)
            elif token.upper() == "OFF":
                ignored.append("off")
            elif area is None and is_value(token):
                area = normalize_value(token)
            else:
                context.error(f"'{name}': unexpected '{token}' after the model name")
        if ignored:
            context.diagnose(f"{context.where()}: '{name}': instance parameters {', '.join(sorted(ignored))} are ignored")
        return area, count

    @staticmethod
    def _model_index(name: str, parts: list[str], context: ParseContext, keywords: frozenset[str]) -> int | None:
        """Index of the model token of a 3/4-terminal SPICE device line, ``None`` for the dialect's own forms.

        The model is the token after the three terminals (index 4) or after a fourth
        terminal (BJT substrate, MOS bulk; index 5). A token naming a card in scope wins;
        otherwise the dialect's keyword and numeric forms keep their meaning, and any
        other identifier is an undefined model (``NETLIST_FATAL`` naming it).
        """

        def model_like(token: str | None) -> bool:
            return token is not None and "=" not in token and not is_value(token)

        first = parts[4] if len(parts) > 4 else None
        if first is None:
            return None
        second = parts[5] if len(parts) > 5 else None
        if context.has_models:
            if model_like(first) and context.lookup_model(first) is not None:
                return 4
            if model_like(second) and context.lookup_model(second) is not None:  # type: ignore[arg-type]
                return 5
        if first.upper() in keywords:
            return None
        second_named = model_like(second) and second.upper() not in keywords | {"OFF"}  # type: ignore[union-attr]
        if is_value(first) and not second_named:
            return None
        missing = second if second_named else first
        context.error(f"model '{missing}' is not defined (referenced by '{name}')")
        return None  # unreachable

    @classmethod
    def _parse_bjt(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        #   Q1 collector base emitter [substrate] MODEL [area] [AREA=] [M=]   (SPICE Gummel-Poon card)
        index = None
        if context.has_models or (len(parts) > 4 and parts[4].upper() not in _BJT_KEYWORDS):
            index = cls._model_index(component.name, parts, context, _BJT_KEYWORDS)
        if index is not None:
            card = context.require_model(parts[index], component.name, ("NPN", "PNP"))
            component.type = "QNPN" if card.canonical_type == "NPN" else "QPNP"
            component.subtype = component.type
            component.node1 = parts[1]
            component.ctrl_node1 = parts[2]
            component.node2 = parts[3]
            if component.node1 == component.node2:
                error_handler(f"COMPONENT_ERROR: '{component.name}' collector and emitter must be different.")
            if index == 5:
                component.ctrl_node2 = parts[4]
                context.diagnose(
                    f"{context.where()}: '{component.name}': substrate node '{parts[4]}' is not modelled"
                )
            area, multiplier = cls._instance_area(component.name, parts[index + 1 :], context, multiplier=True)
            component.value = card.params.get("is", "1e-16")
            component.value2 = card.params.get("bf", "100")
            component.value3 = card.params.get("br", "1")
            component.metadata = cls._card_metadata(card, area)
            if multiplier is not None:
                component.metadata["m"] = multiplier
            return
        cls._parse_legacy_bjt(component, parts)

    @classmethod
    def _parse_legacy_bjt(cls, component: "Component", parts: list[str]) -> None:
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
    def _parse_mos(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        #   M1 drain gate source [bulk] MODEL [L=] [W=] [M=] [AD=] [AS=] [PD=] [PS=] [NRD=] [NRS=]
        #   (SPICE NMOS/PMOS card, LEVEL 1-3; a three-terminal instance has bulk = source)
        index = None
        if context.has_models or (len(parts) > 4 and parts[4].upper() not in _MOS_KEYWORDS):
            index = cls._model_index(component.name, parts, context, _MOS_KEYWORDS)
        if index is not None:
            card = context.require_model(parts[index], component.name, ("NMOS", "PMOS"))
            component.type = card.canonical_type
            component.subtype = component.type
            component.node1 = parts[1]
            component.ctrl_node1 = parts[2]
            component.node2 = parts[3]
            if component.node1 == component.node2:
                error_handler(f"COMPONENT_ERROR: '{component.name}' drain and source must be different.")
            if index == 5:
                component.ctrl_node2 = parts[4]
            metadata = cls._card_metadata(card)
            ignored: list[str] = []
            for token in merge_assignments(parts[index + 1 :]):
                key, separator, text = token.partition("=")
                key = key.lower()
                if separator and key in MOS_INSTANCE_KEYS:
                    metadata[key] = cls._option_value(text, key.upper(), component.name, context)
                elif separator or key == "off":
                    ignored.append(key)
                else:
                    context.error(f"'{component.name}': unexpected '{token}' after the model name")
            if ignored:
                context.diagnose(
                    f"{context.where()}: '{component.name}': instance parameters {', '.join(sorted(ignored))} are ignored"
                )
            component.metadata = metadata
            return
        cls._parse_legacy_mos(component, parts)

    @classmethod
    def _parse_legacy_mos(cls, component: "Component", parts: list[str]) -> None:
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
    def _parse_behavioral(cls, component: "Component", parts: list[str], context: ParseContext) -> None:
        check_behavioural_element(parts, context)
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
        _check_distinct_nodes(component.name, component.node1, component.node2)

    def validate(self) -> bool:
        """Validate node names and parameter shapes."""

        if self.type is None:
            error_handler(f"NETLIST_FATAL: '{self.name}' has unknown component type.")

        if self.type == "GND":
            if self.node1 is None or not _node_name_ok(self.node1):
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
            if node is None or not _node_name_ok(node):
                error_handler(
                    f"NETLIST_FATAL: '{self.name}' has invalid node name '{node}'. "
                    f"Use '0' for ground, or any alphanumeric identifier."
                )

        if self.type in {"V", "I"}:
            if self.subtype not in {"AC", "COS", "SIN", "DC", "FUNC", "STEP", "PULSE", "PWL", "EXP"}:
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid subtype '{self.subtype}'.")
            if self.metadata:
                # SPICE forms: every value was checked while the specification was read.
                return True
            if self.subtype in {"DC", "AC"} and not _independent_value_ok(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid value '{self.value}'.")
            if self.subtype in {"SIN", "COS", "STEP"}:
                if not _dependent_value_ok(self.value or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid source value '{self.value}'.")
                if not _dependent_value_ok(self.value2 or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid parameter '{self.value2}'.")
                if self.subtype == "STEP" and self.value3 is not None and not self.DUTY_CYCLE.match(self.value3):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid duty cycle '{self.value3}'.")
            return True

        if self.type in {"R", "C", "L", "D"}:
            if not _independent_value_ok(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid value format '{self.value}'.")
            return True

        if self.type in {"QNPN", "QPNP", "NMOS", "PMOS"} and self.metadata.get("model") in _CARD_EQUATION_SETS:
            # SPICE model cards: the card and instance values were checked when they were read.
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
                if value is not None and not _independent_value_ok(value):
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
                if value is None or not _dependent_value_ok(value):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid {label} value '{value}'.")
            return True

        if self.type in {"VCVS", "VCCS", "CCVS", "CCCS"}:
            if self.type in {"VCVS", "VCCS"} and not _dependent_value_ok(self.value or ""):
                error_handler(f"NETLIST_FATAL: '{self.name}' has invalid gain value '{self.value}'.")
            if self.type in {"CCVS", "CCCS"}:
                if not self.COMPONENT_PATTERN.match(self.ctrl_source or ""):
                    error_handler(f"NETLIST_FATAL: '{self.name}' has invalid controlling source '{self.ctrl_source}'.")
                if not _dependent_value_ok(self.value or ""):
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


# Single-letter name prefixes (``Component.get_type``); V, C and G also have the
# multi-letter prefixes VCVS/VCCS, CCVS/CCCS and GND.
_SINGLE_LETTER_TYPES = {
    "R": "R",
    "L": "L",
    "I": "I",
    "E": "VCVS",
    "F": "CCCS",
    "H": "CCVS",
    "D": "D",
    "B": "B",
    "Q": "QNPN",
    "M": "NMOS",
}

# Bound on each memo of regex results below (a long-lived server sees many node names).
_MATCH_MEMO_LIMIT = 8192


def _memoized_match(pattern: re.Pattern[str]):
    """``bool(pattern.match(token))`` with a per-process memo on ``str`` tokens.

    Netlists repeat node names and value literals many times; the regex runs once
    per distinct token. Non-``str`` arguments go straight to the regex (and raise
    what it raises).
    """

    memo: dict[str, bool] = {}
    match = pattern.match

    def matches(token: str) -> bool:
        if type(token) is not str:
            return match(token) is not None
        ok = memo.get(token)
        if ok is None:
            ok = match(token) is not None
            if len(memo) >= _MATCH_MEMO_LIMIT:
                memo.clear()
            memo[token] = ok
        return ok

    matches.memo = memo  # type: ignore[attr-defined]  # read inline by parse_netlist_text
    return matches


_node_name_ok = _memoized_match(Component.NODE_PATTERN)
_independent_value_ok = _memoized_match(Component.INDEPENDENT_VAL_PATTERN)
_dependent_value_ok = _memoized_match(Component.DEPENDENT_VAL_PATTERN)


def _name_and_type(name_token: str) -> tuple[str, str]:
    """Component name and normalized type of a name token (``R1``, ``Rload``, ``feedback:R``).

    Raises the ``COMPONENT_ERROR``s of an invalid name or an unknown prefix.
    """

    if type(name_token) is str and ":" not in name_token:
        explicit = None  # the explicit form needs a ':'
    else:
        explicit = Component.EXPLICIT_TYPE_PATTERN.match(name_token)
    if explicit is not None:
        return explicit.group(1), explicit.group(2).upper()
    # ASCII identifiers starting with a letter always match NAME_BARE; anything else
    # (Unicode letters that case-fold to ASCII under IGNORECASE, ...) asks the regex.
    if not (
        type(name_token) is str and name_token.isascii() and name_token.isidentifier() and name_token[0] != "_"
    ) and not Component.COMPONENT_PATTERN.match(name_token):
        error_handler(f"COMPONENT_ERROR: Invalid component name: '{name_token}'")
    type_ = Component.get_type(name_token)
    if type_ is None:
        error_handler(
            f"COMPONENT_ERROR: Cannot infer type from '{name_token}'. "
            f"Use '<name>:<TYPE>' to declare the type explicitly."
        )
    return name_token, type_


def _check_distinct_nodes(name: str, node1: str | None, node2: str | None) -> None:
    if node1 == node2:
        error_handler(f"COMPONENT_ERROR: '{name}' has conflict in nodes: '{node1, node2}'")


# ``metadata["model"]`` of the devices built from SPICE ``.model`` cards.
_CARD_EQUATION_SETS = frozenset(EQUATION_SETS.values())
# Tokens after the terminals that select the dialect's own transistor forms.
_BJT_KEYWORDS = frozenset({"QNPN", "QPNP", "LEVEL1", "SMALLSIG", "SMALL_SIGNAL", "HYBRID_PI"})
_MOS_KEYWORDS = frozenset({"NMOS", "PMOS", "LEVEL1", "SMALLSIG", "SMALL_SIGNAL"})

# SPICE element letters this simulator does not model: (what, fewest tokens of a real
# element line). Shorter lines keep the "Cannot infer type" error of the dialect.
UNSUPPORTED_ELEMENTS = {
    "J": ("JFETs (J elements)", 5),
    "Z": ("MESFETs (Z elements)", 5),
    "S": ("voltage-controlled switches (S elements)", 6),
    "W": ("current-controlled switches (W elements)", 5),
    "T": ("lossless transmission lines (T elements)", 5),
    "O": ("lossy transmission lines (O elements)", 6),
    "U": ("uniform RC lines (U elements)", 5),
    "K": ("coupled inductors (K elements)", 4),
    "P": ("coupled multiconductor lines (P elements)", 5),
    "Y": ("transmission lines (Y elements)", 5),
    "A": ("XSPICE code models (A elements)", 3),
    "N": ("Verilog-A devices (N elements)", 3),
}
# Behavioural controlled-source forms that would change the answer if ignored
# (``E1 o 0 TABLE {V(i)} = ...``, ``VALUE={...}``, ``POLY(2) ...``, ``LAPLACE=...``).
_BEHAVIOURAL_FORM = re.compile(r"(?<![\w.])(TABLE|LAPLACE|VALUE|POLY|FREQ|CHEBYSHEV)\s*[=({]", re.IGNORECASE)


def check_unsupported_element(name: str, parts: list[str], context: ParseContext) -> None:
    """``NETLIST_FATAL`` for a SPICE element type this simulator does not model (J, Z, S, W, T, O, U, K, ...)."""

    initial = name[:1].upper() if name[:1].isascii() else ""
    unsupported = UNSUPPORTED_ELEMENTS.get(initial)
    if unsupported is not None and len(parts) >= unsupported[1]:
        context.error(f"'{name}': {unsupported[0]} are not supported")


def check_behavioural_element(parts: list[str], context: ParseContext) -> None:
    """``NETLIST_FATAL`` for TABLE/LAPLACE/VALUE/POLY controlled sources and ``B ... V=``/``I=`` sources."""

    name = parts[0] if parts else ""
    initial = name[:1].upper()
    if initial == "B":
        if len(parts) >= 4 and re.match(r"^[VvIi]\s*=", " ".join(parts[3:])):
            context.error(f"'{name}': behavioural B sources with V=/I= expressions are not supported")
        return
    match = _BEHAVIOURAL_FORM.search(" ".join(parts[3:]))
    if match is not None:
        context.error(f"'{name}': {match.group(1).upper()} controlled sources are not supported")


@dataclass(slots=True)
class Netlist:
    """Validated netlist container.

    ``parse_netlist_text`` fills it with ``ComponentRecord``s (``add_component`` and
    ``validate`` read only the ``type``/node fields both classes share); ``to_ir``
    needs ``Component``s.
    """

    components: list[Component] = field(default_factory=list)
    voltage_source_node_sets: list[frozenset[str]] = field(default_factory=list)
    current_source_nodes: list[str] = field(default_factory=list)
    other_nodes: list[str] = field(default_factory=list)

    def add_component(self, component: Component | ComponentRecord) -> None:
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
            if component.ctrl_node2:  # substrate / bulk of a SPICE-card transistor
                self.other_nodes.append(component.ctrl_node2)
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

        normalized_v_sets = [
            frozenset(("0" if n in ground_aliases else n) for n in s) for s in self.voltage_source_node_sets
        ]
        for s in normalized_v_sets:
            if len(s) == 1 and "0" in s:
                error_handler(
                    "NETLIST_FATAL: Voltage source has both terminals on ground."
                )
        if normalized_v_sets and len(normalized_v_sets) > len(set(normalized_v_sets)):
            error_handler("NETLIST_FATAL: Voltage source parallel conflict: Two or more voltage sources are in parallel.")

        normalized_current_nodes = [("0" if n in ground_aliases else n) for n in self.current_source_nodes]
        normalized_other_nodes = {("0" if n in ground_aliases else n) for n in self.other_nodes}
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

    # The first four tokens of ``split()`` and ``split(maxsplit=5)`` are the same.
    parts = line.split()
    if len(parts) >= 4 and parts[3].upper() == "FUNC":
        return line.split(maxsplit=5)
    return parts


# First token after the terminals, for the types whose remaining tokens ("tail")
# are parsed and validated independently of the name and the node names.
_TAIL_START = {
    "R": 3,
    "C": 3,
    "L": 3,
    "B": 3,
    "D": 3,
    "V": 3,
    "I": 3,
    "NMOS": 4,
    "PMOS": 4,
    "QNPN": 4,
    "QPNP": 4,
}
# Node-order message of the three-terminal parsers (node1 == node2).
_SAME_TERMINALS_MESSAGE = {
    "NMOS": "drain and source must be different.",
    "PMOS": "drain and source must be different.",
    "QNPN": "collector and emitter must be different.",
    "QPNP": "collector and emitter must be different.",
}


# Directives handled by the SPICE front end (``spice.frontend``): the plain line loop
# restarts there when it meets one.
_FRONTEND_DIRECTIVES = frozenset(
    {".model", ".subckt", ".ends", ".param", ".include", ".inc", ".lib", ".endl", ".global", ".control", ".endc"}
)
# Directives that would change the answer if ignored: NETLIST_FATAL naming the line.
_FATAL_DIRECTIVES = {
    ".func": "user-defined functions are not supported (use .param or write the expression inline)",
    ".if": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".elseif": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".else": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".endif": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".alter": "alternative runs (.alter) are not supported",
}
# Harmless directives: ignored with a diagnostic.
_IGNORED_DIRECTIVES = {
    ".options": "simulator options are not supported",
    ".option": "simulator options are not supported",
    ".opt": "simulator options are not supported",
    ".temp": "the circuit is simulated at 27 C",
    ".nodeset": "node sets are not supported",
    ".ic": "initial conditions are not supported yet",
    ".probe": "outputs are selected by the client",
    ".print": "outputs are selected by the client",
    ".plot": "outputs are selected by the client",
    ".save": "outputs are selected by the client",
    ".width": "outputs are selected by the client",
    ".backanno": "back-annotation is not supported",
    ".step": "parameter sweeps are not supported",
    ".meas": "measurements are not supported",
    ".measure": "measurements are not supported",
    ".four": "Fourier analysis is not supported",
    ".noise": "noise analysis is not supported",
    ".tf": "transfer-function analysis is not supported",
    ".sens": "sensitivity analysis is not supported",
    ".pz": "pole-zero analysis is not supported",
    ".disto": "distortion analysis is not supported",
    ".title": "titles are not used",
}
# SPICE needs arguments for these: the bare directive stays an unsupported directive.
_DIRECTIVES_NEEDING_ARGUMENTS = frozenset(
    {".noise", ".tf", ".four", ".meas", ".measure", ".step", ".temp", ".ic", ".nodeset", ".print", ".plot", ".sens", ".pz", ".disto"}
)
# A line that only the SPICE front end reads: continuation, subcircuit instance, or a
# front-end directive (searched only after the plain loop failed).
_FRONTEND_MARKERS = re.compile(
    r"^[ \t]*(?:\+|[xX]|\.(?:model|subckt|ends|param|include|inc|lib|endl|global|control|endc)\b)",
    re.MULTILINE | re.IGNORECASE,
)


class _NeedsFrontend(Exception):
    """The plain line loop met a construct only the SPICE front end reads (it restarts there)."""


def parse_directive(line: str) -> DirectiveRecord:
    """Parse a single analysis directive line (``.op``, ``.dc``, ``.tran``, ``.ac``, ``.hb``)."""

    try:
        directive = _parse_control_line(line, ParseContext())
    except _NeedsFrontend:
        directive = None  # .model, .subckt, ...: netlist structure, not an analysis
    if directive is None:
        error_handler(f"NETLIST_FATAL: Unsupported directive '{line}'.")
    return directive


def _directive_value(token: str) -> float:
    return parse_value(normalize_value(token))


def _parse_control_line(line: str, context: ParseContext) -> DirectiveRecord | None:
    """A directive line: an analysis record, ``None`` for an ignored directive (diagnostic), or an error."""

    parts = line.split()
    command = parts[0].lower()

    if command == ".op":
        return DirectiveRecord(mode=AnalysisMode.OP, raw_line=line)

    if command == ".dc":
        if len(parts) > 1:
            context.diagnose(f"{context.where()}: '.dc' sweeps are not supported: the circuit runs as .op at its source values")
        return DirectiveRecord(mode=AnalysisMode.OP, raw_line=line)

    if command == ".tran":
        return _parse_tran(line, parts, context)

    if command == ".ac":
        return _parse_ac(line, parts, context)

    if command == ".hb":
        params: dict[str, float | int] = {"harmonics": 16}
        args = parts[1:]
        if len(args) == 1:
            if re.match(r"^\d+$", args[0]):
                params["harmonics"] = int(args[0])
            else:
                params["time_window"] = _directive_value(args[0])
        elif len(args) >= 2:
            params["harmonics"] = int(_directive_value(args[0]))
            params["time_window"] = _directive_value(args[1])
        return DirectiveRecord(mode=AnalysisMode.HB, params=params, raw_line=line)

    if command in _FRONTEND_DIRECTIVES:
        raise _NeedsFrontend(command)
    reason = _FATAL_DIRECTIVES.get(command)
    if reason is not None:
        context.error(f"{command}: {reason}")
    reason = _IGNORED_DIRECTIVES.get(command)
    if reason is not None and (len(parts) > 1 or command not in _DIRECTIVES_NEEDING_ARGUMENTS):
        context.diagnose(f"{context.where()}: '{command}' ignored ({reason})")
        return None

    error_handler(f"NETLIST_FATAL: Unsupported directive '{line}'.")


def _parse_tran(line: str, parts: list[str], context: ParseContext) -> DirectiveRecord:
    """``.tran <t_stop> [<t_step>]`` (the dialect's order) or SPICE ``.tran tstep tstop tstart [tmax]``."""

    args = parts[1:]
    if any(arg.lower() == "uic" for arg in args):
        args = [arg for arg in args if arg.lower() != "uic"]
        context.diagnose(f"{context.where()}: '.tran ... uic' ignored (initial conditions are not supported yet)")
    if len(args) in (1, 2):
        params = {
            "t_stop": _directive_value(args[0]),
            "t_step": _directive_value(args[1]) if len(args) == 2 else parse_value("0.01m"),
        }
        if len(args) == 2 and 0.0 < params["t_stop"] < params["t_step"]:
            context.diagnose(
                f"{context.where()}: '.tran {args[0]} {args[1]}' is read as .tran <t_stop> <t_step> "
                f"(t_stop={args[0]}, t_step={args[1]}); SPICE writes the step first"
            )
        return DirectiveRecord(mode=AnalysisMode.TRAN, params=params, raw_line=line)
    if len(args) in (3, 4):
        # Three or four values: SPICE order, .tran tstep tstop tstart [tmax].
        params = {"t_stop": _directive_value(args[1]), "t_step": _directive_value(args[0])}
        if _directive_value(args[2]) != 0.0:
            context.diagnose(f"{context.where()}: .tran start time {args[2]} ignored (results start at t = 0)")
        if len(args) == 4:
            context.diagnose(f"{context.where()}: .tran maximum step {args[3]} ignored (the step is fixed)")
        return DirectiveRecord(mode=AnalysisMode.TRAN, params=params, raw_line=line)
    error_handler("NETLIST_FATAL: Use .tran <time> (<step_time>).")


def _parse_ac(line: str, parts: list[str], context: ParseContext) -> DirectiveRecord:
    """``.ac <f_start> <f_stop> <points>`` (linear) or SPICE ``.ac dec|oct|lin <n> <f_start> <f_stop>``."""

    if len(parts) == 5 and parts[1].lower() in {"dec", "oct", "lin"}:
        sweep = parts[1].lower()
        count = _directive_value(parts[2])
        f_start = _directive_value(parts[3])
        f_stop = _directive_value(parts[4])
        if sweep == "lin":
            points = int(count)
        else:
            if f_start <= 0.0 or f_stop <= 0.0:
                context.error(f"'.ac {sweep}' needs positive start and stop frequencies")
            span = math.log10(f_stop / f_start) if sweep == "dec" else math.log2(f_stop / f_start)
            points = max(1, int(round(count * abs(span))) + 1)
            context.diagnose(
                f"{context.where()}: '{line}' runs as a linear sweep from {f_start:g} Hz to {f_stop:g} Hz "
                f"with {points} points (logarithmic sweeps are not supported)"
            )
        params = {"f_start": f_start, "f_stop": f_stop, "points": points}
        return DirectiveRecord(mode=AnalysisMode.AC, params=params, raw_line=line)
    if len(parts) != 4:
        error_handler("NETLIST_FATAL: Use .ac <start_freq> <stop_freq> <points>.")
    params = {
        "f_start": _directive_value(parts[1]),
        "f_stop": _directive_value(parts[2]),
        "points": int(_directive_value(parts[3])),
    }
    return DirectiveRecord(mode=AnalysisMode.AC, params=params, raw_line=line)


def parse_netlist_text(netlist_text: str, *, include_resolver=None) -> CircuitIR:  # noqa: ANN001
    """Parse multiline netlist text into a validated IR.

    The dialect of ``README.md`` ("Netlist dialect") is read in one pass by
    :func:`_parse_netlist_lines`. A netlist that uses the structural SPICE constructs
    (``+`` continuations, ``;``/``$`` comments, ``.model``, ``.subckt``/``X``,
    ``.param``/``{...}``, ``.include``/``.lib``, ``.global``) goes through the SPICE
    front end (``mna_simulation.spice.frontend``), which flattens it into the same
    records. ``include_resolver(path, including_file)`` returns ``(resolved_name, text)``
    for ``.include``/``.lib``; without one (netlist text, the HTTP API) those lines are
    a ``NETLIST_FATAL`` error asking for the library to be pasted into the netlist
    (``parse_netlist_file`` resolves files).

    Lines of the types in ``_TAIL_START`` that repeat the parameter tokens of an
    earlier line of the same type (``... PMOS LEVEL1 0.7m 0.4 0.02 2f 1f``, ``4f``,
    ``200Meg``) reuse that line's parsed fields: those tokens are parsed and
    validated independently of the name and the nodes, so only the name and the
    node checks run, in the order ``Component.from_parts`` runs them (same errors),
    followed by ``Netlist.add_component``'s bookkeeping. Every other line goes
    through ``Component.from_parts``.
    """

    if type(netlist_text) is str and ("{" in netlist_text or ";" in netlist_text or "$" in netlist_text):
        return _parse_spice_netlist(netlist_text, include_resolver)
    try:
        return _parse_netlist_lines(netlist_text)
    except _NeedsFrontend:
        return _parse_spice_netlist(netlist_text, include_resolver)
    except NetlistError:
        # A model card, subcircuit or parameter defined further down may be what the
        # failing line needs: the front end reads the whole netlist first.
        if type(netlist_text) is str and _FRONTEND_MARKERS.search(netlist_text):
            return _parse_spice_netlist(netlist_text, include_resolver)
        raise


def _parse_spice_netlist(netlist_text: str, include_resolver) -> CircuitIR:  # noqa: ANN001
    from .spice.frontend import parse_spice_netlist

    return parse_spice_netlist(netlist_text, include_resolver)


def _parse_netlist_lines(netlist_text: str) -> CircuitIR:
    """The single-pass parser of the netlist dialect (see :func:`parse_netlist_text`)."""

    context = ParseContext()
    netlist = Netlist()
    directives: list[DirectiveRecord] = []
    records = netlist.components
    append_record = records.append
    add_component = netlist.add_component
    voltage_source_node_sets = netlist.voltage_source_node_sets
    extend_current_source_nodes = netlist.current_source_nodes.extend
    extend_other_nodes = netlist.other_nodes.extend
    record_class = ComponentRecord
    node_name_ok = _node_name_ok
    node_name_known = _node_name_ok.memo.get  # type: ignore[attr-defined]
    single_letter_types = _SINGLE_LETTER_TYPES
    get_type = Component.get_type
    tail_start = _TAIL_START
    # (type, tail tokens) -> the first record parsed from them
    prototypes: dict[tuple[str, tuple[str, ...]], ComponentRecord] = {}
    lines = netlist_text.splitlines()
    line_iter = iter(lines)
    context.lines = (len(lines), line_iter)  # messages compute the current line number on demand

    for raw_line in line_iter:
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith("#"):
            continue
        if line[0] == ".":  # only such a line can lower() to ".end"
            if line.lower() == ".end":
                break
            directive = _parse_control_line(line, context)
            if directive is not None:
                directives.append(directive)
            continue

        parts = line.split()
        if len(parts) >= 4 and parts[3].upper() == "FUNC":
            parts = line.split(maxsplit=5)

        name = parts[0]
        # A bare ASCII identifier starting with a letter is a valid name (see
        # _name_and_type); its type comes from the prefix.
        if name.isascii() and name.isidentifier() and name[0] != "_":
            type_ = single_letter_types.get(name[0].upper()) or get_type(name)
            if type_ is None:
                if name[0] in "xX":
                    raise _NeedsFrontend(name)
                check_unsupported_element(name, parts, context)
                name, type_ = _name_and_type(name)  # raises "Cannot infer type"
        else:
            if name[0] == "+":
                raise _NeedsFrontend(name)
            name, type_ = _name_and_type(name)
        start = tail_start.get(type_)
        key = None
        if start is not None and len(parts) >= start:
            key = (type_, tuple(parts[start:]))
            prototype = prototypes.get(key)
            if prototype is not None:
                node1 = parts[1]
                if start == 4:
                    ctrl_node1 = parts[2]
                    node2 = parts[3]
                    if node1 == node2:
                        error_handler(f"COMPONENT_ERROR: '{name}' {_SAME_TERMINALS_MESSAGE[type_]}")
                    if node_name_known(node1) is not True and not node_name_ok(node1):
                        _invalid_node_name(name, node1)
                    if node_name_known(node2) is not True and not node_name_ok(node2):
                        _invalid_node_name(name, node2)
                    if node_name_known(ctrl_node1) is not True and not node_name_ok(ctrl_node1):
                        _invalid_node_name(name, ctrl_node1)
                else:
                    ctrl_node1 = None
                    node2 = parts[2]
                    if node1 == node2:
                        _check_distinct_nodes(name, node1, node2)
                    if node_name_known(node1) is not True and not node_name_ok(node1):
                        _invalid_node_name(name, node1)
                    if node_name_known(node2) is not True and not node_name_ok(node2):
                        _invalid_node_name(name, node2)
                final_type = prototype.type
                append_record(
                    record_class(
                        name,
                        final_type,
                        node1,
                        node2,
                        prototype.value,
                        prototype.subtype,
                        ctrl_node1,
                        None,
                        None,
                        prototype.value2,
                        prototype.value3,
                        dict(prototype.metadata),
                    )
                )
                # Netlist.add_component for these types (node tokens are non-empty,
                # so ``node or "0"`` is the node itself).
                if start == 4:
                    extend_other_nodes((node1, node2, ctrl_node1))
                elif final_type == "V":
                    voltage_source_node_sets.append(frozenset((node1, node2)))
                elif final_type == "I":
                    extend_current_source_nodes((node1, node2))
                else:
                    extend_other_nodes((node1, node2))
                continue

        record = Component.from_parts(parts, context).to_record()
        if context.needs_defaults:
            # A SPICE waveform waiting for the .tran defaults: never shared as a prototype.
            context.needs_defaults = False
            context.deferred.append(record)
        elif key is not None:
            prototypes[key] = record
        add_component(record)

    return finish_netlist(netlist, directives, netlist_text, context)


def finish_netlist(netlist: Netlist, directives: list[DirectiveRecord], source_text: str, context: ParseContext) -> CircuitIR:
    """Netlist-wide steps shared by both parsers, then the ``CircuitIR``.

    Node names are matched case-insensitively and keep the spelling of their first
    occurrence; controlling-source names resolve case-insensitively; SPICE waveforms
    get their ``.tran`` defaults; then the topological checks of ``Netlist.validate``.
    """

    records = netlist.components
    if not records:
        raise NetlistError("No components entered in netlist.")
    _fold_node_case(netlist)
    if context.ctrl_refs:
        _resolve_controlling_sources(records)
    if context.deferred:
        _fill_transient_defaults(context.deferred, directives)
    netlist.validate()
    return CircuitIR(components=records, directives=directives, source_text=source_text, diagnostics=context.diagnostics)


def _fold_node_case(netlist: Netlist) -> None:
    """Give every spelling of a node (``in``, ``IN``) the spelling of its first occurrence."""

    # Cheap common case: no node name has an upper-case letter, so no two spellings differ by case.
    joined = "".join(netlist.other_nodes) + "".join(netlist.current_source_nodes)
    joined += "".join(map("".join, netlist.voltage_source_node_sets))
    if joined == joined.lower():
        return
    names = set(netlist.other_nodes)
    names.update(netlist.current_source_nodes)
    names.update(*netlist.voltage_source_node_sets)
    if len(set(map(str.lower, names))) == len(names):
        return
    records = netlist.components
    canonical: dict[str, str] = {}
    for record in records:
        for node in (record.node1, record.node2, record.ctrl_node1, record.ctrl_node2):
            if node is not None:
                canonical.setdefault(node.lower(), node)
    for record in records:
        if record.node1 is not None:
            record.node1 = canonical[record.node1.lower()]
        if record.node2 is not None:
            record.node2 = canonical[record.node2.lower()]
        if record.ctrl_node1 is not None:
            record.ctrl_node1 = canonical[record.ctrl_node1.lower()]
        if record.ctrl_node2 is not None:
            record.ctrl_node2 = canonical[record.ctrl_node2.lower()]
        if record.type != "GND" and record.node1 is not None and record.node1 == record.node2:
            error_handler(f"COMPONENT_ERROR: '{record.name}' has conflict in nodes: '{record.node1, record.node2}'")
    fresh = Netlist()
    for record in records:
        fresh.add_component(record)
    netlist.voltage_source_node_sets[:] = fresh.voltage_source_node_sets
    netlist.current_source_nodes[:] = fresh.current_source_nodes
    netlist.other_nodes[:] = fresh.other_nodes


def _resolve_controlling_sources(records: list[ComponentRecord]) -> None:
    """``F1 a b v1 2`` controls through ``V1``: an inexact-case name maps to the branch of that name."""

    exact: set[str] = set()
    folded: dict[str, str] = {}
    for record in records:
        if record.type in {"V", "VCVS", "CCVS", "L"}:
            exact.add(record.name)
            folded.setdefault(record.name.lower(), record.name)
    for record in records:
        if record.type in {"CCCS", "CCVS"}:
            source = record.ctrl_source
            if source and source not in exact:
                record.ctrl_source = folded.get(source.lower(), source)


def _fill_transient_defaults(records: list[ComponentRecord], directives: list[DirectiveRecord]) -> None:
    """Omitted PULSE/EXP/SIN times take the step and stop time of the netlist's last ``.tran``."""

    tran = next((directive for directive in reversed(directives) if directive.mode == AnalysisMode.TRAN), None)
    if tran is None:
        return
    t_step = float(tran.params["t_step"])
    t_stop = float(tran.params["t_stop"])
    for record in records:
        frequency = fill_transient_defaults(record.metadata, record.subtype or "", t_step, t_stop)
        if record.subtype == "SIN":
            record.value2 = frequency


def _invalid_node_name(name: str, node: str) -> None:
    error_handler(
        f"NETLIST_FATAL: '{name}' has invalid node name '{node}'. "
        f"Use '0' for ground, or any alphanumeric identifier."
    )


def parse_netlist_file(path: str | Path, *, search_dirs=()) -> CircuitIR:  # noqa: ANN001
    """Parse a netlist from a `.cir`/`.sp` file.

    ``.include``/``.lib`` paths resolve relative to the including file, then in
    ``search_dirs`` (the CLI's ``--lib`` directories).
    """

    from .spice.frontend import FileIncludeResolver

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    resolver = FileIncludeResolver(search_dirs=search_dirs, root=file_path)
    return parse_netlist_text(text, include_resolver=resolver)


def select_analysis(circuit: CircuitIR, fallback: AnalysisMode = AnalysisMode.SHOW_MATRIX) -> DirectiveRecord:
    """Select the final directive, or use a fallback mode."""

    if circuit.directives:
        return circuit.directives[-1]
    return DirectiveRecord(mode=fallback, raw_line="")
