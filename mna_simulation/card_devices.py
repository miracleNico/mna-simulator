"""Model-card devices for the MNA builder (``HIERARCHY_AND_MODELS_PLAN.md`` 3.2 and 3.4, phase M2).

A ``ComponentRecord`` selects a SPICE equation set with ``metadata["model"]``:

* ``type="D"``, ``"spice"``: SPICE diode (``cpp/include/mna/devices/spice_diode.hpp``);
* ``type="QNPN"``/``"QPNP"``, ``"gummel_poon"``: Gummel-Poon BJT, DC part
  (``gummel_poon.hpp``);
* ``type="NMOS"``/``"PMOS"``, ``"spice_level1"``: Level-1 MOS with body effect for cards of
  LEVEL 1, 2 and 3 (``mos_card.hpp``).

The card parameters arrive as text in ``metadata`` (lowercase keys, SPICE values with metric
suffixes, see :func:`parse_card_value`). This module applies the SPICE defaults of plan 3.2 to
missing keys, the area / multiplier rules of plan 3.4, validates the values, names the internal
nodes of the series resistances (``<name>#a`` for a diode, ``<name>#b``, ``<name>#c``,
``<name>#e`` for a BJT) and derives the kernel parameters the builder puts into the native
tables. Keys outside the tables below are ignored here (the parser reports them).

Area and multiplier: ``metadata["area"]`` (default 1) scales ``is``, ``ibv``, ``cjo`` of a
diode and ``is``, ``ise``, ``isc``, ``ikf``, ``ikr``, ``cje``, ``cjc`` of a BJT up and ``rs``,
``rb``, ``re``, ``rc`` down. ``metadata["m"]`` (default 1) is the parallel multiplier of a BJT
(it multiplies the area) and of a MOS (``beta`` and the overlap capacitances); on a diode ``m``
is the card's grading coefficient (recorded, unused before phase M4), so a diode has no
instance multiplier besides ``area``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .api.contracts import ComponentRecord
from .errors import error_handler

#: k T / q at T = 300.15 K, the thermal voltage of every model-card equation (plan 3.4).
CARD_VT = 8.617333262e-5 * 300.15

INF = math.inf

#: ``metadata["model"]`` value of each component type that has a model-card equation set.
CARD_MODEL_OF_TYPE = {
    "D": "spice",
    "QNPN": "gummel_poon",
    "QPNP": "gummel_poon",
    "NMOS": "spice_level1",
    "PMOS": "spice_level1",
}
CARD_TYPES = frozenset(CARD_MODEL_OF_TYPE)

#: Native table of each equation set.
KIND_OF_MODEL = {"spice": "spice_diode", "gummel_poon": "gummel_poon", "spice_level1": "mos4"}

#: SPICE defaults (plan 3.2) of the parameters the M2 equations read or record.
DIODE_DEFAULTS: dict[str, float] = {
    "is": 1e-14,
    "n": 1.0,
    "rs": 0.0,
    "bv": INF,
    "ibv": 1e-3,
    "cjo": 0.0,
    "vj": 1.0,
    "m": 0.5,
    "fc": 0.5,
    "tt": 0.0,
}
GUMMEL_POON_DEFAULTS: dict[str, float] = {
    "is": 1e-16,
    "bf": 100.0,
    "nf": 1.0,
    "br": 1.0,
    "nr": 1.0,
    "vaf": INF,
    "var": INF,
    "ikf": INF,
    "ikr": INF,
    "ise": 0.0,
    "ne": 1.5,
    "isc": 0.0,
    "nc": 2.0,
    "rb": 0.0,
    "re": 0.0,
    "rc": 0.0,
    "cje": 0.0,
    "cjc": 0.0,
}
MOS_DEFAULTS: dict[str, float] = {
    "vto": 0.0,
    "kp": 2e-5,
    "w": 100e-6,
    "l": 100e-6,
    "lambda": 0.0,
    "gamma": 0.0,
    "phi": 0.6,
    "cgso": 0.0,
    "cgdo": 0.0,
    "cgbo": 0.0,
    "level": 1.0,
}

# SPICE scale factors (case-insensitive); letters after the scale factor are units and ignored.
_SCALE = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
_NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)")
_INFINITE = frozenset({"inf", "+inf", "infinity", "+infinity"})


def card_kind(component: ComponentRecord) -> str | None:
    """``"spice_diode"``, ``"gummel_poon"`` or ``"mos4"`` for a model-card component, else ``None``."""

    model = CARD_MODEL_OF_TYPE.get(component.type)
    if model is None:
        return None
    metadata = component.metadata
    if not metadata or metadata.get("model") != model:
        return None
    return KIND_OF_MODEL[model]


def parse_card_value(value: Any, component: ComponentRecord | None = None, key: str = "value") -> float:
    """A SPICE number: ``2.52n``, ``100MEG``, ``4pF``, ``1.5e-3``, ``10mil``; ``inf`` for infinity.

    Numbers pass through as ``float``. The scale factors are SPICE's (``T G MEG K M MIL U N P F``,
    case-insensitive, ``M`` is milli); letters after them are units and ignored, as in SPICE.
    The arithmetic is ``float(number) * factor`` like ``utils.parse_value``, so both agree on
    every literal they both accept.
    """

    if isinstance(value, bool):
        _bad_value(value, component, key)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        _bad_value(value, component, key)
    text = value.strip()
    if text.lower() in _INFINITE:
        return INF
    match = _NUMBER.fullmatch(text)
    if match is None:
        _bad_value(value, component, key)
    assert match is not None
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if not suffix:
        return number
    if suffix.startswith("meg"):
        return number * 1e6
    if suffix.startswith("mil"):
        return number * 25.4e-6
    return number * _SCALE.get(suffix[0], 1.0)


def _bad_value(value: Any, component: ComponentRecord | None, key: str) -> None:
    where = f"'{component.name}'{_model_text(component)}: " if component is not None else ""
    error_handler(f"NETLIST_FATAL: {where}cannot read parameter {key} = {value!r} as a number")


def _model_text(component: ComponentRecord | None) -> str:
    if component is None:
        return ""
    name = component.metadata.get("model_name") if component.metadata else None
    return f" (model {name})" if name else ""


@dataclass
class CardDevice:
    """One model-card component with its parsed card values and derived kernel parameters."""

    component: ComponentRecord
    kind: str
    #: card values after the SPICE defaults (before the area rules)
    card: dict[str, float]
    area: float
    multiplier: float
    #: kernel parameters (area rules applied)
    params: dict[str, float]
    #: terminal letter -> internal node name, in allocation order
    internal: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def model_name(self) -> str | None:
        name = self.component.metadata.get("model_name")
        return None if name is None else str(name)

    def summary(self) -> dict[str, Any]:
        """JSON-safe description for ``MnaProblem.metadata["card_devices"]``."""

        return {
            "name": self.name,
            "type": self.component.type,
            "model": self.component.metadata.get("model"),
            "model_name": self.model_name,
            "kind": self.kind,
            "internal_nodes": list(self.internal.values()),
        }


def _fail(component: ComponentRecord, message: str) -> None:
    error_handler(f"NETLIST_FATAL: '{component.name}'{_model_text(component)}: {message}")


def _read(component: ComponentRecord, defaults: dict[str, float], fallbacks: dict[str, Any]) -> dict[str, float]:
    metadata = component.metadata
    values: dict[str, float] = {}
    for key, default in defaults.items():
        raw = metadata.get(key)
        if raw is None:
            raw = fallbacks.get(key)
        values[key] = default if raw is None else parse_card_value(raw, component, key)
    return values


def _instance(component: ComponentRecord, key: str) -> float:
    raw = component.metadata.get(key)
    value = 1.0 if raw is None else parse_card_value(raw, component, key)
    if not (value > 0.0 and math.isfinite(value)):
        _fail(component, f"instance {key} = {value!r} must be a positive finite number")
    return value


def _require(component: ComponentRecord, values: dict[str, float], key: str, rule: str) -> None:
    value = values[key]
    if rule == "positive":
        ok = value > 0.0 and math.isfinite(value)
        text = "a positive finite number"
    elif rule == "positive_or_inf":
        ok = value > 0.0
        text = "positive (omit it for infinity)"
    elif rule == "non_negative":
        ok = value >= 0.0 and math.isfinite(value)
        text = "a finite number >= 0"
    elif rule == "non_negative_or_inf":
        ok = value >= 0.0
        text = ">= 0"
    else:  # finite
        ok = math.isfinite(value)
        text = "finite"
    if not ok:
        _fail(component, f"parameter {key} = {value!r} must be {text}")


def _diode(component: ComponentRecord) -> CardDevice:
    card = _read(component, DIODE_DEFAULTS, {"is": component.value})
    for key in ("is", "n", "ibv"):
        _require(component, card, key, "positive")
    _require(component, card, "bv", "positive_or_inf")
    for key in ("rs", "cjo"):
        _require(component, card, key, "non_negative")
    area = _instance(component, "area")
    params = {
        "is": card["is"] * area,
        "n": card["n"],
        "bv": card["bv"],
        "ibv": card["ibv"] * area,
        "rs": card["rs"] / area,
        "cjo": card["cjo"] * area,
    }
    device = CardDevice(component, "spice_diode", card, area, 1.0, params)
    if params["rs"] > 0.0:
        device.internal["a"] = f"{component.name}#a"
    return device


def _gummel_poon(component: ComponentRecord) -> CardDevice:
    card = _read(
        component,
        GUMMEL_POON_DEFAULTS,
        {"is": component.value, "bf": component.value2, "br": component.value3},
    )
    for key in ("vaf", "var", "ikf", "ikr"):
        if card[key] == 0.0:
            card[key] = INF  # SPICE: 0 means "not given"
    for key in ("is", "bf", "nf", "br", "nr", "ne", "nc"):
        _require(component, card, key, "positive")
    for key in ("vaf", "var", "ikf", "ikr"):
        _require(component, card, key, "positive_or_inf")
    for key in ("ise", "isc", "rb", "re", "rc", "cje", "cjc"):
        _require(component, card, key, "non_negative")
    area = _instance(component, "area")
    multiplier = _instance(component, "m")
    scale = area * multiplier
    params = {key: card[key] for key in ("bf", "nf", "br", "nr", "vaf", "var", "ne", "nc")}
    for key in ("is", "ise", "isc", "ikf", "ikr", "cje", "cjc"):
        params[key] = card[key] * scale
    for key in ("rb", "re", "rc"):
        params[key] = card[key] / scale
    device = CardDevice(component, "gummel_poon", card, area, multiplier, params)
    for terminal, key in (("b", "rb"), ("c", "rc"), ("e", "re")):
        if params[key] > 0.0:
            device.internal[terminal] = f"{component.name}#{terminal}"
    return device


def _mos(component: ComponentRecord) -> CardDevice:
    card = _read(component, MOS_DEFAULTS, {})
    level = card["level"]
    if level not in (1.0, 2.0, 3.0):
        _fail(component, f"MOS LEVEL {level:g} is not supported (LEVEL 1, 2 and 3 map to the Level-1 equations)")
    for key in ("kp", "w", "l", "phi"):
        _require(component, card, key, "positive")
    for key in ("vto", "lambda"):
        _require(component, card, key, "finite")
    for key in ("gamma", "cgso", "cgdo", "cgbo"):
        _require(component, card, key, "non_negative")
    multiplier = _instance(component, "m")
    params = {
        "beta": card["kp"] * card["w"] / card["l"] * multiplier,
        "vto": card["vto"],
        "lambda": card["lambda"],
        "gamma": card["gamma"],
        "phi": card["phi"],
        "cgs": card["cgso"] * card["w"] * multiplier,
        "cgd": card["cgdo"] * card["w"] * multiplier,
        "cgb": card["cgbo"] * card["l"] * multiplier,
    }
    return CardDevice(component, "mos4", card, 1.0, multiplier, params)


_PREPARE = {"spice_diode": _diode, "gummel_poon": _gummel_poon, "mos4": _mos}


def prepare_card_devices(components: list[ComponentRecord]) -> list[CardDevice]:
    """The model-card devices of ``components`` in component order (empty when there are none)."""

    devices: list[CardDevice] = []
    for component in components:
        if component.type not in CARD_TYPES:
            continue
        kind = card_kind(component)
        if kind is not None:
            devices.append(_PREPARE[kind](component))
    return devices
