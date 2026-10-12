"""SPICE ``.model`` cards: supported parameter lists and card parsing.

The parser copies every card parameter into the device record's ``metadata``
(lowercase key, value as text) and never does device physics; the builder applies the
SPICE defaults and derives kernel parameters (``HIERARCHY_AND_MODELS_PLAN.md``
section 3.2). The lists below are that section's tables:

* ``SUPPORTED_KEYS[card type]``: every parameter the tables name (the equations use it
  now or a later phase records it). A referenced card's other parameters are reported
  once per card in ``CircuitIR.diagnostics`` ("... are not used by the diode equations").
* ``MODELLED_KEYS[card type]``: the subset the current device equations evaluate; the
  remaining supported keys of a referenced card are reported once as recorded but not
  modelled yet (transit times, temperature, Level-2/3 terms, ...).
* ``TEMPERATURE_KEYS``: temperature coefficients of the passive cards (a diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .values import is_value, normalize_value

DIODE_MODELLED = frozenset({"is", "n", "rs", "bv", "ibv", "cjo"})
DIODE_LATER = frozenset({"vj", "m", "fc", "tt", "eg", "xti", "tnom"})

BJT_MODELLED = frozenset(
    {"is", "bf", "nf", "br", "nr", "vaf", "var", "ikf", "ikr", "ise", "ne", "isc", "nc", "rb", "re", "rc", "cje", "cjc"}
)
BJT_LATER = frozenset(
    {
        "rbm",
        "irb",
        "vje",
        "mje",
        "vjc",
        "mjc",
        "xcjc",
        "fc",
        "tf",
        "tr",
        "xtf",
        "vtf",
        "itf",
        "ptf",
        "xtb",
        "eg",
        "xti",
        "tnom",
    }
)

# ``w``/``l`` on a card (some vendor cards carry them) share the instance keys in
# ``metadata``: the instance's ``W=``/``L=`` replace them, otherwise they act as defaults.
# ``uo``/``tox``/``nsub`` give ``kp``/``phi``/``gamma`` when a card omits those (SPICE).
MOS_MODELLED = frozenset(
    {"level", "vto", "kp", "lambda", "gamma", "phi", "cgso", "cgdo", "cgbo", "w", "l", "uo", "tox", "nsub"}
)
MOS_LATER = frozenset(
    {
        "cbd",
        "cbs",
        "cj",
        "cjsw",
        "pb",
        "mj",
        "mjsw",
        "js",
        "nss",
        "tpg",
        "xj",
        "ld",
        "vmax",
        "theta",
        "eta",
        "kappa",
        "delta",
        "nfs",
        "rsh",
        "rd",
        "rs",
    }
)
#: MOS instance parameters copied into ``metadata`` (``M1 d g s b NMOD L=.. W=..``).
MOS_INSTANCE_KEYS = ("w", "l", "m", "ad", "as", "pd", "ps", "nrd", "nrs")

TEMPERATURE_KEYS = frozenset({"tc1", "tc2", "tce", "tc", "tnom", "t_abs", "t_measured", "t_rel_global", "t_rel_local"})

MODELLED_KEYS: dict[str, frozenset[str]] = {
    "D": DIODE_MODELLED,
    "NPN": BJT_MODELLED,
    "PNP": BJT_MODELLED,
    "NMOS": MOS_MODELLED,
    "PMOS": MOS_MODELLED,
    "R": frozenset({"r"}),
    "C": frozenset({"c"}),
    "L": frozenset({"l"}),
}
SUPPORTED_KEYS: dict[str, frozenset[str]] = {
    "D": DIODE_MODELLED | DIODE_LATER,
    "NPN": BJT_MODELLED | BJT_LATER,
    "PNP": BJT_MODELLED | BJT_LATER,
    "NMOS": MOS_MODELLED | MOS_LATER,
    "PMOS": MOS_MODELLED | MOS_LATER,
    "R": frozenset({"r"}) | TEMPERATURE_KEYS,
    "C": frozenset({"c"}) | TEMPERATURE_KEYS,
    "L": frozenset({"l"}) | TEMPERATURE_KEYS,
}

#: Card type as written -> canonical type. ``RES``/``CAP``/``IND`` are the PSpice
#: spellings of the passive cards (their ``R=``/``C=``/``L=`` multiplies the instance
#: value); ``R``/``C``/``L`` cards give the value an instance omits.
CARD_TYPES = {
    "D": "D",
    "NPN": "NPN",
    "PNP": "PNP",
    "NMOS": "NMOS",
    "PMOS": "PMOS",
    "R": "R",
    "RES": "R",
    "C": "C",
    "CAP": "C",
    "L": "L",
    "IND": "L",
}
MULTIPLIER_CARD_TYPES = frozenset({"RES", "CAP", "IND"})

#: Equation set names for messages.
KIND_NAMES = {
    "D": "diode",
    "NPN": "BJT",
    "PNP": "BJT",
    "NMOS": "MOS",
    "PMOS": "MOS",
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
}

#: ``metadata["model"]`` per card type.
EQUATION_SETS = {"D": "spice", "NPN": "gummel_poon", "PNP": "gummel_poon", "NMOS": "spice_level1", "PMOS": "spice_level1"}

#: MOS card levels mapped to the Level-1 equations (2 and 3 with a diagnostic).
MOS_LEVELS = (1, 2, 3)


_PUNCTUATION = ("(", ")", "=")
# An infinite parameter (VAF=inf, IKF=inf): passed on as "inf", which the builder reads.
_INFINITY = frozenset({"inf", "+inf", "infinity", "+infinity"})


class CardError(ValueError):
    """A ``.model`` line could not be read (the message names the problem)."""


@dataclass(slots=True)
class ModelCard:
    """One evaluated ``.model`` card."""

    name: str
    type: str  # card type as written, upper case (D, NPN, NMOS, RES, VDMOS, ...)
    params: dict[str, str] = field(default_factory=dict)
    where: str = ""
    key: object = None  # identity of the definition (once-per-card diagnostics)

    @property
    def canonical_type(self) -> str | None:
        return CARD_TYPES.get(self.type)

    def mos_level(self) -> int:
        """The ``LEVEL`` of a MOS card (1 when absent)."""

        text = self.params.get("level")
        if text is None:
            return 1
        try:
            level = float(text)
        except ValueError:
            from ..utils import parse_value

            level = parse_value(text)
        return int(level) if float(level).is_integer() else -1


def parse_card(tokens: list[str], where: str) -> ModelCard:
    """``.model NAME TYPE(k=v ...)`` (parenthesised or not) from ``fine_tokens`` of the line tail."""

    if len(tokens) < 2 or tokens[0] in _PUNCTUATION or tokens[1] in _PUNCTUATION:
        raise CardError("use .model <name> <type>(<parameter>=<value> ...)")
    name, card_type = tokens[0], tokens[1].upper()
    if card_type.startswith("AKO:"):
        raise CardError(f"model '{name}': AKO (a-kind-of) models are not supported")
    body = tokens[2:]
    if body and body[0] == "(":
        if body[-1] != ")":
            raise CardError(f"model '{name}': missing ')'")
        body = body[1:-1]
    params: dict[str, str] = {}
    index = 0
    while index < len(body):
        key = body[index]
        if key in _PUNCTUATION or index + 1 >= len(body) or body[index + 1] != "=":
            raise CardError(f"model '{name}': expected <parameter>=<value> at '{key}'")
        if index + 2 >= len(body) or body[index + 2] in _PUNCTUATION:
            raise CardError(f"model '{name}': missing value for parameter '{key}'")
        value = body[index + 2]
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        params[key.lower()] = "inf" if value.lower() in _INFINITY else normalize_value(value)
        index += 3
    card = ModelCard(name=name, type=card_type, params=params, where=where)
    canonical = card.canonical_type
    if canonical is not None:
        for key in sorted(SUPPORTED_KEYS[canonical] & params.keys()):
            if not is_value(params[key]) and params[key] != "inf":
                raise CardError(f"model '{name}': parameter '{key}' has invalid value '{params[key]}'")
    return card


def card_diagnostics(card: ModelCard) -> list[str]:
    """Diagnostics of a referenced card: unknown parameters, recorded-only ones, MOS level mapping."""

    canonical = card.canonical_type
    if canonical is None:
        return []
    messages: list[str] = []
    keys = set(card.params)
    kind = KIND_NAMES[canonical]
    if canonical in {"R", "C", "L"}:
        temperature = sorted(keys & TEMPERATURE_KEYS)
        unknown = sorted(keys - SUPPORTED_KEYS[canonical])
        if temperature:
            messages.append(f"model {card.name}: temperature coefficients {', '.join(temperature)} are ignored")
        if unknown:
            messages.append(f"model {card.name}: parameters {', '.join(unknown)} are not used by the {kind} equations")
        return messages
    unknown = sorted(keys - SUPPORTED_KEYS[canonical])
    later = sorted((keys & SUPPORTED_KEYS[canonical]) - MODELLED_KEYS[canonical])
    if canonical in {"NMOS", "PMOS"}:
        level = card.mos_level()
        if level in (2, 3):
            messages.append(f"model {card.name}: LEVEL={level} card mapped to the Level-1 MOS equations")
    if unknown:
        messages.append(f"model {card.name}: parameters {', '.join(unknown)} are not used by the {kind} equations")
    if later:
        messages.append(f"model {card.name}: parameters {', '.join(later)} are recorded but not modelled yet")
    return messages
