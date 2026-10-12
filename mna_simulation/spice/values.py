"""SPICE value literals and token helpers.

Values travel through ``ComponentRecord`` as text that ``utils.parse_value`` and the
netlist value pattern accept. SPICE writes more than that: the scale factors ``T``
(tera) and ``MIL`` (25.4e-6), any case of ``MEG``, and trailing unit letters that SPICE
ignores (``10kOhm``, ``5V``, ``1uF``, ``100nS``). :func:`normalize_value` rewrites such a
literal into the accepted form (``10kOhm`` -> ``10k``, ``1T`` -> ``1000000000000.0``) and
leaves every other token unchanged, so a literal the netlist dialect already accepts is
never touched and an invalid one still fails with the existing message.

Unit letters are accepted from a fixed list of unit words (:data:`UNIT_WORDS`), not as
"any letters": ``1x`` stays an invalid literal (``.tran 1x`` and a ``2x`` duty cycle are
pinned errors of the dialect).
"""

from __future__ import annotations

import re

# The netlist dialect's own value pattern (netlist.Component.INDEPENDENT_VAL_PATTERN).
_LEGACY_VALUE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)(([eE][+-]?\d+)|(u|U|m|M|k|K|Meg|MEG|g|G|n|N|p|P|f|F|%))?$")

# number, optional scale factor, optional unit word (case-insensitive like SPICE).
_SPICE_VALUE = re.compile(
    r"([+-]?(?:\d+\.?\d*|\.\d+))([eE][+-]?\d+)?(meg|mil|[tgkmunpf])?([a-z]*)\Z",
    re.IGNORECASE,
)

#: Unit words SPICE ignores after a value (lower case). ``f`` and ``m`` are scale factors
#: when they come first (``1F`` is 1 femto, ``1mA`` 1 milli-ampere), as in SPICE.
UNIT_WORDS = frozenset(
    {
        "",
        "v",
        "volt",
        "volts",
        "a",
        "amp",
        "amps",
        "ampere",
        "amperes",
        "ohm",
        "ohms",
        "f",
        "farad",
        "farads",
        "h",
        "henry",
        "henrys",
        "henries",
        "s",
        "sec",
        "secs",
        "second",
        "seconds",
        "hz",
        "w",
        "watt",
        "watts",
        "deg",
        "degree",
        "degrees",
    }
)

_SCALE_FACTORS = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "mil": 25.4e-6,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

_MEMO: dict[str, str | None] = {}
_MEMO_LIMIT = 8192


def _spice_literal(token: str) -> str | None:
    """The accepted spelling of a SPICE literal, or ``None`` when ``token`` is not one."""

    cached = _MEMO.get(token)
    if cached is not None or token in _MEMO:
        return cached
    result: str | None
    if _LEGACY_VALUE.match(token):
        result = token
    else:
        match = _SPICE_VALUE.match(token)
        if match is None or match.group(4).lower() not in UNIT_WORDS:
            result = None
        else:
            mantissa, exponent, scale, _unit = match.groups()
            scale_key = (scale or "").lower()
            if not scale_key:
                result = mantissa + (exponent or "")
            elif exponent is None and scale_key in {"g", "k", "m", "u", "n", "p", "f"}:
                result = mantissa + scale
            elif exponent is None and scale_key == "meg":
                result = mantissa + "Meg"
            else:
                result = repr(float(mantissa + (exponent or "")) * _SCALE_FACTORS[scale_key])
    if len(_MEMO) >= _MEMO_LIMIT:
        _MEMO.clear()
    _MEMO[token] = result
    return result


def normalize_value(token: str) -> str:
    """``token`` in the spelling ``parse_value`` accepts, or unchanged when it is no value.

    ``2.2K`` and ``1e-9`` stay as they are; ``10kOhm`` -> ``10k``, ``5V`` -> ``5``,
    ``1MEG`` -> ``1Meg``, ``1meg`` -> ``1Meg``, ``2T`` -> ``2000000000000.0``,
    ``10mil`` -> ``0.000254``.
    """

    if type(token) is not str:
        return token
    literal = _MEMO.get(token)
    if literal is None:
        literal = _spice_literal(token)
    return token if literal is None else literal


def is_value(token: str) -> bool:
    """True for a numeric literal (with optional scale factor and unit word)."""

    return type(token) is str and _spice_literal(token) is not None


def format_number(value: float) -> str:
    """Text of a computed value (``repr`` of the float, accepted by the value pattern)."""

    return repr(float(value))


def merge_assignments(tokens: list[str]) -> list[str]:
    """Join ``k = v``, ``k= v`` and ``k =v`` token runs into single ``k=v`` tokens."""

    if not any("=" in token for token in tokens):
        return list(tokens)
    return re.sub(r"\s*=\s*", "=", " ".join(tokens)).split()


def fine_tokens(text: str) -> list[str]:
    """Split a source specification or model card body.

    ``(``, ``)`` and ``=`` are tokens of their own, commas and whitespace separate,
    quoted strings and ``{...}`` stay whole.
    """

    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in " \t,\r\n":
            i += 1
            continue
        if char in "()=":
            tokens.append(char)
            i += 1
            continue
        if char in "'\"":
            end = text.find(char, i + 1)
            end = n - 1 if end < 0 else end
            tokens.append(text[i : end + 1])
            i = end + 1
            continue
        if char == "{":
            depth = 0
            j = i
            while j < n:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < n and text[j] not in " \t,\r\n()='\"{":
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class AssignmentError(ValueError):
    """A ``name=value`` list could not be read (the message says where)."""


def parse_assignments(text: str) -> list[tuple[str, str]]:
    """``name=value`` pairs of a ``.param`` line, a ``PARAMS:`` list or an instance line.

    A value is ``{expr}``, a quoted ``'expr'``/``"expr"`` (returned as ``{expr}``), or a
    run of characters up to the next whitespace outside parentheses (``a*2``,
    ``max(1, 2)``, ``10k``). Spaces around ``=`` are allowed.
    """

    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while True:
        while i < n and text[i] in " \t,":
            i += 1
        if i >= n:
            return pairs
        match = _IDENT.match(text, i)
        if match is None:
            raise AssignmentError(f"expected name=value at '{text[i:].strip()}'")
        name = match.group(0)
        i = match.end()
        while i < n and text[i] in " \t":
            i += 1
        if i >= n or text[i] != "=" or text.startswith("==", i):
            raise AssignmentError(f"expected '=' after '{name}'")
        i += 1
        while i < n and text[i] in " \t":
            i += 1
        if i >= n:
            raise AssignmentError(f"missing value for '{name}'")
        char = text[i]
        if char == "{":
            depth = 0
            j = i
            while j < n:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:
                raise AssignmentError(f"unbalanced '{{' in the value of '{name}'")
            value = text[i : j + 1]
            i = j + 1
        elif char in "'\"":
            end = text.find(char, i + 1)
            if end < 0:
                raise AssignmentError(f"unterminated quote in the value of '{name}'")
            value = "{" + text[i + 1 : end] + "}"
            i = end + 1
        else:
            depth = 0
            j = i
            while j < n and (depth > 0 or text[j] not in " \t,"):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            value = text[i:j]
            i = j
        pairs.append((name, value))
