"""SPICE forms of independent ``V``/``I`` sources: parsing and waveform expressions.

``parse_source_spec`` reads the tokens after the two nodes (``5``, ``DC 5 AC 1 0``,
``SIN(0 1 1k)``, ``PULSE(0 5 1n 1n 1n 5n 10n)``, ``PWL(0 0 1u 5) R=0 TD=1n``,
``EXP(0 1 1u 1u 5u 1u)``) into the record fields of ``HIERARCHY_AND_MODELS_PLAN.md``
section 3.2. The ``render_*`` functions turn the stored text back into expression text
for the native time-source programs (``Min``, ``Max``, ``Mod``, ``Piecewise``,
``np.heaviside``, ``np.exp``, ``np.sin``), with linear ramps.

Transient defaults follow SPICE: an omitted (or zero) ``tr``/``tf``/``tau`` is the
analysis step and an omitted (or zero) ``pw``/``per`` is the stop time, an omitted EXP
``td2`` is ``td1`` plus the step, an omitted SIN frequency is ``1/tstop``. The parser
fills them from the netlist's last ``.tran`` directive (:func:`fill_transient_defaults`);
without one they stay ``0`` and the renderers read ``0`` as an ideal edge
(``tr``/``tf``/``tau``), a pulse that never ends (``pw``), a single pulse (``per``), an
EXP that does not fall (``td2``) or a constant (SIN frequency).

A line without a ``DC`` value takes, as in SPICE, its waveform's value at t = 0 as the
``.op`` value (:func:`transient_value`, the renderers' formulas evaluated in Python).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils import parse_value
from .values import format_number, is_value, normalize_value

TRANSIENT_FUNCTIONS = ("SIN", "PULSE", "PWL", "EXP")
#: Transient functions of other simulators that would change the answer if ignored.
UNSUPPORTED_FUNCTIONS = ("SFFM", "AM", "TRNOISE", "TRRANDOM", "PAT", "EXTERNAL")
#: Distortion-analysis inputs: ignored with a diagnostic.
IGNORED_FUNCTIONS = ("DISTOF1", "DISTOF2")
_KEYWORDS = frozenset(("DC", "AC", "SINE") + TRANSIENT_FUNCTIONS + UNSUPPORTED_FUNCTIONS + IGNORED_FUNCTIONS)

# Arity (minimum, maximum) and the slots that take a transient default when zero.
_ARITY = {"SIN": (2, 6), "PULSE": (2, 7), "EXP": (2, 6)}
_DEFAULTED_SLOTS = {"SIN": (2,), "PULSE": (3, 4, 5, 6), "EXP": (3, 4, 5)}


class SourceError(ValueError):
    """The source specification is invalid (the message names the problem)."""

    def __init__(self, message: str, invalid_subtype: str | None = None) -> None:
        super().__init__(message)
        self.invalid_subtype = invalid_subtype


@dataclass(slots=True)
class SourceSpec:
    """Record fields of a SPICE-form source line."""

    subtype: str
    value: str | None
    value2: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    needs_defaults: bool = False
    diagnostics: list[str] = field(default_factory=list)


def _value(token: str, what: str) -> str:
    text = normalize_value(token)
    if not is_value(text):
        raise SourceError(f"invalid {what} '{token}'")
    return text


def _is_keyword(token: str) -> bool:
    return token.upper() in _KEYWORDS


def parse_source_spec(tokens: list[str]) -> SourceSpec:
    """Fields of the source specification ``tokens`` (``values.fine_tokens`` output)."""

    if not tokens:
        raise SourceError("missing source value")
    dc: str | None = None
    ac_mag: str | None = None
    ac_phase: str | None = None
    transient: tuple[str, list[str], dict[str, str]] | None = None
    diagnostics: list[str] = []
    index = 0
    if tokens[0] not in ("(", ")", "=") and not _is_keyword(tokens[0]):
        if not is_value(normalize_value(tokens[0])):
            raise SourceError(f"invalid subtype '{tokens[0].upper()}'", invalid_subtype=tokens[0].upper())
        dc = _value(tokens[0], "DC value")
        index = 1
    while index < len(tokens):
        token = tokens[index]
        keyword = "SIN" if token.upper() == "SINE" else token.upper()  # SINE: the LTspice spelling
        if keyword == "DC":
            if dc is not None:
                raise SourceError("the DC value is given twice")
            if index + 1 >= len(tokens) or not is_value(normalize_value(tokens[index + 1])):
                raise SourceError("DC needs a value")
            dc = _value(tokens[index + 1], "DC value")
            index += 2
        elif keyword == "AC":
            if ac_mag is not None:
                raise SourceError("the AC value is given twice")
            index += 1
            ac_mag = "1"
            if index < len(tokens) and is_value(normalize_value(tokens[index])):
                ac_mag = _value(tokens[index], "AC magnitude")
                index += 1
                if index < len(tokens) and is_value(normalize_value(tokens[index])):
                    ac_phase = _value(tokens[index], "AC phase")
                    index += 1
        elif keyword in TRANSIENT_FUNCTIONS:
            if transient is not None:
                raise SourceError(f"a second transient function '{keyword}' (the source already has {transient[0]})")
            index, args, options = _function_arguments(tokens, index + 1, keyword)
            transient = (keyword, args, options)
        elif keyword in IGNORED_FUNCTIONS:
            index, _, _ = _function_arguments(tokens, index + 1, keyword)
            diagnostics.append(f"{keyword} (distortion analysis input) is ignored")
        elif keyword in UNSUPPORTED_FUNCTIONS:
            raise SourceError(f"{keyword} sources are not supported")
        elif index == 0:
            raise SourceError(f"invalid subtype '{keyword}'", invalid_subtype=keyword)
        else:
            raise SourceError(f"unexpected '{token}' in the source specification")

    metadata: dict[str, object] = {}
    if ac_mag is not None:
        metadata["ac_mag"] = ac_mag
        metadata["ac_phase"] = ac_phase or "0"
    if transient is None:
        if dc is not None:
            return SourceSpec("DC", dc, metadata=metadata, diagnostics=diagnostics)
        if ac_mag is None:
            raise SourceError("missing source value")
        if ac_phase is None:
            # ``AC mag`` alone: the dialect's own AC source (value = magnitude, no metadata).
            return SourceSpec("AC", ac_mag, diagnostics=diagnostics)
        return SourceSpec("AC", ac_mag, metadata=metadata, diagnostics=diagnostics)

    kind, args, options = transient
    spec = SourceSpec(kind, dc, metadata=metadata, diagnostics=diagnostics)
    if kind == "PWL":
        if "file" in options:
            raise SourceError("PWL FILE= sources are not supported (write the points inline)")
        if len(args) < 2 or len(args) % 2:
            raise SourceError("PWL needs time-value pairs")
        points = [_value(arg, "PWL point") for arg in args]
        times = [parse_value(points[i]) for i in range(0, len(points), 2)]
        for earlier, later in zip(times, times[1:]):
            if later < earlier:
                raise SourceError("PWL times must not decrease")
        spec.metadata["points"] = points
        for key, value in options.items():
            if key == "r":
                repeat = _value(value, "PWL repeat time")
                start = parse_value(repeat)
                if not times[0] <= start < times[-1]:
                    raise SourceError("PWL R= must lie between the first and the last time point")
                spec.metadata["repeat"] = repeat
            elif key == "td":
                spec.metadata["td"] = _value(value, "PWL delay")
            elif key == "file":
                raise SourceError("PWL FILE= sources are not supported (write the points inline)")
            else:
                raise SourceError(f"unknown PWL option '{key.upper()}='")
        return spec

    if options:
        raise SourceError(f"{kind} takes no {', '.join(k.upper() + '=' for k in options)} option")
    minimum, maximum = _ARITY[kind]
    if not minimum <= len(args) <= maximum:
        raise SourceError(f"{kind} takes {minimum} to {maximum} values, got {len(args)}")
    values = [_value(arg, f"{kind} value") for arg in args] + ["0"] * (maximum - len(args))
    spec.metadata[kind.lower()] = values
    spec.needs_defaults = any(parse_value(values[slot]) == 0.0 for slot in _DEFAULTED_SLOTS[kind])
    if kind == "SIN":
        spec.value2 = values[2]
    return spec


def _function_arguments(tokens: list[str], index: int, keyword: str) -> tuple[int, list[str], dict[str, str]]:
    """Arguments of a transient function, parenthesised or not, plus trailing ``KEY=value`` options."""

    args: list[str] = []
    options: dict[str, str] = {}
    if index < len(tokens) and tokens[index] == "(":
        index += 1
        while index < len(tokens) and tokens[index] != ")":
            index = _argument(tokens, index, args, options, keyword)
        if index >= len(tokens):
            raise SourceError(f"missing ')' after {keyword}(")
        index += 1
    else:
        while index < len(tokens) and not _is_keyword(tokens[index]):
            if tokens[index] in ("(", ")"):
                raise SourceError(f"unexpected '{tokens[index]}' after {keyword}")
            index = _argument(tokens, index, args, options, keyword)
    # Options written after the parenthesised list (``PWL(...) R=0 TD=1n``).
    while index + 2 < len(tokens) and tokens[index + 1] == "=" and not _is_keyword(tokens[index]):
        options[tokens[index].lower()] = tokens[index + 2]
        index += 3
    return index, args, options


def _argument(tokens: list[str], index: int, args: list[str], options: dict[str, str], keyword: str) -> int:
    token = tokens[index]
    if index + 1 < len(tokens) and tokens[index + 1] == "=":
        if index + 2 >= len(tokens) or tokens[index + 2] in ("(", ")", "="):
            raise SourceError(f"missing value for {token.upper()}= in {keyword}")
        options[token.lower()] = tokens[index + 2]
        return index + 3
    if token == "=":
        raise SourceError(f"unexpected '=' in {keyword}")
    if options and keyword == "PWL":
        raise SourceError("PWL points must come before its options")
    args.append(token)
    return index + 1


def fill_transient_defaults(metadata: dict[str, object], subtype: str, t_step: float, t_stop: float) -> str | None:
    """Replace the zero (omitted) defaulted slots by the ``.tran`` values; returns the new SIN frequency text."""

    kind = subtype.lower()
    values = list(metadata[kind])  # type: ignore[arg-type]
    if subtype == "PULSE":
        for slot, default in ((3, t_step), (4, t_step), (5, t_stop), (6, t_stop)):
            if parse_value(values[slot]) == 0.0 and default > 0.0:
                values[slot] = format_number(default)
    elif subtype == "EXP":
        td1 = parse_value(values[2])
        if parse_value(values[3]) == 0.0 and t_step > 0.0:
            values[3] = format_number(t_step)
        if parse_value(values[4]) == 0.0 and t_step > 0.0:
            values[4] = format_number(td1 + t_step)
        if parse_value(values[5]) == 0.0 and t_step > 0.0:
            values[5] = format_number(t_step)
    elif subtype == "SIN":
        if parse_value(values[2]) == 0.0 and t_stop > 0.0:
            values[2] = format_number(1.0 / t_stop)
    metadata[kind] = values
    return values[2] if subtype == "SIN" else None


# ---------------------------------------------------------------------------
# Expression rendering
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    text = repr(float(value))
    return f"({text})" if text.startswith("-") else text


def _delayed_time(delay: float) -> str:
    return "t" if delay == 0.0 else f"Max(t - {_num(delay)}, 0)"


def render_pulse(values: list[str]) -> str:
    """``PULSE(v1 v2 td tr tf pw per)``: trapezoid with linear edges, periodic when ``per > 0``."""

    v1, v2, td, tr, tf, pw, per = (parse_value(v) for v in values)
    tau = _delayed_time(td)
    if per > 0.0:
        tau = f"Mod({tau}, {_num(per)})"
    ramps: list[str] = []
    gates: list[str] = []
    if tr > 0.0:
        ramps.append(f"({tau})/{_num(tr)}")
    else:
        gates.append(f"np.heaviside(t - {_num(td)}, 1)")
    if pw > 0.0:
        if tf > 0.0:
            ramps.append(f"({_num(tr + pw + tf)} - ({tau}))/{_num(tf)}")
        else:
            gates.append(f"np.heaviside({_num(tr + pw)} - ({tau}), 0)")
    shape = f"Max(0, Min(1, {', '.join(ramps)}))" if ramps else "1"
    for gate in gates:
        shape = f"({shape})*{gate}"
    return f"({_num(v1)} + {_num(v2 - v1)}*({shape}))"


def render_pwl(points: list[str], repeat: str | None = None, delay: str | None = None) -> str:
    """``PWL(t1 v1 t2 v2 ...)``: ``Piecewise`` segments (linear between points, flat outside)."""

    numbers = [parse_value(p) for p in points]
    times, levels = numbers[0::2], numbers[1::2]
    td = parse_value(delay) if delay else 0.0
    t_expr = "t" if td == 0.0 else f"(t - {_num(td)})"
    if repeat is not None:
        start, last = parse_value(repeat), times[-1]
        t_expr = f"Piecewise(({t_expr}, {t_expr} < {_num(last)}), ({_num(start)} + Mod({t_expr} - {_num(last)}, {_num(last - start)}), True))"
    pieces = [f"({_num(levels[0])}, {t_expr} < {_num(times[0])})"]
    for (t0, v0), (t1, v1) in zip(zip(times, levels), zip(times[1:], levels[1:])):
        if t1 <= t0:
            continue  # a vertical step: the next segment starts at the new level
        slope = (v1 - v0) / (t1 - t0)
        segment = _num(v0) if slope == 0.0 else f"{_num(v0)} + {_num(slope)}*({t_expr} - {_num(t0)})"
        pieces.append(f"({segment}, {t_expr} < {_num(t1)})")
    pieces.append(f"({_num(levels[-1])}, True)")
    return f"Piecewise({', '.join(pieces)})"


def render_exp(values: list[str]) -> str:
    """``EXP(v1 v2 td1 tau1 td2 tau2)``: rise from ``td1``, fall from ``td2``.

    ``td2`` still 0 after the ``.tran`` defaults (SPICE: ``td1`` plus the step) means the
    netlist has no ``.tran``: the waveform does not fall.
    """

    v1, v2, td1, tau1, td2, tau2 = (parse_value(v) for v in values)

    def edge(delay: float, tau: float) -> str:
        if tau > 0.0:
            return f"(1 - np.exp(-{_delayed_time(delay)}/{_num(tau)}))"
        return f"np.heaviside(t - {_num(delay)}, 1)"

    rise = f"{_num(v1)} + {_num(v2 - v1)}*{edge(td1, tau1)}"
    if td2 == 0.0:
        return f"({rise})"
    return f"({rise} + {_num(v1 - v2)}*{edge(td2, tau2)})"


def render_sin(values: list[str]) -> str:
    """``SIN(vo va freq td theta phase)``: ``vo + va*exp(-theta (t-td)) sin(2 pi f (t-td) + phase)`` after ``td``."""

    vo, va, freq, td, theta, phase = (parse_value(v) for v in values)
    tau = _delayed_time(td)
    phase_rad = math.radians(phase)
    argument = f"2*np.pi*{_num(freq)}*{tau}"
    if phase_rad:
        argument += f" + {_num(phase_rad)}"
    wave = f"np.sin({argument})"
    if theta:
        wave = f"np.exp(-{_num(theta)}*{tau})*{wave}"
    return f"({_num(vo)} + {_num(va)}*{wave})"


def render_transient(subtype: str, metadata: dict[str, object]) -> str | None:
    """Waveform expression of a SPICE-form source, or ``None`` when it has no transient function."""

    if subtype == "PULSE":
        return render_pulse(metadata["pulse"])  # type: ignore[arg-type]
    if subtype == "PWL":
        return render_pwl(metadata["points"], metadata.get("repeat"), metadata.get("td"))  # type: ignore[arg-type]
    if subtype == "EXP":
        return render_exp(metadata["exp"])  # type: ignore[arg-type]
    if subtype == "SIN" and "sin" in metadata:
        return render_sin(metadata["sin"])  # type: ignore[arg-type]
    return None


# ---------------------------------------------------------------------------
# Waveform values (the same formulas as the renderers, evaluated in Python)
# ---------------------------------------------------------------------------


def _heaviside(x: float, at_zero: float) -> float:
    return 1.0 if x > 0.0 else (at_zero if x == 0.0 else 0.0)


def _pulse_value(values: list[str], t: float) -> float:
    v1, v2, td, tr, tf, pw, per = (parse_value(v) for v in values)
    tau = t if td == 0.0 else max(t - td, 0.0)
    if per > 0.0:
        tau = tau % per
    ramps: list[float] = []
    gate = 1.0
    if tr > 0.0:
        ramps.append(tau / tr)
    else:
        gate *= _heaviside(t - td, 1.0)
    if pw > 0.0:
        if tf > 0.0:
            ramps.append((tr + pw + tf - tau) / tf)
        else:
            gate *= _heaviside(tr + pw - tau, 0.0)
    shape = max(0.0, min([1.0, *ramps])) if ramps else 1.0
    return v1 + (v2 - v1) * shape * gate


def _pwl_value(points: list[str], repeat: str | None, delay: str | None, t: float) -> float:
    numbers = [parse_value(p) for p in points]
    times, levels = numbers[0::2], numbers[1::2]
    local = t - (parse_value(delay) if delay else 0.0)
    if repeat is not None:
        start, last = parse_value(repeat), times[-1]
        if not local < last:
            local = start + (local - last) % (last - start)
    if local < times[0]:
        return levels[0]
    for (t0, v0), (t1, v1) in zip(zip(times, levels), zip(times[1:], levels[1:])):
        if t1 > t0 and local < t1:
            return v0 + (v1 - v0) / (t1 - t0) * (local - t0)
    return levels[-1]


def _exp_value(values: list[str], t: float) -> float:
    v1, v2, td1, tau1, td2, tau2 = (parse_value(v) for v in values)

    def edge(delay: float, tau: float) -> float:
        if tau > 0.0:
            return 1.0 - math.exp(-(t if delay == 0.0 else max(t - delay, 0.0)) / tau)
        return _heaviside(t - delay, 1.0)

    rise = v1 + (v2 - v1) * edge(td1, tau1)
    return rise if td2 == 0.0 else rise + (v1 - v2) * edge(td2, tau2)


def _sin_value(values: list[str], t: float) -> float:
    vo, va, freq, td, theta, phase = (parse_value(v) for v in values)
    tau = t if td == 0.0 else max(t - td, 0.0)
    return vo + va * math.exp(-theta * tau) * math.sin(2.0 * math.pi * freq * tau + math.radians(phase))


def transient_value(subtype: str, metadata: dict[str, object], t: float) -> float | None:
    """Value of a SPICE-form source's waveform at ``t`` (``None`` without a transient function)."""

    if subtype == "PULSE":
        return _pulse_value(metadata["pulse"], t)  # type: ignore[arg-type]
    if subtype == "PWL":
        return _pwl_value(metadata["points"], metadata.get("repeat"), metadata.get("td"), t)  # type: ignore[arg-type]
    if subtype == "EXP":
        return _exp_value(metadata["exp"], t)  # type: ignore[arg-type]
    if subtype == "SIN" and "sin" in metadata:
        return _sin_value(metadata["sin"], t)  # type: ignore[arg-type]
    return None


def source_terms(subtype: str, value: str | None, metadata: dict[str, object]) -> tuple[float, complex, str | None]:
    """``(dc, ac phasor, time expression)`` of a SPICE-form source record.

    ``.op`` uses ``dc``: the explicit ``DC`` value, or (as in SPICE) the waveform's value
    at t = 0 when the line gives none (``PULSE(5 0 ...)`` gives 5). ``.ac`` uses the phasor
    ``ac_mag * exp(j ac_phase deg)``. ``.tran`` and ``.hb`` evaluate ``b_dc + b_time(t)``,
    so the time expression is the waveform minus ``dc`` (``None`` when the source has no
    transient function).
    """

    if subtype == "AC":
        dc = 0.0
    elif value is not None:
        dc = parse_value(value)
    else:
        dc = transient_value(subtype, metadata, 0.0) or 0.0
    ac: complex = 0j
    if "ac_mag" in metadata:
        magnitude = parse_value(metadata["ac_mag"])  # type: ignore[arg-type]
        phase = math.radians(parse_value(metadata.get("ac_phase") or "0"))  # type: ignore[arg-type]
        ac = complex(magnitude * math.cos(phase), magnitude * math.sin(phase)) if phase else complex(magnitude, 0.0)
    elif subtype == "AC" and value is not None:
        ac = complex(parse_value(value), 0.0)
    waveform = render_transient(subtype, metadata)
    if waveform is not None and dc != 0.0:
        waveform = f"({waveform} - {_num(dc)})"
    return dc, ac, waveform
