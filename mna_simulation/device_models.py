"""Built-in device stamp helpers and behavioral source expressions."""

from __future__ import annotations

import functools
import numpy as np
import re

from .api.contracts import ComponentRecord
from .spice.sources import render_transient, source_terms
from .utils import parse_value

V_T = "0.025"


def _periodize_time_symbol(expression: str, period: float) -> str:
    """Apply FUNC periodization to standalone ``t`` tokens only."""

    return re.sub(r"\bt\b", f"(np.mod(t, {period}))", expression)


def source_expression(component: ComponentRecord) -> str:
    """Build the time-domain expression string for a source.

    SPICE forms (``PULSE``, ``PWL``, ``EXP`` and the parenthesised ``SIN``, recognised by
    their ``metadata``) render as piecewise-linear / exponential / sinusoidal expressions
    of ``t`` (``spice.sources``); the dialect's own forms keep their expressions.
    """

    subtype = (component.subtype or "").upper()
    if component.metadata:
        waveform = render_transient(subtype, component.metadata)
        if waveform is not None:
            return waveform
    amplitude = parse_value(component.value or "0")

    if subtype == "SIN":
        frequency = parse_value(component.value2 or "0")
        return f"({amplitude} * np.sin(2 * np.pi * {frequency} * t))"

    if subtype == "COS":
        frequency = parse_value(component.value2 or "0")
        return f"({amplitude} * np.cos(2 * np.pi * {frequency} * t))"

    if subtype == "STEP":
        if component.value3 is None:
            t_step = parse_value(component.value2 or "0")
            return f"({amplitude} * np.heaviside(t - {t_step}, 1))"
        frequency = parse_value(component.value2 or "0")
        period = 1.0 / frequency
        duty_cycle = min(parse_value(component.value3), 1.0)
        t_step = (1.0 - duty_cycle) * period
        waveform = f"np.heaviside(np.mod(t, {period}) - {t_step}, 1)"
        return f"({amplitude} * {waveform})"

    if subtype == "FUNC":
        period = parse_value(component.value or "0")
        expression = component.value2 or "0"
        if period == 0:
            return f"({expression})"
        return f"({_periodize_time_symbol(expression, period)})"

    return "0"


def spice_source_terms(component: ComponentRecord) -> tuple[float, complex, str | None]:
    """``(dc, ac phasor, time expression or None)`` of a SPICE-form ``V``/``I`` record.

    A SPICE-form record is one with ``metadata`` (``DC x AC mag phase``, ``PULSE``,
    ``PWL``, ``EXP``, ``SIN(...)``); ``.op`` uses ``dc``, ``.ac`` the phasor, and
    ``.tran``/``.hb`` ``dc`` plus the time expression (the waveform minus ``dc``).
    """

    return source_terms((component.subtype or "").upper(), component.value, component.metadata)


def diode_current_expression(component: ComponentRecord, j_expr: str, k_expr: str) -> str:
    """Build a diode current expression string."""

    saturation_current = parse_value(component.value or "1e-15")
    voltage_expr = f"({j_expr} - {k_expr})"
    return f"({saturation_current} * (np.exp({voltage_expr} / {V_T}) - 1))"


def _nmos_forward_level1(vgs: str, vds: str, beta: float, vth: float, lambda_: float) -> str:
    vov = f"(({vgs}) - {vth})"
    triode = f"({beta} * (({vov}) * ({vds}) - 0.5 * ({vds})**2) * (1 + {lambda_} * ({vds})))"
    saturation = f"(0.5 * {beta} * ({vov})**2 * (1 + {lambda_} * ({vds})))"
    return (
        f"Piecewise((0, ({vgs}) <= {vth}), "
        f"({triode}, ({vds}) < ({vov})), "
        f"({saturation}, True))"
    )


def mos_level1_current_expression(component: ComponentRecord, drain_expr: str, gate_expr: str, source_expr: str) -> str:
    """Return drain-to-source current for a three-terminal Level-1 MOSFET.

    The expression uses a symmetric source/drain convention so access devices
    still conduct when the schematic swaps source and drain potentials.
    """

    try:
        template = mos_level1_current_template(component.value, component.value2, component.value3, component.type)
    except TypeError:
        # Unhashable parameter objects cannot key the memo: build the text directly
        # (which raises whatever the uncached construction raises).
        return _mos_level1_current_text(
            component.value, component.value2, component.value3, component.type, drain_expr, gate_expr, source_expr
        )
    return render_mos_level1_current(template, component.type, drain_expr, gate_expr, source_expr)


def render_mos_level1_current(template: str, mos_type: str, drain_expr: str, gate_expr: str, source_expr: str) -> str:
    """``mos_level1_current_expression`` text from a ``mos_level1_current_template`` of the same type."""

    vgs_forward, vds_forward, vgs_reverse, vds_reverse, _ = _mos_level1_terminals(
        mos_type, drain_expr, gate_expr, source_expr
    )
    return template % {"a": vgs_forward, "b": vds_forward, "c": vgs_reverse, "e": vds_reverse}


# Placeholders for the four terminal-voltage expressions while a template is
# generated, and their ``%`` keys. The expression is pure string concatenation of
# these and of the parameter literals (float reprs, which never contain NUL or "%").
_TEMPLATE_VOLTAGES = (("\x00A\x00", "%(a)s"), ("\x00B\x00", "%(b)s"), ("\x00C\x00", "%(c)s"), ("\x00E\x00", "%(e)s"))


@functools.lru_cache(maxsize=1024)
def mos_level1_current_template(value: str | None, value2: str | None, value3: str | None, mos_type: str) -> str:
    """``%``-template of the Level-1 current for these parameter strings.

    The keys are the four terminal-voltage expressions of ``_mos_level1_terminals``
    (``a`` vgs forward, ``b`` vds forward, ``c`` vgs reverse, ``e`` vds reverse);
    ``render_mos_level1_current`` fills them, giving character-for-character the
    text ``_mos_level1_current_text`` builds (the same two functions build both,
    and substituting a placeholder of a pure concatenation is exact). Memoized on
    the parameter strings, so the devices of a netlist that share a parameter set
    (the SRAM has four) parse and format them once. Parse errors are raised in the
    same order as before and on every call (``lru_cache`` does not store exceptions).
    """

    beta, vth, lambda_ = _mos_level1_parameters(value, value2, value3)
    sign = _mos_level1_terminals(mos_type, "", "", "")[4]
    (vgs_forward, _), (vds_forward, _), (vgs_reverse, _), (vds_reverse, _) = _TEMPLATE_VOLTAGES
    text = _mos_level1_current_from_terminals(vgs_forward, vds_forward, vgs_reverse, vds_reverse, sign, beta, vth, lambda_)
    text = text.replace("%", "%%")
    for placeholder, key in _TEMPLATE_VOLTAGES:
        text = text.replace(placeholder, key)
    return text


def _mos_level1_parameters(value: str | None, value2: str | None, value3: str | None) -> tuple[float, float, float]:
    beta = parse_value(value or "1m")
    vth = abs(parse_value(value2 or "0.4"))
    lambda_ = parse_value(value3 or "0")
    return beta, vth, lambda_


def _mos_level1_terminals(
    mos_type_name: str, drain_expr: str, gate_expr: str, source_expr: str
) -> tuple[str, str, str, str, str]:
    """``(vgs_forward, vds_forward, vgs_reverse, vds_reverse, sign)`` of the symmetric Level-1 model."""

    mos_type = mos_type_name.upper()

    if mos_type == "PMOS":
        d = f"(-({drain_expr}))"
        g = f"(-({gate_expr}))"
        s = f"(-({source_expr}))"
        sign = "-"
    else:
        d = drain_expr
        g = gate_expr
        s = source_expr
        sign = ""

    vgs_forward = f"(({g}) - ({s}))"
    vds_forward = f"(({d}) - ({s}))"
    vgs_reverse = f"(({g}) - ({d}))"
    vds_reverse = f"(({s}) - ({d}))"
    return vgs_forward, vds_forward, vgs_reverse, vds_reverse, sign


def _mos_level1_current_from_terminals(
    vgs_forward: str,
    vds_forward: str,
    vgs_reverse: str,
    vds_reverse: str,
    sign: str,
    beta: float,
    vth: float,
    lambda_: float,
) -> str:
    forward = _nmos_forward_level1(vgs_forward, vds_forward, beta, vth, lambda_)
    reverse = _nmos_forward_level1(vgs_reverse, vds_reverse, beta, vth, lambda_)
    current = f"Piecewise(({forward}, ({vds_forward}) >= 0), (-({reverse}), True))"
    return f"({sign}({current}))" if sign else f"({current})"


def _mos_level1_current_text(
    value: str | None,
    value2: str | None,
    value3: str | None,
    mos_type_name: str,
    drain_expr: str,
    gate_expr: str,
    source_expr: str,
) -> str:
    """The uncached expression text (parameters parsed first, then the type, as before)."""

    beta, vth, lambda_ = _mos_level1_parameters(value, value2, value3)
    vgs_forward, vds_forward, vgs_reverse, vds_reverse, sign = _mos_level1_terminals(
        mos_type_name, drain_expr, gate_expr, source_expr
    )
    return _mos_level1_current_from_terminals(vgs_forward, vds_forward, vgs_reverse, vds_reverse, sign, beta, vth, lambda_)
