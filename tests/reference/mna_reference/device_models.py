"""Built-in device stamp helpers and behavioral source expressions."""

from __future__ import annotations

import numpy as np
import re

from .api.contracts import ComponentRecord
from .utils import parse_value

V_T = "0.025"


def _periodize_time_symbol(expression: str, period: float) -> str:
    """Apply FUNC periodization to standalone ``t`` tokens only."""

    return re.sub(r"\bt\b", f"(np.mod(t, {period}))", expression)


def source_expression(component: ComponentRecord) -> str:
    """Build the time-domain expression string for a source."""

    subtype = (component.subtype or "").upper()
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

    beta = parse_value(component.value or "1m")
    vth = abs(parse_value(component.value2 or "0.4"))
    lambda_ = parse_value(component.value3 or "0")
    mos_type = component.type.upper()

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
    forward = _nmos_forward_level1(vgs_forward, vds_forward, beta, vth, lambda_)
    reverse = _nmos_forward_level1(vgs_reverse, vds_reverse, beta, vth, lambda_)
    current = f"Piecewise(({forward}, ({vds_forward}) >= 0), (-({reverse}), True))"
    return f"({sign}({current}))" if sign else f"({current})"
