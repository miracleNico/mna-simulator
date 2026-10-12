"""SPICE diode (SPICE3f5 ``dioload.c`` DC current, ``diotemp.c`` breakdown voltage).

Junction voltage v = V(anode) - V(cathode), emission voltage vte = N * VT:

* v >= -3 vte: I = IS (exp(v / vte) - 1)
* -BV_eff <= v < -3 vte: I = -IS (1 + (3 vte / (e v))^3)  (SPICE's smooth reverse region)
* v < -BV_eff: I = -IS exp(-(BV_eff + v) / VT)  (breakdown; plain VT as in SPICE3f5)

with the exponentials continued linearly beyond |x| = 80 (``common.limexp``).
"""

from __future__ import annotations

import math

import numpy as np

from .common import VT, limexp

RELTOL = 1e-3


def breakdown_voltage(is_: float, bv: float, ibv: float, vt: float = VT) -> float:
    """SPICE3f5's continuity-shifted breakdown voltage BV_eff (``diotemp.c``).

    BV_eff is chosen so that IS (exp((BV - BV_eff) / vt) - 1 + BV_eff / vt) = IBV, i.e. the
    breakdown current at v = -BV is IBV (to RELTOL, by SPICE's fixed-point iteration of at most
    25 steps). When IBV < IS BV / vt SPICE raises IBV to that value and keeps BV_eff = BV.
    No breakdown (BV infinite) gives +inf.
    """

    if math.isinf(bv):
        return math.inf
    if ibv < is_ * bv / vt:
        return bv
    target = ibv
    shifted = bv - vt * math.log(1.0 + target / is_)
    for _ in range(25):
        shifted = bv - vt * math.log(target / is_ + 1.0 - shifted / vt)
        matched = is_ * (math.exp((bv - shifted) / vt) - 1.0 + shifted / vt)
        if abs(matched - target) <= RELTOL * target:
            break
    return shifted


def spice_diode(v: np.ndarray | float, is_: float, n: float, bv_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """``(I, dI/dv)`` of the junction at the voltages ``v`` (``is_`` already area-scaled)."""

    v = np.asarray(v, dtype=float)
    vte = n * VT
    forward = v >= -3.0 * vte
    breakdown = ~forward & (v < -bv_eff)
    reverse = ~forward & ~breakdown

    e_fwd, de_fwd = limexp(np.where(forward, v, 0.0) / vte)
    i_fwd = is_ * (e_fwd - 1.0)
    g_fwd = is_ / vte * de_fwd

    v_rev = np.where(reverse, v, -1.0)
    cube = (3.0 * vte / (math.e * v_rev)) ** 3
    i_rev = -is_ * (1.0 + cube)
    g_rev = 3.0 * is_ * cube / v_rev

    x_brk = np.where(breakdown, -(bv_eff + v), 0.0) / VT
    e_brk, de_brk = limexp(x_brk)
    i_brk = -is_ * e_brk
    g_brk = is_ / VT * de_brk

    current = np.where(forward, i_fwd, np.where(breakdown, i_brk, i_rev))
    conductance = np.where(forward, g_fwd, np.where(breakdown, g_brk, g_rev))
    return current, conductance
