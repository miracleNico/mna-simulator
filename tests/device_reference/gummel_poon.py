"""Gummel-Poon BJT, DC part (SPICE3f5 ``bjtload.c`` at the nominal temperature).

In the NPN frame (PNP: every terminal voltage and current negated) with vbe = vb - ve,
vbc = vb - vc:

* transport currents  If = IS (exp(vbe / (NF VT)) - 1),  Ir = IS (exp(vbc / (NR VT)) - 1)
* leakage currents    Ile = ISE (exp(vbe / (NE VT)) - 1), Ilc = ISC (exp(vbc / (NC VT)) - 1)
* base charge         q1 = 1 / (1 - vbc / VAF - vbe / VAR), q2 = If / IKF + Ir / IKR,
                      qb = q1 (1 + sqrt(max(1 + 4 q2, 0))) / 2
* Ic = (If - Ir) / qb - Ir / BR - Ilc,  Ib = If / BF + Ile + Ir / BR + Ilc,  Ie = -(Ic + Ib)

Infinite (or zero, SPICE's "not given") VAF, VAR, IKF, IKR drop their terms. Every exponential
continues linearly beyond |x| = 80 (``common.limexp``). The Jacobian is formed by the chain
rule over (vbe, vbc) and mapped to the terminals (collector, base, emitter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import VT, inverse, limexp


@dataclass(frozen=True)
class GummelPoonCard:
    """Intrinsic parameters with the area already applied to IS, ISE, ISC, IKF, IKR."""

    is_: float = 1e-16
    bf: float = 100.0
    nf: float = 1.0
    br: float = 1.0
    nr: float = 1.0
    vaf: float = np.inf
    var: float = np.inf
    ikf: float = np.inf
    ikr: float = np.inf
    ise: float = 0.0
    ne: float = 1.5
    isc: float = 0.0
    nc: float = 2.0


def _diode(v: np.ndarray, saturation: float, emission: float) -> tuple[np.ndarray, np.ndarray]:
    if saturation == 0.0:
        zero = np.zeros_like(v)
        return zero, zero
    e, de = limexp(v / (emission * VT))
    return saturation * (e - 1.0), saturation * de / (emission * VT)


def gummel_poon(
    vc: np.ndarray | float, vb: np.ndarray | float, ve: np.ndarray | float, card: GummelPoonCard, polarity: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """``(currents, jacobian)``: currents ``(..., 3)`` into (c, b, e), jacobian ``(..., 3, 3)`` d I_row / d V_col."""

    s = 1.0 if polarity >= 0 else -1.0
    vc, vb, ve = (np.asarray(value, dtype=float) for value in (vc, vb, ve))
    vbe = s * (vb - ve)
    vbc = s * (vb - vc)

    i_f, g_f = _diode(vbe, card.is_, card.nf)
    i_r, g_r = _diode(vbc, card.is_, card.nr)
    i_le, g_le = _diode(vbe, card.ise, card.ne)
    i_lc, g_lc = _diode(vbc, card.isc, card.nc)

    inv_vaf, inv_var = inverse(card.vaf), inverse(card.var)
    inv_ikf, inv_ikr = inverse(card.ikf), inverse(card.ikr)
    q1 = 1.0 / (1.0 - vbc * inv_vaf - vbe * inv_var)
    dq1_dvbe = q1 * q1 * inv_var
    dq1_dvbc = q1 * q1 * inv_vaf
    radicand = 1.0 + 4.0 * (i_f * inv_ikf + i_r * inv_ikr)
    positive = radicand > 0.0
    root = np.sqrt(np.where(positive, radicand, 0.0))
    safe_root = np.where(positive, root, 1.0)
    droot_dvbe = np.where(positive, 2.0 * inv_ikf * g_f / safe_root, 0.0)
    droot_dvbc = np.where(positive, 2.0 * inv_ikr * g_r / safe_root, 0.0)
    qb = 0.5 * q1 * (1.0 + root)
    dqb_dvbe = 0.5 * (dq1_dvbe * (1.0 + root) + q1 * droot_dvbe)
    dqb_dvbc = 0.5 * (dq1_dvbc * (1.0 + root) + q1 * droot_dvbc)

    transfer = (i_f - i_r) / qb
    dtransfer_dvbe = (g_f - transfer * dqb_dvbe) / qb
    dtransfer_dvbc = (-g_r - transfer * dqb_dvbc) / qb

    ic = transfer - i_r / card.br - i_lc
    ib = i_f / card.bf + i_le + i_r / card.br + i_lc
    dic = (dtransfer_dvbe, dtransfer_dvbc - g_r / card.br - g_lc)
    dib = (g_f / card.bf + g_le, g_r / card.br + g_lc)

    def terminals(d_vbe: np.ndarray, d_vbc: np.ndarray) -> np.ndarray:
        # d/dvc = -d/dvbc, d/dvb = d/dvbe + d/dvbc, d/dve = -d/dvbe (frame signs cancel: s * s = 1).
        return np.stack((-d_vbc, d_vbe + d_vbc, -d_vbe), axis=-1)

    row_c = terminals(*dic)
    row_b = terminals(*dib)
    currents = s * np.stack((ic, ib, -(ic + ib)), axis=-1)
    jacobian = np.stack((row_c, row_b, -(row_c + row_b)), axis=-2)
    return currents, jacobian
