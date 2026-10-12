"""Four-terminal Level-1 MOSFET with body effect (SPICE3f5 ``mos1load.c`` DC current).

In the n-channel frame (PMOS: every terminal voltage negated, the current negated) the
terminal at the lower potential acts as the source (drain and source swap when vd < vs).
With vgs, vds >= 0, vbs measured from that source and vth0 = type * VTO:

* threshold  vth = vth0 + GAMMA (sqrt(PHI - vbs) - sqrt(PHI)) for vbs <= 0, vth0 for vbs > 0
* off        vgs <= vth:          I = 0
* triode     vds < vgs - vth:     I = beta (vov vds - vds^2 / 2) (1 + LAMBDA vds)
* saturation otherwise:           I = beta / 2 vov^2 (1 + LAMBDA vds),  vov = vgs - vth

The gradient is formed over (vov, vds) and mapped back to the four terminals.
"""

from __future__ import annotations

import numpy as np


def threshold(vbs: np.ndarray, vth0: float, gamma: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """``(vth, dvth/dvbs)`` in the n-channel frame (held at vth0 for vbs > 0)."""

    vbs = np.asarray(vbs, dtype=float)
    depleted = vbs <= 0.0
    root = np.sqrt(phi - np.where(depleted, vbs, 0.0))
    vth = np.where(depleted, vth0 + gamma * (root - np.sqrt(phi)), vth0)
    slope = np.where(depleted, -gamma / (2.0 * root), 0.0)
    return vth, slope


def mos_card(
    vd: np.ndarray | float,
    vg: np.ndarray | float,
    vs: np.ndarray | float,
    vb: np.ndarray | float,
    polarity: int,
    beta: float,
    vto: float,
    lam: float,
    gamma: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, gradient)``: drain-to-source current and d ids / d(vd, vg, vs, vb) as ``(..., 4)``."""

    s = 1.0 if polarity >= 0 else -1.0
    vd, vg, vs, vb = (s * np.asarray(value, dtype=float) for value in (vd, vg, vs, vb))
    forward = vd >= vs
    source = np.where(forward, vs, vd)
    drain = np.where(forward, vd, vs)
    vgs = vg - source
    vds = drain - source
    vbs = vb - source
    vth, dvth_dvbs = threshold(vbs, s * vto, gamma, phi)
    vov = vgs - vth
    on = vgs > vth
    triode = on & (vds < vov)
    modulation = 1.0 + lam * vds
    channel = vov * vds - 0.5 * vds * vds
    current = np.where(on, np.where(triode, beta * channel * modulation, 0.5 * beta * vov * vov * modulation), 0.0)
    di_dvov = np.where(on, np.where(triode, beta * vds * modulation, beta * vov * modulation), 0.0)
    di_dvds = np.where(
        on, np.where(triode, beta * ((vov - vds) * modulation + lam * channel), 0.5 * beta * vov * vov * lam), 0.0
    )
    # vov = vg - source - vth(vb - source), vds = drain - source.
    d_gate = di_dvov
    d_bulk = -di_dvov * dvth_dvbs
    d_drain_side = di_dvds
    d_source_side = -di_dvov * (1.0 - dvth_dvbs) - di_dvds
    ids = np.where(forward, current, -current)
    sign = np.where(forward, 1.0, -1.0)
    # Map (drain side, source side) back to the terminals d and s; the current changes sign
    # with the orientation, the frame sign cancels (s * s = 1).
    grad_d = sign * np.where(forward, d_drain_side, d_source_side)
    grad_s = sign * np.where(forward, d_source_side, d_drain_side)
    gradient = np.stack((grad_d, sign * d_gate, grad_s, sign * d_bulk), axis=-1)
    return s * ids, gradient
