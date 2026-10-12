"""Constants and the overflow-safe exponential shared by the reference equations."""

from __future__ import annotations

import numpy as np

#: Boltzmann constant / electron charge (V/K, CODATA 2018) times 300.15 K (plan 3.4).
VT = 8.617333262e-5 * 300.15

#: |x| beyond which exp(x) continues along its tangent at the clip point (the P3b rule of the
#: Level-1 BJT, ``CPP_MIGRATION_PLAN.md`` 2.4.1, reused by every model-card exponential).
EXP_CLIP = 80.0


def limexp(x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """``(f(x), f'(x))`` with f = exp inside [-80, 80] and its tangent line outside.

    For x > 80: f(x) = e^80 + e^80 (x - 80); for x < -80: f(x) = e^-80 + e^-80 (x + 80).
    Value and slope are continuous at both clip points.
    """

    x = np.asarray(x, dtype=float)
    upper = np.exp(EXP_CLIP)
    lower = np.exp(-EXP_CLIP)
    inside = np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))
    value = np.where(x > EXP_CLIP, upper + upper * (x - EXP_CLIP), np.where(x < -EXP_CLIP, lower + lower * (x + EXP_CLIP), inside))
    slope = np.where(x > EXP_CLIP, upper, np.where(x < -EXP_CLIP, lower, inside))
    return value, slope


def inverse(x: float) -> float:
    """1 / x with SPICE's "not given" values 0 and inf mapped to 0."""

    return 0.0 if x == 0.0 or np.isinf(x) else 1.0 / x
