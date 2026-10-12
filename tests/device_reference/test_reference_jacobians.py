"""The reference Jacobians agree with central finite differences of the reference currents.

Bias grids cover every region of each equation set (forward, reverse, breakdown, the clip
region beyond |x| = 80; cutoff, forward/reverse active, saturation, high injection; off,
triode, saturation, reverse mode, body bias). Points closer than a few steps to a kink of the
first derivative (the diode breakdown switch, the MOS bulk at vbs = 0) are left out, since a
central difference straddling the kink measures the mean of the two slopes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from device_reference import VT, GummelPoonCard, breakdown_voltage, gummel_poon, mos_card, spice_diode

RTOL = 2e-6


def _assert_close(analytic: np.ndarray, numeric: np.ndarray, scale: np.ndarray, label: str, atol: float = 0.0) -> None:
    """|analytic - numeric| <= RTOL max(|analytic|, scale) + atol (atol: the difference quotient's rounding)."""

    error = np.abs(analytic - numeric)
    allowed = RTOL * np.maximum(np.abs(analytic), scale) + atol
    bad = error > allowed
    assert not bad.any(), (label, np.argwhere(bad)[:5], analytic[bad][:5], numeric[bad][:5])


DIODE_CARDS = [
    # (is, n, bv, ibv): 1N4148-like, a zener-like card, no breakdown
    (2.52e-9, 1.752, 100.0, 100e-6),
    (1e-14, 1.0, 5.1, 5e-3),
    (1e-15, 1.3, np.inf, 1e-3),
]


@pytest.mark.parametrize("card", DIODE_CARDS)
def test_diode_conductance_matches_finite_differences(card) -> None:
    is_, n, bv, ibv = card
    bv_eff = breakdown_voltage(is_, bv, ibv)
    vte = n * VT
    v = np.concatenate(
        (
            np.linspace(-3.5 * vte, 1.2, 97),  # forward and the reverse/forward joint at -3 vte
            np.linspace(-min(bv_eff, 50.0) * 0.98, -4.0 * vte, 41),  # reverse region
            np.array([90.0 * vte, 150.0 * vte, 400.0 * vte]),  # beyond the forward clip
        )
    )
    if np.isfinite(bv_eff):
        v = np.concatenate((v, -bv_eff - np.array([0.01, 0.1, 0.5, 1.0, 85.0 * VT, 200.0 * VT])))
    h = 1e-6 * np.maximum(1.0, np.abs(v)) * vte
    current, conductance = spice_diode(v, is_, n, bv_eff)
    numeric = (spice_diode(v + h, is_, n, bv_eff)[0] - spice_diode(v - h, is_, n, bv_eff)[0]) / (2.0 * h)
    _assert_close(conductance, numeric, np.full_like(v, is_ / vte), f"diode {card}")
    assert np.all(np.isfinite(current)) and np.all(np.isfinite(conductance))


def test_breakdown_voltage_matches_ibv_at_minus_bv() -> None:
    for is_, n, bv, ibv in DIODE_CARDS[:2]:
        bv_eff = breakdown_voltage(is_, bv, ibv)
        if ibv < is_ * bv / VT:
            assert bv_eff == bv
            continue
        # SPICE's matching condition at v = -BV (to RELTOL).
        matched = is_ * (np.exp((bv - bv_eff) / VT) - 1.0 + bv_eff / VT)
        assert abs(matched - ibv) <= 1e-3 * ibv
    assert breakdown_voltage(1e-14, np.inf, 1e-3) == np.inf
    assert breakdown_voltage(2.52e-9, 100.0, 1e-7) == 100.0  # ibv below is * bv / vt


GP_CARDS = {
    "simple": GummelPoonCard(is_=1e-15, bf=100.0, br=2.0),
    "q2n2222": GummelPoonCard(
        is_=14.34e-15, bf=255.9, nf=1.0, br=6.092, nr=1.0, vaf=74.03, ikf=0.2847, ise=14.34e-15, ne=1.307, ikr=0.5, isc=1e-14, nc=2.0
    ),
    "early_reverse": GummelPoonCard(is_=1e-16, bf=80.0, nf=1.02, br=3.0, nr=1.05, vaf=50.0, var=8.0, ikf=0.01, ikr=0.002),
}
GP_BIAS = [
    # (vbe, vbc): cutoff, forward active, saturation, reverse active, high injection, clip region
    (-0.3, -2.0),
    (0.65, -3.0),
    (0.72, -0.5),
    (0.75, 0.6),
    (-0.5, 0.68),
    (0.95, -1.0),
    (2.4, -1.0),
    (0.5, 2.3),
    (-3.0, -4.0),
]


@pytest.mark.parametrize("name", sorted(GP_CARDS))
@pytest.mark.parametrize("polarity", [1, -1])
def test_gummel_poon_jacobian_matches_finite_differences(name: str, polarity: int) -> None:
    card = GP_CARDS[name]
    for vbe, vbc in GP_BIAS:
        ve = 0.3
        vb = ve + polarity * vbe
        vc = vb - polarity * vbc
        voltages = np.array([vc, vb, ve])
        currents, jacobian = gummel_poon(vc, vb, ve, card, polarity)
        assert np.all(np.isfinite(currents)) and np.all(np.isfinite(jacobian))
        assert abs(currents.sum()) <= 1e-12 * np.abs(currents).max()
        numeric = np.zeros((3, 3))
        for col in range(3):
            h = 1e-7
            plus, minus = voltages.copy(), voltages.copy()
            plus[col] += h
            minus[col] -= h
            numeric[:, col] = (gummel_poon(*plus, card, polarity)[0] - gummel_poon(*minus, card, polarity)[0]) / (2 * h)
        scale = np.full((3, 3), np.abs(jacobian).max())
        rounding = 64 * np.finfo(float).eps * np.abs(currents).max() / h
        _assert_close(jacobian, numeric, scale, f"{name} polarity={polarity} vbe={vbe} vbc={vbc}", rounding)


MOS_CARDS = [
    # (beta, vto, lambda, gamma, phi)
    (2e-4, 0.7, 0.02, 0.45, 0.65),
    (1e-3, 1.8, 0.0, 0.0, 0.6),
    (5e-5, -0.8, 0.05, 0.6, 0.7),  # PMOS enhancement card (VTO < 0) when used with polarity -1
]


@pytest.mark.parametrize("card", MOS_CARDS)
@pytest.mark.parametrize("polarity", [1, -1])
def test_mos_card_gradient_matches_finite_differences(card, polarity: int) -> None:
    beta, vto, lam, gamma, phi = card
    s = float(polarity)
    checked = 0
    for vd, vg, vs, vb in itertools.product((0.05, 0.4, 1.3, 3.0), (0.0, 1.1, 2.05, 3.5), (0.0, 0.2), (-2.0, -0.6, 0.1)):
        point = s * np.array([vd, vg, vs, vb])
        # Stay away from the vbs = 0 kink (both orientations) and from the vd = vs swap.
        if min(abs(vb - vs), abs(vb - vd)) < 1e-3 or abs(vd - vs) < 1e-3:
            continue
        ids, gradient = mos_card(*point, polarity, beta, vto, lam, gamma, phi)
        numeric = np.zeros(4)
        for col in range(4):
            h = 1e-7
            plus, minus = point.copy(), point.copy()
            plus[col] += h
            minus[col] -= h
            numeric[col] = (
                mos_card(*plus, polarity, beta, vto, lam, gamma, phi)[0]
                - mos_card(*minus, polarity, beta, vto, lam, gamma, phi)[0]
            ) / (2 * h)
        scale = np.full(4, max(np.abs(gradient).max(), beta * 1e-3))
        _assert_close(gradient, numeric, scale, f"mos {card} polarity={polarity} at {point}")
        # Kirchhoff: the current depends on voltage differences only.
        assert abs(gradient.sum()) <= 1e-9 * max(np.abs(gradient).max(), 1e-30)
        checked += 1
    assert checked > 50
