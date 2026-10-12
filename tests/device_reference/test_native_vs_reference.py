"""Native model-card kernels (``mna_simulation._core``) against the NumPy reference, to 1e-12.

Two entry points are compared: the scalar kernels (``eval_spice_diode``,
``spice_diode_breakdown_voltage``, ``eval_gummel_poon``, ``eval_mos_card``) on bias grids that
include the region boundaries exactly (diode -3 N vt and -BV_eff, the exponent clip at
|x| = 80 and beyond, BJT junctions at 0 V, MOS vgs = vth, vds = vgs - vth, vd = vs, vbs = 0),
and the compiled-circuit path (``build_circuit`` with the ``spice_diodes``, ``gummel_poon`` and
``mos4`` tables, ``device_f_jacobian_dense``), which also covers the table plumbing, the PNP /
PMOS signs, ground terminals and the assembly.

Tolerance: |native - reference| <= 1e-12 max(|reference|, s) + a, elementwise, where s is the
largest magnitude among the outputs of the same device at the same bias (one current vector or
one Jacobian), so a component that is the small difference of large terms (Ie = -(Ic + Ib) in
cutoff, a near-zero conductance next to a large one) is held to the same relative accuracy as
its row, and a = 1e-12 times a device scale (is, or beta * 1 V^2) only matters where every
output is essentially zero (a MOS at threshold).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from device_reference import EXP_CLIP, VT, GummelPoonCard, breakdown_voltage, gummel_poon, mos_card, spice_diode
from mna_simulation import native
from mna_simulation.native.circuit import build_circuit, gummel_poon_table, mos4_table, spice_diode_table

core = native.require_core()
RTOL = 1e-12


def _assert_matches(native_values, reference_values, label: str, floor: float = 0.0) -> None:
    current = np.asarray(native_values, dtype=float)
    expected = np.asarray(reference_values, dtype=float)
    assert current.shape == expected.shape, (label, current.shape, expected.shape)
    scale = np.abs(expected).max(initial=0.0)
    allowed = RTOL * np.maximum(np.abs(expected), scale) + RTOL * floor
    error = np.abs(current - expected)
    bad = ~(error <= allowed)
    assert not bad.any(), (label, np.argwhere(bad)[:5], current[bad][:5], expected[bad][:5])


def test_constants_match() -> None:
    assert core.card_vt == VT == 8.617333262e-5 * 300.15
    assert core.exp_clip == EXP_CLIP


DIODE_CARDS = [
    (2.52e-9, 1.752, 100.0, 100e-6),
    (2.52e-9, 1.752, 100.0, 1e-7),  # ibv below is * bv / vt: bv_eff = bv
    (1e-14, 1.0, 5.1, 5e-3),
    (3e-12, 1.2, 12.0, 20e-3),
    (1e-15, 1.3, math.inf, 1e-3),
]


@pytest.mark.parametrize("card", DIODE_CARDS)
def test_diode_breakdown_voltage_and_current(card) -> None:
    is_, n, bv, ibv = card
    bv_eff = core.spice_diode_breakdown_voltage(is_, bv, ibv)
    expected_bv = breakdown_voltage(is_, bv, ibv)
    if math.isinf(expected_bv):
        assert bv_eff == math.inf
    else:
        assert abs(bv_eff - expected_bv) <= RTOL * expected_bv
    vte = n * VT
    boundaries = [-3.0 * vte, 0.0, EXP_CLIP * vte, -EXP_CLIP * vte]
    if math.isfinite(bv_eff):
        boundaries += [-bv_eff, -bv_eff - EXP_CLIP * VT]
    voltages = list(np.linspace(-1.5 * min(bv_eff, 40.0), 1.5, 301)) + [
        b + d for b in boundaries for d in (0.0, -1e-12, 1e-12, -1e-6, 1e-6)
    ]
    voltages += [100.0 * vte, 1e3 * vte, 5.0, 50.0]  # beyond the forward clip
    if math.isfinite(bv_eff):
        voltages += [-bv_eff - 100.0 * VT, -bv_eff - 3.0, -2.0 * bv_eff]  # beyond the breakdown clip
    v = np.asarray(voltages, dtype=float)
    reference_i, reference_g = spice_diode(v, is_, n, expected_bv)
    for k in range(v.size):
        i, g = core.eval_spice_diode(is_, n, bv_eff, float(v[k]))
        assert np.isfinite(i) and np.isfinite(g)
        _assert_matches([i, g], [reference_i[k], reference_g[k]], f"diode {card} v={v[k]!r}", floor=is_)


GP_CARDS = [
    GummelPoonCard(is_=1e-15, bf=100.0, br=2.0),
    GummelPoonCard(
        is_=14.34e-15, bf=255.9, nf=1.0, br=6.092, nr=1.0, vaf=74.03, ikf=0.2847, ise=14.34e-15, ne=1.307, isc=1e-14, nc=2.0
    ),
    GummelPoonCard(is_=1e-16, bf=80.0, nf=1.02, br=3.0, nr=1.05, vaf=50.0, var=8.0, ikf=0.01, ikr=0.002, isc=3e-15, nc=1.7),
    GummelPoonCard(is_=6.734e-15, bf=416.4, br=0.7371, vaf=74.03, ikf=66.78e-3, ise=6.734e-15, ne=1.259, isc=0.0),
]


def _gp_native(card: GummelPoonCard, polarity: int, vc: float, vb: float, ve: float):
    (currents, rows) = core.eval_gummel_poon(
        polarity,
        card.is_,
        card.bf,
        card.nf,
        card.br,
        card.nr,
        card.vaf,
        card.var,
        card.ikf,
        card.ikr,
        card.ise,
        card.ne,
        card.isc,
        card.nc,
        vc,
        vb,
        ve,
    )
    return np.asarray(currents), np.asarray(rows)


def _junction_values(card: GummelPoonCard) -> list[float]:
    clip_f = EXP_CLIP * card.nf * VT
    clip_r = EXP_CLIP * card.nr * VT
    values = [-5.0, -1.0, -0.2, 0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.1]
    values += [clip_f, -clip_f, clip_r, -clip_r, clip_f * 1.3, -clip_f * 1.5, 3.0 * clip_r]
    return values


@pytest.mark.parametrize("card_index", range(len(GP_CARDS)))
@pytest.mark.parametrize("polarity", [1, -1])
def test_gummel_poon_matches_reference(card_index: int, polarity: int) -> None:
    card = GP_CARDS[card_index]
    values = _junction_values(card)
    for vbe, vbc in itertools.product(values, values):
        ve = -0.25
        vb = ve + polarity * vbe
        vc = vb - polarity * vbc
        currents, jacobian = _gp_native(card, polarity, vc, vb, ve)
        assert np.all(np.isfinite(currents)) and np.all(np.isfinite(jacobian))
        reference_currents, reference_jacobian = gummel_poon(vc, vb, ve, card, polarity)
        label = f"gp[{card_index}] polarity={polarity} vbe={vbe!r} vbc={vbc!r}"
        _assert_matches(currents, reference_currents, label + " currents", floor=card.is_)
        _assert_matches(jacobian, reference_jacobian, label + " jacobian", floor=card.is_ / VT)


def test_gummel_poon_reduces_to_the_simple_transport_form() -> None:
    card = GummelPoonCard(is_=2e-15, bf=150.0, br=3.0)
    for vbe, vbc in itertools.product((-0.4, 0.0, 0.62, 0.7), (-3.0, 0.0, 0.5)):
        vc, vb, ve = -vbc + vbe, vbe, 0.0
        currents, _ = _gp_native(card, 1, vc, vb, ve)
        i_f = card.is_ * (math.exp(vbe / VT) - 1.0)
        i_r = card.is_ * (math.exp(vbc / VT) - 1.0)
        expected = [i_f - i_r - i_r / card.br, i_f / card.bf + i_r / card.br]
        expected.append(-(expected[0] + expected[1]))
        _assert_matches(currents, expected, f"simple vbe={vbe} vbc={vbc}", floor=card.is_)


MOS_CARDS = [
    (2e-4, 0.7, 0.02, 0.45, 0.65),
    (1e-3, 1.8, 0.0, 0.0, 0.6),
    (5e-5, -0.8, 0.05, 0.6, 0.7),
    (3e-4, 0.0, 0.1, 1.2, 0.4),
]


def _mos_points(vto: float, gamma: float, phi: float, polarity: int) -> list[tuple[float, float, float, float]]:
    s = float(polarity)
    vth0 = s * vto
    points = []
    for vd, vg, vs, vb in itertools.product((0.0, 0.1, 0.9, 2.5), (-0.5, 0.6, 1.5, 3.0), (0.0, 0.3), (-1.5, 0.0, 0.3, 0.6)):
        points.append((s * vd, s * vg, s * vs, s * vb))
    # Exact boundaries in the n-channel frame: vgs = vth, vds = vgs - vth, vd = vs, vbs = 0.
    for vs, vbs in itertools.product((0.0, 0.4), (-0.9, 0.0, 0.2)):
        vb = vs + vbs
        root = math.sqrt(phi - vbs) if vbs <= 0.0 else math.sqrt(phi)
        vth = vth0 + gamma * (root - math.sqrt(phi)) if vbs <= 0.0 and gamma != 0.0 else vth0
        vg = vs + vth
        points.append((s * (vs + 1.0), s * vg, s * vs, s * vb))  # vgs == vth
        vg = vs + vth + 0.8
        points.append((s * (vs + 0.8), s * vg, s * vs, s * vb))  # vds == vgs - vth (approximately exact)
        points.append((s * vs, s * vg, s * vs, s * vb))  # vd == vs
    return points


@pytest.mark.parametrize("card", MOS_CARDS)
@pytest.mark.parametrize("polarity", [1, -1])
def test_mos_card_matches_reference(card, polarity: int) -> None:
    beta, vto, lam, gamma, phi = card
    for vd, vg, vs, vb in _mos_points(vto, gamma, phi, polarity):
        result = np.asarray(core.eval_mos_card(polarity, beta, vto, lam, gamma, phi, vd, vg, vs, vb))
        ids, gradient = mos_card(vd, vg, vs, vb, polarity, beta, vto, lam, gamma, phi)
        label = f"mos {card} polarity={polarity} at {(vd, vg, vs, vb)}"
        _assert_matches(result[:1], [ids], label + " ids", floor=beta)
        _assert_matches(result[1:], gradient, label + " gradient", floor=beta)


def test_mos_card_with_zero_gamma_is_bit_identical_to_level1() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        polarity = int(rng.choice([1, -1]))
        beta, lam = float(rng.uniform(1e-5, 1e-2)), float(rng.uniform(0.0, 0.1))
        vto = polarity * float(rng.uniform(0.1, 1.5))
        vd, vg, vs, vb = (float(value) for value in rng.uniform(-3.0, 3.0, 4))
        card = core.eval_mos_card(polarity, beta, vto, lam, 0.0, 0.6, vd, vg, vs, vb)
        level1 = core.eval_mos_level1(polarity, beta, abs(vto), lam, vd, vg, vs)
        assert card[:4] == level1
        assert card[4] == 0.0


# ---- compiled-circuit path -----------------------------------------------------------------


def _circuit_states(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = rng.uniform(-1.2, 1.2, (count, n))
    states[: count // 4] *= 4.0  # some states beyond the exponent clips
    return states


def test_compiled_tables_match_reference() -> None:
    # Unknowns 0..5; -1 is ground. Diodes, BJTs (NPN and PNP, one emitter on ground), MOS
    # (NMOS four-terminal, PMOS with bulk on the source, one source on ground).
    diodes = [
        {"anode": 0, "cathode": 1, "is": 2.52e-9, "n": 1.752, "bv": 100.0, "ibv": 1e-4},
        {"anode": 2, "cathode": -1, "is": 1e-14, "n": 1.0, "bv": 5.1, "ibv": 5e-3},
        {"anode": 3, "cathode": 4, "is": 1e-15, "n": 1.3, "bv": math.inf, "ibv": 1e-3},
    ]
    bjts = [
        {"type": "QNPN", "collector": 0, "base": 1, "emitter": -1, "card": GP_CARDS[1]},
        {"type": "QPNP", "collector": 2, "base": 3, "emitter": 5, "card": GP_CARDS[2]},
    ]
    moses = [
        {"type": "NMOS", "drain": 4, "gate": 0, "source": -1, "bulk": 5, "card": MOS_CARDS[0]},
        {"type": "PMOS", "drain": 1, "gate": 2, "source": 3, "bulk": 3, "card": MOS_CARDS[2]},
    ]
    gp_rows = []
    for bjt in bjts:
        card = bjt["card"]
        row = {key: bjt[key] for key in ("type", "collector", "base", "emitter")}
        row.update(
            is_=card.is_, bf=card.bf, nf=card.nf, br=card.br, nr=card.nr, vaf=card.vaf, var=card.var,
            ikf=card.ikf, ikr=card.ikr, ise=card.ise, ne=card.ne, isc=card.isc, nc=card.nc,
        )
        row["is"] = row.pop("is_")
        gp_rows.append(row)
    mos_rows = []
    for mos in moses:
        beta, vto, lam, gamma, phi = mos["card"]
        row = {key: mos[key] for key in ("type", "drain", "gate", "source", "bulk")}
        row.update(beta=beta, vto=vto, gamma=gamma, phi=phi)
        row["lambda"] = lam
        mos_rows.append(row)
    n = 6
    circuit = build_circuit(
        n,
        spice_diodes=spice_diode_table(diodes),
        gummel_poon=gummel_poon_table(gp_rows),
        mos4=mos4_table(mos_rows),
        node_count=0,
    )
    assert circuit.card_device_counts == {"spice_diode": 3, "gummel_poon": 2, "mos4": 2}
    assert circuit.device_counts == {"diode": 0, "mos": 0, "bjt": 0, "behavioral": 0}
    bv_eff = circuit.spice_diode_breakdown_voltages()

    def volt(x: np.ndarray, index: int) -> float:
        return 0.0 if index < 0 else float(x[index])

    for x in _circuit_states(n, 300, 11):
        f, jacobian = core.device_f_jacobian_dense(circuit, x, 0.0)
        f = f.reshape(-1)
        f_ref = np.zeros(n)
        j_ref = np.zeros((n, n))
        f_mag = np.zeros(n)  # largest single contribution per row (the scale of a sum that may cancel)
        j_mag = np.zeros(n)

        def scatter(rows, cols, currents, jac):
            for a, r in enumerate(rows):
                if r < 0:
                    continue
                f_ref[r] += currents[a]
                f_mag[r] = max(f_mag[r], abs(float(currents[a])))
                for b, c in enumerate(cols):
                    if c >= 0:
                        j_ref[r, c] += jac[a][b]
                        j_mag[r] = max(j_mag[r], abs(float(jac[a][b])))

        for d, eff in zip(diodes, bv_eff):
            v = volt(x, d["anode"]) - volt(x, d["cathode"])
            i, g = spice_diode(v, d["is"], d["n"], breakdown_voltage(d["is"], d["bv"], d["ibv"]))
            assert eff == breakdown_voltage(d["is"], d["bv"], d["ibv"]) or abs(eff - breakdown_voltage(d["is"], d["bv"], d["ibv"])) <= 1e-12 * eff
            scatter([d["anode"], d["cathode"]], [d["anode"], d["cathode"]], [float(i), -float(i)], [[float(g), -float(g)], [-float(g), float(g)]])
        for bjt in bjts:
            terms = [bjt["collector"], bjt["base"], bjt["emitter"]]
            polarity = -1 if bjt["type"] == "QPNP" else 1
            currents, jac = gummel_poon(*(volt(x, t) for t in terms), bjt["card"], polarity)
            scatter(terms, terms, currents, jac)
        for mos in moses:
            terms = [mos["drain"], mos["gate"], mos["source"], mos["bulk"]]
            polarity = -1 if mos["type"] == "PMOS" else 1
            ids, grad = mos_card(*(volt(x, t) for t in terms), polarity, *mos["card"])
            scatter([mos["drain"], mos["source"]], terms, [float(ids), -float(ids)], [grad, -grad])
        # Row-wise tolerance: the largest single device contribution to the row sets the scale.
        for r in range(n):
            _assert_matches(f[r], f_ref[r], f"f[{r}] at {x}", floor=f_mag[r])
            _assert_matches(jacobian[r], j_ref[r], f"J[{r}] at {x}", floor=j_mag[r])
