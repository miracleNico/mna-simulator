"""End-to-end import of vendor-format model cards and macromodels (phase M3).

``tests/vendor/vendor_parts.lib`` holds cards in the formats manufacturers and SPICE
distributions publish (parenthesised and bare parameter lists, ``+`` continuation lines,
string-valued bookkeeping fields such as ``mfg=OnSemi``, LEVEL=3 MOS cards, a VDMOS card,
``.subckt`` macromodels with ``PARAMS:`` and ``{...}`` expressions). The tests check the
physics the equations must reproduce and the parser/builder contract, not datasheet digits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest
from mna_simulation.api.service import SimulationService
from mna_simulation.card_devices import CARD_VT
from mna_simulation.errors import NetlistError
from mna_simulation.netlist import parse_netlist_text

VENDOR_DIR = Path(__file__).parent / "vendor"
LIB = (VENDOR_DIR / "vendor_parts.lib").read_text()


def _run(deck: str, mode: AnalysisMode = AnalysisMode.OP, **options):
    request = AnalysisRequest(netlist_text=LIB + "\n" + deck, mode=mode, options=options)
    return SimulationService().run_request(request)


def _dc(response) -> dict[str, float]:
    return dict(zip(response.labels, response.dc_solution))


def test_library_parses_with_only_bookkeeping_diagnostics() -> None:
    circuit = parse_netlist_text(LIB + "\nR1 a 0 1k\nV1 a 0 1\n.op")
    assert [c.name for c in circuit.components] == ["R1", "V1"]
    # Unreferenced cards are not reported one by one; the unsupported VDMOS card is named once.
    assert any("BS170" in d and "VDMOS" in d for d in circuit.diagnostics)


def test_1n4148_forward_drop_follows_the_card() -> None:
    response = _run("V1 in 0 5\nR1 in a 4.3k\nD1 a 0 1N4148\n.op")
    assert response.status == "ok"
    v = _dc(response)
    i = (5.0 - v["V(a)"]) / 4.3e3
    expected = 1.752 * CARD_VT * math.log(i / 2.52e-9 + 1.0) + i * 0.568
    assert v["V(a)"] == pytest.approx(expected, rel=1e-6)
    # String-valued bookkeeping fields (mfg=, type=) are reported, not rejected.
    assert any("1N4148" in d and "mfg" in d for d in response.diagnostics)


def test_1n4007_half_wave_rectifier_charges_the_filter_capacitor() -> None:
    response = _run(
        "V1 in 0 SIN(0 10 50)\nD1 in out D1N4007\nR1 out 0 1k\nC1 out 0 100u\n.tran 60m 50u",
        mode=AnalysisMode.TRAN,
    )
    assert response.status == "ok"
    waveform = response.waveform
    out = np.asarray(waveform["values"][waveform["labels"].index("V(out)")])
    assert 8.0 < out.max() < 10.0  # one diode drop below the 10 V peak
    assert out[-1] > 6.0  # tau = 100 ms holds most of the charge over a 20 ms period


def test_zener_card_regulates_near_bv() -> None:
    response = _run("V1 in 0 12\nR1 in z 1k\nDz 0 z DZ5V1\n.op")
    assert response.status == "ok"
    assert _dc(response)["V(z)"] == pytest.approx(5.1, abs=0.1)


def test_2n2222a_common_emitter_bias_is_forward_active() -> None:
    response = _run("Vcc vcc 0 12\nRc vcc c 2.2k\nRb vcc b 470k\nQ1 c b 0 Q2N2222A\n.op")
    assert response.status == "ok"
    v = _dc(response)
    ib = (12.0 - v["V(b)"]) / 470e3
    ic = (12.0 - v["V(c)"]) / 2.2e3
    assert 0.6 < v["V(b)"] < 0.75
    assert 100 < ic / ib < 300  # BF=255.9 with the Early and leakage terms of the card
    assert v["V(c)"] > v["V(b)"]  # forward active, not saturated


def test_2n2907a_pnp_emitter_follower_stage() -> None:
    response = _run("Vcc vcc 0 12\nRe vcc e 1k\nRb b 0 100k\nQ1 c b e Q2N2907A\nRc c 0 1k\n.op")
    assert response.status == "ok"
    v = _dc(response)
    assert 0.55 < v["V(e)"] - v["V(b)"] < 0.8  # emitter-base forward biased (PNP)
    assert v["V(c)"] < v["V(e)"]


def test_level3_card_turns_the_switch_fully_on_and_off() -> None:
    on = _run("Vdd vdd 0 10\nRd vdd d 1k\nVg g 0 5\nM1 d g 0 0 M2N7000\n.op")
    off = _run("Vdd vdd 0 10\nRd vdd d 1k\nVg g 0 1\nM1 d g 0 0 M2N7000\n.op")
    assert _dc(on)["V(d)"] < 0.1
    assert _dc(off)["V(d)"] == pytest.approx(10.0, abs=1e-6)
    assert any("LEVEL=3" in d for d in on.diagnostics)


def test_kp_is_derived_from_uo_and_tox_when_the_card_omits_it() -> None:
    response = _run("Vdd vdd 0 5\nRd vdd d 500\nVg g 0 3\nM1 d g 0 0 MNOKP W=10u L=1u\n.op")
    assert response.status == "ok"
    kp = 550e-4 * 3.9 * 8.854214871e-12 / 20e-9  # SPICE: UO * eps_ox / TOX
    beta = kp * 10e-6 / 1e-6
    # Saturation: Id = beta/2 (Vgs - Vto)^2 (1 + lambda Vds), bulk tied to source (no body effect).
    vd = _dc(response)["V(d)"]
    i_d = (5.0 - vd) / 500.0
    assert i_d == pytest.approx(0.5 * beta * (3.0 - 0.7) ** 2 * (1 + 0.05 * vd), rel=1e-6)


def test_vdmos_card_is_a_clear_error_when_used() -> None:
    with pytest.raises(NetlistError) as info:
        _run("Vdd vdd 0 10\nRd vdd d 1k\nVg g 0 5\nM1 d g 0 BS170\n.op")
    assert "BS170" in info.value.message and "VDMOS" in info.value.message


def test_opamp_macromodel_inverting_gain_and_bandwidth() -> None:
    op = _run(
        "Vcc vcc 0 15\nVee vee 0 -15\nVin in 0 0.1\nRi in inn 1k\nRf inn out 10k\n"
        "X1 0 inn out vcc vee OPAMP1 PARAMS: Aol=2e5\n.op"
    )
    assert _dc(op)["V(out)"] == pytest.approx(-1.0, rel=1e-3)
    ac = _run(
        "Vcc vcc 0 15\nVee vee 0 -15\nVin in 0 AC 1\nRi in inn 1k\nRf inn out 10k\n"
        "X1 0 inn out vcc vee OPAMP1\n.ac 10 1e7 100",
        mode=AnalysisMode.AC,
    )
    assert ac.status == "ok"
    spectrum = ac.spectrum
    freqs = np.asarray(spectrum["frequencies"])
    gain = np.abs(np.asarray(spectrum["magnitudes"][spectrum["labels"].index("V(out)")]))
    assert gain[0] == pytest.approx(10.0, rel=1e-2)
    # Closed-loop bandwidth GBW / (1 + Rf/Ri) = 1 MHz / 11: -3 dB near 91 kHz.
    f3 = freqs[np.argmin(np.abs(gain - 10.0 / math.sqrt(2.0)))]
    assert 6e4 < f3 < 1.3e5


def test_mirror_macromodel_bias_current_has_spice_polarity() -> None:
    response = _run("Vcc vcc 0 5\nX1 n1 n2 vcc MIRROR\nRl vcc n2 1k\n.op")
    assert response.status == "ok"
    v = _dc(response)
    assert 0.6 < v["V(n1)"] < 0.75  # the diode-connected transistor carries the bias current
    assert (5.0 - v["V(n2)"]) / 1e3 == pytest.approx(1e-3, rel=0.1)  # mirrored into the load


def test_include_through_the_cli_search_path(tmp_path: Path) -> None:
    import subprocess
    import sys

    deck = tmp_path / "deck" / "top.cir"
    deck.parent.mkdir()
    deck.write_text('.include "vendor_parts.lib"\nV1 in 0 5\nR1 in a 4.3k\nD1 a 0 1N4148\n.op\n')
    result = subprocess.run(
        [sys.executable, "-m", "mna_simulation.cli", "run", str(deck), "--lib", str(VENDOR_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.lower()
