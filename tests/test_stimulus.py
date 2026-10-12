"""Python-defined digital stimulus (MIXED_SIGNAL.md phase D1): ``mna_simulation.stimulus``.

* ``DigitalSignal``: constructors, transforms (NRZI, bit stuffing, invert, delay, differential),
  rendering (corner points, PWL netlist line, request stimulus).
* ``stimulus.usb``: PID bytes, CRC5/CRC16 (residuals and hand-derived vectors), bit stuffing,
  NRZI of SYNC, EOP timing, packet length in bit times, polarity.
* Request stimuli: validation, ``.op``/``.ac``/``.tran`` semantics.
* Acceptance (section 4 item 4): a 64-byte DATA0 full-speed packet into a receiver front end
  (terminations, series resistors, ESD diodes, 10 pF loads, a Gummel-Poon differential pair)
  at ``t_step = 1 ns``; the D+ waveform read at the bit centres is the transmitted NRZI stream
  and the differential pair's output decodes to the packet; under 2 s with ``--run-perf``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, CircuitIR, ComponentRecord
from mna_simulation.core_basic import run_basic_analysis
from mna_simulation.errors import NetlistError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.stimulus import DigitalSignal, usb
from mna_simulation.table_sources import stimulus_records

# ---------------------------------------------------------------------------
# DigitalSignal
# ---------------------------------------------------------------------------


def test_from_bits_renders_two_corners_per_edge() -> None:
    signal = DigitalSignal.from_bits([0, 1, 1, 0], bit_period=10e-9, v_low=0.0, v_high=3.3, t_rise=1e-9, t_fall=2e-9)
    np.testing.assert_allclose(
        signal.points(),
        [[0.0, 0.0], [10e-9, 0.0], [11e-9, 3.3], [30e-9, 3.3], [32e-9, 0.0], [40e-9, 0.0]],
        rtol=1e-15,
        atol=0,
    )
    assert signal.bits == (0, 1, 1, 0) and signal.duration == 40e-9
    # The instance form uses the instance's timing; t_start shifts everything.
    template = DigitalSignal(10e-9, t_rise=1e-9, t_fall=2e-9, t_start=5e-9)
    shifted = template.from_bits([0, 1, 1, 0])
    np.testing.assert_allclose(shifted.points()[:, 0], signal.points()[:, 0] + 5e-9, rtol=0, atol=1e-24)
    # Default edges: a tenth of the bit.
    assert DigitalSignal(1e-6).t_rise == pytest.approx(1e-7)


def test_from_string_events_and_numpy() -> None:
    a = DigitalSignal.from_string("0b0110", bit_period=1e-6)
    b = DigitalSignal.from_bits([0, 1, 1, 0], bit_period=1e-6)
    np.testing.assert_array_equal(a.points(), b.points())
    assert DigitalSignal.from_string("01 10_1", bit_period=1e-6).bits == (0, 1, 1, 0, 1)
    with pytest.raises(ValueError, match="unexpected character '2'"):
        DigitalSignal.from_string("0120", bit_period=1e-6)
    events = DigitalSignal.from_events([(0.0, 1), (5e-9, 0), (9e-9, 0), (12e-9, 1)], v_high=1.8, t_rise=1e-9, t_fall=1e-9)
    np.testing.assert_allclose(
        events.points(), [[0.0, 1.8], [5e-9, 1.8], [6e-9, 0.0], [12e-9, 0.0], [13e-9, 1.8]], rtol=0, atol=1e-24
    )
    with pytest.raises(ValueError, match="starts before the previous edge"):
        DigitalSignal.from_events([(0.0, 0), (1e-9, 1), (1.5e-9, 0)], t_rise=1e-9, t_fall=1e-9)
    with pytest.raises(ValueError, match="strictly increasing"):
        DigitalSignal.from_events([(1.0, 0), (1.0, 1)], t_rise=1e-9, t_fall=1e-9)
    raw = DigitalSignal.from_numpy([0.0, 1.0, 2.0], [0.0, 2.0, 1.0], t_start=1.0)
    np.testing.assert_array_equal(raw.points(), [[1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        DigitalSignal.from_numpy([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="bit 1 must be 0 or 1"):
        DigitalSignal.from_bits([0, 2], bit_period=1e-6)
    with pytest.raises(ValueError, match="must not exceed bit_period"):
        DigitalSignal(1e-9, t_rise=2e-9)
    with pytest.raises(ValueError, match="needs a bit_period"):
        DigitalSignal(t_rise=1e-9, t_fall=1e-9).from_bits([1])


def test_nrzi_bit_stuff_invert_delay_differential() -> None:
    signal = DigitalSignal.from_bits([0, 0, 0, 0, 0, 0, 0, 1], bit_period=1.0, t_rise=0.1, t_fall=0.1)
    # USB SYNC: from idle J (1), each 0 toggles: K J K J K J K, then the 1 holds K.
    assert signal.nrzi().bits == (0, 1, 0, 1, 0, 1, 0, 0)
    assert signal.nrzi(initial=0).bits == (1, 0, 1, 0, 1, 0, 1, 1)
    ones = DigitalSignal.from_bits([1] * 13 + [0, 1], bit_period=1.0)
    assert ones.bit_stuff().bits == (1,) * 6 + (0,) + (1,) * 6 + (0,) + (1, 0, 1)
    assert ones.bit_stuff(max_run=3).bits == (1, 1, 1, 0) * 4 + (1, 0, 1)
    inverted = signal.invert()
    assert inverted.bits == (1, 1, 1, 1, 1, 1, 1, 0)
    # Inversion mirrors the voltages and keeps every edge in time.
    asym = DigitalSignal.from_bits([0, 1, 0], bit_period=10.0, t_rise=1.0, t_fall=3.0, v_low=0.5, v_high=2.5)
    mirrored = asym.invert().points()
    np.testing.assert_array_equal(mirrored[:, 0], asym.points()[:, 0])
    np.testing.assert_array_equal(mirrored[:, 1], 3.0 - asym.points()[:, 1])
    delayed = asym.delay(2.5)
    np.testing.assert_array_equal(delayed.points()[:, 0], asym.points()[:, 0] + 2.5)
    positive, negative = asym.differential()
    np.testing.assert_array_equal(positive.points(), asym.points())
    np.testing.assert_array_equal(negative.points(), mirrored)
    pos, neg = asym.differential(v_common=1.0)
    assert (pos.v_low, pos.v_high, neg.v_low, neg.v_high) == (0.0, 2.0, 0.0, 2.0)
    np.testing.assert_allclose(pos.points()[:, 1] + neg.points()[:, 1], 2.0)
    with pytest.raises(ValueError, match="needs a signal built from bits"):
        DigitalSignal.from_numpy([0.0], [1.0]).nrzi()


def test_netlist_line_and_stimulus_entry() -> None:
    signal = DigitalSignal.from_bits([0, 1], bit_period=1.0, t_rise=0.25, t_fall=0.25, v_high=1.2)
    line = signal.netlist_line("Vdp", "dp", "0")
    assert line == "Vdp dp 0 PWL(0.0 0.0 1.0 0.0 1.25 1.2 2.0 1.2)"
    # Float literals round-trip exactly.
    fine = DigitalSignal.from_bits([0, 1], bit_period=1e-6, t_rise=1e-8, t_fall=1e-8).netlist_line("V1", "a", "0")
    numbers = [float(token) for token in fine[fine.index("(") + 1 : -1].split()]
    np.testing.assert_array_equal(np.reshape(numbers, (-1, 2)), DigitalSignal.from_bits([0, 1], bit_period=1e-6, t_rise=1e-8, t_fall=1e-8).points())
    entry = signal.to_stimulus("Vdp", "dp", "0")
    assert entry == {"name": "Vdp", "node_p": "dp", "node_n": "0", "points": signal.points().tolist(), "period": None}
    assert signal.to_stimulus("Vclk", "clk", "0", period=2.0)["period"] == 2.0
    with pytest.raises(ValueError, match="must start with V"):
        signal.to_stimulus("I1", "a", "0")
    with pytest.raises(ValueError, match="not a valid node name"):
        signal.netlist_line("V1", "a b", "0")


# ---------------------------------------------------------------------------
# USB
# ---------------------------------------------------------------------------


def test_pid_bytes() -> None:
    assert {name: usb.pid_byte(name) for name in usb.PID_CODES} == {
        "OUT": 0xE1,
        "IN": 0x69,
        "SOF": 0xA5,
        "SETUP": 0x2D,
        "DATA0": 0xC3,
        "DATA1": 0x4B,
        "ACK": 0xD2,
        "NAK": 0x5A,
        "STALL": 0x1E,
    }
    with pytest.raises(ValueError, match="unknown PID"):
        usb.pid_byte("PING")


def test_crc5_hand_derived_vector_and_residual() -> None:
    # SETUP to address 0, endpoint 0: the 11 field bits are all 0. Register preset 11111;
    # per bit: feedback = register MSB XOR bit, shift left, XOR 00101 when feedback is 1:
    #   bit  1: fb 1  11110 ^ 00101 = 11011     bit  7: fb 1  10000 ^ 00101 = 10101
    #   bit  2: fb 1  10110 ^ 00101 = 10011     bit  8: fb 1  01010 ^ 00101 = 01111
    #   bit  3: fb 1  00110 ^ 00101 = 00011     bit  9: fb 0  11110
    #   bit  4: fb 0  00110                     bit 10: fb 1  11100 ^ 00101 = 11001
    #   bit  5: fb 0  01100                     bit 11: fb 1  10010 ^ 00101 = 10111
    #   bit  6: fb 0  11000
    # Remainder 10111, inverted: 01000 = 0x08, sent MSB first: 0 1 0 0 0. On the wire the token
    # bytes are 2D 00 10 (0x10 = bits 0000 1000 LSB first: endpoint bits 1-3 = 000, CRC 01000).
    fields = [0] * 11
    assert usb.crc_register(fields, usb.CRC5_POLY, 5) == 0b10111
    assert usb.crc5(fields) == 0x08 and usb.crc5_bits(fields) == [0, 1, 0, 0, 0]
    bits = usb.packet_bits("SETUP", address=0, endpoint=0)
    wire = bytes(sum(bit << i for i, bit in enumerate(bits[k : k + 8])) for k in range(0, 24, 8))
    assert wire == bytes([0x2D, 0x00, 0x10])
    # Published vectors (USB CRC white paper): address 0x15 endpoint 0xE -> 0x17, address 0x3A endpoint 0xA -> 0x1C.
    assert usb.crc5(usb.lsb_first(0x15, 7) + usb.lsb_first(0xE, 4)) == 0x17
    assert usb.crc5(usb.lsb_first(0x3A, 7) + usb.lsb_first(0xA, 4)) == 0x1C
    # Residual: field bits plus the transmitted CRC leave 01100 in the register.
    for address, endpoint in ((0, 0), (0x15, 0xE), (127, 15), (42, 3)):
        token = usb.packet_bits("IN", address=address, endpoint=endpoint)[8:]
        assert usb.crc_register(token, usb.CRC5_POLY, 5) == usb.CRC5_RESIDUAL
    for frame in (0, 1, 0x710, 2047):
        assert usb.crc_register(usb.packet_bits("SOF", frame=frame)[8:], usb.CRC5_POLY, 5) == usb.CRC5_RESIDUAL


def test_crc16_vectors_and_residual() -> None:
    # Empty payload: the register stays 0xFFFF (nothing shifted in), inverted 0x0000 -> wire C3 00 00.
    assert usb.crc16(b"") == 0x0000
    # Published vectors (USB CRC white paper): 00 01 02 03 -> 0xF75E, 23 45 67 89 -> 0x7038.
    assert usb.crc16(bytes([0x00, 0x01, 0x02, 0x03])) == 0xF75E
    assert usb.crc16(bytes([0x23, 0x45, 0x67, 0x89])) == 0x7038
    rng = np.random.default_rng(3)
    for size in (0, 1, 7, 64, 1023):
        payload = bytes(rng.integers(0, 256, size).tolist())
        bits = usb.packet_bits("DATA1", payload)[8:]
        assert len(bits) == 8 * size + 16
        assert usb.crc_register(bits, usb.CRC16_POLY, 16) == usb.CRC16_RESIDUAL


def test_bit_stuffing_of_an_all_ones_payload() -> None:
    payload = b"\xff" * 4
    packet = usb.packet_bits("DATA0", payload)
    stream = list(usb.SYNC_BITS) + packet
    stuffed = usb.stuff_bits(stream)
    # The 32 payload ones follow the PID byte C3 (LSB first 1100 0011: it ends in two 1s).
    # SYNC ends in a 1, so runs start counting there; every sixth 1 is followed by a stuffed 0.
    run = 0
    inserted = 0
    for index, bit in enumerate(stuffed):
        if run == 6:
            assert bit == 0, index
            inserted += 1
            run = 0
            continue
        run = run + 1 if bit else 0
        assert run <= 6
    assert len(stuffed) == len(stream) + inserted
    # PID ...11 + 32 ones = a run of 34 ones (5 stuffed zeros), CRC16 of FF FF FF FF adds its own.
    assert inserted >= 5
    assert usb.unstuff_bits(stuffed) == stream
    with pytest.raises(ValueError, match="bit stuffing error"):
        usb.unstuff_bits([1] * 7)


def test_sync_nrzi_eop_and_packet_length() -> None:
    states = usb.line_states("ACK", idle_before=3, idle_after=2)
    assert states[:3] == ["J"] * 3
    assert "".join(s[0] for s in states[3:11]) == "KJKJKJKK"  # SYNC on the bus
    # ACK: PID D2 = 0100 1011 LSB first; NRZI from K continues the stream, no stuffing needed.
    assert len(states) == 3 + 8 + 8 + 3 + 2
    assert states[-5:] == ["SE0", "SE0", "J", "J", "J"]
    assert usb.decode_line_states(states) == usb.packet_bits("ACK")
    payload = bytes(range(64))
    data = usb.line_states("DATA0", payload)
    stuffed = len(usb.stuff_bits(list(usb.SYNC_BITS) + usb.packet_bits("DATA0", payload))) - (8 + 8 + 64 * 8 + 16)
    assert len(data) == 2 + 8 + 8 + 64 * 8 + 16 + stuffed + 3 + 2
    assert usb.decode_line_states(data) == usb.packet_bits("DATA0", payload)


def test_full_speed_packet_levels_and_eop_timing() -> None:
    dplus, dminus = usb.full_speed_packet("SETUP", address=5, endpoint=1, t_start=1e-6)
    states = usb.line_states("SETUP", address=5, endpoint=1)
    bit = usb.FULL_SPEED_BIT_PERIOD
    assert dplus.bit_period == bit and dplus.t_rise == 8e-9 and dplus.t_start == 1e-6
    assert dplus.bits == tuple(1 if s == "J" else 0 for s in states)
    assert dminus.bits == tuple(1 if s == "K" else 0 for s in states)
    # EOP: SE0 (both lines low) for two bit times, then J for one bit time (and the idle J after).
    eop = states.index("SE0")
    assert states[eop : eop + 3] == ["SE0", "SE0", "J"]
    p, m = dplus.points(), dminus.points()

    def level(points: np.ndarray, t: float) -> float:
        return float(np.interp(t, points[:, 0], points[:, 1]))

    t_eop = 1e-6 + eop * bit
    for t in (t_eop + 0.5 * bit, t_eop + 1.5 * bit):
        assert level(p, t) == 0.0 and level(m, t) == 0.0
    assert level(p, t_eop + 2.5 * bit) == 3.3 and level(m, t_eop + 2.5 * bit) == 0.0
    # D+ is driven high again at the start of the EOP's J bit (8 ns edge).
    boundary = 1e-6 + (eop + 2) * bit
    assert level(p, boundary) == 0.0 and level(p, boundary + 8e-9) == 3.3
    # Low speed: J is D- high, 1.5 Mbit/s, 150 ns edges, at most 8 payload bytes.
    lp, lm = usb.low_speed_packet("DATA1", b"\x01\x02")
    assert lp.bit_period == pytest.approx(666.6666666666666e-9) and lp.t_rise == 150e-9
    assert lm.bits[0] == 1 and lp.bits[0] == 0
    with pytest.raises(ValueError, match="exceeds the maximum of 8"):
        usb.low_speed_packet("DATA0", bytes(9))
    with pytest.raises(ValueError, match="carries no payload"):
        usb.full_speed_packet("ACK", b"\x00")
    with pytest.raises(ValueError, match="address must be an integer 0..127"):
        usb.full_speed_packet("OUT", address=128)
    with pytest.raises(ValueError, match="carries no frame number"):
        usb.full_speed_packet("IN", frame=3)


# ---------------------------------------------------------------------------
# Request stimuli (record level)
# ---------------------------------------------------------------------------


def _load() -> list[ComponentRecord]:
    return [ComponentRecord("R1", "R", "dp", "0", "1k")]


def test_stimulus_validation_errors() -> None:
    good = {"name": "Vdp", "node_p": "dp", "node_n": "0", "points": [[0.0, 0.0], [1e-9, 3.3]]}
    assert len(stimulus_records([good], _load())) == 1
    cases = [
        ({**good, "points": [[0.0, 0.0], [0.0, 1.0]]}, "stimulus 'Vdp': times must be strictly increasing"),
        ({**good, "points": [[0.0, float("nan")]]}, "stimulus 'Vdp': value 0 is not finite"),
        ({**good, "points": []}, "stimulus 'Vdp': the table has no points"),
        ({**good, "points": [[0.0]]}, "every point must be a \\[t, v\\] pair"),
        ({**good, "points": [[0.0, True]]}, "every point must be a \\[t, v\\] pair"),
        ({**good, "name": "R1"}, "must be a voltage-source name"),
        ({**good, "name": "V1 x"}, "must be a voltage-source name"),
        ({**good, "node_p": "a b"}, "node_p 'a b' is not a valid node name"),
        ({**good, "period": -1.0}, "period must be finite and non-negative"),
        ({**good, "period": "1u"}, "period must be a number or null"),
    ]
    for entry, message in cases:
        with pytest.raises(NetlistError, match=message) as exc:
            stimulus_records([entry], _load())
        assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(NetlistError, match="has the name of an existing component"):
        stimulus_records([{**good, "name": "VR1"}], [*_load(), ComponentRecord("vr1", "V", "x", "0", "1", subtype="DC")])
    with pytest.raises(NetlistError, match="has the name of an existing component \\('Vdp'\\)"):
        stimulus_records([good, {**good, "name": "VDP"}], _load())
    with pytest.raises(NetlistError, match="at most 1000000"):
        stimulus_records([{**good, "points": np.zeros((1_000_001, 2))}], _load())


def test_stimulus_semantics_in_op_ac_and_tran() -> None:
    signal = DigitalSignal.from_events([(0.0, 1), (10e-9, 0)], v_high=2.0, t_rise=1e-9, t_fall=1e-9)
    records = _load() + stimulus_records([signal.to_stimulus("VDP", "DP", "0")], _load())
    assert records[1].node1 == "dp"  # node names match case-insensitively, the netlist spelling wins
    problem = build_mna_problem(CircuitIR(records))
    labels = problem.metadata["labels"]
    op = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.OP))
    assert op.dc_solution[labels.index("V(dp)"), 0] == 2.0  # .op: the value at t = 0
    assert "_stimulus_op_rhs" not in op.metadata and op.metadata["table_sources"][0]["kind"] == "table"
    ac = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.AC, ac_start=1.0, ac_stop=1e6, ac_points=3))
    assert np.all(ac.spectrum.magnitudes == 0.0)  # .ac: no AC part
    tran = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=20e-9, tran_step=1e-9))
    v = tran.waveform.values[labels.index("V(dp)")]
    np.testing.assert_allclose(
        np.interp([0.0, 10e-9, 10.5e-9, 11e-9, 20e-9], tran.waveform.time, v), [2.0, 2.0, 1.0, 0.0, 0.0], atol=1e-12
    )


# ---------------------------------------------------------------------------
# Acceptance: a 64-byte DATA0 full-speed packet into a receiver front end
# ---------------------------------------------------------------------------

D_ESD = {"model": "spice", "model_name": "D1N4148", "is": "2.52n", "n": "1.752", "rs": "0.568", "bv": "100", "ibv": "0.1u", "cjo": "4p"}
Q_RX = {
    "model": "gummel_poon",
    "model_name": "Q2N2222",
    "is": "14.34f",
    "bf": "255.9",
    "nf": "1",
    "br": "6.092",
    "nr": "1",
    "vaf": "74.03",
    "ikf": ".2847",
    "ise": "14.34f",
    "ne": "1.307",
    "isc": "0",
    "nc": "2",
    "ikr": "0",
    "rb": "10",
    "rc": "1",
    "cje": "22.01p",
    "cjc": "7.306p",
}
PAYLOAD = bytes((37 * i + 11) & 0xFF for i in range(64))


def _receiver_front_end() -> list[ComponentRecord]:
    def r(name: str, a: str, b: str, value: str) -> ComponentRecord:
        return ComponentRecord(name, "R", a, b, value)

    def c(name: str, a: str, b: str, value: str) -> ComponentRecord:
        return ComponentRecord(name, "C", a, b, value)

    def d(name: str, anode: str, cathode: str) -> ComponentRecord:
        return ComponentRecord(name, "D", anode, cathode, D_ESD["is"], metadata=dict(D_ESD))

    def q(name: str, collector: str, base: str, emitter: str) -> ComponentRecord:
        return ComponentRecord(name, "QNPN", collector, emitter, None, ctrl_node1=base, metadata=dict(Q_RX))

    return [
        ComponentRecord("VCC", "V", "vcc", "0", "3.3", subtype="DC"),
        ComponentRecord("VCC5", "V", "vcc5", "0", "5", subtype="DC"),
        r("RSP", "txp", "dp", "33"),  # series resistors
        r("RSM", "txm", "dm", "33"),
        r("RTP", "dp", "0", "90"),  # terminations
        r("RTM", "dm", "0", "90"),
        c("CLP", "dp", "0", "10p"),  # loads
        c("CLM", "dm", "0", "10p"),
        d("DP1", "dp", "vcc"),  # ESD diodes to the rails
        d("DP2", "0", "dp"),
        d("DM1", "dm", "vcc"),
        d("DM2", "0", "dm"),
        r("RBP", "dp", "bp", "100"),  # differential pair
        r("RBM", "dm", "bm", "100"),
        q("Q1", "c1", "bp", "em"),
        q("Q2", "c2", "bm", "em"),
        r("RC1", "vcc5", "c1", "1k"),
        r("RC2", "vcc5", "c2", "1k"),
        r("RE", "em", "0", "1.5k"),
    ]


def _usb_acceptance_run(num_threads: int = 1) -> tuple[object, list[str], object, float]:
    dplus, dminus = usb.full_speed_packet("DATA0", PAYLOAD)
    states = usb.line_states("DATA0", PAYLOAD)
    front_end = _receiver_front_end()
    stimuli = [dplus.to_stimulus("VDP", "txp", "0"), dminus.to_stimulus("VDM", "txm", "0")]
    circuit = CircuitIR(front_end + stimulus_records(stimuli, front_end))
    started = time.perf_counter()
    problem = build_mna_problem(circuit)
    t_stop = len(states) * usb.FULL_SPEED_BIT_PERIOD
    options = AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=t_stop, tran_step=1e-9, num_threads=num_threads)
    result = run_basic_analysis(problem, options)
    elapsed = time.perf_counter() - started
    return result, states, dplus, elapsed


def _bus_states(result: object, count: int) -> tuple[np.ndarray, list[str]]:
    waveform = result.waveform  # type: ignore[attr-defined]
    centres = (np.arange(count) + 0.5) * usb.FULL_SPEED_BIT_PERIOD

    def sample(label: str) -> np.ndarray:
        return np.interp(centres, waveform.time, waveform.values[waveform.labels.index(label)])

    v_dp, v_dm = sample("V(dp)"), sample("V(dm)")
    differential = sample("V(c2)") - sample("V(c1)")
    states = []
    for dp_level, dm_level, diff in zip(v_dp, v_dm, differential):
        if dp_level < 0.8 and dm_level < 0.8:
            states.append("SE0")
        else:
            states.append("J" if diff > 0.0 else "K")
    return v_dp, states


def test_usb_full_speed_packet_through_a_receiver_front_end() -> None:
    result, states, dplus, elapsed = _usb_acceptance_run()
    assert len(states) > 540  # 8 + 8 + 512 + 16 bits plus stuffing, EOP and idle
    time_grid = result.waveform.time
    corners = dplus.points()[:, 0]
    assert set(corners[corners <= time_grid[-1]].tolist()) <= set(time_grid.tolist())
    v_dp, receiver_states = _bus_states(result, len(states))
    # D+ read at the bit centres is the transmitted NRZI stream (J = high on full speed).
    decoded = [1 if level > 1.2 else 0 for level in v_dp]
    assert decoded == list(dplus.bits)
    # The differential pair's output decodes back to the packet.
    assert receiver_states == states
    assert usb.decode_line_states(receiver_states) == usb.packet_bits("DATA0", PAYLOAD)
    # 1 and 4 threads agree bitwise.
    four, _, _, _ = _usb_acceptance_run(num_threads=4)
    np.testing.assert_array_equal(four.waveform.values, result.waveform.values)


@pytest.mark.perf
def test_usb_acceptance_run_is_fast() -> None:
    _usb_acceptance_run()  # warm-up (imports, compile caches)
    _, _, _, elapsed = _usb_acceptance_run()
    assert elapsed < 2.0, f"acceptance run took {elapsed:.3f} s"
