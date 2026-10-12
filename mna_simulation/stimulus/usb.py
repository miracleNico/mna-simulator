"""USB 2.0 low-speed and full-speed packets as D+/D- waveforms.

A packet on the bus (USB 2.0 chapter 7 and 8) is, bit time by bit time:

1. idle ``J`` (``idle_before`` bit times);
2. SYNC: the eight bits ``00000001`` (``KJKJKJKK`` on the bus);
3. the PID byte: the 4-bit code and its complement in the upper nibble, LSB first;
4. the fields of the packet type, LSB first: a token (``OUT`` ``IN`` ``SETUP``) carries the
   7-bit address and the 4-bit endpoint, ``SOF`` the 11-bit frame number, both followed by
   CRC5; a data packet (``DATA0`` ``DATA1``) its payload bytes followed by CRC16; a
   handshake (``ACK`` ``NAK`` ``STALL``) nothing;
5. bit stuffing over SYNC, PID, fields and CRC: a 0 after every six consecutive 1s
   (the final 1 of SYNC counts), then NRZI from the idle ``J`` (a 0 toggles J/K, a 1 holds);
6. EOP: SE0 (both lines low) for two bit times, then ``J`` for one bit time;
7. idle ``J`` (``idle_after`` bit times).

CRCs (USB 2.0 section 8.3.5): a shift register preset to all ones, the field bits fed in
transmission order, the remainder inverted and sent most significant bit first; CRC5 uses
``x^5 + x^2 + 1`` and leaves the residual ``0b01100`` when a receiver runs the field and its
CRC through the register, CRC16 uses ``x^16 + x^15 + x^2 + 1`` with residual ``0x800D``.

Line levels are single-ended ``v_low``/``v_high`` (0 / 3.3 V by default). Full speed:
``J`` is D+ high (12 Mbit/s, 83.33 ns bits, 8 ns edges by default); low speed: ``J`` is D-
high (1.5 Mbit/s, 666.7 ns bits, 150 ns edges). ``j_is_dplus_high`` overrides the polarity.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence

from .signal import DigitalSignal

__all__ = [
    "CRC16_POLY",
    "CRC16_RESIDUAL",
    "CRC5_POLY",
    "CRC5_RESIDUAL",
    "FULL_SPEED_BIT_PERIOD",
    "FULL_SPEED_EDGE",
    "LOW_SPEED_BIT_PERIOD",
    "LOW_SPEED_EDGE",
    "PID_CODES",
    "SYNC_BITS",
    "crc16",
    "crc16_bits",
    "crc5",
    "crc5_bits",
    "crc_register",
    "decode_line_states",
    "full_speed_packet",
    "line_states",
    "low_speed_packet",
    "nrzi_states",
    "packet_bits",
    "pid_byte",
    "stuff_bits",
    "unstuff_bits",
]

FULL_SPEED_BIT_PERIOD = 1.0 / 12e6
LOW_SPEED_BIT_PERIOD = 1.0 / 1.5e6
FULL_SPEED_EDGE = 8e-9
LOW_SPEED_EDGE = 150e-9
FULL_SPEED_MAX_PAYLOAD = 1023
LOW_SPEED_MAX_PAYLOAD = 8

PID_CODES = {
    "OUT": 0b0001,
    "IN": 0b1001,
    "SOF": 0b0101,
    "SETUP": 0b1101,
    "DATA0": 0b0011,
    "DATA1": 0b1011,
    "ACK": 0b0010,
    "NAK": 0b1010,
    "STALL": 0b1110,
}
TOKEN_PIDS = frozenset({"OUT", "IN", "SETUP"})
DATA_PIDS = frozenset({"DATA0", "DATA1"})
HANDSHAKE_PIDS = frozenset({"ACK", "NAK", "STALL"})

SYNC_BITS = (0, 0, 0, 0, 0, 0, 0, 1)
CRC5_POLY = 0b00101
CRC5_RESIDUAL = 0b01100
CRC16_POLY = 0x8005
CRC16_RESIDUAL = 0x800D
J, K, SE0 = "J", "K", "SE0"


def lsb_first(value: int, width: int) -> list[int]:
    """The ``width`` low bits of ``value``, least significant first (USB field order)."""

    return [(int(value) >> i) & 1 for i in range(width)]


def bytes_to_bits(data: bytes | bytearray | Sequence[int]) -> list[int]:
    """Bytes as bits in transmission order (each byte LSB first)."""

    bits: list[int] = []
    for index, byte in enumerate(data):
        if isinstance(byte, bool) or not isinstance(byte, numbers.Integral) or not 0 <= int(byte) <= 255:
            raise ValueError(f"payload byte {index} must be an integer 0..255 (got {byte!r})")
        bits.extend(lsb_first(int(byte), 8))
    return bits


def crc_register(bits: Iterable[int], poly: int, width: int) -> int:
    """The CRC shift register after ``bits`` (preset to all ones; feedback = incoming bit XOR MSB)."""

    mask = (1 << width) - 1
    register = mask
    for bit in bits:
        feedback = ((register >> (width - 1)) & 1) ^ (int(bit) & 1)
        register = (register << 1) & mask
        if feedback:
            register ^= poly
    return register


def _msb_first(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def crc5(bits: Iterable[int]) -> int:
    """CRC5 of token/SOF field bits (transmission order); bit 4 of the result is sent first."""

    return crc_register(bits, CRC5_POLY, 5) ^ 0x1F


def crc5_bits(bits: Iterable[int]) -> list[int]:
    """The five CRC5 bits in transmission order."""

    return _msb_first(crc5(bits), 5)


def crc16(data: bytes | bytearray | Sequence[int]) -> int:
    """CRC16 of a data payload; bit 15 of the result is sent first."""

    return crc_register(bytes_to_bits(data), CRC16_POLY, 16) ^ 0xFFFF


def crc16_bits(data: bytes | bytearray | Sequence[int]) -> list[int]:
    """The sixteen CRC16 bits in transmission order."""

    return _msb_first(crc16(data), 16)


def pid_byte(pid: str) -> int:
    """The PID byte: the 4-bit code in the low nibble, its complement in the high nibble."""

    code = PID_CODES[_pid_name(pid)]
    return code | ((~code & 0xF) << 4)


def _pid_name(pid: str) -> str:
    if not isinstance(pid, str) or pid.upper() not in PID_CODES:
        raise ValueError(f"unknown PID {pid!r} (one of {', '.join(PID_CODES)})")
    return pid.upper()


def _field(value: object, name: str, width: int) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= int(value) < (1 << width):
        raise ValueError(f"{name} must be an integer 0..{(1 << width) - 1} (got {value!r})")
    return int(value)


def packet_bits(
    pid: str,
    payload: bytes | bytearray | Sequence[int] = b"",
    address: int | None = None,
    endpoint: int | None = None,
    frame: int | None = None,
    *,
    max_payload: int = FULL_SPEED_MAX_PAYLOAD,
) -> list[int]:
    """PID, fields and CRC in transmission order (no SYNC, not stuffed, not NRZI-coded).

    ``address``/``endpoint`` (default 0) belong to token PIDs, ``frame`` (default 0) to
    ``SOF``, ``payload`` to ``DATA0``/``DATA1`` (at most ``max_payload`` bytes); giving a
    field the PID does not carry is a ``ValueError``.
    """

    name = _pid_name(pid)
    payload_bytes = bytes(payload) if isinstance(payload, (bytes, bytearray)) else payload
    if name not in TOKEN_PIDS and (address is not None or endpoint is not None):
        raise ValueError(f"{name} carries no address or endpoint (token PIDs: OUT IN SETUP)")
    if name != "SOF" and frame is not None:
        raise ValueError(f"{name} carries no frame number (SOF does)")
    if name not in DATA_PIDS and len(payload_bytes):
        raise ValueError(f"{name} carries no payload (data PIDs: DATA0 DATA1)")
    bits = lsb_first(pid_byte(name), 8)
    if name in TOKEN_PIDS:
        fields = lsb_first(_field(address, "address", 7), 7) + lsb_first(_field(endpoint, "endpoint", 4), 4)
        bits += fields + crc5_bits(fields)
    elif name == "SOF":
        fields = lsb_first(_field(frame, "frame", 11), 11)
        bits += fields + crc5_bits(fields)
    elif name in DATA_PIDS:
        if len(payload_bytes) > max_payload:
            raise ValueError(f"payload of {len(payload_bytes)} bytes exceeds the maximum of {max_payload}")
        bits += bytes_to_bits(payload_bytes) + crc16_bits(payload_bytes)
    return bits


def stuff_bits(bits: Iterable[int], max_run: int = 6) -> list[int]:
    """A 0 inserted after every ``max_run`` consecutive 1s."""

    out: list[int] = []
    run = 0
    for bit in bits:
        out.append(int(bit))
        run = run + 1 if bit else 0
        if run == max_run:
            out.append(0)
            run = 0
    return out


def unstuff_bits(bits: Iterable[int], max_run: int = 6) -> list[int]:
    """Inverse of :func:`stuff_bits`: drops the 0 after every ``max_run`` 1s (``ValueError`` if it is a 1)."""

    out: list[int] = []
    run = 0
    skip = False
    for index, bit in enumerate(bits):
        if skip:
            if bit:
                raise ValueError(f"bit stuffing error at bit {index}: expected a stuffed 0 after {max_run} ones")
            skip = False
            run = 0
            continue
        out.append(int(bit))
        run = run + 1 if bit else 0
        if run == max_run:
            skip = True
    return out


def nrzi_states(bits: Iterable[int], initial: str = J) -> list[str]:
    """NRZI: a 0 toggles between ``J`` and ``K``, a 1 keeps the state (``initial`` before the first bit)."""

    state = initial
    states: list[str] = []
    for bit in bits:
        if not bit:
            state = K if state == J else J
        states.append(state)
    return states


def line_states(
    pid: str,
    payload: bytes | bytearray | Sequence[int] = b"",
    address: int | None = None,
    endpoint: int | None = None,
    frame: int | None = None,
    *,
    idle_before: int = 2,
    idle_after: int = 2,
    max_payload: int = FULL_SPEED_MAX_PAYLOAD,
) -> list[str]:
    """Bus state (``"J"``, ``"K"``, ``"SE0"``) of every bit time of a packet, idle included."""

    for label, count in (("idle_before", idle_before), ("idle_after", idle_after)):
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
            raise ValueError(f"{label} must be a non-negative integer (got {count!r})")
    bits = list(SYNC_BITS) + packet_bits(pid, payload, address, endpoint, frame, max_payload=max_payload)
    coded = nrzi_states(stuff_bits(bits), initial=J)
    return [J] * int(idle_before) + coded + [SE0, SE0, J] + [J] * int(idle_after)


def decode_line_states(states: Sequence[str]) -> list[int]:
    """The packet bits (PID onward, as :func:`packet_bits`) of a bus-state sequence.

    Skips the idle ``J`` before the packet, NRZI-decodes up to the EOP's SE0, removes the
    stuffed bits and the SYNC; ``ValueError`` for a missing SYNC or a stuffing error.
    """

    start = 0
    while start < len(states) and states[start] == J:
        start += 1
    end = start
    while end < len(states) and states[end] != SE0:
        end += 1
    previous = J
    bits: list[int] = []
    for state in states[start:end]:
        bits.append(1 if state == previous else 0)
        previous = state
    bits = unstuff_bits(bits)
    if tuple(bits[:8]) != SYNC_BITS:
        raise ValueError(f"no SYNC at the start of the packet (got {bits[:8]})")
    return bits[8:]


def _signals(
    states: Sequence[str],
    *,
    bit_period: float,
    t_rise: float,
    t_fall: float,
    t_start: float,
    v_low: float,
    v_high: float,
    j_is_dplus_high: bool,
) -> tuple[DigitalSignal, DigitalSignal]:
    j_plus = 1 if j_is_dplus_high else 0
    dplus = [j_plus if s == J else 1 - j_plus if s == K else 0 for s in states]
    dminus = [1 - j_plus if s == J else j_plus if s == K else 0 for s in states]
    template = DigitalSignal(bit_period, v_low=v_low, v_high=v_high, t_rise=t_rise, t_fall=t_fall, t_start=t_start)
    return template.from_bits(dplus), template.from_bits(dminus)


def _packet(
    pid: str,
    payload: bytes | bytearray | Sequence[int],
    address: int | None,
    endpoint: int | None,
    frame: int | None,
    j_is_dplus_high: bool,
    *,
    bit_period: float,
    t_rise: float,
    t_fall: float,
    t_start: float,
    idle_before: int,
    idle_after: int,
    v_low: float,
    v_high: float,
    max_payload: int,
) -> tuple[DigitalSignal, DigitalSignal]:
    states = line_states(
        pid,
        payload,
        address,
        endpoint,
        frame,
        idle_before=idle_before,
        idle_after=idle_after,
        max_payload=max_payload,
    )
    return _signals(
        states,
        bit_period=bit_period,
        t_rise=t_rise,
        t_fall=t_fall,
        t_start=t_start,
        v_low=v_low,
        v_high=v_high,
        j_is_dplus_high=bool(j_is_dplus_high),
    )


def full_speed_packet(
    pid: str,
    payload: bytes | bytearray | Sequence[int] = b"",
    address: int | None = None,
    endpoint: int | None = None,
    frame: int | None = None,
    j_is_dplus_high: bool = True,
    *,
    t_rise: float = FULL_SPEED_EDGE,
    t_fall: float = FULL_SPEED_EDGE,
    t_start: float = 0.0,
    idle_before: int = 2,
    idle_after: int = 2,
    v_low: float = 0.0,
    v_high: float = 3.3,
    bit_period: float = FULL_SPEED_BIT_PERIOD,
) -> tuple[DigitalSignal, DigitalSignal]:
    """``(dplus, dminus)`` of a full-speed packet (12 Mbit/s; J = D+ high by default).

    ``payload`` holds at most 1023 bytes. The first bit time (idle J) starts at ``t_start``.
    """

    return _packet(
        pid,
        payload,
        address,
        endpoint,
        frame,
        j_is_dplus_high,
        bit_period=bit_period,
        t_rise=t_rise,
        t_fall=t_fall,
        t_start=t_start,
        idle_before=idle_before,
        idle_after=idle_after,
        v_low=v_low,
        v_high=v_high,
        max_payload=FULL_SPEED_MAX_PAYLOAD,
    )


def low_speed_packet(
    pid: str,
    payload: bytes | bytearray | Sequence[int] = b"",
    address: int | None = None,
    endpoint: int | None = None,
    frame: int | None = None,
    j_is_dplus_high: bool = False,
    *,
    t_rise: float = LOW_SPEED_EDGE,
    t_fall: float = LOW_SPEED_EDGE,
    t_start: float = 0.0,
    idle_before: int = 2,
    idle_after: int = 2,
    v_low: float = 0.0,
    v_high: float = 3.3,
    bit_period: float = LOW_SPEED_BIT_PERIOD,
) -> tuple[DigitalSignal, DigitalSignal]:
    """``(dplus, dminus)`` of a low-speed packet (1.5 Mbit/s; J = D- high by default).

    ``payload`` holds at most 8 bytes. The first bit time (idle J) starts at ``t_start``.
    """

    return _packet(
        pid,
        payload,
        address,
        endpoint,
        frame,
        j_is_dplus_high,
        bit_period=bit_period,
        t_rise=t_rise,
        t_fall=t_fall,
        t_start=t_start,
        idle_before=idle_before,
        idle_after=idle_after,
        v_low=v_low,
        v_high=v_high,
        max_payload=LOW_SPEED_MAX_PAYLOAD,
    )
