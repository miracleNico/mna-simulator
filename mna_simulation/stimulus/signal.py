"""Digital waveforms as piecewise-linear sources (``DigitalSignal``).

A :class:`DigitalSignal` is a logic waveform with analog edges: two levels
(``v_low``/``v_high``), a rise time and a fall time. It is built from a bit
sequence (one level per ``bit_period`` slot), from transition events, or from
raw ``(t, v)`` points, transformed (NRZI, bit stuffing, inversion, delay,
differential pair), and rendered as the corner points of a ``PWL`` source:
:meth:`DigitalSignal.points` (an ``(N, 2)`` array), :meth:`DigitalSignal.netlist_line`
(a ``V... PWL(...)`` netlist line) or :meth:`DigitalSignal.to_stimulus` (the
``stimuli`` entry of an ``AnalysisRequest``).

Timing convention: an edge starts at its bit boundary (or event time) and reaches
the new level ``t_rise``/``t_fall`` later; edges must fit in a bit (``t_rise``,
``t_fall`` at most ``bit_period``). Pure Python and NumPy; no simulator import.

The constructors work on the class (timing as keywords) and on an instance (its
timing)::

    DigitalSignal.from_bits([1, 0, 1, 1], bit_period=1e-6, t_rise=10e-9, t_fall=10e-9)
    DigitalSignal(1e-6, t_rise=10e-9, t_fall=10e-9).from_string("1011")
"""

from __future__ import annotations

import functools
import math
import numbers
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = ["DigitalSignal"]

_TIMING = ("bit_period", "v_low", "v_high", "t_rise", "t_fall", "t_start")
_SOURCE_NAME = re.compile(r"[VvIi][A-Za-z0-9_]*\Z")
_NODE_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(f"{what} must be a finite number (got {value!r})")
    return float(value)


class _constructor:
    """A constructor usable on the class (timing keywords) and on an instance (its timing)."""

    def __init__(self, func: Any) -> None:
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj: Any, cls: type) -> Any:
        func = self.func
        if obj is None:

            @functools.wraps(func)
            def build(*args: Any, **kwargs: Any) -> Any:
                timing = {key: kwargs.pop(key) for key in _TIMING if key in kwargs}
                return func(cls(**timing), *args, **kwargs)

            return build

        @functools.wraps(func)
        def bound(*args: Any, **kwargs: Any) -> Any:
            return func(obj, *args, **kwargs)

        return bound


class DigitalSignal:
    """A logic waveform with analog edges, rendered as ``PWL`` corner points.

    ``bit_period`` (seconds, needed by the bit constructors), ``v_low``/``v_high``
    (the two levels), ``t_rise``/``t_fall`` (edge durations, default a tenth of
    ``bit_period``), ``t_start`` (the time of the first bit or the offset of the
    event and point times). A signal built from bits keeps them (``bits``), so
    :meth:`nrzi`, :meth:`bit_stuff` and :meth:`invert` act on the bit level.
    """

    __slots__ = ("bit_period", "v_low", "v_high", "t_rise", "t_fall", "t_start", "_bits", "_events", "_points")

    def __init__(
        self,
        bit_period: float | None = None,
        v_low: float = 0.0,
        v_high: float = 3.3,
        t_rise: float | None = None,
        t_fall: float | None = None,
        t_start: float = 0.0,
    ) -> None:
        if bit_period is not None:
            bit_period = _finite(bit_period, "bit_period")
            if bit_period <= 0.0:
                raise ValueError(f"bit_period must be positive (got {bit_period!r})")
        default_edge = None if bit_period is None else 0.1 * bit_period
        rise = default_edge if t_rise is None else _finite(t_rise, "t_rise")
        fall = default_edge if t_fall is None else _finite(t_fall, "t_fall")
        for name, edge in (("t_rise", rise), ("t_fall", fall)):
            if edge is not None and edge <= 0.0:
                raise ValueError(f"{name} must be positive (got {edge!r})")
            if edge is not None and bit_period is not None and edge > bit_period:
                raise ValueError(f"{name} ({edge!r}) must not exceed bit_period ({bit_period!r})")
        self.bit_period = bit_period
        self.v_low = _finite(v_low, "v_low")
        self.v_high = _finite(v_high, "v_high")
        self.t_rise = rise
        self.t_fall = fall
        self.t_start = _finite(t_start, "t_start")
        self._bits: tuple[int, ...] | None = None
        self._events: tuple[tuple[float, int], ...] | None = None
        self._points: np.ndarray | None = None

    # -- construction helpers ------------------------------------------------

    def _timing(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _TIMING}

    def _copy(self, **changes: Any) -> DigitalSignal:
        timing = self._timing()
        data = {key: changes.pop(key) for key in ("_bits", "_events", "_points") if key in changes}
        timing.update(changes)
        other = DigitalSignal(**timing)
        other._bits = data.get("_bits", self._bits)
        other._events = data.get("_events", self._events)
        other._points = data.get("_points", self._points)
        return other

    def _edges(self) -> tuple[float, float]:
        if self.t_rise is None or self.t_fall is None:
            raise ValueError("t_rise and t_fall are needed (or a bit_period, whose tenth is the default)")
        return self.t_rise, self.t_fall

    # -- constructors ----------------------------------------------------------

    @_constructor
    def from_bits(self, bits: Iterable[object]) -> DigitalSignal:
        """One logic level (0/1, ``True``/``False``) per ``bit_period`` slot from ``t_start``."""

        if self.bit_period is None:
            raise ValueError("from_bits needs a bit_period")
        levels: list[int] = []
        for index, bit in enumerate(bits):
            if isinstance(bit, (bool, np.bool_)):
                levels.append(int(bit))
            elif isinstance(bit, numbers.Integral) and int(bit) in (0, 1):
                levels.append(int(bit))
            else:
                raise ValueError(f"bit {index} must be 0 or 1 (got {bit!r})")
        self._edges()
        return self._copy(_bits=tuple(levels), _events=None, _points=None)

    @_constructor
    def from_string(self, text: str) -> DigitalSignal:
        """Bits from a ``"1011..."`` string (optional ``0b`` prefix; spaces and ``_`` ignored)."""

        if not isinstance(text, str):
            raise ValueError(f"from_string needs a string (got {type(text).__name__})")
        cleaned = re.sub(r"[\s_]", "", text)
        if cleaned[:2].lower() == "0b":
            cleaned = cleaned[2:]
        bad = re.search(r"[^01]", cleaned)
        if bad:
            raise ValueError(f"from_string: unexpected character {bad.group(0)!r} (only 0 and 1)")
        return self.from_bits(int(char) for char in cleaned)

    @_constructor
    def from_events(self, events: Iterable[Sequence[object]]) -> DigitalSignal:
        """Transitions ``[(t, level), ...]``: the first sets the level from its time, each later one
        starts an edge at ``t`` (times strictly increasing, relative to ``t_start``)."""

        parsed: list[tuple[float, int]] = []
        for index, event in enumerate(events):
            if not isinstance(event, (tuple, list)) or len(event) != 2:
                raise ValueError(f"event {index} must be a (t, level) pair (got {event!r})")
            time = _finite(event[0], f"event {index} time")
            level = event[1]
            if isinstance(level, (bool, np.bool_)) or (isinstance(level, numbers.Integral) and int(level) in (0, 1)):
                level = int(level)
            else:
                raise ValueError(f"event {index} level must be 0 or 1 (got {level!r})")
            if parsed and not time > parsed[-1][0]:
                raise ValueError(f"event {index}: times must be strictly increasing ({time!r} after {parsed[-1][0]!r})")
            parsed.append((time, level))
        if not parsed:
            raise ValueError("from_events needs at least one event")
        signal = self._copy(_bits=None, _events=tuple(parsed), _points=None)
        signal.points()  # edges must not overlap
        return signal

    @_constructor
    def from_numpy(self, t: Any, v: Any) -> DigitalSignal:
        """Raw corner points ``(t[i], v[i])`` (strictly increasing finite times, relative to ``t_start``)."""

        times = np.array(t, dtype=float).reshape(-1)
        values = np.array(v, dtype=float).reshape(-1)
        if times.size == 0 or times.size != values.size:
            raise ValueError(f"from_numpy needs t and v of the same non-zero length ({times.size} and {values.size})")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise ValueError("from_numpy: t and v must be finite")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("from_numpy: t must be strictly increasing")
        points = np.column_stack((times, values))
        points.setflags(write=False)
        return self._copy(_bits=None, _events=None, _points=points)

    # -- properties ------------------------------------------------------------

    @property
    def bits(self) -> tuple[int, ...] | None:
        """The bit levels of a signal built from bits (``None`` otherwise)."""

        return self._bits

    @property
    def duration(self) -> float:
        """End time of the last corner (for bits: of the last slot)."""

        return float(self.points()[-1, 0])

    def _require_bits(self, operation: str) -> tuple[int, ...]:
        if self._bits is None:
            raise ValueError(f"{operation} needs a signal built from bits (from_bits / from_string)")
        return self._bits

    # -- transforms --------------------------------------------------------------

    def nrzi(self, initial: int = 1) -> DigitalSignal:
        """NRZI line levels (USB convention): a 0 bit toggles the line, a 1 bit holds it.

        ``initial`` is the line level before the first bit (1: the USB idle J state when
        J is the high level).
        """

        bits = self._require_bits("nrzi")
        if initial not in (0, 1):
            raise ValueError(f"initial must be 0 or 1 (got {initial!r})")
        level = int(initial)
        levels = []
        for bit in bits:
            if bit == 0:
                level ^= 1
            levels.append(level)
        return self._copy(_bits=tuple(levels))

    def bit_stuff(self, max_run: int = 6) -> DigitalSignal:
        """Insert a 0 after every ``max_run`` consecutive 1s (USB: six, applied before NRZI)."""

        bits = self._require_bits("bit_stuff")
        if isinstance(max_run, bool) or not isinstance(max_run, numbers.Integral) or max_run < 1:
            raise ValueError(f"max_run must be a positive integer (got {max_run!r})")
        out: list[int] = []
        run = 0
        for bit in bits:
            out.append(bit)
            run = run + 1 if bit == 1 else 0
            if run == max_run:
                out.append(0)
                run = 0
        return self._copy(_bits=tuple(out))

    def invert(self) -> DigitalSignal:
        """The complementary signal: levels swapped, edges kept in time (``t_rise``/``t_fall`` swap)."""

        if self._points is not None:
            mirrored = self._points.copy()
            mirrored[:, 1] = self.v_low + self.v_high - mirrored[:, 1]
            mirrored.setflags(write=False)
            return self._copy(_points=mirrored, t_rise=self.t_fall, t_fall=self.t_rise)
        if self._events is not None:
            return self._copy(_events=tuple((t, 1 - level) for t, level in self._events), t_rise=self.t_fall, t_fall=self.t_rise)
        bits = self._bits if self._bits is not None else ()
        return self._copy(_bits=tuple(1 - bit for bit in bits), t_rise=self.t_fall, t_fall=self.t_rise)

    def delay(self, td: float) -> DigitalSignal:
        """The same signal ``td`` seconds later (``td`` may be negative)."""

        return self._copy(t_start=self.t_start + _finite(td, "td"))

    def differential(self, v_common: float | None = None) -> tuple[DigitalSignal, DigitalSignal]:
        """``(positive, negative)``: the signal and its complement (:meth:`invert`).

        With ``v_common`` both swing by ``v_high - v_low`` around it (levels
        ``v_common -+ swing / 2``); without, they keep ``v_low``/``v_high``.
        """

        positive = self
        if v_common is not None:
            centre = _finite(v_common, "v_common")
            half = 0.5 * (self.v_high - self.v_low)
            positive = self._copy(v_low=centre - half, v_high=centre + half)
            if self._points is not None:
                shifted = self._points.copy()
                shifted[:, 1] = shifted[:, 1] - 0.5 * (self.v_low + self.v_high) + centre
                shifted.setflags(write=False)
                positive = positive._copy(_points=shifted)
        return positive, positive.invert()

    # -- rendering ---------------------------------------------------------------

    def _level(self, bit: int) -> float:
        return self.v_high if bit else self.v_low

    def points(self) -> np.ndarray:
        """``(N, 2)`` array of ``(t, v)`` corners: each edge contributes its two ends, the
        signal is held before the first and after the last corner, adjacent duplicate
        points are removed, and times are strictly increasing."""

        if self._points is not None:
            out = np.array(self._points, dtype=float)
            out[:, 0] += self.t_start
            return out
        rows: list[tuple[float, float]] = []
        if self._events is not None:
            rise, fall = self._edges()
            first_time, level = self._events[0]
            rows.append((self.t_start + first_time, self._level(level)))
            for time, new in self._events[1:]:
                if new == level:
                    continue
                start = self.t_start + time
                if start < rows[-1][0]:
                    raise ValueError(f"the edge at t = {time!r} starts before the previous edge has ended")
                edge = rise if new > level else fall
                rows.append((start, self._level(level)))
                rows.append((start + edge, self._level(new)))
                level = new
        elif self._bits:
            rise, fall = self._edges()
            bits = self._bits
            period = float(self.bit_period)  # type: ignore[arg-type]
            rows.append((self.t_start, self._level(bits[0])))
            for index in range(1, len(bits)):
                previous, new = bits[index - 1], bits[index]
                if new == previous:
                    continue
                start = self.t_start + index * period
                edge = rise if new > previous else fall
                # An edge as long as the bit ends exactly at the next boundary (no rounding overlap).
                end = min(start + edge, self.t_start + (index + 1) * period)
                rows.append((start, self._level(previous)))
                rows.append((end, self._level(new)))
            rows.append((self.t_start + len(bits) * period, self._level(bits[-1])))
        else:
            rows.append((self.t_start, self.v_low))
        out = np.asarray(rows, dtype=float)
        if out.shape[0] > 1:
            keep = np.ones(out.shape[0], dtype=bool)
            keep[1:] = np.any(out[1:] != out[:-1], axis=1)
            out = out[keep]
            # An end point that repeats the level reached at the same time is dropped above;
            # one that lands before the last edge has ended would be a modelling error.
            if not np.all(np.diff(out[:, 0]) > 0.0):
                raise ValueError("the signal's corners are not strictly increasing in time (overlapping edges)")
        return out

    def netlist_line(self, name: str, node_p: str, node_n: str) -> str:
        """``"<name> <node_p> <node_n> PWL(t1 v1 t2 v2 ...)"`` (exact float literals)."""

        _check_names(name, node_p, node_n)
        corners = self.points()
        body = " ".join(f"{float(t)!r} {float(v)!r}" for t, v in corners)
        return f"{name} {node_p} {node_n} PWL({body})"

    def to_stimulus(self, name: str, node_p: str, node_n: str, period: float | None = None) -> dict[str, Any]:
        """The ``stimuli`` entry of an ``AnalysisRequest``: ``{"name", "node_p", "node_n", "points", "period"}``."""

        _check_names(name, node_p, node_n)
        if name[0] not in "Vv":
            raise ValueError(f"a stimulus is a voltage source: its name must start with V (got {name!r})")
        if period is not None:
            period = _finite(period, "period")
            if period < 0.0:
                raise ValueError(f"period must not be negative (got {period!r})")
        return {
            "name": name,
            "node_p": node_p,
            "node_n": node_n,
            "points": self.points().tolist(),
            "period": period,
        }

    def __repr__(self) -> str:
        kind = "bits" if self._bits is not None else "events" if self._events is not None else "points" if self._points is not None else "empty"
        size = len(self._bits or self._events or ()) if self._points is None else len(self._points)
        return (
            f"DigitalSignal({kind}={size}, bit_period={self.bit_period!r}, v_low={self.v_low!r}, "
            f"v_high={self.v_high!r}, t_rise={self.t_rise!r}, t_fall={self.t_fall!r}, t_start={self.t_start!r})"
        )


def _check_names(name: str, node_p: str, node_n: str) -> None:
    if not isinstance(name, str) or not _SOURCE_NAME.match(name):
        raise ValueError(f"source name {name!r} must start with V or I followed by letters, digits or '_'")
    for label, node in (("node_p", node_p), ("node_n", node_n)):
        if not isinstance(node, str) or not _NODE_NAME.match(node):
            raise ValueError(f"{label} {node!r} is not a valid node name")
