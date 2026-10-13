"""Table sources: piecewise-linear waveforms evaluated by the VM's ``TABLE`` lookup.

MIXED_SIGNAL.md phase D1. Three kinds of source become native tables instead of
expression chains (O(log N) per evaluation, exact corners):

* ``PULSE`` records (``subtype="PULSE"``, ``metadata["pulse"]`` = the seven SPICE
  values ``v1 v2 td tr tf pw per`` as text, HIERARCHY_AND_MODELS_PLAN.md 3.2):
  the four corners of one period, repeated every ``per``. A zero ``tr``/``tf`` is
  one analysis time step and a zero ``pw`` the stop time (SPICE defaults), a zero
  ``per`` means no repetition. The builder does not know the step, so it stores a
  :class:`TableSpec` and the analysis resolves it (:func:`resolve_tables`).
* ``PWL`` records (``metadata["points"]`` = the flat ``t v`` list as text,
  ``metadata["repeat"]`` for ``R=``, ``metadata["td"]`` for ``TD=``). A PWL with a
  vertical step (two equal times) is not a table (tables need strictly increasing
  times): it keeps its expression rendering.
* request stimuli (``AnalysisRequest.stimuli``): :func:`stimulus_records` validates
  them and returns ``V`` records whose ``metadata["stimulus"]`` is the
  :class:`~mna_simulation.native.expressions.TimeTable`.

The builder replaces the source's waveform by the placeholder ``__table_k(t)`` and
keeps the specs in ``MnaProblem.metadata["_time_tables"]`` (index ``k``); the
solvers resolve them for the analysis (:func:`resolve_tables`), pass the tables to
the expression compiler, and merge their corners into the transient grid
(:func:`merge_breakpoints`).
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .api.contracts import CircuitIR, ComponentRecord
from .errors import error_handler
from .native.expressions import MAX_TABLE_POINTS, TimeTable, table_placeholder
from .utils import parse_value

__all__ = [
    "MAX_BREAKPOINTS",
    "MAX_TABLE_POINTS",
    "RAMP_POINTS",
    "STIMULUS_KEY",
    "TIME_TABLES_KEY",
    "TableSpec",
    "breakpoints",
    "has_table_reference",
    "merge_breakpoints",
    "pulse_table",
    "register_table",
    "resolve_tables",
    "stimulus_op_rhs",
    "stimulus_records",
    "stimulus_table_of",
    "table_spec",
    "with_stimuli",
]

#: ``MnaProblem.metadata`` key of the table specs (index k = placeholder ``__table_k``).
TIME_TABLES_KEY = "_time_tables"
#: ``MnaProblem.metadata`` key of the ``.op`` right-hand side of request stimuli: ``[(row, value at t=0)]``.
STIMULUS_OP_KEY = "_stimulus_op_rhs"
#: ``ComponentRecord.metadata`` key of a request stimulus (value: a ``TimeTable``).
STIMULUS_KEY = "stimulus"
#: Extra grid points across a ramp shorter than the time step (at 1/5, 2/5, 3/5, 4/5 of it).
RAMP_POINTS = 4
#: Largest number of breakpoints the table sources may add to one transient grid.
MAX_BREAKPOINTS = 10_000_000

_PULSE_FIELDS = ("v1", "v2", "td", "tr", "tf", "pw", "per")
_STIMULUS_NAME = re.compile(r"[Vv][A-Za-z0-9_]*\Z")
_NODE_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")


# ---------------------------------------------------------------------------
# Specs: tables whose corners may depend on the analysis step
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TableSpec:
    """A table source as the builder records it.

    Exactly one of ``table`` (a fixed table: PWL, stimulus) and ``pulse`` (the seven
    PULSE values as floats; ``0`` in ``tr``/``tf``/``pw``/``per`` marks the SPICE
    default) is set. ``name`` is the source's name (for messages).
    """

    name: str
    table: TimeTable | None = None
    pulse: tuple[float, float, float, float, float, float, float] | None = None

    @property
    def fixed(self) -> bool:
        return self.table is not None

    def resolve(self, t_step: float, t_stop: float) -> TimeTable:
        """The table for an analysis with time step ``t_step`` and stop time ``t_stop``."""

        if self.table is not None:
            return self.table
        assert self.pulse is not None
        return pulse_table(*self.pulse, t_step=t_step, t_stop=t_stop, name=self.name)

    def summary(self) -> dict[str, object]:
        """JSON-safe description (result metadata)."""

        if self.table is not None:
            return {"name": self.name, "kind": "table", "points": len(self.table), "period": self.table.period}
        return {"name": self.name, "kind": "pulse", "pulse": list(self.pulse or ())}


def pulse_table(
    v1: float,
    v2: float,
    td: float = 0.0,
    tr: float = 0.0,
    tf: float = 0.0,
    pw: float = 0.0,
    per: float = 0.0,
    *,
    t_step: float,
    t_stop: float,
    name: str = "PULSE",
) -> TimeTable:
    """``PULSE(v1 v2 td tr tf pw per)`` as a table: four corners, repeated every ``per``.

    Corners ``(td, v1)``, ``(td + tr, v2)``, ``(td + tr + pw, v2)``,
    ``(td + tr + pw + tf, v1)``; ``v1`` is held before ``td`` and after the fall.
    SPICE defaults: ``tr = 0`` or ``tf = 0`` is one ``t_step``, ``pw = 0`` is
    ``t_stop``; ``per = 0`` means a single pulse. With ``per > 0`` the waveform
    repeats from ``td`` (a period shorter than ``tr + pw + tf`` cuts the pulse).
    """

    values = [float(v) for v in (v1, v2, td, tr, tf, pw, per)]
    for label, value in zip(_PULSE_FIELDS, values):
        if not math.isfinite(value):
            error_handler(f"NETLIST_FATAL: '{name}': PULSE {label} must be finite (got {value!r}).")
    v1, v2, td, tr, tf, pw, per = values
    for label, value in (("tr", tr), ("tf", tf), ("pw", pw), ("per", per)):
        if value < 0.0:
            error_handler(f"NETLIST_FATAL: '{name}': PULSE {label} must not be negative (got {value!r}).")
    step = float(t_step)
    if tr == 0.0 or tf == 0.0:
        if not (math.isfinite(step) and step > 0.0):
            error_handler(f"NETLIST_FATAL: '{name}': PULSE with tr = 0 or tf = 0 needs a positive time step.")
    if tr == 0.0:
        tr = step
    if tf == 0.0:
        tf = step
    if pw == 0.0:
        stop = float(t_stop)
        pw = stop if math.isfinite(stop) and stop > 0.0 else 0.0
    rise_end = td + tr
    fall_start = rise_end + pw
    fall_end = fall_start + tf
    if pw > 0.0:
        times = [td, rise_end, fall_start, fall_end]
        levels = [v1, v2, v2, v1]
    else:  # a triangle (t_stop unknown or zero)
        times = [td, rise_end, fall_end]
        levels = [v1, v2, v1]
    return TimeTable(times, levels, period=per, origin=td, name=f"'{name}' PULSE")


def _number(value: object, what: str, name: str) -> float:
    if isinstance(value, bool):
        error_handler(f"NETLIST_FATAL: '{name}': invalid {what} {value!r}.")
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        return parse_value(str(value))
    except Exception:  # noqa: BLE001 - parse_value raises NetlistError for bad text
        error_handler(f"NETLIST_FATAL: '{name}': invalid {what} {value!r}.")
        raise


def _pulse_spec(name: str, raw: Sequence[object]) -> TableSpec:
    items = list(raw)
    if not 2 <= len(items) <= 7:
        error_handler(f"NETLIST_FATAL: '{name}': PULSE takes 2 to 7 values, got {len(items)}.")
    values = [_number(item, f"PULSE {label}", name) for label, item in zip(_PULSE_FIELDS, items)]
    values += [0.0] * (7 - len(values))
    # Validate now (the step only replaces zeros, so a spec that resolves for one step resolves for all).
    pulse_table(*values, t_step=1.0, t_stop=1.0, name=name)
    return TableSpec(name=name, pulse=tuple(values))  # type: ignore[arg-type]


def _pwl_spec(name: str, metadata: Mapping[str, object]) -> TableSpec | None:
    raw = list(metadata["points"])  # type: ignore[arg-type]
    if len(raw) < 2 or len(raw) % 2:
        error_handler(f"NETLIST_FATAL: '{name}': PWL needs time-value pairs.")
    numbers_ = [_number(item, "PWL point", name) for item in raw]
    times = np.asarray(numbers_[0::2], dtype=float)
    values = np.asarray(numbers_[1::2], dtype=float)
    steps = np.diff(times)
    if np.any(steps < 0.0):
        error_handler(f"NETLIST_FATAL: '{name}': PWL times must not decrease.")
    if np.any(steps == 0.0):
        return None  # a vertical step: keep the expression rendering (tables need increasing times)
    delay = metadata.get("td")
    shift = _number(delay, "PWL TD", name) if delay not in (None, "") else 0.0
    period = 0.0
    origin = None
    repeat = metadata.get("repeat")
    if repeat not in (None, ""):
        start = _number(repeat, "PWL R", name)
        if not times[0] <= start < times[-1]:
            error_handler(f"NETLIST_FATAL: '{name}': PWL R= must lie between the first and the last time point.")
        period = float(times[-1] - start)
        origin = start + shift
    table = TimeTable(times + shift, values, period=period, origin=origin, name=f"'{name}' PWL")
    return TableSpec(name=name, table=table)


def table_spec(component: ComponentRecord) -> TableSpec | None:
    """The table source of a ``V``/``I`` record, or ``None`` when it is not one.

    Request stimuli (``metadata["stimulus"]``), ``PULSE`` records and ``PWL`` records
    with strictly increasing times are table sources; invalid data is ``NETLIST_FATAL``.
    """

    metadata = component.metadata
    if not metadata:
        return None
    stimulus = metadata.get(STIMULUS_KEY)
    if isinstance(stimulus, TimeTable):
        return TableSpec(name=component.name, table=stimulus)
    subtype = (component.subtype or "").upper()
    if subtype == "PULSE" and "pulse" in metadata:
        return _pulse_spec(component.name, metadata["pulse"])  # type: ignore[arg-type]
    if subtype == "PWL" and "points" in metadata:
        return _pwl_spec(component.name, metadata)
    return None


def register_table(tables: list[TableSpec], spec: TableSpec) -> str:
    """Append ``spec`` to the builder's table list and return its placeholder ``__table_k(t)``."""

    tables.append(spec)
    return table_placeholder(len(tables) - 1)


def stimulus_table_of(component: ComponentRecord) -> TimeTable | None:
    """The table of a request-stimulus record (``metadata["stimulus"]``), else ``None``."""

    table = component.metadata.get(STIMULUS_KEY) if component.metadata else None
    return table if isinstance(table, TimeTable) else None


def has_table_reference(text: object) -> bool:
    """True when an expression text reads a table placeholder."""

    return "__table_" in str(text)


def resolve_tables(metadata: Mapping[str, object] | None, t_step: float, t_stop: float) -> tuple[TimeTable, ...]:
    """The problem's tables (``metadata["_time_tables"]``) for an analysis step and stop time."""

    specs = (metadata or {}).get(TIME_TABLES_KEY) or ()
    return tuple(spec.resolve(t_step, t_stop) for spec in specs)  # type: ignore[union-attr]


def stimulus_op_rhs(metadata: Mapping[str, object] | None) -> list[tuple[int, float]]:
    """``[(row, value)]`` the ``.op`` right-hand side gets from request stimuli (their value at ``t = 0``)."""

    return list((metadata or {}).get(STIMULUS_OP_KEY) or ())


# ---------------------------------------------------------------------------
# Breakpoint-aware transient grid
# ---------------------------------------------------------------------------


def _ramp_points(starts: np.ndarray, durations: np.ndarray, ramp_points: int) -> np.ndarray:
    if starts.size == 0 or ramp_points <= 0:
        return np.zeros(0)
    fractions = np.arange(1, ramp_points + 1, dtype=float) / float(ramp_points + 1)
    return (starts[:, None] + durations[:, None] * fractions[None, :]).reshape(-1)


def _short_ramps(
    times: np.ndarray, values: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, t_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Starts and durations of the pieces ``[seg_start, seg_end]`` that are ramps shorter than ``t_step``."""

    durations = seg_end - seg_start
    if seg_start.size == 0:
        return seg_start, durations
    middle = 0.5 * (seg_start + seg_end)
    index = np.searchsorted(times, middle, side="right") - 1
    inside = (index >= 0) & (index < times.size - 1)
    safe = np.clip(index, 0, max(times.size - 2, 0))
    ramp = inside & (values[safe] != values[np.minimum(safe + 1, times.size - 1)])
    short = ramp & (durations > 0.0) & (durations < t_step)
    return seg_start[short], durations[short]


def _table_breakpoints(
    table: TimeTable, t_stop: float, t_step: float, ramp_points: int, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(corners, ramp points)`` of one table in ``[0, t_stop]`` (unfiltered at the ends)."""

    times, values = table.times, table.values
    period, origin = table.period, table.origin
    end = origin + period if period > 0.0 else math.inf
    # Up to ``end`` the waveform is the table itself.
    literal = times[times < end]
    corners = [literal]
    seg_start = times[:-1]
    seg_end = np.minimum(times[1:], end)
    keep = seg_start < end
    starts, durations = _short_ramps(times, values, seg_start[keep], seg_end[keep], t_step)
    ramps = [_ramp_points(starts, durations, ramp_points)]
    if period > 0.0 and end <= t_stop:
        # From ``end`` on, the window [origin, origin + period) repeats.
        inner = times[(times > origin) & (times < end)]
        bounds = np.concatenate(([origin], inner, [end]))
        offsets = bounds[:-1] - origin
        w_starts, w_durations = _short_ramps(times, values, bounds[:-1], bounds[1:], t_step)
        w_ramps = _ramp_points(w_starts - origin, w_durations, ramp_points)
        first = max(1, math.floor(-origin / period) if origin < 0.0 else 1)
        last = math.floor((t_stop - origin) / period)
        count = max(0, last - first + 1)
        per_window = offsets.size + w_ramps.size
        if count * per_window > budget:
            error_handler(
                f"TRAN_ERROR: the table source {table.name} repeats {count} times in [0, t_stop], "
                f"which adds more than {MAX_BREAKPOINTS} breakpoints to the time grid; turn breakpoints off "
                "(breakpoints=false) or shorten t_stop."
            )
        if count > 0:
            window_starts = origin + np.arange(first, last + 1, dtype=float) * period
            corners.append((window_starts[:, None] + offsets[None, :]).reshape(-1))
            if w_ramps.size:
                ramps.append((window_starts[:, None] + w_ramps[None, :]).reshape(-1))
    return np.concatenate(corners), np.concatenate(ramps)


def breakpoints(
    tables: Iterable[TimeTable], t_stop: float, t_step: float, ramp_points: int = RAMP_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique ``(corner times, ramp points)`` of ``tables`` inside ``[0, t_stop]``.

    Corners: every table point (and, for a repeating table, every point of every
    repetition and each repetition start). Ramp points: ``ramp_points`` evenly spaced
    interior points across every ramp (a segment whose two values differ) shorter
    than ``t_step``. More than :data:`MAX_BREAKPOINTS` is ``TRAN_ERROR``.
    """

    corner_parts: list[np.ndarray] = []
    ramp_parts: list[np.ndarray] = []
    total = 0
    for table in tables:
        corners, ramps = _table_breakpoints(table, float(t_stop), float(t_step), int(ramp_points), MAX_BREAKPOINTS - total)
        total += corners.size + ramps.size
        if total > MAX_BREAKPOINTS:
            error_handler(
                f"TRAN_ERROR: the table sources add more than {MAX_BREAKPOINTS} breakpoints to the time grid; "
                "turn breakpoints off (breakpoints=false) or shorten t_stop."
            )
        corner_parts.append(corners)
        ramp_parts.append(ramps)

    def inside(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.zeros(0)
        merged = np.concatenate(parts)
        return np.unique(merged[(merged >= 0.0) & (merged <= t_stop)])

    return inside(corner_parts), inside(ramp_parts)


def _far_from(points: np.ndarray, reference: np.ndarray, tolerance: float) -> np.ndarray:
    """Mask of ``points`` farther than ``tolerance`` from every entry of the sorted ``reference``."""

    if reference.size == 0 or points.size == 0:
        return np.ones(points.size, dtype=bool)
    index = np.searchsorted(reference, points)
    left = np.where(index > 0, points - reference[np.maximum(index - 1, 0)], np.inf)
    right = np.where(index < reference.size, reference[np.minimum(index, reference.size - 1)] - points, np.inf)
    return np.minimum(left, right) > tolerance


def _dedup(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Sorted ``points`` without those within ``tolerance`` of their predecessor."""

    if points.size < 2:
        return points
    keep = np.ones(points.size, dtype=bool)
    keep[1:] = np.diff(points) > tolerance
    return points[keep]


def merge_breakpoints(
    uniform: np.ndarray,
    tables: Iterable[TimeTable],
    t_stop: float,
    t_step: float,
    *,
    rel_tol: float,
    ramp_points: int = RAMP_POINTS,
) -> np.ndarray:
    """The uniform transient grid merged with the breakpoints of ``tables``.

    Points closer than ``max(|t_stop|, |t_step|, eps) * rel_tol`` (the core's grid
    tolerance) are merged; precedence: the end points ``0`` and ``t_stop``, then the
    table corners (which therefore appear in the grid exactly unless they lie within
    the tolerance of an end point or of an earlier corner), then the ramp points, then
    the uniform points. The result is strictly increasing, starts at ``0`` and ends at
    ``t_stop``; without breakpoints it is ``uniform`` itself.
    """

    grid = np.asarray(uniform, dtype=float)
    corners, ramps = breakpoints(tables, t_stop, t_step, ramp_points)
    if corners.size == 0 and ramps.size == 0:
        return grid
    tolerance = max(abs(float(t_stop)), abs(float(t_step)), float(np.finfo(float).eps)) * float(rel_tol)

    def interior(points: np.ndarray) -> np.ndarray:
        return points[(points > tolerance) & (points < t_stop - tolerance)]

    corners = _dedup(interior(corners), tolerance)
    ramps = interior(ramps)
    ramps = _dedup(ramps[_far_from(ramps, corners, tolerance)], tolerance)
    extra = np.sort(np.concatenate((corners, ramps)))
    if extra.size == 0:
        return grid
    keep = _far_from(grid, extra, tolerance)
    keep[0] = True
    keep[-1] = True
    merged = np.sort(np.concatenate((grid[keep], extra)))
    return merged


# ---------------------------------------------------------------------------
# Request stimuli
# ---------------------------------------------------------------------------


def _field(entry: object, key: str) -> object:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


def stimulus_table(entry: object, index: int = 0) -> tuple[str, str, str, TimeTable]:
    """``(name, node_p, node_n, table)`` of one stimulus (a dict or an object with those fields)."""

    if not isinstance(entry, Mapping) and not hasattr(entry, "points"):
        error_handler(f"NETLIST_FATAL: stimulus {index}: expected an object with name, node_p, node_n and points.")
    name = _field(entry, "name")
    if not isinstance(name, str) or not _STIMULUS_NAME.match(name):
        error_handler(
            f"NETLIST_FATAL: stimulus {index}: name {name!r} must be a voltage-source name "
            "(a 'V' followed by letters, digits or '_')."
        )
    nodes = []
    for key in ("node_p", "node_n"):
        node = _field(entry, key)
        if isinstance(node, numbers.Integral) and not isinstance(node, bool):
            node = str(node)
        if not isinstance(node, str) or not _NODE_NAME.match(node):
            error_handler(f"NETLIST_FATAL: stimulus '{name}': {key} {node!r} is not a valid node name.")
        nodes.append(node)
    raw_points = _field(entry, "points")
    label = f"stimulus '{name}'"
    if not isinstance(raw_points, (list, tuple, np.ndarray)):
        error_handler(f"NETLIST_FATAL: {label}: points must be a list of [t, v] pairs.")
    if len(raw_points) == 0:  # type: ignore[arg-type]
        error_handler(f"NETLIST_FATAL: {label}: the table has no points.")
    if len(raw_points) > MAX_TABLE_POINTS:  # type: ignore[arg-type]
        error_handler(f"NETLIST_FATAL: {label}: the table has {len(raw_points)} points (at most {MAX_TABLE_POINTS}).")  # type: ignore[arg-type]
    if isinstance(raw_points, np.ndarray):
        array = np.asarray(raw_points, dtype=float)
    else:
        for pair in raw_points:  # type: ignore[union-attr]
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(item, numbers.Real) and not isinstance(item, bool) for item in pair)
            ):
                error_handler(f"NETLIST_FATAL: {label}: every point must be a [t, v] pair of numbers (got {pair!r}).")
        array = np.asarray(raw_points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        error_handler(f"NETLIST_FATAL: {label}: points must be a list of [t, v] pairs.")
    period = _field(entry, "period")
    if period is not None:
        if isinstance(period, bool) or not isinstance(period, numbers.Real):
            error_handler(f"NETLIST_FATAL: {label}: period must be a number or null (got {period!r}).")
        if not math.isfinite(float(period)) or float(period) < 0.0:
            error_handler(f"NETLIST_FATAL: {label}: the repetition period must be finite and non-negative (got {period!r}).")
    table = TimeTable(array[:, 0], array[:, 1], period=None if period is None else float(period), name=label)
    return name, nodes[0], nodes[1], table


def stimulus_records(stimuli: Sequence[object] | None, components: Sequence[ComponentRecord] = ()) -> list[ComponentRecord]:
    """``V`` records (table sources) for request stimuli, validated against ``components``.

    Names are unique (case-insensitively) and must not repeat a component name;
    node names that match a netlist node case-insensitively take its spelling.
    """

    if not stimuli:
        return []
    if not isinstance(stimuli, (list, tuple)):
        error_handler("NETLIST_FATAL: stimuli must be a list of stimulus objects.")
    taken = {component.name.upper(): component.name for component in components}
    spelling: dict[str, str] = {}
    for component in components:
        for node in (component.node1, component.node2, component.ctrl_node1, component.ctrl_node2):
            if node is not None:
                spelling.setdefault(str(node).upper(), str(node))
    records: list[ComponentRecord] = []
    for index, entry in enumerate(stimuli):
        name, node_p, node_n, table = stimulus_table(entry, index)
        key = name.upper()
        if key in taken:
            error_handler(f"NETLIST_FATAL: stimulus '{name}' has the name of an existing component ('{taken[key]}').")
        taken[key] = name
        record = ComponentRecord(
            name=name,
            type="V",
            node1=spelling.get(node_p.upper(), node_p),
            node2=spelling.get(node_n.upper(), node_n),
            value=None,
            subtype="PWL",
            metadata={STIMULUS_KEY: table},
        )
        records.append(record)
    return records


def with_stimuli(circuit: CircuitIR, stimuli: Sequence[object] | None) -> CircuitIR:
    """``circuit`` with the validated request ``stimuli`` appended as table voltage sources."""

    records = stimulus_records(stimuli, circuit.components)
    if not records:
        return circuit
    return replace(circuit, components=[*circuit.components, *records])


def public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """``metadata`` without the private table keys, plus a JSON-safe ``table_sources`` summary."""

    result = {key: value for key, value in metadata.items() if key not in (TIME_TABLES_KEY, STIMULUS_OP_KEY)}
    specs = metadata.get(TIME_TABLES_KEY)
    if specs:
        result["table_sources"] = [spec.summary() for spec in specs]
    return result
