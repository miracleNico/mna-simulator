"""Native table source (MIXED_SIGNAL.md phase D1): the VM's TABLE lookup and the table records.

* ``__table_k(t)`` compiles to one TABLE instruction with the table data attached; the value
  matches the equivalent ``min``/``max``/``heaviside`` expression to 1e-12 and the NumPy
  reference (``TimeTable.__call__``) bit for bit; hold before/after, periodic repetition.
* A 1e5-point table compiles and runs.
* ``PULSE`` records become four corners per period (``tr``/``tf`` = 0 is one time step, ``pw``
  = 0 the stop time, ``per`` = 0 no repetition); ``PWL`` records with ``R=``/``TD=``.
* Invalid tables are ``NETLIST_FATAL`` with texts naming the source.
* ``.hb`` with a PULSE equals ``.hb`` with the equivalent ``STEP`` source.
"""

from __future__ import annotations

import numpy as np
import pytest

from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, CircuitIR, ComponentRecord
from mna_simulation.core_advanced import build_and_run_advanced
from mna_simulation.core_basic import build_and_run_basic, run_basic_analysis
from mna_simulation.device_models import source_expression
from mna_simulation.errors import NetlistError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.expressions import TimeTable, compile_expression
from mna_simulation.netlist import parse_netlist_text
from mna_simulation.stimulus import DigitalSignal
from mna_simulation.table_sources import TIME_TABLES_KEY, pulse_table, resolve_tables, table_spec
from mna_simulation.utils import parse_value

core = native.require_core()


def _program(table: TimeTable, text: str = "__table_0(t)"):
    return compile_expression(text, time_dependent=True, n_unknowns=0, tables=[table])


def _ramp_expression(points: np.ndarray) -> str:
    """The waveform of ``points`` as a sum of clamped ramps (``Min``/``Max``) and steps (``heaviside``)."""

    terms = [repr(float(points[0, 1]))]
    for (t0, v0), (t1, v1) in zip(points[:-1].tolist(), points[1:].tolist()):
        if v1 == v0:
            continue
        terms.append(f"({v1 - v0!r})*Min(1, Max(0, (t - {t0!r})/{t1 - t0!r}))")
    return " + ".join(terms)


def _sixteen_edge_points() -> np.ndarray:
    signal = DigitalSignal.from_string("0110 1001 0111 0100 1101 0010", bit_period=10e-9, t_rise=1e-9, t_fall=2e-9)
    points = signal.points()
    edges = int(np.sum(np.diff(points[:, 1]) != 0.0))
    assert edges == 16
    return points


def test_table_matches_the_equivalent_ramp_expression_for_16_edges() -> None:
    points = _sixteen_edge_points()
    table = TimeTable(points[:, 0], points[:, 1])
    program = _program(table)
    assert program.handle.n_tables == 1
    assert list(program.opcodes) == [core.expr_opcodes()["table"]]
    reference = compile_expression(_ramp_expression(points), time_dependent=True, n_unknowns=0)
    times = np.concatenate((np.linspace(-5e-9, 200e-9, 20001), points[:, 0]))
    got = program.evaluate_batch(times)
    want = reference.evaluate_batch(times)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    # The VM and the NumPy reference implementation agree bit for bit.
    np.testing.assert_array_equal(got, table(times))
    # Corners are exact.
    np.testing.assert_array_equal(program.evaluate_batch(points[:, 0]), points[:, 1])


def test_a_heaviside_expression_matches_a_steep_table() -> None:
    # A 1 fs edge: the table is the heaviside step except inside the edge.
    table = TimeTable([1e-6, 1e-6 + 1e-15], [0.0, 2.5])
    step = compile_expression("2.5*np.heaviside(t - 1e-6 - 1e-15, 1)", time_dependent=True, n_unknowns=0)
    times = np.array([0.0, 5e-7, 1e-6 - 1e-12, 1e-6 + 1e-12, 2e-6, 1.0])
    np.testing.assert_allclose(_program(table).evaluate_batch(times), step.evaluate_batch(times), atol=1e-12, rtol=0)


def test_hold_before_after_and_periodic_repetition() -> None:
    table = TimeTable([2.0, 3.0, 5.0], [1.0, 4.0, -2.0])
    program = _program(table)
    for t, want in ((-1e9, 1.0), (0.0, 1.0), (2.0, 1.0), (2.5, 2.5), (3.0, 4.0), (4.0, 1.0), (5.0, -2.0), (1e9, -2.0)):
        assert program.evaluate(None, t) == want
    assert np.isnan(program.evaluate(None, float("nan")))
    # Periodic from the first point (a PULSE): t >= origin folds into [origin, origin + period).
    periodic = TimeTable([1.0, 2.0, 4.0, 5.0], [0.0, 1.0, 1.0, 0.0], period=10.0)
    assert periodic.origin == 1.0
    p = _program(periodic)
    for t, want in ((0.5, 0.0), (1.5, 0.5), (11.5, 0.5), (101.5, 0.5), (7.0, 0.0), (13.0, 1.0), (11.0, 0.0)):
        assert p.evaluate(None, t) == want
    times = np.linspace(-3.0, 60.0, 6301)
    np.testing.assert_array_equal(p.evaluate_batch(times), periodic(times))
    # Repetition from an interior origin (SPICE PWL R=r): the tail [r, t_last) repeats after t_last.
    tail = TimeTable([0.0, 2.0, 3.0, 4.0], [7.0, 0.0, 1.0, 0.0], period=2.0, origin=2.0)
    q = _program(tail)
    for t, want in ((1.0, 3.5), (3.0, 1.0), (4.5, 0.5), (5.0, 1.0), (7.0, 1.0)):
        assert q.evaluate(None, t) == want


def test_a_row_mixing_a_table_and_an_expression_is_one_program() -> None:
    table = TimeTable([0.0, 1e-3], [0.0, 1.0])
    text = "(__table_0(t)) - (0.5 * np.sin(2 * np.pi * 1000 * t))"
    program = _program(table, text)
    for t in (0.0, 2.5e-4, 7.5e-4, 2e-3):
        assert program.evaluate(None, t) == pytest.approx(float(table(t)) - 0.5 * np.sin(2 * np.pi * 1000 * t), abs=1e-15)
    # Several tables in one program; tables are matched by content in the compile cache.
    two = compile_expression(
        "__table_0(t) + 2*__table_1(t)",
        time_dependent=True,
        n_unknowns=0,
        tables=[table, TimeTable([0.0, 1.0], [5.0, 5.0])],
    )
    assert two.evaluate(None, 5e-4) == 0.5 + 10.0
    again = compile_expression(text, time_dependent=True, n_unknowns=0, tables=[TimeTable([0.0, 1e-3], [0.0, 1.0])])
    assert again is program


def test_a_1e5_point_table_compiles_and_runs() -> None:
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.uniform(1e-10, 1e-9, 100_000))
    values = rng.uniform(-1.0, 3.3, 100_000)
    table = TimeTable(times, values)
    program = _program(table)
    probe = np.sort(rng.uniform(times[0] - 1e-9, times[-1] + 1e-9, 100_000))
    np.testing.assert_array_equal(program.evaluate_batch(probe), table(probe))
    # And in a transient run: an RC low-pass driven by the table (breakpoints off: 1e5 corners).
    records = [
        ComponentRecord("V1", "V", "in", "0", None, subtype="PWL", metadata={"stimulus": table}),
        ComponentRecord("R1", "R", "in", "out", "1k"),
        ComponentRecord("C1", "C", "out", "0", "1p"),
    ]
    problem = build_mna_problem(CircuitIR(records))
    waveform = solvers.solve_transient(problem, float(times[-1]), 1e-9, breakpoints=False)
    v_in = waveform.values[waveform.labels.index("V(in)")]
    np.testing.assert_array_equal(v_in, table(waveform.time))


def test_the_compile_cache_keys_on_table_content() -> None:
    a = TimeTable([0.0, 1.0], [0.0, 1.0])
    b = TimeTable([0.0, 1.0], [0.0, 1.0])
    c = TimeTable([0.0, 1.0], [0.0, 2.0])
    assert a == b and hash(a) == hash(b) and a != c
    assert _program(a) is _program(b)
    assert _program(c).evaluate(None, 0.5) == 1.0


def test_pulse_corner_points_and_the_zero_rise_rule() -> None:
    table = pulse_table(0.0, 5.0, 1e-9, 2e-9, 3e-9, 10e-9, 50e-9, t_step=1e-10, t_stop=1e-6)
    np.testing.assert_array_equal(table.times, [1e-9, 1e-9 + 2e-9, 1e-9 + 2e-9 + 10e-9, 1e-9 + 2e-9 + 10e-9 + 3e-9])
    np.testing.assert_array_equal(table.values, [0.0, 5.0, 5.0, 0.0])
    assert table.period == 50e-9 and table.origin == 1e-9
    # tr = tf = 0: one time step each; pw = 0: the stop time; per = 0: a single pulse.
    zero = pulse_table(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, t_step=1e-9, t_stop=1e-6)
    np.testing.assert_array_equal(zero.times, [0.0, 1e-9, 1e-9 + 1e-6, 1e-9 + 1e-6 + 1e-9])
    np.testing.assert_array_equal(zero.values, [1.0, 2.0, 2.0, 1.0])
    assert zero.period == 0.0
    # A record keeps the zeros as markers until the analysis knows its step.
    record = ComponentRecord("V1", "V", "a", "0", None, subtype="PULSE", metadata={"pulse": ["0", "1", "5n", "0", "0", "20n", "100n"]})
    spec = table_spec(record)
    assert spec is not None and spec.pulse == tuple(parse_value(v) for v in ("0", "1", "5n", "0", "0", "20n", "100n"))
    td, pw = parse_value("5n"), parse_value("20n")
    for step in (1e-9, 2e-10):
        resolved = spec.resolve(step, 1e-6)
        np.testing.assert_array_equal(resolved.times, [td, td + step, td + step + pw, td + step + pw + step])
    # Through the builder and a transient run: the source rises over exactly one step.
    records = [record, ComponentRecord("R1", "R", "a", "0", "1k")]
    problem = build_mna_problem(CircuitIR(records))
    assert problem.b_time_str[problem.index_map.node_to_index["a"] + 1, 0] == "__table_0(t)"
    tables = resolve_tables(problem.metadata, 1e-9, 200e-9)
    assert tables[0] == spec.resolve(1e-9, 200e-9)
    result = build_and_run_basic(CircuitIR(records), AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=200e-9, tran_step=1e-9))
    t = result.waveform.time
    v = result.waveform.values[result.waveform.labels.index("V(a)")]
    probes = [5e-9, 6e-9, 26e-9, 27e-9, 105e-9, 106e-9, 120e-9, 127e-9]
    np.testing.assert_allclose(np.interp(probes, t, v), [0, 1, 1, 0, 0, 1, 1, 0], atol=1e-9)
    # .op uses the DC value of the record (none: 0); .ac sees no AC part.
    op = build_and_run_basic(CircuitIR(records), AnalysisOptions(mode=AnalysisMode.OP))
    assert np.all(op.dc_solution == 0.0)


def test_pwl_records_with_repeat_and_delay() -> None:
    record = ComponentRecord(
        "V1", "V", "a", "0", None, subtype="PWL", metadata={"points": ["0", "0", "1u", "1", "2u", "0"], "repeat": "1u", "td": "0.5u"}
    )
    spec = table_spec(record)
    assert spec is not None and spec.table is not None
    table = spec.table
    np.testing.assert_allclose(table.times, [0.5e-6, 1.5e-6, 2.5e-6])
    assert table.period == pytest.approx(1e-6) and table.origin == pytest.approx(1.5e-6)
    program = _program(table)
    assert program.evaluate(None, 0.0) == 0.0
    assert program.evaluate(None, 1.5e-6) == 1.0
    assert program.evaluate(None, 3.0e-6) == pytest.approx(0.5)
    assert program.evaluate(None, 3.5e-6) == pytest.approx(1.0, abs=1e-12)
    # A vertical step (two equal times) is not a table: the record keeps its expression form.
    step = ComponentRecord("V2", "V", "a", "0", None, subtype="PWL", metadata={"points": ["0", "0", "1u", "0", "1u", "1"]})
    assert table_spec(step) is None
    # Records without table data are untouched.
    assert table_spec(ComponentRecord("V3", "V", "a", "0", "1", subtype="DC")) is None


def _fatal(match: str):
    return pytest.raises(NetlistError, match=match)


def test_invalid_tables_are_netlist_fatal() -> None:
    with _fatal("NETLIST_FATAL|times must be strictly increasing") as exc:
        TimeTable([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], name="stimulus 'Vx'")
    assert exc.value.code == "NETLIST_FATAL"
    assert exc.value.message == "stimulus 'Vx': times must be strictly increasing (point 2: 1.0 after 1.0)."
    with _fatal("times must be strictly increasing"):
        TimeTable([1.0, 0.0], [0.0, 1.0])
    with _fatal("value 1 is not finite"):
        TimeTable([0.0, 1.0], [0.0, float("nan")])
    with _fatal("time 0 is not finite"):
        TimeTable([float("inf")], [0.0])
    with _fatal("the table has no points"):
        TimeTable([], [])
    with _fatal("2 times but 1 values"):
        TimeTable([0.0, 1.0], [0.0])
    with _fatal("repetition period must be finite and non-negative"):
        TimeTable([0.0, 1.0], [0.0, 1.0], period=-1.0)
    with _fatal("too steep"):
        TimeTable([0.0, 1e-300], [-1e300, 1e300])
    with _fatal("PULSE tr must not be negative"):
        table_spec(ComponentRecord("V1", "V", "a", "0", None, subtype="PULSE", metadata={"pulse": ["0", "1", "0", "-1n"]}))
    with _fatal("PULSE v2 must be finite"):
        pulse_table(0.0, float("nan"), t_step=1e-9, t_stop=1e-6)
    with _fatal("PWL times must not decrease"):
        table_spec(ComponentRecord("V1", "V", "a", "0", None, subtype="PWL", metadata={"points": ["1", "0", "0", "1"]}))
    with _fatal("PWL needs time-value pairs"):
        table_spec(ComponentRecord("V1", "V", "a", "0", None, subtype="PWL", metadata={"points": ["1", "0", "2"]}))
    # Placeholders without data, or not called with t.
    with _fatal("has no table data"):
        compile_expression("__table_0(t)", time_dependent=True, n_unknowns=0)
    with _fatal("must be called with the time 't'"):
        compile_expression("__table_0(2*t)", time_dependent=True, n_unknowns=0, tables=[TimeTable([0.0], [1.0])])
    # The core validates on its own as well.
    table_op = core.expr_opcodes()["table"]
    ops = np.array([table_op], dtype=np.int32)
    args = np.zeros(1, dtype=np.int32)
    with pytest.raises(core.CoreError) as error:
        core.compile_program(ops, args, np.zeros(0), 0, True, [(np.array([0.0, 0.0]), np.array([1.0, 2.0]), 0.0, 0.0)])
    assert error.value.code == "NETLIST_FATAL" and "strictly increasing" in str(error.value)
    with pytest.raises(core.CoreError):
        core.compile_program(ops, args, np.zeros(0), 0, True)  # no table 0
    with pytest.raises(core.CoreError):
        core.compile_program(ops, args, np.zeros(0), 0, False, [(np.array([0.0]), np.array([1.0]), 0.0, 0.0)])


def _hb_circuit(source: ComponentRecord) -> CircuitIR:
    return CircuitIR(
        [
            ComponentRecord("VS", "V", "a", "0", "1", subtype="SIN", value2="1k"),
            ComponentRecord("R1", "R", "a", "o", "1k"),
            source,
            ComponentRecord("R2", "R", "b", "o", "2k"),
            ComponentRecord("C1", "C", "o", "0", "100n"),
        ]
    )


def test_hb_with_a_pulse_matches_hb_with_the_equivalent_step() -> None:
    period = 1e-3
    duty = 0.5
    step = ComponentRecord("V2", "V", "b", "0", "2", subtype="STEP", value2="1k", value3=str(duty))
    td = (1.0 - duty) * period
    edge = 1e-12
    pulse = ComponentRecord(
        "V2",
        "V",
        "b",
        "0",
        None,
        subtype="PULSE",
        metadata={"pulse": ["0", "2", repr(td), repr(edge), repr(edge), repr(period - td - 2 * edge), repr(period)]},
    )
    options = AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=8)
    with_step = build_and_run_advanced(_hb_circuit(step), options)
    with_pulse = build_and_run_advanced(_hb_circuit(pulse), options)
    np.testing.assert_allclose(with_pulse.harmonic_state, with_step.harmonic_state, rtol=1e-12, atol=1e-15)
    assert with_pulse.metadata["table_sources"] == [{"name": "V2", "kind": "pulse", "pulse": [0.0, 2.0, td, edge, edge, period - td - 2 * edge, period]}]
    assert TIME_TABLES_KEY not in with_pulse.metadata


def test_hb_resolves_a_zero_rise_to_one_sample_spacing() -> None:
    pulse = ComponentRecord("V2", "V", "b", "0", None, subtype="PULSE", metadata={"pulse": ["0", "1", "0.25m", "0", "0", "0.25m", "1m"]})
    problem = build_mna_problem(_hb_circuit(pulse))
    tables = solvers.hb_tables(problem, 2 * np.pi * 1e3, 8)
    spacing = 1e-3 / 17
    np.testing.assert_allclose(tables[0].times, [0.25e-3, 0.25e-3 + spacing, 0.5e-3 + spacing, 0.5e-3 + 2 * spacing])
    result = build_and_run_advanced(_hb_circuit(pulse), AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=8))
    assert np.all(np.isfinite(result.harmonic_state))


def test_netlists_without_table_sources_have_no_table_metadata() -> None:
    records = [ComponentRecord("V1", "V", "a", "0", "1", subtype="SIN", value2="1k"), ComponentRecord("R1", "R", "a", "0", "1k")]
    problem = build_mna_problem(CircuitIR(records))
    assert TIME_TABLES_KEY not in problem.metadata
    result = run_basic_analysis(problem, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=1e-3, tran_step=1e-5))
    assert "table_sources" not in result.metadata
    np.testing.assert_array_equal(result.waveform.time, solvers._transient_time_points(1e-3, 1e-5))


# ---------------------------------------------------------------------------
# Netlist text: the parser's PULSE/PWL records become tables in the build
# ---------------------------------------------------------------------------


def test_netlist_pulse_and_pwl_become_tables() -> None:
    text = "V1 a 0 PULSE(0 5 1u 1u 1u 5u 10u)\nV2 b 0 DC 1 PWL(0 0 1u 4 3u 0) R=0 TD=1u\nV3 c 0 PWL(0 0 1u 0 1u 5 2u 5)\n"
    text += "R1 a 0 1k\nR2 b 0 1k\nR3 c 0 1k\n"
    circuit = parse_netlist_text(text)
    problem = build_mna_problem(circuit)
    rows = {name: problem.b_time_str[problem.index_map.branch_to_index[name], 0] for name in ("V1", "V2", "V3")}
    assert rows["V1"] == "__table_0(t)"
    assert rows["V2"] == "(__table_1(t) - 1.0)"  # the waveform minus the DC value (the .op value)
    assert "__table_" not in rows["V3"]  # a vertical step keeps the expression rendering
    assert problem.b_dc[problem.index_map.branch_to_index["V2"], 0] == 1.0
    # The tables and the parser's expression rendering agree.
    for record, index in ((circuit.components[0], 0), (circuit.components[1], 1)):
        reference = compile_expression(source_expression(record), time_dependent=True, n_unknowns=0)
        table = resolve_tables(problem.metadata, 1e-7, 1e-5)[index]
        times = np.linspace(0.0, 25e-6, 2501)
        np.testing.assert_allclose(_program(table).evaluate_batch(times), reference.evaluate_batch(times), atol=1e-12, rtol=0)
    # source_expression with the builder's list gives the placeholder as well.
    tables: list = []
    assert source_expression(circuit.components[0], tables) == "__table_0(t)" and len(tables) == 1


def test_a_digital_signal_netlist_line_round_trips_through_the_parser() -> None:
    signal = DigitalSignal.from_string("0110100", bit_period=83.33e-9, t_rise=8e-9, t_fall=8e-9)
    text = signal.netlist_line("VDP", "dp", "0") + "\nR1 dp 0 90\n.tran 600n 1n\n"
    problem = build_mna_problem(parse_netlist_text(text))
    table = resolve_tables(problem.metadata, 1e-9, 600e-9)[0]
    np.testing.assert_array_equal(np.column_stack((table.times, table.values)), signal.points())
