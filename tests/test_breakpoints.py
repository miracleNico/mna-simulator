"""Breakpoint-aware transient grid (MIXED_SIGNAL.md phase D1).

* Every corner time of every table source inside [0, t_stop] is a grid point, exactly.
* Ramps shorter than ``t_step`` get ``RAMP_POINTS`` (4) extra points at 1/5 ... 4/5 of the ramp;
  longer ramps only their corners (the uniform grid samples them).
* ``breakpoints=False`` restores the uniform grid; a netlist without table sources keeps the
  uniform grid and bitwise the same values.
* Points closer than the core's grid tolerance merge (corners win over ramp and uniform points).
* Results are identical for 1 and 4 threads.
"""

from __future__ import annotations

import numpy as np
import pytest

from mna_simulation import native, solvers
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, CircuitIR, ComponentRecord
from mna_simulation.core_basic import run_basic_analysis
from mna_simulation.errors import NetlistError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.expressions import TimeTable
from mna_simulation.netlist import parse_netlist_text
from mna_simulation.table_sources import MAX_BREAKPOINTS, RAMP_POINTS, breakpoints, merge_breakpoints
from mna_simulation.utils import parse_value

core = native.require_core()
REL_TOL = float(core.TIME_GRID_REL_TOL)


def _stimulus(name: str, node: str, times, values, period=None) -> ComponentRecord:
    table = TimeTable(times, values, period=period, name=name)
    return ComponentRecord(name, "V", node, "0", None, subtype="PWL", metadata={"stimulus": table})


def _rc(*sources: ComponentRecord, diode: bool = False) -> CircuitIR:
    records = list(sources)
    for source in sources:
        records.append(ComponentRecord(f"R{source.name}", "R", source.node1, "out", "1k"))
    records.append(ComponentRecord("C1", "C", "out", "0", "10p"))
    if diode:
        records.append(ComponentRecord("D1", "D", "out", "0", "1e-14"))
    return CircuitIR(records)


def _run(circuit: CircuitIR, t_stop: float, t_step: float, **extra) -> object:
    problem = build_mna_problem(circuit)
    return solvers.solve_transient(problem, t_stop, t_step, **extra)


def test_every_corner_time_appears_in_the_grid_exactly() -> None:
    pwl_times = [3.3e-9, 10e-9, 10.7e-9, 50.123456789e-9, 77e-9]
    stimulus = _stimulus("VA", "a", pwl_times, [0.0, 1.0, 0.0, 0.0, 2.0])
    pulse = ComponentRecord(
        "VB", "V", "b", "0", None, subtype="PULSE", metadata={"pulse": ["0", "1", "2.5n", "0.3n", "0.4n", "7.77n", "0"]}
    )
    pwl = ComponentRecord("VC", "V", "c", "0", None, subtype="PWL", metadata={"points": ["1.111n", "0", "61.7n", "1"]})
    waveform = _run(_rc(stimulus, pulse, pwl), 100e-9, 1e-9)
    grid = waveform.time
    assert grid[0] == 0.0 and grid[-1] == 100e-9 and np.all(np.diff(grid) > 0.0)
    td, tr, pw, tf = (parse_value(v) for v in ("2.5n", "0.3n", "7.77n", "0.4n"))
    pulse_corners = [td, td + tr, td + tr + pw, td + tr + pw + tf]
    for corner in [*pwl_times, *pulse_corners, parse_value("1.111n"), parse_value("61.7n")]:
        assert corner in set(grid.tolist()), corner
    # The source follows its table at every grid point (exactly: the corners are grid points).
    v_a = waveform.values[waveform.labels.index("V(a)")]
    np.testing.assert_array_equal(v_a[np.isin(grid, pwl_times)], [0.0, 1.0, 0.0, 0.0, 2.0])


def test_periodic_corners_are_in_the_grid() -> None:
    period = 20e-9
    table = TimeTable([1e-9, 1.5e-9, 6e-9, 6.5e-9], [0.0, 1.0, 1.0, 0.0], period=period)
    grid = _run(_rc(ComponentRecord("VP", "V", "p", "0", None, subtype="PWL", metadata={"stimulus": table})), 100e-9, 2e-9).time
    for m in range(5):
        start = 1e-9 + m * period
        for offset in (0.0, 0.5e-9, 5e-9, 5.5e-9):
            expected = start + offset
            assert np.min(np.abs(grid - expected)) <= 1e-21, (m, offset)
    corners, ramps = breakpoints([table], 100e-9, 2e-9)
    assert corners.size == 20 and ramps.size == 2 * RAMP_POINTS * 5
    assert set(corners.tolist()) <= set(grid.tolist())


def test_ramp_refinement_counts() -> None:
    # A 1 ns edge under a 10 ns step: four points at 0.2, 0.4, 0.6, 0.8 of the ramp.
    fast = _stimulus("VF", "f", [33e-9, 34e-9], [0.0, 3.3])
    grid = _run(_rc(fast), 100e-9, 10e-9).time
    inside = grid[(grid > 33e-9) & (grid < 34e-9)]
    np.testing.assert_allclose(inside, 33e-9 + 1e-9 * np.array([0.2, 0.4, 0.6, 0.8]), rtol=1e-14, atol=0)
    assert grid.size == 11 + 2 + RAMP_POINTS
    # A 5 ns edge under a 1 ns step: no extra points, the uniform grid samples it at least five times.
    slow = _stimulus("VS", "s", [20.5e-9, 25.5e-9], [0.0, 3.3])
    grid = _run(_rc(slow), 50e-9, 1e-9).time
    on_ramp = grid[(grid >= 20.5e-9) & (grid <= 25.5e-9)]
    assert on_ramp.size >= 5
    assert grid.size == 51 + 2
    # A flat segment shorter than the step is not a ramp: corners only.
    flat = _stimulus("VZ", "z", [10e-9, 10.5e-9, 11e-9], [1.0, 1.0, 1.0])
    assert _run(_rc(flat), 50e-9, 1e-9).time.size == 51 + 1  # 10 ns is a uniform point; 10.5 ns and 11 ns ...
    # ... 11 ns is a uniform point too: only 10.5 ns is new.


def test_breakpoints_false_restores_the_uniform_grid() -> None:
    stimulus = _stimulus("VA", "a", [3.3e-9, 4e-9, 50e-9], [0.0, 1.0, 0.0])
    circuit = _rc(stimulus)
    uniform = solvers._transient_time_points(100e-9, 1e-9)
    waveform = _run(circuit, 100e-9, 1e-9, breakpoints=False)
    np.testing.assert_array_equal(waveform.time, uniform)
    merged = _run(circuit, 100e-9, 1e-9)
    assert merged.time.size > uniform.size
    # Through the analysis driver: options without the field default to breakpoints on.
    result = run_basic_analysis(build_mna_problem(circuit), AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=100e-9, tran_step=1e-9))
    assert result.metadata["time_points"] == merged.time.size
    np.testing.assert_array_equal(result.waveform.time, merged.time)


NO_TABLES = """V1 a 0 SIN 1 1e5
V2 b 0 STEP 2 2e5 0.3
R1 a c 1k
R2 b c 2k
D1 c 0 1e-14
C1 c 0 1n
"""


@pytest.mark.parametrize("breakpoints_on", [True, False])
def test_netlists_without_table_sources_keep_the_grid_and_the_values(breakpoints_on: bool) -> None:
    problem = build_mna_problem(parse_netlist_text(NO_TABLES))
    uniform = solvers._transient_time_points(2e-5, 1e-7)
    waveform = solvers.solve_transient(problem, 2e-5, 1e-7, breakpoints=breakpoints_on)
    np.testing.assert_array_equal(waveform.time, uniform)
    # The values are those of the native core on the uniform grid with the plain programs.
    rows, programs = solvers._time_sources(problem.b_time_str)
    circuit, gmin = solvers._problem_circuit(problem, True)
    b_dc = np.ascontiguousarray(problem.b_dc.reshape(-1))
    x0 = solvers._native_op(
        circuit,
        gmin,
        core.time_source_vector(circuit, b_dc, rows, programs, 0.0),
        np.zeros(circuit.n),
        max_iter=10000,
        v_tol=1e-9,
        f_tol=1e-9,
        use_krylov=False,
        krylov_tol=1e-9,
        krylov_max_iter=2000,
        krylov_restart=80,
        krylov_rank="auto",
        krylov_method="auto",
        krylov_stats=None,
        num_threads=1,
    ).reshape(-1)
    values, _info = core.solve_transient(circuit, x0, uniform, b_dc, rows, programs, gmin, 10000, 1e-9, 1e-9, 1)
    np.testing.assert_array_equal(waveform.values, values)


def test_merge_rules_tolerance_and_end_points() -> None:
    uniform = solvers._transient_time_points(10.0, 1.0)
    tol = 10.0 * REL_TOL
    # A corner within the tolerance of a uniform point replaces it; one within the tolerance of an
    # end point is dropped (the end points stay exact); two corners closer than it merge to the first.
    table = TimeTable([0.5 * tol, 3.0 + 0.5 * tol, 5.5, 5.5 + 0.5 * tol, 10.0 - 0.5 * tol], [0.0, 0.0, 1.0, 1.0, 1.0])
    grid = merge_breakpoints(uniform, [table], 10.0, 1.0, rel_tol=REL_TOL)
    assert grid[0] == 0.0 and grid[-1] == 10.0
    assert 3.0 + 0.5 * tol in grid and 3.0 not in grid
    assert 5.5 in grid and 5.5 + 0.5 * tol not in grid
    assert np.all(np.diff(grid) > tol)
    # No tables, or none inside [0, t_stop]: the uniform grid itself.
    assert merge_breakpoints(uniform, [], 10.0, 1.0, rel_tol=REL_TOL) is uniform
    outside = TimeTable([20.0, 30.0], [0.0, 1.0])
    np.testing.assert_array_equal(merge_breakpoints(uniform, [outside], 10.0, 1.0, rel_tol=REL_TOL), uniform)
    # t_stop = 0: the one-point grid.
    np.testing.assert_array_equal(
        merge_breakpoints(solvers._transient_time_points(0.0, 1.0), [table], 0.0, 1.0, rel_tol=REL_TOL), [0.0]
    )


def test_too_many_breakpoints_is_a_tran_error() -> None:
    clock = TimeTable([0.0, 1e-12, 5e-10, 5.01e-10], [0.0, 1.0, 1.0, 0.0], period=1e-9)
    with pytest.raises(NetlistError) as exc:
        breakpoints([clock], 1.0, 1e-6)
    assert exc.value.code == "TRAN_ERROR" and str(MAX_BREAKPOINTS) in exc.value.message
    assert "breakpoints=false" in exc.value.message


def test_results_are_identical_for_1_and_4_threads() -> None:
    stimulus = _stimulus("VA", "a", np.arange(0, 200) * 5e-9 + 1e-9, np.tile([0.0, 3.3], 100))
    pulse = ComponentRecord(
        "VB", "V", "b", "0", None, subtype="PULSE", metadata={"pulse": ["0", "1", "3n", "0.2n", "0.2n", "4n", "10n"]}
    )
    circuit = _rc(stimulus, pulse, diode=True)
    problem = build_mna_problem(circuit)
    one = solvers.solve_transient(problem, 1e-6, 1e-9, num_threads=1)
    four = solvers.solve_transient(problem, 1e-6, 1e-9, num_threads=4)
    np.testing.assert_array_equal(one.time, four.time)
    np.testing.assert_array_equal(one.values, four.values)


def test_tpwl_and_linear_mor_use_the_merged_grid() -> None:
    stimulus = _stimulus("VA", "a", [3.3e-9, 4e-9, 50e-9], [0.0, 1.0, 0.0])
    linear = _rc(stimulus)
    options = AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=100e-9, tran_step=1e-9, use_mor=True, mor_output_nodes=["out"])
    result = run_basic_analysis(build_mna_problem(linear), options)
    expected = _run(linear, 100e-9, 1e-9).time
    np.testing.assert_array_equal(result.waveform.time, expected)
    assert result.metadata["time_points"] == expected.size
    nonlinear = _rc(stimulus, diode=True)
    options = AnalysisOptions(
        mode=AnalysisMode.TRAN, tran_stop=100e-9, tran_step=1e-9, use_mor=True, mor_method="tpwl", mor_output_nodes=["out"]
    )
    result = run_basic_analysis(build_mna_problem(nonlinear), options)
    np.testing.assert_array_equal(result.waveform.time, _run(nonlinear, 100e-9, 1e-9).time)
