"""Native device kernels and deterministic assembly vs the frozen reference.

* ``solvers.compile_nl_functions`` (native) vs ``mna_reference.solvers.compile_nl_functions``
  on every nonlinear corpus case at 1000 random states: Level-1 MOS circuits are
  bit-identical; circuits with exponentials agree to 1e-13 (the only differences come
  from NumPy's SIMD ``exp`` versus the C library ``exp``: with the reference patched to
  the C library ``exp`` the Level-1 BJT results are bit-identical too).
* ``_level1_*`` facades vs the reference functions.
* Thread-count determinism of ``_core.evaluate_devices``.
* SRAM 10x10 (622 MOS) evaluation time vs the reference Python loop.
"""

from __future__ import annotations

import math
import timeit
import types
import warnings

import numpy as np
import pytest
from corpus import iter_cases, sram_netlist
from mna_reference import solvers as ref_solvers
from mna_reference.mna_builder import build_mna_problem as reference_build
from mna_reference.netlist import parse_netlist_text as reference_parse

from mna_simulation import native, solvers
from mna_simulation.core_basic import _bjt_level1_region, _mos_level1_region
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.native.circuit import bjt_table, build_circuit, diode_table, mos_table

core = native.require_core()
N_STATES = 1000


def _nonlinear_cases():
    seen: set[str] = set()
    for case in iter_cases():
        if case.netlist_text in seen:
            continue
        try:
            reference = reference_build(reference_parse(case.netlist_text))
        except Exception:  # noqa: BLE001
            continue
        if ref_solvers.is_nonlinear(reference.solver_f_str) or reference.level1_mos_devices or reference.level1_bjt_devices:
            seen.add(case.netlist_text)
            yield case


NONLINEAR_CASES = list(_nonlinear_cases())


class _NumpyWithLibmExp(types.ModuleType):
    """numpy with exp replaced by the C library exp (what the native kernels call)."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def exp(value):
        return np.float64(math.exp(float(value)))


def _close(current: np.ndarray, reference: np.ndarray, rtol: float, atol: float | np.ndarray = 0.0) -> None:
    same = (current == reference) | (np.isnan(current) & np.isnan(reference))
    with np.errstate(invalid="ignore"):
        close = np.abs(current - reference) <= atol + rtol * np.abs(reference)
    bad = ~(same | close)
    assert not bad.any(), (np.flatnonzero(bad)[:5], current[bad][:5], reference[bad][:5])


BJT_VT = 0.02585
BJT_EXP_CLIP = 80.0


def _bjt_inside_clip(devices, x) -> bool:
    """True when every Level-1 BJT junction of ``devices`` has |v / vt| <= 80 at state ``x``.

    Beyond the clip the native kernel continues the exponential linearly (P3b,
    CPP_MIGRATION_PLAN.md 2.4.1) while the oracle clips it, so oracle parity is only
    checked inside; the extrapolation itself is tested in tests/test_native_convergence.py.
    """

    def v(index) -> float:
        return 0.0 if int(index) == -1 else float(x[int(index)])

    for device in devices:
        vc, vb, ve = v(device["collector"]), v(device["base"]), v(device["emitter"])
        if abs(vb - ve) / BJT_VT > BJT_EXP_CLIP or abs(vb - vc) / BJT_VT > BJT_EXP_CLIP:
            return False
    return True


def _terminals_inside_clip(vc: float, vb: float, ve: float) -> bool:
    return abs(vb - ve) / BJT_VT <= BJT_EXP_CLIP and abs(vb - vc) / BJT_VT <= BJT_EXP_CLIP


def _bitwise(current: np.ndarray, reference: np.ndarray) -> bool:
    return current.shape == reference.shape and np.array_equal(
        np.ascontiguousarray(current, dtype=np.float64).view(np.uint64),
        np.ascontiguousarray(reference, dtype=np.float64).view(np.uint64),
    )


@pytest.mark.parametrize("case", NONLINEAR_CASES, ids=[case.name for case in NONLINEAR_CASES])
def test_compiled_f_and_jacobian_match_reference(case, monkeypatch: pytest.MonkeyPatch) -> None:
    ir = reference_parse(case.netlist_text)
    reference = reference_build(ir)
    problem = build_mna_problem(ir)
    n = problem.G.shape[0]
    f_func, j_func = solvers.compile_nl_functions(
        problem.solver_f_str, n, problem.level1_mos_devices, problem.level1_bjt_devices
    )
    ref_f, ref_j = ref_solvers.compile_nl_functions(
        reference.solver_f_str, n, reference.level1_mos_devices, reference.level1_bjt_devices
    )
    exponential = ref_solvers.is_nonlinear(reference.solver_f_str) or bool(reference.level1_bjt_devices)
    states = np.random.default_rng(sum(map(ord, case.name))).uniform(-1.5, 1.5, (N_STATES, n))
    # Oracle parity only where every BJT junction is inside the exponent clip (P3b extrapolates beyond).
    states = np.array([x for x in states if _bjt_inside_clip(reference.level1_bjt_devices, x)])
    assert len(states) >= 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for x in states:
            f, jac = f_func(*x), j_func(*x)
            assert f.shape == (n, 1) and jac.shape == (n, n)
            rf = np.asarray(ref_f(*x), dtype=float).reshape(n, 1)
            rj = np.asarray(ref_j(*x), dtype=float).reshape(n, n)
            if not exponential:
                assert _bitwise(f, rf) and _bitwise(jac, rj)
                continue
            _close(f, rf, rtol=1e-13)
            row_scale = np.max(np.abs(rj), axis=1, keepdims=True)
            _close(jac, rj, rtol=1e-13, atol=1e-14 * row_scale)

    if reference.level1_bjt_devices and not ref_solvers.is_nonlinear(reference.solver_f_str):
        # Same arithmetic as the reference: bit-identical once both use the C library exp.
        monkeypatch.setattr(ref_solvers, "np", _NumpyWithLibmExp("numpy_libm_exp"))
        for x in states[:200]:
            assert _bitwise(f_func(*x), np.asarray(ref_f(*x), dtype=float).reshape(n, 1))
            assert _bitwise(j_func(*x), np.asarray(ref_j(*x), dtype=float).reshape(n, n))


def test_compiled_functions_check_the_state_length() -> None:
    problem = build_mna_problem(reference_parse("V1 a 0 DC 1\nR1 a b 1k\nD1 b 0 1e-14\n"))
    f_func, j_func = solvers.compile_nl_functions(problem.solver_f_str, problem.G.shape[0])
    with pytest.raises(TypeError):
        f_func(1.0, 2.0)
    with pytest.raises(TypeError):
        j_func(1.0, 2.0, 3.0, 4.0)
    assert f_func(0.0, 0.0, 0.0).shape == (3, 1)


MOS_DEVICES = [
    {"type": "NMOS", "beta": 1e-3, "vth": 0.4, "lambda": 0.02},
    {"type": "PMOS", "beta": 3e-3, "vth": 0.4, "lambda": 0.02},
    {"type": "nmos", "beta": 2.5e-3, "vth": 0.7, "lambda": 0.0},
]
BJT_DEVICES = [
    {"type": "QNPN", "is": 1e-15, "bf": 150, "br": 3, "vaf": 100, "var": 25},
    {"type": "QPNP", "is": 2e-15, "bf": 80.0, "br": 2.0, "vaf": 0.0, "var": 0.0},
    {"type": "QNPN", "is": 1e-14, "bf": 0.0, "br": 0.0, "vaf": 50.0, "var": 10.0},
]


def _terminal_points(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = [(1.0, 0.4, 0.0), (0.8, 1.2, 0.0), (0.0, 1.2, 0.1), (0.3, 1.2, 0.3), (0.2, 0.0, 1.2), (5.0, 80 * 0.02585, 0.0)]
    return np.vstack([np.asarray(edges), rng.uniform(-2.5, 2.5, (3000, 3))])


def test_level1_mos_facade_is_bit_identical() -> None:
    for device in MOS_DEVICES:
        for vd, vg, vs in _terminal_points(1):
            current = solvers._level1_mos_current_and_derivatives(device, vd, vg, vs)
            reference = ref_solvers._level1_mos_current_and_derivatives(device, vd, vg, vs)
            assert isinstance(current, tuple) and len(current) == 4
            assert current == reference
    for vd, vg, vs in _terminal_points(2)[:500]:
        assert solvers._nmos_level1_current_and_derivatives(vd, vg, vs, 1e-3, 0.4, 0.02) == (
            ref_solvers._nmos_level1_current_and_derivatives(vd, vg, vs, 1e-3, 0.4, 0.02)
        )


def test_level1_bjt_facade_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inside the exponent clip only (P3b extrapolates beyond it, the oracle clips).
    points = np.array([p for p in _terminal_points(3) if _terminals_inside_clip(*p)])
    assert len(points) >= 1000
    for device in BJT_DEVICES:
        for vc, vb, ve in points:
            currents, derivatives = solvers._level1_bjt_currents_and_derivatives(device, vc, vb, ve)
            ref_currents, ref_derivatives = ref_solvers._level1_bjt_currents_and_derivatives(device, vc, vb, ve)
            assert len(currents) == 3 and len(derivatives) == 3 and all(len(row) == 3 for row in derivatives)
            current = np.array([currents, *derivatives], dtype=float)
            reference = np.array([ref_currents, *ref_derivatives], dtype=float)
            _close(current, reference, rtol=1e-13, atol=1e-14 * np.max(np.abs(reference), axis=1, keepdims=True))
    monkeypatch.setattr(ref_solvers, "np", _NumpyWithLibmExp("numpy_libm_exp"))
    for device in BJT_DEVICES:
        for vc, vb, ve in points[:1000]:
            assert solvers._level1_bjt_currents_and_derivatives(device, vc, vb, ve) == _float_tuples(
                ref_solvers._level1_bjt_currents_and_derivatives(device, vc, vb, ve)
            )
            assert solvers._bjt_level1_npn_currents_and_derivatives(vc, vb, ve, 1e-15, 150, 3, 100, 25) == _float_tuples(
                ref_solvers._bjt_level1_npn_currents_and_derivatives(vc, vb, ve, 1e-15, 150, 3, 100, 25)
            )


def _float_tuples(result):
    """((ic, ib, ie), ((...), (...), (...))) with NumPy scalars turned into Python floats."""

    currents, derivatives = result
    return tuple(float(v) for v in currents), tuple(tuple(float(v) for v in row) for row in derivatives)


def test_device_list_facades_match_reference() -> None:
    reference = reference_build(reference_parse(sram_netlist(2, 2)))
    n = reference.G.shape[0]
    rng = np.random.default_rng(4)
    for _ in range(20):
        x = rng.uniform(-0.2, 1.4, n)
        f, jac = solvers._evaluate_level1_mos_devices(reference.level1_mos_devices, n, x)
        rf, rj = ref_solvers._evaluate_level1_mos_devices(reference.level1_mos_devices, n, x)
        assert _bitwise(f, rf) and _bitwise(jac, rj)
    bjt = reference_build(reference_parse(next(c for c in iter_cases() if c.name == "bjt_three_stage_test").netlist_text))
    n = bjt.G.shape[0]
    for _ in range(20):
        x = rng.uniform(-1.0, 1.0, n)
        f, jac = solvers._evaluate_level1_bjt_devices(bjt.level1_bjt_devices, n, x)
        rf, rj = ref_solvers._evaluate_level1_bjt_devices(bjt.level1_bjt_devices, n, x)
        _close(f, rf, rtol=1e-13)
        _close(jac, rj, rtol=1e-13, atol=1e-14 * np.max(np.abs(rj), axis=1, keepdims=True))


def test_region_boundaries_through_the_facade() -> None:
    nmos = {"type": "NMOS", "beta": 1e-3, "vth": 0.4, "lambda": 0.02}
    assert solvers._level1_mos_current_and_derivatives(nmos, 1.0, 0.4, 0.0) == (0.0, 0.0, 0.0, 0.0)
    at = solvers._level1_mos_current_and_derivatives(nmos, 1.2 - 0.4, 1.2, 0.0)
    assert at[0] == 0.5 * 1e-3 * (1.2 - 0.4) ** 2 * (1.0 + 0.02 * (1.2 - 0.4))
    reverse = solvers._level1_mos_current_and_derivatives(nmos, 0.0, 1.2, 0.1)
    forward = solvers._level1_mos_current_and_derivatives(nmos, 0.1, 1.2, 0.0)
    assert reverse == (-forward[0], -forward[3], -forward[2], -forward[1])
    bjt = BJT_DEVICES[0]
    clip, d_clip = solvers._level1_bjt_currents_and_derivatives(bjt, 5.0, 80 * 0.02585, 0.0)
    beyond, d_beyond = solvers._level1_bjt_currents_and_derivatives(bjt, 5.0, 6.0, 0.0)
    # Beyond the clip the exponential continues linearly (P3b, 2.4.1): the current keeps growing
    # with the clip-point slope, which is also the reported derivative.
    assert d_beyond[0][1] == pytest.approx(d_clip[0][1], rel=1e-12)
    slope = (beyond[0] - clip[0]) / (6.0 - 80 * 0.02585)
    assert slope == pytest.approx(d_clip[0][1], rel=1e-6)
    currents, _ = solvers._level1_bjt_currents_and_derivatives(bjt, 0.3, 0.7, 0.1)
    assert currents[2] == -(currents[0] + currents[1])


def test_operating_points_binding_matches_the_facades() -> None:
    netlist = "VDD vdd 0 DC 1.2\nM1 out in vdd PMOS LEVEL1 2m 0.4 0.02 2f 1f\nM2 out in 0 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"
    netlist += "VIN in 0 DC 0.6\nQ1 c b e QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5\n"
    netlist += "Q2 c2 b2 e2 QPNP LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5\nR1 c 0 1k\n"
    netlist += "R2 b 0 1k\nR3 e 0 1k\nR4 c2 0 1k\nR5 b2 0 1k\nR6 e2 0 1k\n"
    problem = build_mna_problem(reference_parse(netlist))
    rng = np.random.default_rng(6)
    for _ in range(50):
        x = rng.uniform(-1.5, 1.5, problem.G.shape[0])
        op = core.operating_points(problem.native, x)

        def v(index: int) -> float:
            return 0.0 if index == -1 else float(x[index])

        for i, device in enumerate(problem.level1_mos_devices):
            vd, vg, vs = v(device["drain"]), v(device["gate"]), v(device["source"])
            ids, gds, gm, d_source = solvers._level1_mos_current_and_derivatives(device, vd, vg, vs)
            assert (op["mos"]["ids"][i], op["mos"]["gds"][i], op["mos"]["gm"][i], op["mos"]["d_source"][i]) == (ids, gds, gm, d_source)
            region = core.mos_region_names[op["mos"]["region"][i]]
            assert region == _mos_level1_region(device["type"], vd, vg, vs, device["vth"])
        for i, device in enumerate(problem.level1_bjt_devices):
            vc, vb, ve = v(device["collector"]), v(device["base"]), v(device["emitter"])
            currents, derivatives = solvers._level1_bjt_currents_and_derivatives(device, vc, vb, ve)
            assert tuple(op["bjt"][key][i] for key in ("ic", "ib", "ie")) == currents
            assert op["bjt"]["gm"][i] == derivatives[0][1] and op["bjt"]["go"][i] == derivatives[0][0]
            assert op["bjt"]["rpi"][i] == (1.0 / derivatives[1][1] if abs(derivatives[1][1]) > 1e-30 else math.inf)
            assert core.bjt_region_names[op["bjt"]["region"][i]] == _bjt_level1_region(device["type"], vc, vb, ve)


# ---------------------------------------------------------------------------
# Determinism and evaluation buffers
# ---------------------------------------------------------------------------


def _large_random_circuit(seed: int):
    rng = np.random.default_rng(seed)
    n = 400

    def nodes(count: int) -> np.ndarray:
        return rng.integers(-1, n, count)

    mos = [
        {"type": rng.choice(["NMOS", "PMOS"]), "drain": d, "gate": g, "source": s, "beta": 1e-3, "vth": 0.4, "lambda": 0.02}
        for d, g, s in zip(nodes(3000), nodes(3000), nodes(3000))
    ]
    bjt = [
        {"type": rng.choice(["QNPN", "QPNP"]), "collector": c, "base": b, "emitter": e, "is": 1e-15, "bf": 150.0, "br": 3.0, "vaf": 100.0, "var": 25.0}
        for c, b, e in zip(nodes(800), nodes(800), nodes(800))
    ]
    from mna_simulation.native.circuit import behavioral_table
    from mna_simulation.native.expressions import compile_expression

    programs = [compile_expression(f"x[{i}]*x[{i + 1}] - sin(x[{i + 2}])", time_dependent=False, n_unknowns=n) for i in range(50)]
    return n, build_circuit(
        n,
        mos=mos_table(mos),
        bjt=bjt_table(bjt),
        diodes=diode_table(nodes(500), nodes(500), np.full(500, 1e-14)),
        behavioral=behavioral_table(nodes(50), nodes(50), programs),
    )


@pytest.mark.parametrize("which", ["sram", "random"])
def test_evaluation_is_bitwise_identical_for_any_thread_count(which: str) -> None:
    if which == "sram":
        problem = build_mna_problem(reference_parse(sram_netlist(10, 10)))
        n, circuit = problem.G.shape[0], problem.native
    else:
        n, circuit = _large_random_circuit(11)
    rng = np.random.default_rng(12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(5):
            x = rng.uniform(-1.0, 1.4, n)
            outputs = []
            for threads in (1, 2, 3, 4, 7):
                f = np.full(n, np.nan)
                jvals = np.full(circuit.nnz, np.nan)
                core.evaluate_devices(circuit, x, 0.0, threads, f, jvals)
                outputs.append((f, jvals))
            for f, jvals in outputs[1:]:
                assert _bitwise(f, outputs[0][0]) and _bitwise(jvals, outputs[0][1])
            # Dense helpers agree with the sparse values.
            dense = core.device_jacobian_dense(circuit, x, 0.0)
            col_ptr, row_idx = circuit.pattern()
            expected = np.zeros((n, n))
            for col in range(n):
                expected[row_idx[col_ptr[col] : col_ptr[col + 1]], col] = outputs[0][1][col_ptr[col] : col_ptr[col + 1]]
            assert _bitwise(dense, expected)
            assert _bitwise(core.device_f(circuit, x, 0.0).reshape(-1), outputs[0][0])


def test_evaluate_devices_validates_buffers() -> None:
    problem = build_mna_problem(reference_parse(sram_netlist(1, 1)))
    circuit, n = problem.native, problem.G.shape[0]
    x = np.zeros(n)
    core.evaluate_devices(circuit, x, 0.0, 1, None, None)
    f = np.zeros(n)
    core.evaluate_devices(circuit, x, 0.0, 1, f, None)
    with pytest.raises(core.CoreError):
        core.evaluate_devices(circuit, np.zeros(n + 1), 0.0, 1, f, None)
    with pytest.raises(core.CoreError):
        core.evaluate_devices(circuit, x, 0.0, 1, np.zeros(n - 1), None)
    with pytest.raises(core.CoreError):
        core.evaluate_devices(circuit, x, 0.0, 1, None, np.zeros(circuit.nnz + 1))
    with pytest.raises(core.CoreError):
        core.evaluate_devices(circuit, x, 0.0, 1, np.zeros(n, dtype=np.float32), None)
    readonly = np.zeros(n)
    readonly.setflags(write=False)
    with pytest.raises(core.CoreError):
        core.evaluate_devices(circuit, x, 0.0, 1, readonly, None)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.perf  # timing-sensitive: opt in with --run-perf (the perf gate covers the same units)
def test_sram_device_evaluation_speedup() -> None:
    reference = reference_build(reference_parse(sram_netlist(10, 10)))
    problem = build_mna_problem(reference_parse(sram_netlist(10, 10)))
    n = problem.G.shape[0]
    assert len(problem.level1_mos_devices) == 622
    x = np.random.default_rng(7).uniform(0.0, 1.2, n)
    f_func, j_func = solvers.compile_nl_functions(problem.solver_f_str, n, problem.level1_mos_devices)
    f_buffer, j_buffer = np.zeros(n), np.zeros(problem.native.nnz)

    def best(fn, number: int) -> float:
        return min(timeit.repeat(fn, number=number, repeat=7)) / number

    reference_time = best(lambda: ref_solvers._evaluate_level1_mos_devices(reference.level1_mos_devices, n, x), 10)
    facade_time = best(lambda: (f_func(*x), j_func(*x)), 100)
    kernel_time = best(lambda: core.evaluate_devices(problem.native, x, 0.0, 1, f_buffer, j_buffer), 1000)
    print(
        f"\nSRAM 10x10 (622 MOS, n={n}) f + J: reference loop {reference_time * 1e6:.0f} us, "
        f"compile_nl_functions f_func + j_func (dense J) {facade_time * 1e6:.0f} us "
        f"({reference_time / facade_time:.0f}x), _core.evaluate_devices (pattern values) "
        f"{kernel_time * 1e6:.1f} us ({reference_time / kernel_time:.0f}x)"
    )
    assert reference_time / facade_time >= 20.0
    assert reference_time / kernel_time >= 20.0
