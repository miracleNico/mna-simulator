"""Expression compiler (``mna_simulation.native.expressions``) and the native bytecode VM.

The reference evaluates expressions with ``sympify`` + ``lambdify(..., "numpy")``
(``mna_reference.solvers.compile_time_source_func`` / ``compile_nl_functions``).
Every expression form used in the repository (``device_models.py``,
``tests/corpus.py``, ``app/web/src/lib/demoPresets.ts``) is compared against that
path over 10 000 random points at 1e-13 relative; most points agree bit for bit
(the only differences come from NumPy's SIMD ``exp``/``log``/``tan`` versus the C
library). Gradients (forward-mode dual numbers) are compared against the
lambdified SymPy derivative.
"""

from __future__ import annotations

import inspect
import math
import re
import warnings

import numpy as np
import pytest
import sympy as sp
from mna_reference import solvers as ref_solvers

from mna_simulation import native, solvers
from mna_simulation.api.contracts import ComponentRecord
from mna_simulation.device_models import diode_current_expression, mos_level1_current_expression, source_expression
from mna_simulation.errors import NetlistError
from mna_simulation.native.expressions import (
    _prepare_sympy_expression,
    _sympy_locals,
    compile_expression,
    lambdify_code,
)

core = native.require_core()
N_POINTS = 10_000
RTOL = 1e-13


def _source(subtype: str, value: str, value2: str | None = None, value3: str | None = None) -> str:
    return source_expression(
        ComponentRecord(name="V1", type="V", node1="a", node2="0", value=value, subtype=subtype, value2=value2, value3=value3)
    )


def _mos(mos_type: str, d: str, g: str, s: str) -> str:
    component = ComponentRecord(
        name="M1", type=mos_type, node1="d", node2="s", value="1m", value2="0.4", value3="0.02", ctrl_node1="g"
    )
    return mos_level1_current_expression(component, d, g, s)


def _diode(is_: str, a: str, k: str) -> str:
    return diode_current_expression(ComponentRecord(name="D1", type="D", node1="a", node2="k", value=is_), a, k)


V_HIGH = "1.2"
SRAM_WL = f"Piecewise((0, t < 1e-9), ({V_HIGH}, t < 2.5e-9), (0, t < 4.5e-9), ({V_HIGH}, t < 6e-9), (0, True))"

# (expression, time window) of every FUNC/SIN/COS/STEP form in the repository plus VM coverage forms.
TIME_FORMS: list[tuple[str, float]] = [
    (_source("SIN", "1.5", "1k"), 4e-3),
    (_source("COS", "0.02", "1k"), 4e-3),
    (_source("SIN", "40u", "1.005G"), 4e-9),
    (_source("STEP", "2", "1m"), 2e-3),
    (_source("STEP", "1", "1k", "0.5"), 4e-3),
    (_source("STEP", "1", "1k", "0.25"), 4e-3),
    (_source("FUNC", "1m", "0.8*t/0.001"), 4e-3),
    (_source("FUNC", "1m", "Piecewise((0.0005, t < 0.0005), (0, True))"), 4e-3),
    (_source("FUNC", "0", "0.5*exp(-t/0.0005) + 0.1"), 4e-3),
    (_source("FUNC", "0", "Piecewise((0, t < 1e-9), (1.2, t < 3.5e-9), (0, t < 4.5e-9), (1.2, True))"), 8e-9),
    (_source("FUNC", "0", SRAM_WL), 8e-9),
    (_source("FUNC", "0", f"Piecewise((0, t < 1e-9), ({V_HIGH}, t < 2.5e-9), (0, True))"), 8e-9),
    (_source("FUNC", "0", f"Piecewise(({V_HIGH}, t < 1e-9), (0, t < 2.5e-9), ({V_HIGH}, True))"), 8e-9),
    (_source("FUNC", "0", "Piecewise((0, t < 1e-3), (sin(t) + cos(t) + Max(t, 0), True))"), 4e-3),
    (_source("FUNC", "0", "Piecewise((0, t < 1e-9), (1.2, True))"), 4e-9),
    (_source("FUNC", "0", "t"), 1.0),
    (_source("FUNC", "1m", "t"), 4e-3),
    (_source("FUNC", "0", "1.5"), 1.0),
    # VM coverage: Min/Max, np.* spellings, mod, logic, powers, pi, tan/log/sqrt/abs.
    ("Min(t, 2e-3) + Max(t - 1e-3, 0) + np.minimum(t, 1e-3) - np.maximum(t, 3e-3)", 4e-3),
    ("np.heaviside(t - 1e-3, 0.5) + 2*np.heaviside(np.mod(t, 3e-4) - 1e-4, 1)", 4e-3),
    ("Piecewise((1, (t > 1e-3) & (t < 2e-3)), (2, (t < 5e-4) | (t > 3e-3)), (3, True))", 4e-3),
    ("Piecewise((t**2, t <= 1e-3), (sqrt(t)*pi, t >= 2e-3), (-t, True))", 4e-3),
    ("abs(sin(2*pi*1000*t)) + tan(t) + 1/(1 + t) - 2**(-t)", 4e-3),
    ("exp(-t/1e-3)*cos(2*pi*500*t) + Mod(t, 7e-4)/7e-4", 4e-3),
    ("Heaviside(t - 2e-3)", 4e-3),
]


def _time_points(window: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(0.0, window, N_POINTS - 1000)
    # A transient-style accumulated grid hits source breakpoints exactly.
    grid = [0.0]
    step = window / 999
    for _ in range(999):
        grid.append(grid[-1] + step)
    return np.concatenate([np.asarray(grid), random_points])


def _assert_close(current: np.ndarray, reference: np.ndarray, rtol: float = RTOL, atol: float | np.ndarray = 0.0) -> float:
    """|cur - ref| <= atol + rtol |ref| elementwise (NaN == NaN, inf == inf); returns the bitwise fraction."""

    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert current.shape == reference.shape
    same = (current == reference) | (np.isnan(current) & np.isnan(reference))
    with np.errstate(invalid="ignore"):
        close = np.abs(current - reference) <= atol + rtol * np.abs(reference)
    bad = ~(same | close)
    if bad.any():
        index = np.flatnonzero(bad)[0]
        raise AssertionError(
            f"{bad.sum()} mismatches, first at {index}: current={current.flat[index]!r} reference={reference.flat[index]!r}"
        )
    return float(same.mean())


@pytest.mark.parametrize(("text", "window"), TIME_FORMS, ids=[f"time{i}" for i in range(len(TIME_FORMS))])
def test_time_expressions_match_lambdify(text: str, window: float) -> None:
    t_values = _time_points(window, seed=len(text))
    reference_func = ref_solvers.compile_time_source_func(np.array([[text]], dtype=object))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.array([float(reference_func(float(t))[0, 0]) for t in t_values])
    program = compile_expression(text, time_dependent=True, n_unknowns=0)
    batch = program.evaluate_batch(t_values)
    bitwise = _assert_close(batch, reference)
    # The time-source facade evaluates through the same programs.
    facade = solvers.compile_time_source_func(np.array([["0"], [text]], dtype=object))
    for t in t_values[:: N_POINTS // 50]:
        values = facade(float(t))
        assert values.shape == (2, 1) and values[0, 0] == 0.0
        assert values[1, 0] == program.evaluate(None, float(t))
    if not re.search(r"exp|log|tan|\*\*", text):
        # Arithmetic, sin/cos, sqrt, mod and select are bit-identical to the NumPy evaluation.
        assert bitwise == 1.0
    else:
        assert bitwise > 0.9


# ---------------------------------------------------------------------------
# Expressions of the unknowns (nonlinear rows and behavioral sources)
# ---------------------------------------------------------------------------

DIODE_PAIR = f"{_diode('1e-14', 'x[0]', 'x[1]')} - ({_diode('2e-15', 'x[2]', 'x[0]')})"
NL_FORMS: list[tuple[str, bool]] = [
    # (expression with unknowns x[0..2], reference SymPy jacobian is lambdify-able)
    (_diode("1e-14", "x[0]", "0"), True),
    (_diode("1e-15", "x[0]", "x[1]"), True),
    (DIODE_PAIR, True),
    (f"-({_diode('3e-15', 'x[1]', 'x[2]')}) + ({_diode('1e-14', 'x[0]', 'x[1]')})", True),
    (_mos("NMOS", "x[0]", "x[1]", "x[2]"), True),
    (_mos("PMOS", "x[0]", "x[1]", "x[2]"), True),
    (_mos("NMOS", "x[0]", "x[1]", "0"), True),
    ("x[0]**2*x[1] - 3*x[2]/(1 + x[0]**2)", True),
    ("x_0*x_1 + x_2**3", True),
    ("Max(x[0], 0)*Min(x[1], 1) + np.maximum(x[2], -0.5)", True),
    ("sqrt(x[0]**2 + 1e-3) + exp(-x[2])*sin(x[1])*cos(x[0])", True),  # terms O(1), see NL_ATOL
    ("pi*x[0] - tan(x[1]) + log(x[2] + 2)", True),
    ("Piecewise((x[0], x[0] < 0.5), (0.5 + 0.1*x[1], True))", True),
    ("Piecewise((x[0]*x[1], (x[0] > 0) & (x[1] < 0.2)), (x[2], True))", True),
    ("abs(x[1]) + x[0]", False),
    ("np.heaviside(x[0], 1)*x[1]", False),
    ("np.mod(x[0], 0.3) + x[1]", False),
]


# Absolute tolerance for forms whose O(1) terms can cancel: a 1-ulp difference between NumPy's
# SIMD exp and the C library exp then exceeds 1e-13 relative to the (small) result.
NL_ATOL = {"sqrt(x[0]**2 + 1e-3) + exp(-x[2])*sin(x[1])*cos(x[0])": 2.3e-16}


def _nl_reference(text: str, size: int, with_jacobian: bool):
    f_str = np.array([[text]] + [["0"]] * (size - 1), dtype=object)
    if with_jacobian:
        return ref_solvers.compile_nl_functions(f_str, size)
    # Build f alone (the reference jacobian of abs/heaviside/mod cannot be lambdified).
    x_vars = sp.symbols(f"x_0:{size}")
    expression = text
    for i in range(size):
        expression = expression.replace(f"x[{i}]", f"x_{i}")
    f_sym = sp.Matrix([sp.sympify(_prepare_sympy_expression(expression), locals=_sympy_locals())] + [0] * (size - 1))
    return sp.lambdify(x_vars, f_sym, "numpy"), None


@pytest.mark.parametrize(("text", "jacobian_ok"), NL_FORMS, ids=[f"nl{i}" for i in range(len(NL_FORMS))])
def test_unknown_expressions_and_gradients_match_lambdify(text: str, jacobian_ok: bool) -> None:
    size = 3
    rng = np.random.default_rng(len(text))
    points = rng.uniform(-1.0, 1.0, (N_POINTS, size))
    ref_f, ref_j = _nl_reference(text, size, jacobian_ok)
    program = compile_expression(text, time_dependent=False, n_unknowns=size)
    wrt = np.arange(size, dtype=np.int32)
    values = np.empty(N_POINTS)
    reference = np.empty(N_POINTS)
    grads = np.empty((N_POINTS, size))
    ref_grads = np.empty((N_POINTS, size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, x in enumerate(points):
            values[i] = core.evaluate_program(program.handle, x, 0.0)
            reference[i] = float(np.asarray(ref_f(*x), dtype=float).reshape(-1)[0])
            value, grads[i] = core.evaluate_program_gradient(program.handle, x, 0.0, wrt)
            assert value == values[i]
            if ref_j is not None:
                ref_grads[i] = np.asarray(ref_j(*x), dtype=float).reshape(size, size)[0]
    _assert_close(values, reference, atol=NL_ATOL.get(text, 0.0))
    if ref_j is not None:
        # Dual numbers vs the lambdified symbolic derivative: same mathematics, different rounding
        # (observed: <= 1.6e-15 of the largest partial at the same point).
        row_scale = np.max(np.abs(ref_grads), axis=1, keepdims=True)
        _assert_close(grads, ref_grads, rtol=RTOL, atol=1e-14 * row_scale)
    else:
        # Central differences away from kinks.
        h = 1e-6
        for x, g in zip(points[:200], grads[:200]):
            for k in range(size):
                xp, xm = x.copy(), x.copy()
                xp[k] += h
                xm[k] -= h
                fd = (core.evaluate_program(program.handle, xp, 0.0) - core.evaluate_program(program.handle, xm, 0.0)) / (2 * h)
                if abs(fd - g[k]) > 1e-5 * (1 + abs(g[k])):
                    # Only legitimate near a kink (|x| ~ h) or a mod wrap.
                    assert min(abs(x[0]), abs(x[1]), abs(math.remainder(x[0], 0.3))) < 1e-5


def test_gradient_conventions_at_kinks() -> None:
    def grad(text: str, x: list[float]) -> np.ndarray:
        program = compile_expression(text, time_dependent=False, n_unknowns=len(x))
        return program.gradient(np.asarray(x))[1]

    # Heaviside derivative is 0 (SymPy's DiracDelta cannot be lambdified at all).
    assert grad("np.heaviside(x[0], 1)*x[1]", [0.0, 2.0]).tolist() == [0.0, 1.0]
    assert grad("np.heaviside(x[0], 1)*x[1]", [-1.0, 2.0]).tolist() == [0.0, 0.0]
    # abs: sign(x), 0 at 0.
    assert grad("abs(x[0])", [0.0]).tolist() == [0.0]
    assert grad("abs(x[0])", [-3.0]).tolist() == [-1.0]
    # Max/Min: the operand the value comes from; numpy.maximum returns the second operand at ties
    # (Max(x_0, 0) prints as reduce(maximum, [0, x_0])).
    assert grad("Max(x[0], 0)", [0.0]).tolist() == [1.0]
    assert grad("Max(x[0], 0)", [-1e-300]).tolist() == [0.0]
    assert grad("Max(x[0], x[1])", [2.0, 2.0]).tolist() in ([1.0, 0.0], [0.0, 1.0])
    # Piecewise: the selected branch only.
    assert grad("Piecewise((3*x[0], x[0] < 0), (5*x[0], True))", [0.0]).tolist() == [5.0]


def test_diode_jacobian_overflows_like_the_reference() -> None:
    program = compile_expression(_diode("1e-14", "x[0]", "0"), time_dependent=False, n_unknowns=2)
    value, grad = program.gradient(np.array([20.0, 0.0]), wrt=[0, 1])
    assert value == math.inf and grad[0] == math.inf and grad[1] == 0.0
    # Two diodes in one row, the first overflowing: the second diode's partial stays finite
    # (a zero partial is skipped, not multiplied by inf), as in the symbolic Jacobian.
    pair = compile_expression(DIODE_PAIR, time_dependent=False, n_unknowns=3)
    ref_f, ref_j = ref_solvers.compile_nl_functions(np.array([[DIODE_PAIR], ["0"], ["0"]], dtype=object), 3)
    x = np.array([20.0, 0.0, 19.9])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value, grad = pair.gradient(x, wrt=[0, 1, 2])
        reference = np.asarray(ref_j(*x), dtype=float)[0]
        reference_value = float(np.asarray(ref_f(*x), dtype=float)[0, 0])
    assert value == reference_value == math.inf
    assert grad[0] == reference[0] == math.inf and grad[1] == reference[1] == -math.inf
    assert math.isfinite(grad[2]) and grad[2] == pytest.approx(reference[2], rel=1e-15)


# ---------------------------------------------------------------------------
# Scalar semantics of individual opcodes vs NumPy
# ---------------------------------------------------------------------------

EDGE_VALUES = np.array([0.0, -0.0, 1.0, -1.0, 0.3, -0.7, 1e-300, -1e-300, 1e300, math.inf, -math.inf, math.nan, 0.1, 3.0])


def _pairs(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = np.concatenate([np.repeat(EDGE_VALUES, EDGE_VALUES.size), rng.uniform(-5, 5, 5000), rng.integers(-9, 9, 2000) * 0.1])
    b = np.concatenate([np.tile(EDGE_VALUES, EDGE_VALUES.size), rng.uniform(-5, 5, 5000), rng.choice([0.1, -0.1, 0.3, 1e-3], 2000)])
    return a, b


def _raw_program(op: str, n_unknowns: int = 2, consts: list[float] | None = None, extra: list[tuple[str, int]] | None = None):
    table = core.expr_opcodes()
    code = [("load_x", 0), ("load_x", 1)] + (extra or []) + [(op, 0)]
    return core.compile_program(
        np.array([table[name] for name, _ in code], dtype=np.int32),
        np.array([arg for _, arg in code], dtype=np.int32),
        np.array(consts or [], dtype=np.float64),
        n_unknowns,
        False,
    )


@pytest.mark.parametrize(
    ("op", "numpy_fn"),
    [
        ("pymod", np.mod),
        ("heaviside", np.heaviside),
        ("max", np.maximum),
        ("min", np.minimum),
        ("add", np.add),
        ("sub", np.subtract),
        ("mul", np.multiply),
        ("div", np.true_divide),
        ("lt", np.less),
        ("le", np.less_equal),
        ("gt", np.greater),
        ("ge", np.greater_equal),
        ("eq", np.equal),
        ("ne", np.not_equal),
        ("and", np.logical_and),
        ("or", np.logical_or),
    ],
)
def test_binary_opcodes_match_numpy_bitwise(op: str, numpy_fn) -> None:
    program = _raw_program(op)
    a, b = _pairs(seed=len(op))
    with np.errstate(all="ignore"):
        # NumPy's scalar path, which is what lambdified code runs on np.float64 arguments.
        expected = np.array([float(numpy_fn(np.float64(x), np.float64(y))) for x, y in zip(a, b)])
        got = np.array([core.evaluate_program(program, np.array([x, y]), 0.0) for x, y in zip(a, b)])
    same = (got == expected) | (np.isnan(got) & np.isnan(expected))
    same &= (np.signbit(got) == np.signbit(expected)) | np.isnan(got)  # bit for bit, including the sign of zero
    assert same.all(), (op, a[~same][:5], b[~same][:5], got[~same][:5], expected[~same][:5])


def test_pow_matches_numpy_to_rounding() -> None:
    # NumPy's float64 power (also on scalars) is its own SIMD implementation on AVX-512
    # machines; it and the C library pow agree to an ulp or two, special cases exactly.
    program = _raw_program("pow")
    a, b = _pairs(seed=3)
    with np.errstate(all="ignore"):
        expected = np.array([float(np.power(np.float64(x), np.float64(y))) for x, y in zip(a, b)])
        got = np.array([core.evaluate_program(program, np.array([x, y]), 0.0) for x, y in zip(a, b)])
    _assert_close(got, expected, rtol=5e-16)
    special = ~np.isfinite(expected) | (expected == 0.0)
    assert np.array_equal(got[special], expected[special], equal_nan=True)


def test_unary_opcodes_match_numpy() -> None:
    table = core.expr_opcodes()
    rng = np.random.default_rng(5)
    values = np.concatenate([EDGE_VALUES, rng.uniform(-20, 20, 5000)])
    for op, numpy_fn in (("neg", np.negative), ("sqrt", np.sqrt), ("abs", np.abs), ("sin", np.sin), ("cos", np.cos), ("not", np.logical_not)):
        program = core.compile_program(
            np.array([table["load_x"], table[op]], dtype=np.int32), np.zeros(2, dtype=np.int32), np.zeros(0), 1, False
        )
        with np.errstate(all="ignore"):
            expected = numpy_fn(values).astype(float)
        got = np.array([core.evaluate_program(program, np.array([v]), 0.0) for v in values])
        assert ((got == expected) | (np.isnan(got) & np.isnan(expected))).all(), op
    for op, numpy_fn in (("exp", np.exp), ("log", np.log), ("tan", np.tan)):
        program = core.compile_program(
            np.array([table["load_x"], table[op]], dtype=np.int32), np.zeros(2, dtype=np.int32), np.zeros(0), 1, False
        )
        with np.errstate(all="ignore"):
            expected = numpy_fn(values)
        got = np.array([core.evaluate_program(program, np.array([v]), 0.0) for v in values])
        _assert_close(got, expected, rtol=4e-16)


def test_select_without_a_true_branch_is_nan_like_numpy_select() -> None:
    program = compile_expression("Piecewise((1, t < 1), (2, t < 2))", time_dependent=True, n_unknowns=0)
    assert program.code.startswith("select(")
    values = program.evaluate_batch(np.array([0.5, 1.5, 2.5]))
    assert values[0] == 1.0 and values[1] == 2.0 and math.isnan(values[2])
    reference = ref_solvers.compile_time_source_func(np.array([["Piecewise((1, t < 1), (2, t < 2))"]], dtype=object))
    assert math.isnan(float(reference(2.5)[0, 0]))


def test_numpy_mod_is_not_the_floor_formula() -> None:
    program = compile_expression("Mod(t, 0.1)", time_dependent=True, n_unknowns=0)
    assert program.evaluate(None, 1.0) == float(np.mod(1.0, 0.1)) == 0.09999999999999995
    assert 1.0 - 0.1 * math.floor(1.0 / 0.1) == 0.0  # what the naive formula would give


# ---------------------------------------------------------------------------
# Compiler behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [TIME_FORMS[0][0], TIME_FORMS[4][0], TIME_FORMS[8][0], TIME_FORMS[13][0], "Heaviside(t - 2e-3)", "x_0**-2 + Mod(t, 3)"],
)
def test_printed_code_is_what_lambdify_runs(text: str) -> None:
    t = sp.Symbol("t")
    expr = sp.sympify(_prepare_sympy_expression(text), locals=_sympy_locals(t))
    x0 = sp.Symbol("x_0")
    source = inspect.getsource(sp.lambdify((x0, t), sp.Matrix([expr]), "numpy"))
    body = re.search(r"return array\(\[\[(.*)\]\]\)\s*$", source.strip(), re.S)
    assert body is not None, source
    assert lambdify_code(text, time_dependent=True, n_unknowns=1) == body.group(1)


def test_programs_serialize_to_int32_and_float64_arrays_and_are_cached() -> None:
    text = "(1e-14 * (np.exp((x[0] - x[1]) / 0.025) - 1))"
    program = compile_expression(text, time_dependent=False, n_unknowns=2)
    assert compile_expression(text, time_dependent=False, n_unknowns=2) is program
    assert compile_expression(text, time_dependent=False, n_unknowns=3) is not program
    assert program.opcodes.dtype == np.int32 and program.args.dtype == np.int32 and program.consts.dtype == np.float64
    assert not program.opcodes.flags.writeable and not program.consts.flags.writeable
    assert program.unknowns == (0, 1)
    assert program.code == "1.0e-15*exp(40.0*x_0 - 40.0*x_1) - 1.0e-15".replace("1.0e-15", "1.0e-14")
    handle = program.handle
    assert handle.unknowns == [0, 1] and handle.n_unknowns == 2 and not handle.time_dependent
    np.testing.assert_array_equal(handle.opcodes(), program.opcodes)
    np.testing.assert_array_equal(handle.args(), program.args)
    np.testing.assert_array_equal(handle.consts(), program.consts)
    assert len(handle) == program.opcodes.size
    # x[i] and x_i spell the same unknown.
    assert compile_expression("x_0*x_1", time_dependent=False, n_unknowns=2).code == compile_expression(
        "x[0]*x[1]", time_dependent=False, n_unknowns=2
    ).code
    # Constant subexpressions fold exactly as Python would evaluate them.
    folded = compile_expression("2*np.pi*1000*t", time_dependent=True, n_unknowns=0)
    assert folded.consts.tolist() == [2000 * math.pi]
    assert folded.evaluate(None, 0.25) == 2000 * math.pi * 0.25


def test_opcode_table_matches_the_core() -> None:
    table = core.expr_opcodes()
    assert table["const"] == 0 and table["load_x"] == 1 and table["load_t"] == 2 and table["select"] == 29
    assert len(table) == 30 and sorted(table.values()) == list(range(30))


def test_unsupported_functions_raise_netlist_fatal() -> None:
    for text, name in (("sinh(x[0])", "sinh"), ("floor(t) + 1", "floor"), ("gamma(t)", "gamma"), ("atan(t)", "arctan")):
        with pytest.raises(NetlistError) as exc:
            compile_expression(text, time_dependent=True, n_unknowns=1)
        assert exc.value.code == "NETLIST_FATAL"
        assert f"Unsupported function '{name}'" in exc.value.message
        assert text in exc.value.message


def test_unparsable_text_raises_netlist_fatal() -> None:
    with pytest.raises(NetlistError) as exc:
        compile_expression("1 +* (2", time_dependent=True, n_unknowns=0)
    assert exc.value.code == "NETLIST_FATAL" and "Could not parse expression" in exc.value.message


def test_unknown_symbols_raise_type_error_like_the_reference() -> None:
    # The reference fails with TypeError when it evaluates such an expression (e.g. a SIN source
    # with a 1e999 frequency prints 'inf' as a free symbol; tests/test_edge_cases pins it).
    for text, time_dependent in (("foo*t", True), ("2*pi*inf*t", True), ("t*x[0]", False), ("x[5]", False), ("x_00", False)):
        with pytest.raises(TypeError, match="Unsupported symbol"):
            compile_expression(text, time_dependent=time_dependent, n_unknowns=2)


def test_core_rejects_malformed_programs() -> None:
    table = core.expr_opcodes()
    with pytest.raises(core.CoreError) as exc:
        core.compile_program(np.array([table["add"]], dtype=np.int32), np.zeros(1, dtype=np.int32), np.zeros(0), 0, False)
    assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError):
        core.compile_program(np.array([table["load_x"]], dtype=np.int32), np.array([3], dtype=np.int32), np.zeros(0), 2, False)
    with pytest.raises(core.CoreError):
        core.compile_program(np.array([table["load_t"]], dtype=np.int32), np.zeros(1, dtype=np.int32), np.zeros(0), 0, False)
    with pytest.raises(core.CoreError):
        core.compile_program(np.array([0, 0], dtype=np.int32), np.zeros(1, dtype=np.int32), np.zeros(1), 0, False)
    program = compile_expression("x[0] + t", time_dependent=True, n_unknowns=1)
    with pytest.raises(core.CoreError):
        core.evaluate_program_batch(program.handle, np.zeros(3))  # reads an unknown
    with pytest.raises(core.CoreError):
        core.evaluate_program(program.handle, np.zeros(0), 0.0)  # x too short


def test_time_source_facade_keeps_zero_rows_and_shape() -> None:
    evaluate = solvers.compile_time_source_func(np.array([["0"], ["0"]], dtype=object))
    assert evaluate(1.0).shape == (2, 1) and not evaluate(1.0).any()
    nan_source = solvers.compile_time_source_func(np.array([["(nan)"]], dtype=object))
    assert math.isnan(nan_source(0.0)[0, 0])
