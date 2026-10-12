"""Compile circuit expressions to the native bytecode VM of ``mna_simulation._core``.

FUNC time sources, ``B`` behavioral sources and the nonlinear current rows of
``MnaProblem.solver_f_str`` are strings. The reference core evaluated them with
SymPy: ``sympify`` with :func:`_sympy_locals`, then ``lambdify(..., "numpy")``.
This module keeps SymPy as the *parser* only and lowers to the postfix program
that ``cpp/src/expr/program.cpp`` executes:

1. normalize the text exactly as before (:func:`_prepare_sympy_expression`,
   ``x[i]`` becomes ``x_i``) and ``sympify`` it with the same locals;
2. print it with the NumPy code printer configured exactly as ``lambdify`` does;
3. walk the Python AST of that printed code into postfix bytecode.

Step 2 matters for parity: the printer rewrites the tree before lambdify ever
runs it (``Heaviside`` becomes ``select([x < 0, ...], ...)``, ``Mod(t, 0.001)``
becomes ``0.001*mod(1000*t, 1)``, rationals print as ``(1/3)``, floats print
with 15 significant digits, negative integer powers become float powers), so
lowering the printed code reproduces the reference evaluation operation by
operation, in the same left-to-right order. Results then agree bit for bit
wherever the C library and NumPy agree (they differ in the last bit for some
``exp``/``log``/``tan`` arguments, where NumPy uses its own SIMD code).

Table sources (MIXED_SIGNAL.md phase D1): a placeholder call ``__table_k(t)``
lowers to the VM's ``TABLE k`` instruction, a binary-search lookup in the
``k``-th :class:`TimeTable` passed to :func:`compile_expression` (``tables=``).
The table data is attached to the program at compile time, so a row that mixes
a table with other terms (``(__table_0(t)) - (0.5 * sin(...))``) is still one
program; the placeholder text itself never changes with the data.

Errors: text SymPy cannot parse and functions or constructs the VM does not
support raise :class:`~mna_simulation.errors.NetlistError` (``NETLIST_FATAL``);
a free symbol other than ``x_i`` (``i < n_unknowns``) or ``t`` (time-dependent
programs only) raises ``TypeError``, the exception the reference core raised
for such expressions when it evaluated them.
"""

from __future__ import annotations

import ast
import hashlib
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ..errors import error_handler
from . import require_core

__all__ = [
    "MAX_TABLE_POINTS",
    "Program",
    "TimeTable",
    "compile_expression",
    "table_placeholder",
    "lambdify_code",
    "_prepare_sympy_expression",
    "_sympy_locals",
]


def _sympy_locals(t_symbol: sp.Symbol | None = None) -> dict[str, object]:
    """Symbols/functions allowed in generated nonlinear and time expressions."""

    locals_map: dict[str, object] = {
        "sp": sp,
        "np": sp,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "Heaviside": sp.Heaviside,
        "heaviside": sp.Heaviside,
        "Piecewise": sp.Piecewise,
        "Mod": sp.Mod,
        "mod": sp.Mod,
        "Min": sp.Min,
        "Max": sp.Max,
        "pi": sp.pi,
        "True": True,
        "False": False,
    }
    if t_symbol is not None:
        locals_map["t"] = t_symbol
    return locals_map


def _prepare_sympy_expression(expr: str) -> str:
    """Normalize numpy-style generated expressions for SymPy parsing."""

    prepared = str(expr)
    replacements = (
        ("np.heaviside", "Heaviside"),
        ("np.Heaviside", "Heaviside"),
        ("np.mod", "Mod"),
        ("np.Mod", "Mod"),
        ("np.minimum", "Min"),
        ("np.maximum", "Max"),
        ("sp.heaviside", "Heaviside"),
        ("sp.Heaviside", "Heaviside"),
        ("sp.mod", "Mod"),
        ("sp.Mod", "Mod"),
        ("sp.minimum", "Min"),
        ("sp.maximum", "Max"),
    )
    for old, new in replacements:
        prepared = prepared.replace(old, new)
    prepared = prepared.replace("np.", "").replace("sp.", "")
    prepared = re.sub(r"\b(Min|Max)\(\[(.*?)\]\)", r"\1(\2)", prepared)
    return prepared


_X_INDEX = re.compile(r"\bx\[\s*(\d+)\s*\]")
_X_SYMBOL = re.compile(r"x_(0|[1-9][0-9]*)\Z")
_TABLE_NAME = re.compile(r"__table_(0|[1-9][0-9]*)\Z")

#: Largest number of points of one table (stimuli, PWL records, the VM's TABLE data).
MAX_TABLE_POINTS = 1_000_000


def table_placeholder(index: int) -> str:
    """Expression text of table ``index`` evaluated at ``t`` (``__table_<index>(t)``)."""

    return f"__table_{int(index)}(t)"


class TimeTable:
    """A piecewise-linear table read by the VM's ``TABLE`` instruction.

    ``times`` strictly increasing and finite, ``values`` finite (one per time, at
    least one point, at most :data:`MAX_TABLE_POINTS`); the first value is held
    before ``times[0]`` and the last after ``times[-1]``. ``period > 0`` repeats
    the waveform: a time ``t >= origin`` is folded to
    ``origin + mod(t - origin, period)`` (``origin`` defaults to ``times[0]``; a
    SPICE ``PWL ... R=r`` uses ``origin = r``). Invalid data raises
    ``NETLIST_FATAL`` naming ``name``. Instances are immutable and compare by
    content (``key``, a digest of the data), so programs built from equal tables
    share the compile cache.
    """

    __slots__ = ("times", "values", "period", "origin", "name", "key")

    def __init__(
        self,
        times: Any,
        values: Any,
        period: float | None = None,
        origin: float | None = None,
        *,
        name: str = "table",
    ) -> None:
        try:
            t = np.array(times, dtype=float).reshape(-1)
            v = np.array(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            error_handler(f"NETLIST_FATAL: {name}: table points must be numbers ({exc}).")
        if t.size == 0:
            error_handler(f"NETLIST_FATAL: {name}: the table has no points.")
        if t.size != v.size:
            error_handler(f"NETLIST_FATAL: {name}: the table has {t.size} times but {v.size} values.")
        if t.size > MAX_TABLE_POINTS:
            error_handler(f"NETLIST_FATAL: {name}: the table has {t.size} points (at most {MAX_TABLE_POINTS}).")
        bad = np.flatnonzero(~np.isfinite(t))
        if bad.size:
            error_handler(f"NETLIST_FATAL: {name}: time {int(bad[0])} is not finite ({float(t[bad[0]])!r}).")
        bad = np.flatnonzero(~np.isfinite(v))
        if bad.size:
            error_handler(f"NETLIST_FATAL: {name}: value {int(bad[0])} is not finite ({float(v[bad[0]])!r}).")
        if t.size > 1:
            steps = np.diff(t)
            bad = np.flatnonzero(~(steps > 0.0))
            if bad.size:
                i = int(bad[0]) + 1
                error_handler(
                    f"NETLIST_FATAL: {name}: times must be strictly increasing "
                    f"(point {i}: {float(t[i])!r} after {float(t[i - 1])!r})."
                )
            with np.errstate(all="ignore"):
                slopes = np.diff(v) / steps
            bad = np.flatnonzero(~np.isfinite(slopes))
            if bad.size:
                error_handler(f"NETLIST_FATAL: {name}: segment {int(bad[0])} is too steep (its slope is not finite).")
        p = 0.0 if period is None else float(period)
        if not np.isfinite(p) or p < 0.0:
            error_handler(f"NETLIST_FATAL: {name}: the repetition period must be finite and non-negative (got {p!r}).")
        o = float(t[0]) if origin is None else float(origin)
        if not np.isfinite(o):
            error_handler(f"NETLIST_FATAL: {name}: the repetition origin must be finite (got {o!r}).")
        t.setflags(write=False)
        v.setflags(write=False)
        digest = hashlib.sha256()
        digest.update(struct.pack("<qdd", t.size, p, o))
        digest.update(t.tobytes())
        digest.update(v.tobytes())
        self.times = t
        self.values = v
        self.period = p
        self.origin = o
        self.name = str(name)
        self.key = digest.hexdigest()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimeTable) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __len__(self) -> int:
        return int(self.times.size)

    def __repr__(self) -> str:
        return (
            f"TimeTable(name={self.name!r}, points={self.times.size}, period={self.period!r}, "
            f"origin={self.origin!r}, key={self.key[:16]})"
        )

    def native(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """The ``(times, values, period, origin)`` tuple ``_core.compile_program`` takes."""

        return (self.times, self.values, self.period, self.origin)

    def __call__(self, t: Any) -> np.ndarray:
        """Value at ``t`` (scalar or array) computed with NumPy, the VM's semantics (reference)."""

        times, values = self.times, self.values
        x = np.array(t, dtype=float)
        if self.period > 0.0:
            fold = x >= self.origin
            x = np.where(fold, self.origin + np.mod(x - self.origin, self.period), x)
        out = np.empty_like(x)
        flat_x = x.reshape(-1)
        flat = out.reshape(-1)
        index = np.searchsorted(times, flat_x, side="right") - 1
        inner = (index >= 0) & (index < times.size - 1)
        flat[index < 0] = values[0]
        flat[index >= times.size - 1] = values[-1]
        if np.any(inner):
            i = index[inner]
            slopes = (values[i + 1] - values[i]) / (times[i + 1] - times[i])
            flat[inner] = values[i] + slopes * (flat_x[inner] - times[i])
        flat[np.isnan(flat_x)] = np.nan
        return out
_T_SYMBOL = sp.Symbol("t")

# Printed NumPy names -> VM mnemonics (see cpp/include/mna/expr/program.hpp).
_UNARY = {
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sqrt": "sqrt",
    "abs": "abs",
    "absolute": "abs",
    "logical_not": "not",
}
_BINARY = {
    "heaviside": "heaviside",
    "mod": "pymod",
    "remainder": "pymod",
    "power": "pow",
    "minimum": "min",
    "maximum": "max",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
    "logical_and": "and",
    "logical_or": "or",
}
_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_CONSTANT_NAMES = {"pi": float(np.pi), "e": float(np.e), "nan": float("nan"), "inf": float("inf")}
_FOLDABLE = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
}


def _printer() -> NumPyPrinter:
    """A NumPy code printer configured exactly as ``sympy.lambdify(..., "numpy")`` configures it."""

    return NumPyPrinter(
        {
            "fully_qualified_modules": False,
            "inline": True,
            "allow_unknown_functions": True,
            "user_functions": {},
        }
    )


def _opcodes() -> dict[str, int]:
    table = _OPCODE_CACHE.get("table")
    if table is None:
        table = {str(name): int(code) for name, code in require_core().expr_opcodes().items()}
        _OPCODE_CACHE["table"] = table
    return table


_OPCODE_CACHE: dict[str, dict[str, int]] = {}

Instruction = tuple[str, float]


@dataclass(frozen=True, eq=False)
class Program:
    """A compiled expression: postfix bytecode plus the metadata the core needs.

    ``opcodes``/``args`` (int32) and ``consts`` (float64) are the serialized
    form consumed by ``_core.compile_program``; ``handle`` is the validated
    native program (created on first use). Instances are shared through the
    compile cache and must be treated as immutable.
    """

    text: str
    code: str
    opcodes: np.ndarray
    args: np.ndarray
    consts: np.ndarray
    n_unknowns: int
    time_dependent: bool
    unknowns: tuple[int, ...]
    tables: tuple[TimeTable, ...] = ()

    @cached_property
    def handle(self) -> Any:
        core = require_core()
        if self.tables:
            return core.compile_program(
                self.opcodes,
                self.args,
                self.consts,
                self.n_unknowns,
                self.time_dependent,
                [table.native() for table in self.tables],
            )
        return core.compile_program(self.opcodes, self.args, self.consts, self.n_unknowns, self.time_dependent)

    def evaluate(self, x: np.ndarray | None = None, t: float = 0.0) -> float:
        values = np.zeros(self.n_unknowns) if x is None else np.ascontiguousarray(x, dtype=float).reshape(-1)
        return float(require_core().evaluate_program(self.handle, values, float(t)))

    def evaluate_batch(self, t_array: np.ndarray) -> np.ndarray:
        return require_core().evaluate_program_batch(self.handle, np.ascontiguousarray(t_array, dtype=float).reshape(-1))

    def gradient(self, x: np.ndarray, t: float = 0.0, wrt: Any = None) -> tuple[float, np.ndarray]:
        """Value and gradient with respect to ``wrt`` (default: :attr:`unknowns`)."""

        indices = self.unknowns if wrt is None else wrt
        value, grad = require_core().evaluate_program_gradient(
            self.handle,
            np.ascontiguousarray(x, dtype=float).reshape(-1),
            float(t),
            np.ascontiguousarray(indices, dtype=np.int32).reshape(-1),
        )
        return float(value), grad


def _fail_parse(text: str, exc: BaseException) -> None:
    error_handler(f"NETLIST_FATAL: Could not parse expression '{text}': {exc}")


def _unsupported(name: str, text: str) -> None:
    error_handler(f"NETLIST_FATAL: Unsupported function '{name}' in expression '{text}'.")


def _unknown_symbol(name: str, text: str) -> TypeError:
    return TypeError(f"Unsupported symbol '{name}' in expression '{text}'.")


def _sympify(text: str, time_dependent: bool, n_unknowns: int) -> sp.Basic:
    prepared = _X_INDEX.sub(r"x_\1", _prepare_sympy_expression(text))
    try:
        expr = sp.sympify(prepared, locals=_sympy_locals(_T_SYMBOL if time_dependent else None))
    except Exception as exc:  # noqa: BLE001 - SymPy raises many types for bad text
        _fail_parse(text, exc)
    if isinstance(expr, bool):
        expr = sp.true if expr else sp.false
    if not isinstance(expr, sp.Basic):
        _fail_parse(text, TypeError(f"not a scalar expression ({type(expr).__name__})"))
    for symbol in sorted(expr.free_symbols, key=str):
        name = str(symbol)
        match = _X_SYMBOL.match(name)
        if match is not None and int(match.group(1)) < n_unknowns:
            continue
        if time_dependent and name == "t":
            continue
        raise _unknown_symbol(name, text)
    return expr


def lambdify_code(text: str, *, time_dependent: bool, n_unknowns: int) -> str:
    """The NumPy code ``lambdify`` runs for ``text`` (after the reference normalization)."""

    expr = _sympify(text, time_dependent, n_unknowns)
    printer = _printer()
    code = printer.doprint(expr)
    not_supported = getattr(printer, "_not_supported", set())
    if not_supported:
        _unsupported(sorted(type(item).__name__ for item in not_supported)[0], text)
    return code


class _Lowering:
    """Python AST of printed NumPy code -> postfix instructions."""

    def __init__(self, text: str, time_dependent: bool, n_unknowns: int, n_tables: int = 0) -> None:
        self.text = text
        self.time_dependent = time_dependent
        self.n_unknowns = n_unknowns
        self.n_tables = n_tables

    def lower(self, node: ast.AST) -> list[Instruction]:
        if isinstance(node, ast.Expression):
            return self.lower(node.body)
        if isinstance(node, ast.Constant):
            return [("const", self._constant(node.value))]
        if isinstance(node, ast.Name):
            return [self._name(node.id)]
        if isinstance(node, ast.UnaryOp):
            operand = self.lower(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                if len(operand) == 1 and operand[0][0] == "const":
                    return [("const", float(-np.float64(operand[0][1])))]
                return operand + [("neg", 0)]
            if isinstance(node.op, ast.Not):
                return operand + [("not", 0)]
            _unsupported(type(node.op).__name__, self.text)
        if isinstance(node, ast.BinOp):
            name = _BINOPS.get(type(node.op))
            if name is None:
                _unsupported(type(node.op).__name__, self.text)
            return self._binary(name, self.lower(node.left), self.lower(node.right))
        if isinstance(node, ast.Call):
            return self._call(node)
        _unsupported(type(node).__name__, self.text)
        raise AssertionError("unreachable")

    def _constant(self, value: object) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except OverflowError:
                _unsupported(f"integer constant {value}", self.text)
        _unsupported(f"constant {value!r}", self.text)
        raise AssertionError("unreachable")

    def _name(self, name: str) -> Instruction:
        match = _X_SYMBOL.match(name)
        if match is not None and int(match.group(1)) < self.n_unknowns:
            return ("load_x", int(match.group(1)))
        if name == "t" and self.time_dependent:
            return ("load_t", 0)
        if name in _CONSTANT_NAMES:
            return ("const", _CONSTANT_NAMES[name])
        raise _unknown_symbol(name, self.text)

    @staticmethod
    def _binary(name: str, left: list[Instruction], right: list[Instruction]) -> list[Instruction]:
        if name in _FOLDABLE and len(left) == 1 and len(right) == 1 and left[0][0] == "const" and right[0][0] == "const":
            with np.errstate(all="ignore"):
                folded = _FOLDABLE[name](np.float64(left[0][1]), np.float64(right[0][1]))
            return [("const", float(folded))]
        return left + right + [(name, 0)]

    def _chain(self, name: str, items: list[ast.AST]) -> list[Instruction]:
        if not items:
            _unsupported(name, self.text)
        code = self.lower(items[0])
        for item in items[1:]:
            code = self._binary(name, code, self.lower(item))
        return code

    def _call(self, node: ast.Call) -> list[Instruction]:
        func = node.func
        if isinstance(func, ast.Attribute):
            # logical_and.reduce((a, b, ...)) / logical_or.reduce((...))
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if owner in {"logical_and", "logical_or"} and func.attr == "reduce" and len(node.args) == 1 and not node.keywords:
                seq = node.args[0]
                if isinstance(seq, (ast.Tuple, ast.List)):
                    return self._chain(_BINARY[owner], list(seq.elts))
            _unsupported(f"{owner}.{func.attr}" if owner else func.attr, self.text)
        if not isinstance(func, ast.Name):
            _unsupported(type(func).__name__, self.text)
        name = func.id
        if name == "select":
            return self._select(node)
        table = _TABLE_NAME.match(name)
        if table is not None:
            return [self._table(int(table.group(1)), node)]
        if node.keywords:
            _unsupported(name, self.text)
        if name == "reduce":
            if (
                len(node.args) == 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in {"maximum", "minimum"}
                and isinstance(node.args[1], (ast.List, ast.Tuple))
            ):
                return self._chain(_BINARY[node.args[0].id], list(node.args[1].elts))
            _unsupported("reduce", self.text)
        if name in _UNARY and len(node.args) == 1:
            return self.lower(node.args[0]) + [(_UNARY[name], 0)]
        if name in _BINARY and len(node.args) == 2:
            return self._binary(_BINARY[name], self.lower(node.args[0]), self.lower(node.args[1]))
        _unsupported(name, self.text)
        raise AssertionError("unreachable")

    def _table(self, index: int, node: ast.Call) -> Instruction:
        """``__table_k(t)`` -> ``TABLE k`` (the argument must be the time symbol itself)."""

        argument = node.args[0] if len(node.args) == 1 and not node.keywords else None
        if not (isinstance(argument, ast.Name) and argument.id == "t" and self.time_dependent):
            error_handler(
                f"NETLIST_FATAL: Table placeholder '__table_{index}' in expression '{self.text}' "
                "must be called with the time 't' in a time-dependent expression."
            )
        if index >= self.n_tables:
            error_handler(
                f"NETLIST_FATAL: Table placeholder '__table_{index}' in expression '{self.text}' has no table data "
                f"({self.n_tables} table{'s' if self.n_tables != 1 else ''} supplied)."
            )
        return ("table", index)

    def _select(self, node: ast.Call) -> list[Instruction]:
        # select([c1, ..., ck], [v1, ..., vk], default=nan)
        if len(node.args) != 2:
            _unsupported("select", self.text)
        conds, values = node.args
        if not isinstance(conds, (ast.List, ast.Tuple)) or not isinstance(values, (ast.List, ast.Tuple)):
            _unsupported("select", self.text)
        if len(conds.elts) != len(values.elts) or not conds.elts:
            _unsupported("select", self.text)
        for keyword in node.keywords:
            if keyword.arg != "default" or not (isinstance(keyword.value, ast.Name) and keyword.value.id == "nan"):
                _unsupported("select", self.text)
        code: list[Instruction] = []
        for value, cond in zip(values.elts, conds.elts):
            code += self.lower(value)
            code += self.lower(cond)
        return code + [("select", len(conds.elts))]


def _assemble(instructions: list[Instruction]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opcodes = _opcodes()
    ops: list[int] = []
    args: list[int] = []
    consts: list[float] = []
    const_index: dict[str, int] = {}
    for name, arg in instructions:
        if name == "const":
            value = float(arg)
            key = value.hex()
            index = const_index.get(key)
            if index is None:
                index = len(consts)
                const_index[key] = index
                consts.append(value)
            ops.append(opcodes["const"])
            args.append(index)
        else:
            ops.append(opcodes[name])
            args.append(int(arg))
    arrays = (
        np.asarray(ops, dtype=np.int32),
        np.asarray(args, dtype=np.int32),
        np.asarray(consts, dtype=np.float64),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _lower(text: str, time_dependent: bool, n_unknowns: int, n_tables: int) -> Program:
    """The program of ``text`` without table data (``tables`` empty), not yet validated by the core."""

    code = lambdify_code(text, time_dependent=time_dependent, n_unknowns=n_unknowns)
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError as exc:
        _fail_parse(text, exc)
    instructions = _Lowering(text, time_dependent, n_unknowns, n_tables).lower(tree)
    opcodes, args, consts = _assemble(instructions)
    unknowns = tuple(sorted({int(arg) for name, arg in instructions if name == "load_x"}))
    return Program(
        text=text,
        code=code,
        opcodes=opcodes,
        args=args,
        consts=consts,
        n_unknowns=n_unknowns,
        time_dependent=time_dependent,
        unknowns=unknowns,
    )


@lru_cache(maxsize=4096)
def _compile_cached(text: str, time_dependent: bool, n_unknowns: int) -> Program:
    program = _lower(text, time_dependent, n_unknowns, 0)
    program.handle  # noqa: B018 - validate in the core now so errors surface at compile time
    return program


@lru_cache(maxsize=4096)
def _lower_with_tables(text: str, time_dependent: bool, n_unknowns: int, n_tables: int) -> Program:
    return _lower(text, time_dependent, n_unknowns, n_tables)


# Programs that carry table data: a small cache (one table may hold MAX_TABLE_POINTS points).
@lru_cache(maxsize=64)
def _compile_with_tables(text: str, time_dependent: bool, n_unknowns: int, tables: tuple[TimeTable, ...]) -> Program:
    base = _lower_with_tables(text, time_dependent, n_unknowns, len(tables))
    program = Program(
        text=base.text,
        code=base.code,
        opcodes=base.opcodes,
        args=base.args,
        consts=base.consts,
        n_unknowns=base.n_unknowns,
        time_dependent=base.time_dependent,
        unknowns=base.unknowns,
        tables=tables,
    )
    program.handle  # noqa: B018 - validate in the core now so errors surface at compile time
    return program


def compile_expression(
    text: str,
    *,
    time_dependent: bool,
    n_unknowns: int,
    tables: Sequence[TimeTable] | None = None,
) -> Program:
    """Compile ``text`` to a native :class:`Program` (cached per ``(text, time_dependent, n_unknowns)``).

    ``x[i]`` and ``x_i`` both denote unknown ``i`` (``0 <= i < n_unknowns``);
    ``t`` is allowed only when ``time_dependent``. ``tables`` (default none) holds
    the :class:`TimeTable` data of the placeholders ``__table_k(t)`` in ``text``
    (``k`` indexes this sequence); a placeholder without data is ``NETLIST_FATAL``.
    The program carries every table given, whether it reads it or not.
    """

    if n_unknowns < 0:
        raise ValueError("n_unknowns must be non-negative")
    if not tables:
        return _compile_cached(str(text), bool(time_dependent), int(n_unknowns))
    table_tuple = tuple(tables)
    for table in table_tuple:
        if not isinstance(table, TimeTable):
            raise TypeError(f"tables must be TimeTable instances, got {type(table).__name__}")
    return _compile_with_tables(str(text), bool(time_dependent), int(n_unknowns), table_tuple)
