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

Errors: text SymPy cannot parse and functions or constructs the VM does not
support raise :class:`~mna_simulation.errors.NetlistError` (``NETLIST_FATAL``);
a free symbol other than ``x_i`` (``i < n_unknowns``) or ``t`` (time-dependent
programs only) raises ``TypeError``, the exception the reference core raised
for such expressions when it evaluated them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ..errors import error_handler
from . import require_core

__all__ = [
    "Program",
    "compile_expression",
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

    @cached_property
    def handle(self) -> Any:
        return require_core().compile_program(self.opcodes, self.args, self.consts, self.n_unknowns, self.time_dependent)

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

    def __init__(self, text: str, time_dependent: bool, n_unknowns: int) -> None:
        self.text = text
        self.time_dependent = time_dependent
        self.n_unknowns = n_unknowns

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


@lru_cache(maxsize=4096)
def _compile_cached(text: str, time_dependent: bool, n_unknowns: int) -> Program:
    code = lambdify_code(text, time_dependent=time_dependent, n_unknowns=n_unknowns)
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except SyntaxError as exc:
        _fail_parse(text, exc)
    instructions = _Lowering(text, time_dependent, n_unknowns).lower(tree)
    opcodes, args, consts = _assemble(instructions)
    unknowns = tuple(sorted({int(arg) for name, arg in instructions if name == "load_x"}))
    program = Program(
        text=text,
        code=code,
        opcodes=opcodes,
        args=args,
        consts=consts,
        n_unknowns=n_unknowns,
        time_dependent=time_dependent,
        unknowns=unknowns,
    )
    program.handle  # noqa: B018 - validate in the core now so errors surface at compile time
    return program


def compile_expression(text: str, *, time_dependent: bool, n_unknowns: int) -> Program:
    """Compile ``text`` to a native :class:`Program` (cached per ``(text, time_dependent, n_unknowns)``).

    ``x[i]`` and ``x_i`` both denote unknown ``i`` (``0 <= i < n_unknowns``);
    ``t`` is allowed only when ``time_dependent``.
    """

    if n_unknowns < 0:
        raise ValueError("n_unknowns must be non-negative")
    return _compile_cached(str(text), bool(time_dependent), int(n_unknowns))
