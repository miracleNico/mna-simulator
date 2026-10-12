"""Restricted evaluator for SPICE ``.param`` values and ``{...}`` expressions.

A tokenizer and a recursive-descent parser build a small tuple tree that is evaluated
with ``math`` functions; nothing is passed to ``eval``. Supported:

* numbers with SPICE scale factors and unit words (``2.2k``, ``1meg``, ``10uF``);
* ``+ - * /``, ``**`` and ``^`` (power, right-associative), unary ``+ - !``, parentheses;
* comparisons ``< <= > >= == !=`` (1.0 or 0.0), ``&&``, ``||`` and ``c ? a : b``;
* functions ``sqrt exp ln log log10 abs min max pow int floor ceil if(c, a, b)`` and
  ``sin cos tan asin acos atan sinh cosh tanh sgn``; ``log`` is the natural logarithm as
  in SPICE;
* references to parameters (case-insensitive) and the constants ``pi`` and ``e``
  (a parameter of the same name takes precedence).

Errors raise :class:`ExpressionError` with a message naming the problem (an unresolved
reference names the parameter).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import lru_cache

from .values import normalize_value


class ExpressionError(ValueError):
    """An expression could not be parsed or evaluated."""


class UnknownParameter(ExpressionError):
    """A referenced parameter is not defined in scope."""

    def __init__(self, name: str) -> None:
        super().__init__(f"parameter '{name}' is not defined")
        self.name = name


CONSTANTS = {"pi": math.pi, "e": math.e}

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*)
  | (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<op>\*\*|<=|>=|==|!=|&&|\|\||[-+*/^()<>!?:,])
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, object]]:
    tokens: list[tuple[str, object]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise ExpressionError(f"unexpected character '{text[position]}'")
        position = match.end()
        kind = match.lastgroup
        value = match.group(0)
        if kind == "space":
            continue
        if kind == "number":
            literal = normalize_value(value)
            try:
                number = float(literal)
            except ValueError:
                number = _scaled(literal, value)
            tokens.append(("num", number))
        elif kind == "name":
            tokens.append(("name", value.lower()))
        else:
            tokens.append(("op", value))
    tokens.append(("end", None))
    return tokens


_SUFFIX_FACTORS = {"g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


def _scaled(literal: str, original: str) -> float:
    """Value of a normalized literal with a scale suffix (``2.2k``, ``1Meg``)."""

    match = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(meg|[gkmunpf]|%)?", literal, re.IGNORECASE)
    if match is None:
        raise ExpressionError(f"invalid number '{original}'")
    mantissa, suffix = match.groups()
    if suffix is None:
        return float(mantissa)
    if suffix == "%":
        return float(mantissa) * 1e-2
    return float(mantissa) * _SUFFIX_FACTORS[suffix.lower()]


class _Parser:
    def __init__(self, tokens: list[tuple[str, object]]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> tuple[str, object]:
        return self.tokens[self.index]

    def take(self) -> tuple[str, object]:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def accept(self, op: str) -> bool:
        kind, value = self.peek()
        if kind == "op" and value == op:
            self.index += 1
            return True
        return False

    def expect(self, op: str) -> None:
        if not self.accept(op):
            kind, value = self.peek()
            found = "end of expression" if kind == "end" else f"'{_token_text(kind, value)}'"
            raise ExpressionError(f"expected '{op}' but found {found}")

    def parse(self):  # noqa: ANN201
        node = self.ternary()
        kind, value = self.peek()
        if kind != "end":
            raise ExpressionError(f"unexpected '{_token_text(kind, value)}'")
        return node

    def ternary(self):  # noqa: ANN201
        condition = self.logical_or()
        if self.accept("?"):
            when_true = self.ternary()
            self.expect(":")
            when_false = self.ternary()
            return ("if", condition, when_true, when_false)
        return condition

    def logical_or(self):  # noqa: ANN201
        node = self.logical_and()
        while self.accept("||"):
            node = ("or", node, self.logical_and())
        return node

    def logical_and(self):  # noqa: ANN201
        node = self.comparison()
        while self.accept("&&"):
            node = ("and", node, self.comparison())
        return node

    def comparison(self):  # noqa: ANN201
        node = self.additive()
        while True:
            kind, value = self.peek()
            if kind == "op" and value in {"<", "<=", ">", ">=", "==", "!="}:
                self.take()
                node = ("cmp", value, node, self.additive())
            else:
                return node

    def additive(self):  # noqa: ANN201
        node = self.multiplicative()
        while True:
            kind, value = self.peek()
            if kind == "op" and value in {"+", "-"}:
                self.take()
                node = ("bin", value, node, self.multiplicative())
            else:
                return node

    def multiplicative(self):  # noqa: ANN201
        node = self.unary()
        while True:
            kind, value = self.peek()
            if kind == "op" and value in {"*", "/"}:
                self.take()
                node = ("bin", value, node, self.unary())
            else:
                return node

    def unary(self):  # noqa: ANN201
        if self.accept("-"):
            return ("neg", self.unary())
        if self.accept("+"):
            return self.unary()
        if self.accept("!"):
            return ("not", self.unary())
        return self.power()

    def power(self):  # noqa: ANN201
        base = self.atom()
        kind, value = self.peek()
        if kind == "op" and value in {"**", "^"}:
            self.take()
            return ("bin", "**", base, self.unary())
        return base

    def atom(self):  # noqa: ANN201
        kind, value = self.take()
        if kind == "num":
            return ("num", value)
        if kind == "name":
            if self.accept("("):
                args = []
                if not self.accept(")"):
                    while True:
                        args.append(self.ternary())
                        if self.accept(")"):
                            break
                        self.expect(",")
                return ("call", value, tuple(args))
            return ("var", value)
        if kind == "op" and value == "(":
            node = self.ternary()
            self.expect(")")
            return node
        if kind == "end":
            raise ExpressionError("unexpected end of expression")
        raise ExpressionError(f"unexpected '{_token_text(kind, value)}'")


def _token_text(kind: str, value: object) -> str:
    return repr(value) if kind == "num" else str(value)


@lru_cache(maxsize=2048)
def compile_expression(text: str):  # noqa: ANN201
    """Parse ``text`` into the evaluator's tuple tree (memoized)."""

    if not text.strip():
        raise ExpressionError("empty expression")
    return _Parser(_tokenize(text)).parse()


def _sgn(x: float) -> float:
    return float((x > 0) - (x < 0))


_FUNCTIONS: dict[str, tuple[Callable[..., float], int, int]] = {
    # name: (function, minimum arguments, maximum arguments or -1)
    "sqrt": (math.sqrt, 1, 1),
    "exp": (math.exp, 1, 1),
    "ln": (math.log, 1, 1),
    "log": (math.log, 1, 1),
    "log10": (math.log10, 1, 1),
    "abs": (abs, 1, 1),
    "min": (min, 1, -1),
    "max": (max, 1, -1),
    "pow": (math.pow, 2, 2),
    "int": (lambda x: float(math.trunc(x)), 1, 1),
    "floor": (lambda x: float(math.floor(x)), 1, 1),
    "ceil": (lambda x: float(math.ceil(x)), 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "asin": (math.asin, 1, 1),
    "acos": (math.acos, 1, 1),
    "atan": (math.atan, 1, 1),
    "sinh": (math.sinh, 1, 1),
    "cosh": (math.cosh, 1, 1),
    "tanh": (math.tanh, 1, 1),
    "sgn": (_sgn, 1, 1),
}


def evaluate(node, lookup: Callable[[str], float]) -> float:  # noqa: ANN001
    """Value of a compiled tree; ``lookup(name)`` resolves parameters (raise to reject)."""

    kind = node[0]
    if kind == "num":
        return float(node[1])
    if kind == "var":
        try:
            return float(lookup(node[1]))
        except UnknownParameter:
            # SPICE constants, unless a parameter of the same name is in scope.
            if node[1] in CONSTANTS:
                return CONSTANTS[node[1]]
            raise
    if kind == "neg":
        return -evaluate(node[1], lookup)
    if kind == "not":
        return 0.0 if evaluate(node[1], lookup) != 0.0 else 1.0
    if kind == "bin":
        op = node[1]
        left = evaluate(node[2], lookup)
        right = evaluate(node[3], lookup)
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            result = left**right
        except ZeroDivisionError:
            raise ExpressionError("division by zero") from None
        except OverflowError:
            raise ExpressionError("numeric overflow") from None
        if isinstance(result, complex):
            raise ExpressionError("negative number raised to a fractional power")
        return float(result)
    if kind == "cmp":
        op = node[1]
        left = evaluate(node[2], lookup)
        right = evaluate(node[3], lookup)
        outcome = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
            "==": left == right,
            "!=": left != right,
        }[op]
        return 1.0 if outcome else 0.0
    if kind == "and":
        return 1.0 if evaluate(node[1], lookup) != 0.0 and evaluate(node[2], lookup) != 0.0 else 0.0
    if kind == "or":
        return 1.0 if evaluate(node[1], lookup) != 0.0 or evaluate(node[2], lookup) != 0.0 else 0.0
    if kind == "if":
        return evaluate(node[2], lookup) if evaluate(node[1], lookup) != 0.0 else evaluate(node[3], lookup)
    if kind == "call":
        name, args = node[1], node[2]
        if name == "if":
            if len(args) != 3:
                raise ExpressionError("if() takes 3 arguments (condition, value if true, value if false)")
            return evaluate(args[1], lookup) if evaluate(args[0], lookup) != 0.0 else evaluate(args[2], lookup)
        entry = _FUNCTIONS.get(name)
        if entry is None:
            raise ExpressionError(f"unknown function '{name}'")
        function, minimum, maximum = entry
        if len(args) < minimum or (maximum >= 0 and len(args) > maximum):
            expected = str(minimum) if minimum == maximum else f"at least {minimum}"
            raise ExpressionError(f"{name}() takes {expected} argument(s), got {len(args)}")
        values = [evaluate(arg, lookup) for arg in args]
        try:
            result = function(*values)
        except (ValueError, ZeroDivisionError):
            raise ExpressionError(f"math domain error in {name}()") from None
        except OverflowError:
            raise ExpressionError(f"numeric overflow in {name}()") from None
        return float(result)
    raise ExpressionError(f"cannot evaluate node {kind!r}")  # pragma: no cover - parser output only


def evaluate_text(text: str, lookup: Callable[[str], float]) -> float:
    """Parse and evaluate ``text``; the result must be finite."""

    value = evaluate(compile_expression(text), lookup)
    if not math.isfinite(value):
        raise ExpressionError(f"the value is not finite ({value})")
    return value
