"""SPICE-dialect front end of ``netlist.parse_netlist_text``.

Reads the structural SPICE constructs and flattens them into the same
``ComponentRecord``s the netlist dialect produces (``HIERARCHY_AND_MODELS_PLAN.md``
section 3.2):

1. **Lexing** (:func:`logical_lines`): ``*``/``#`` comment lines, ``;`` and ``$``
   (after whitespace) inline comments, ``+`` continuation lines joined to the line
   they continue, ``.end`` stops. There is no title line: the first line is parsed like
   every other line.
2. **Libraries**: ``.include "file"`` and ``.lib "file" section`` (``.lib section`` ...
   ``.endl`` blocks) through an include resolver; recursive includes are errors; netlist
   text without a resolver asks for the library to be pasted in.
3. **Structure**: ``.subckt NAME ports [PARAMS:] k=v ... / .ends [NAME]`` (nested
   definitions are local to their parent), ``.model`` cards and ``.param`` lines (global
   or per subcircuit), ``.global`` nets.
4. **Elaboration**: ``.param`` values and ``{...}`` expressions are evaluated
   (``spice.expressions``), ``X`` instances are flattened recursively (``X1.R1``,
   internal nets ``X1.n``, ports bound to the parent's nets, ``0`` and ``.global`` nets
   kept), every other line is parsed by ``Component.from_parts`` with the cards in scope.
   Independent ``I`` sources inside a ``.subckt`` body follow SPICE (current from the
   first node through the source to the second): their nodes are swapped when flattened,
   while top-level ``I`` lines keep the simulator's convention (current into the first node).

Errors are ``NETLIST_FATAL`` (or the dialect's own codes) naming the line, the file of
an included line, and the subcircuit instance.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..api.contracts import CircuitIR, ComponentRecord, DirectiveRecord
from ..errors import NetlistError
from ..netlist import (
    Component,
    Netlist,
    _node_name_ok,
    _parse_control_line,
    check_behavioural_element,
    check_unsupported_element,
    finish_netlist,
)
from .context import ParseContext, located_error
from .expressions import ExpressionError, UnknownParameter, evaluate_text
from .model_cards import CARD_TYPES, CardError, ModelCard, parse_card
from .values import AssignmentError, fine_tokens, format_number, parse_assignments

#: ``resolver(path as written, resolved name of the including file or None)`` ->
#: ``(resolved name, text)``; raises :class:`IncludeNotFound` when there is no such file.
IncludeResolver = Callable[[str, "str | None"], "tuple[str, str]"]

_INSTANCE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SPACED_EQUALS = re.compile(r"\s*=\s*")
_INLINE_DOLLAR = re.compile(r"(?:^|\s)\$")
_ANALYSIS_DIRECTIVES = frozenset({".op", ".dc", ".tran", ".ac", ".hb"})
_CONDITIONAL_DIRECTIVES = {
    ".func": "user-defined functions are not supported (use .param or write the expression inline)",
    ".if": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".elseif": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".else": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".endif": "conditional netlists (.if/.elseif/.else/.endif) are not supported",
    ".alter": "alternative runs (.alter) are not supported",
}


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Line:
    """One logical line (continuations joined, comments removed)."""

    text: str
    number: int
    source: str | None = None  # file of an included line, as written in the .include

    def where(self) -> str:
        return f"line {self.number}" if self.source is None else f"line {self.number} of {self.source}"


def logical_lines(text: str, source: str | None = None) -> list[Line]:
    """Comment-free logical lines of ``text`` up to ``.end``, ``+`` continuations joined."""

    lines: list[Line] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "*#":
            continue
        cut = line.find(";")
        if cut >= 0:
            line = line[:cut]
        if "$" in line:
            match = _INLINE_DOLLAR.search(line)
            if match is not None:
                line = line[: match.start()]
        line = line.strip()
        if not line:
            continue
        if line[0] == "+":
            if not lines:
                raise located_error(f"{Line('', number, source).where()}: continuation line '+' has no line to continue")
            previous = lines[-1]
            previous.text = f"{previous.text} {line[1:].strip()}".rstrip()
            continue
        if line.lower() == ".end":
            break
        lines.append(Line(line, number, source))
    return lines


def _command(text: str) -> tuple[str, str]:
    """``(".command" in lower case, rest of the line)``."""

    head, _, rest = text.partition(" ")
    if "\t" in head:
        head, _, more = head.partition("\t")
        rest = f"{more} {rest}"
    return head.lower(), rest.strip()


# ---------------------------------------------------------------------------
# .include / .lib
# ---------------------------------------------------------------------------


class IncludeNotFound(LookupError):
    """The include resolver found no file for a ``.include``/``.lib`` path."""

    def __init__(self, requested: str, searched: Sequence[str]) -> None:
        super().__init__(requested)
        self.requested = requested
        self.searched = list(searched)


class FileIncludeResolver:
    """Resolve ``.include``/``.lib`` paths on disk.

    A relative path is looked up next to the including file (the main netlist file for
    its own lines, ``base_dir`` or the current directory for netlist text), then in each
    of ``search_dirs`` in order (the CLI's ``--lib`` directories).
    """

    def __init__(
        self,
        search_dirs: Sequence[str | Path] = (),
        root: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.search_dirs = [Path(d) for d in search_dirs]
        self.root_key = str(Path(root).resolve()) if root is not None else None
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif root is not None:
            self.base_dir = Path(root).resolve().parent
        else:
            self.base_dir = Path.cwd()

    def __call__(self, requested: str, including: str | None) -> tuple[str, str]:
        path = Path(requested).expanduser()
        if path.is_absolute():
            candidates = [path]
        else:
            first = Path(including).parent if including else self.base_dir
            candidates = [first / path] + [directory / path for directory in self.search_dirs]
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate.resolve()
                return str(resolved), resolved.read_text(encoding="utf-8", errors="replace")
        raise IncludeNotFound(requested, [str(candidate.parent) for candidate in candidates])


_LIB_ARGUMENT = re.compile(r'"([^"]*)"|\'([^\']*)\'|(\S+)')


def _arguments(rest: str) -> list[tuple[str, bool]]:
    """``(text, quoted)`` of each argument of a ``.include``/``.lib`` line."""

    arguments = []
    for match in _LIB_ARGUMENT.finditer(rest):
        if match.group(3) is not None:
            arguments.append((match.group(3), False))
        else:
            arguments.append((match.group(1) if match.group(1) is not None else match.group(2), True))
    return arguments


def _is_path(argument: tuple[str, bool]) -> bool:
    text, quoted = argument
    return quoted or any(char in text for char in "./\\")


def _section_name(line: Line) -> str | None:
    """Name of the library section a ``.lib <name>`` line opens, else ``None``."""

    command, rest = _command(line.text)
    if command != ".lib":
        return None
    arguments = _arguments(rest)
    if len(arguments) == 1 and not _is_path(arguments[0]):
        return arguments[0][0]
    return None


class _Includes:
    """Expands ``.include`` and ``.lib`` lines into the lines they name."""

    def __init__(self, resolver: IncludeResolver | None, context: ParseContext) -> None:
        self.resolver = resolver
        self.context = context

    def expand(self, lines: list[Line], stack: list[tuple[str | None, str | None]]) -> list[Line]:
        out: list[Line] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.text[0] != ".":
                out.append(line)
                continue
            command, rest = _command(line.text)
            if command in {".include", ".inc"}:
                arguments = _arguments(rest)
                if len(arguments) != 1:
                    raise located_error(f"{line.where()}: use {command} \"<file>\"")
                out.extend(self.include(arguments[0][0], None, line, stack))
            elif command == ".lib":
                arguments = _arguments(rest)
                if len(arguments) >= 2:
                    out.extend(self.include(arguments[0][0], arguments[1][0], line, stack))
                elif len(arguments) == 1 and _is_path(arguments[0]):
                    out.extend(self.include(arguments[0][0], None, line, stack))
                elif len(arguments) == 1:
                    end = _section_end(lines, index - 1)
                    name = arguments[0][0]
                    self.context.diagnose(
                        f"{line.where()}: library section '{name}' skipped (a section is read with .lib \"<file>\" {name})"
                    )
                    index = end + 1
                else:
                    raise located_error(f"{line.where()}: use .lib \"<file>\" <section>")
            elif command == ".endl":
                raise located_error(f"{line.where()}: .endl without a .lib section")
            else:
                out.append(line)
        return out

    def include(self, target: str, section: str | None, line: Line, stack: list[tuple[str | None, str | None]]) -> list[Line]:
        what = f'.lib "{target}" {section}' if section is not None else f'.include "{target}"'
        if self.resolver is None:
            raise located_error(
                f"{line.where()}: {what} cannot be read from netlist text: paste the contents of {target} into the "
                f"netlist (a netlist file run from the CLI reads it; --lib adds search directories)"
            )
        including = stack[-1][0] if stack else None
        try:
            key, text = self.resolver(target, including)
        except IncludeNotFound as exc:
            raise located_error(f"{line.where()}: file '{target}' not found (searched {', '.join(exc.searched)})") from None
        except OSError as exc:
            raise located_error(f"{line.where()}: cannot read '{target}': {exc}") from None
        frame = (key, section.lower() if section is not None else None)
        if frame in stack:
            raise located_error(f"{line.where()}: recursive {what} (the file is already being read)")
        lines = logical_lines(text, target)
        if section is not None:
            lines = self.section(lines, section, target, line)
        return self.expand(lines, [*stack, frame])

    @staticmethod
    def section(lines: list[Line], name: str, target: str, line: Line) -> list[Line]:
        wanted = name.lower()
        for index, candidate in enumerate(lines):
            opened = _section_name(candidate)
            if opened is not None and opened.lower() == wanted:
                end = _section_end(lines, index)
                return lines[index + 1 : end]
        raise located_error(f"{line.where()}: library section '{name}' not found in '{target}'")


def _section_end(lines: list[Line], start: int) -> int:
    """Index of the ``.endl`` closing the section opened at ``lines[start]``."""

    for index in range(start + 1, len(lines)):
        command, _ = _command(lines[index].text)
        if command == ".endl":
            return index
        if _section_name(lines[index]) is not None:
            raise located_error(f"{lines[index].where()}: library section inside library section '{lines[start].text}'")
    raise located_error(f"{lines[start].where()}: library section has no .endl")


# ---------------------------------------------------------------------------
# Structure: .subckt definitions, .model cards, .param lines
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Definition:
    """The top level (``name`` None) or one ``.subckt`` definition."""

    name: str | None
    ports: list[str] = field(default_factory=list)
    params: list[tuple[str, str, Line]] = field(default_factory=list)  # PARAMS: defaults
    line: Line | None = None
    parent: _Definition | None = None
    elements: list[Line] = field(default_factory=list)
    param_lines: list[tuple[str, str, Line]] = field(default_factory=list)  # .param inside
    models: dict[str, tuple[Line, str]] = field(default_factory=dict)  # lower name -> (line, card text)
    subckts: dict[str, _Definition] = field(default_factory=dict)
    controls: list[Line] = field(default_factory=list)  # directives (top level only)


def _subckt_header(rest: str, line: Line, parent: _Definition) -> _Definition:
    tokens = _SPACED_EQUALS.sub("=", rest).split()
    if not tokens:
        raise located_error(f"{line.where()}: use .subckt <name> <ports> [PARAMS: <name>=<value> ...]")
    index = len(tokens)
    for position, token in enumerate(tokens):
        if token.lower().startswith("params:") or "=" in token:
            index = position
            break
    head, tail = tokens[:index], " ".join(tokens[index:])
    if tail.lower().startswith("params:"):
        tail = tail[len("params:") :]
    name, ports = head[0], head[1:]
    if not _INSTANCE_NAME.match(name):
        raise located_error(f"{line.where()}: invalid subcircuit name '{name}'")
    seen: set[str] = set()
    for port in ports:
        if not _node_name_ok(port):
            raise located_error(f"{line.where()}: subcircuit {name} has an invalid port name '{port}'")
        if port == "0":
            raise located_error(f"{line.where()}: subcircuit {name}: port '0' (ground is global and needs no port)")
        if port.lower() in seen:
            raise located_error(f"{line.where()}: subcircuit {name} lists port '{port}' twice")
        seen.add(port.lower())
    try:
        params = parse_assignments(tail)
    except AssignmentError as exc:
        raise located_error(f"{line.where()}: subcircuit {name}: {exc}") from None
    names: set[str] = set()
    for key, _ in params:
        if key.lower() in names:
            raise located_error(f"{line.where()}: subcircuit {name} defines parameter '{key}' twice")
        names.add(key.lower())
    return _Definition(name=name, ports=ports, params=[(k, v, line) for k, v in params], line=line, parent=parent)


def _structure(lines: list[Line], context: ParseContext) -> tuple[_Definition, set[str]]:
    """The definition tree of the expanded lines and the ``.global`` net names (lower case)."""

    root = _Definition(name=None)
    stack = [root]
    global_nets: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        current = stack[-1]
        if line.text[0] != ".":
            current.elements.append(line)
            continue
        command, rest = _command(line.text)
        if command == ".subckt":
            definition = _subckt_header(rest, line, current)
            key = definition.name.lower()  # type: ignore[union-attr]
            if key in current.subckts:
                first = current.subckts[key].line
                raise located_error(
                    f"{line.where()}: subcircuit {definition.name} is defined twice (first at {first.where()})"  # type: ignore[union-attr]
                )
            current.subckts[key] = definition
            stack.append(definition)
        elif command == ".ends":
            if len(stack) == 1:
                raise located_error(f"{line.where()}: .ends without .subckt")
            name = rest.split()[0] if rest.split() else None
            if name is not None and name.lower() != current.name.lower():  # type: ignore[union-attr]
                raise located_error(f"{line.where()}: '.ends {name}' closes .subckt {current.name}")
            stack.pop()
        elif command == ".model":
            tokens = fine_tokens(rest)
            if len(tokens) < 2:
                raise located_error(f"{line.where()}: use .model <name> <type>(<parameter>=<value> ...)")
            key = tokens[0].lower()
            if key in current.models:
                first = current.models[key][0]
                raise located_error(f"{line.where()}: model {tokens[0]} is defined twice (first at {first.where()})")
            current.models[key] = (line, rest)
        elif command == ".param":
            try:
                pairs = parse_assignments(rest)
            except AssignmentError as exc:
                raise located_error(f"{line.where()}: .param: {exc}") from None
            taken = {k.lower() for k, _, _ in current.param_lines} | {k.lower() for k, _, _ in current.params}
            for key, value in pairs:
                if key.lower() in taken:
                    raise located_error(f"{line.where()}: parameter '{key}' is defined twice")
                taken.add(key.lower())
                current.param_lines.append((key, value, line))
        elif command == ".global":
            for net in rest.split():
                if not _node_name_ok(net):
                    raise located_error(f"{line.where()}: invalid .global net name '{net}'")
                global_nets.add(net.lower())
        elif command in _CONDITIONAL_DIRECTIVES:
            raise located_error(f"{line.where()}: {command}: {_CONDITIONAL_DIRECTIVES[command]}")
        elif command == ".control":
            start = line
            while index < len(lines) and _command(lines[index].text)[0] != ".endc":
                index += 1
            if index >= len(lines):
                raise located_error(f"{start.where()}: .control block has no .endc")
            index += 1
            context.diagnose(f"{start.where()}: '.control' block ignored (simulator commands are not supported)")
        elif command == ".endc":
            raise located_error(f"{line.where()}: .endc without .control")
        elif command in _ANALYSIS_DIRECTIVES and len(stack) > 1:
            raise located_error(f"{line.where()}: '{command}' is not allowed inside .subckt {current.name}")
        else:
            root.controls.append(line)
    if len(stack) > 1:
        open_definition = stack[-1]
        raise located_error(f"{open_definition.line.where()}: .subckt {open_definition.name} has no .ends")  # type: ignore[union-attr]
    return root, global_nets


# ---------------------------------------------------------------------------
# Parameter scopes
# ---------------------------------------------------------------------------


class _Scope:
    """Parameters of the top level or of one subcircuit instance."""

    __slots__ = ("definition", "parent", "instance", "values", "pending", "_active")

    def __init__(self, definition: _Definition, parent: _Scope | None, instance: str | None) -> None:
        self.definition = definition
        self.parent = parent
        self.instance = instance  # hierarchical instance name (X1.X2), None at the top level
        self.values: dict[str, float] = {}
        self.pending: dict[str, tuple[str, Line]] = {}
        self._active: set[str] = set()

    def lookup(self, name: str) -> float:
        scope: _Scope | None = self
        while scope is not None:
            value = scope.values.get(name)
            if value is not None:
                return value
            if name in scope.pending:
                return scope._evaluate(name)
            scope = scope.parent
        raise UnknownParameter(name)  # the evaluator then tries the constants pi and e

    def _evaluate(self, name: str) -> float:
        text, line = self.pending[name]
        where = line.where() if self.instance is None else f"{line.where()} (instance {self.instance})"
        if name in self._active:
            raise located_error(f"{where}: parameter '{name}' depends on itself")
        self._active.add(name)
        try:
            value = evaluate_value(text, self)
        except ExpressionError as exc:
            raise located_error(f"{where}: parameter '{name}' = {text}: {exc}") from None
        finally:
            self._active.discard(name)
        self.values[name] = value
        del self.pending[name]
        return value

    def evaluate_all(self) -> None:
        for name in list(self.pending):
            if name in self.pending:
                self._evaluate(name)


def evaluate_value(text: str, scope: _Scope) -> float:
    """A ``.param``/instance value: ``{expr}``, ``'expr'``, a number or a bare expression."""

    text = text.strip()
    if len(text) >= 2 and ((text[0] == "{" and text[-1] == "}") or (text[0] == text[-1] and text[0] in "'\"")):
        text = text[1:-1]
    return evaluate_text(text, scope.lookup)


# ---------------------------------------------------------------------------
# Elaboration
# ---------------------------------------------------------------------------


class _Context(ParseContext):
    """``ParseContext`` with ``.model`` lookup through subcircuit scopes and full locations."""

    __slots__ = ("frontend", "line", "definition", "scope")
    has_models = True

    def __init__(self, frontend: _Frontend) -> None:
        super().__init__()
        self.frontend = frontend
        self.line: Line | None = None
        self.definition: _Definition | None = None
        self.scope: _Scope | None = None

    def where(self) -> str:
        if self.line is None:
            return "netlist"
        text = self.line.where()
        if self.scope is not None and self.scope.instance is not None:
            text += f" (subcircuit {self.definition.name}, instance {self.scope.instance})"  # type: ignore[union-attr]
        return text

    def lookup_model(self, token: str) -> ModelCard | None:
        return self.frontend.lookup_model(token, self.definition, self.scope)  # type: ignore[arg-type]


class _Frontend:
    def __init__(self, text: str, resolver: IncludeResolver | None) -> None:
        self.text = text
        self.resolver = resolver
        self.context = _Context(self)
        self.netlist = Netlist()
        self.cards: dict[tuple[int, int], ModelCard] = {}
        self.referenced_cards: set[int] = set()
        self.instance_names: dict[str, set[str]] = {}
        self.global_nets: set[str] = set()
        self.global_scope: _Scope | None = None

    # -- driver ------------------------------------------------------------

    def run(self) -> CircuitIR:
        context = self.context
        root_key = getattr(self.resolver, "root_key", None)
        lines = logical_lines(self.text)
        lines = _Includes(self.resolver, context).expand(lines, [(root_key, None)] if root_key else [])
        root, self.global_nets = _structure(lines, context)
        scope = _Scope(root, None, None)
        self.global_scope = scope
        for key, value, line in root.param_lines:
            scope.pending[key.lower()] = (value, line)
        scope.evaluate_all()

        directives: list[DirectiveRecord] = []
        for line in root.controls:
            self._enter(line, root, scope)
            try:
                directive = _parse_control_line(self.substitute(line.text, scope), context)
            except NetlistError as exc:
                raise self.locate(exc) from None
            if directive is not None:
                directives.append(directive)

        for line in root.elements:
            self.element(line, root, scope, "", None, ())

        self._report_unused_cards(root)
        context.line = None
        context.scope = None
        return finish_netlist(self.netlist, directives, self.text, context)

    def _enter(self, line: Line, definition: _Definition, scope: _Scope) -> None:
        context = self.context
        context.line = line
        context.definition = definition
        context.scope = scope

    def locate(self, exc: NetlistError) -> NetlistError:
        if getattr(exc, "located", False):
            return exc
        return located_error(f"{self.context.where()}: {exc.message}", code=exc.code)

    # -- values ------------------------------------------------------------

    def substitute(self, text: str, scope: _Scope) -> str:
        """``text`` with every ``{expression}`` replaced by its value."""

        if "{" not in text:
            return text
        pieces: list[str] = []
        position = 0
        while True:
            start = text.find("{", position)
            if start < 0:
                pieces.append(text[position:])
                return "".join(pieces)
            depth = 0
            end = -1
            for index in range(start, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end < 0:
                self.context.error(f"unbalanced '{{' in '{text}'")
            expression = text[start + 1 : end]
            try:
                value = evaluate_text(expression, scope.lookup)
            except ExpressionError as exc:
                self.context.error(f"{{{expression}}}: {exc}")
            pieces.append(text[position:start])
            pieces.append(format_number(value))
            position = end + 1

    def evaluate(self, text: str, scope: _Scope) -> float:
        try:
            return evaluate_value(text, scope)
        except ExpressionError as exc:
            self.context.error(f"{text}: {exc}")

    # -- models and subcircuits ---------------------------------------------

    def lookup_model(self, token: str, definition: _Definition, scope: _Scope) -> ModelCard | None:
        key = token.lower()
        owner: _Definition | None = definition
        entry = None
        while owner is not None:
            entry = owner.models.get(key)
            if entry is not None:
                break
            owner = owner.parent
        if entry is None:
            return None
        card_scope: _Scope | None = scope
        while card_scope is not None and card_scope.definition is not owner:
            card_scope = card_scope.parent
        if card_scope is None:
            card_scope = self.global_scope
        cache_key = (id(entry), id(card_scope))
        card = self.cards.get(cache_key)
        if card is None:
            line, rest = entry
            context = self.context
            saved = (context.line, context.definition, context.scope)
            self._enter(line, owner, card_scope)  # type: ignore[arg-type]
            try:
                text = self.substitute(rest, card_scope)  # type: ignore[arg-type]
                card = parse_card(fine_tokens(text), line.where())
            except CardError as exc:
                raise located_error(f"{context.where()}: {exc}") from None
            finally:
                context.line, context.definition, context.scope = saved
            card.key = id(entry)
            self.cards[cache_key] = card
        self.referenced_cards.add(id(entry))
        return card

    @staticmethod
    def find_subckt(name: str, definition: _Definition) -> _Definition | None:
        key = name.lower()
        owner: _Definition | None = definition
        while owner is not None:
            found = owner.subckts.get(key)
            if found is not None:
                return found
            owner = owner.parent
        return None

    def lexical_scope(self, target: _Definition, caller: _Scope) -> _Scope:
        """Scope of the definition that contains ``target`` (the top level for a global definition)."""

        scope: _Scope | None = caller
        while scope is not None and scope.definition is not target.parent:
            scope = scope.parent
        return scope if scope is not None else self.global_scope  # type: ignore[return-value]

    # -- elements ------------------------------------------------------------

    def map_node(self, node: str | None, prefix: str, node_map: dict[str, str] | None) -> str | None:
        if node is None or node_map is None:
            return node
        if node == "0":
            return node
        lower = node.lower()
        mapped = node_map.get(lower)
        if mapped is not None:
            return mapped
        if lower in self.global_nets:
            return node
        return prefix + node

    def element(
        self,
        line: Line,
        definition: _Definition,
        scope: _Scope,
        prefix: str,
        node_map: dict[str, str] | None,
        chain: tuple[_Definition, ...],
    ) -> None:
        self._enter(line, definition, scope)
        context = self.context
        text = line.text
        name = text.split(None, 1)[0]
        if name[0] in "xX":
            self.instantiate(line, definition, scope, prefix, node_map, chain)
            return
        try:
            raw_parts = text.split()
            if ":" not in name and Component.get_type(name) is None:
                check_unsupported_element(name, raw_parts, context)
            check_behavioural_element(raw_parts, context)
            text = self.substitute(text, scope)
            parts = text.split()
            if len(parts) >= 4 and parts[3].upper() == "FUNC":
                parts = text.split(maxsplit=5)
            record = Component.from_parts(parts, context).to_record()
        except NetlistError as exc:
            raise self.locate(exc) from None
        if context.needs_defaults:
            context.needs_defaults = False
            context.deferred.append(record)
        if node_map is not None and not self._rename(record, prefix, node_map):
            return
        self.netlist.add_component(record)

    def _rename(self, record: ComponentRecord, prefix: str, node_map: dict[str, str]) -> bool:
        """Flattened name and nets of a subcircuit element; False when it is dropped.

        An independent current source inside a ``.subckt`` body follows SPICE: positive
        current flows from its first node through the source to its second node. The
        simulator drives an ``I`` record's current into its first node, so the flattened
        record gets the two nodes swapped (top-level ``I`` lines keep the simulator's
        convention).
        """

        record.name = prefix + record.name
        record.node1 = self.map_node(record.node1, prefix, node_map)
        record.node2 = self.map_node(record.node2, prefix, node_map)
        if record.type == "I":
            record.node1, record.node2 = record.node2, record.node1
        record.ctrl_node1 = self.map_node(record.ctrl_node1, prefix, node_map)
        record.ctrl_node2 = self.map_node(record.ctrl_node2, prefix, node_map)
        if (record.type == "CCCS" or record.type == "CCVS") and record.ctrl_source:
            record.ctrl_source = prefix + record.ctrl_source
        if record.type != "GND" and record.node1 is not None and record.node1.lower() == (record.node2 or "").lower():
            if record.type in {"R", "C"}:
                self.context.diagnose(
                    f"{self.context.where()}: '{record.name}' dropped: both terminals are on net '{record.node1}'"
                )
                return False
            self.context.error(f"'{record.name}' has both terminals on net '{record.node1}' (instance ports tied together)")
        return True

    def instantiate(
        self,
        line: Line,
        definition: _Definition,
        scope: _Scope,
        prefix: str,
        node_map: dict[str, str] | None,
        chain: tuple[_Definition, ...],
    ) -> None:
        context = self.context
        tokens = _SPACED_EQUALS.sub("=", line.text).split()
        name = tokens[0]
        index = len(tokens)
        for position, token in enumerate(tokens):
            if token.lower().startswith("params:") or "=" in token:
                index = position
                break
        head, tail = tokens[:index], " ".join(tokens[index:])
        if tail.lower().startswith("params:"):
            tail = tail[len("params:") :]
        try:
            assignments = parse_assignments(tail)
        except AssignmentError as exc:
            context.error(f"'{name}': {exc}")
        if not _INSTANCE_NAME.match(name):
            context.error(f"invalid subcircuit instance name '{name}'")
        if len(head) < 2:
            context.error(f"'{name}' names no subcircuit (use X<name> <nodes> <subcircuit> [<name>=<value> ...])")
        nodes, subcircuit = head[1:-1], head[-1]
        for node in nodes:
            if not _node_name_ok(node):
                context.error(f"'{name}' has invalid node name '{node}'")
        target = self.find_subckt(subcircuit, definition)
        if target is None:
            context.error(f"subcircuit '{subcircuit}' is not defined (instance '{name}')")
        if len(nodes) != len(target.ports):
            context.error(
                f"instance '{name}' connects {len(nodes)} node(s) but subcircuit {target.name} has "
                f"{len(target.ports)} port(s) ({' '.join(target.ports) or 'none'})"
            )
        if target in chain:
            cycle = [d.name for d in chain[chain.index(target) :]] + [target.name]
            context.error(f"subcircuit {target.name} instantiates itself ({' -> '.join(cycle)})")  # type: ignore[arg-type]
        seen = self.instance_names.setdefault(prefix, set())
        if name.lower() in seen:
            context.error(f"duplicate subcircuit instance name '{name}'")
        seen.add(name.lower())

        known = {key.lower() for key, _, _ in target.params}
        overrides: dict[str, float] = {}
        for key, value in assignments:
            if key.lower() not in known:
                context.error(f"subcircuit {target.name} has no parameter '{key}' (instance '{name}')")
            overrides[key.lower()] = self.evaluate(value, scope)
        path = prefix + name
        child = _Scope(target, self.lexical_scope(target, scope), path)
        child.values.update(overrides)
        for key, value, param_line in target.params:
            if key.lower() not in overrides:
                child.pending[key.lower()] = (value, param_line)
        for key, value, param_line in target.param_lines:
            child.pending[key.lower()] = (value, param_line)
        child.evaluate_all()
        nets = [self.map_node(node, prefix, node_map) for node in nodes]
        child_map = {port.lower(): net for port, net in zip(target.ports, nets)}  # type: ignore[misc]
        for element in target.elements:
            self.element(element, target, child, path + ".", child_map, (*chain, target))

    # -- end-of-parse diagnostics ------------------------------------------

    def _report_unused_cards(self, root: _Definition) -> None:
        unused: list[str] = []
        pending = [root]
        while pending:
            definition = pending.pop()
            pending.extend(definition.subckts.values())
            for entry in definition.models.values():
                if id(entry) in self.referenced_cards:
                    continue
                tokens = fine_tokens(entry[1])
                card_type = tokens[1].upper() if len(tokens) > 1 else "?"
                if CARD_TYPES.get(card_type) is None:
                    unused.append(f"{tokens[0]} ({card_type})")
        if unused:
            self.context.diagnose(f"models of unsupported types were not used: {', '.join(sorted(unused))}")


def parse_spice_netlist(text: str, include_resolver: IncludeResolver | None = None) -> CircuitIR:
    """Parse SPICE-dialect netlist text into a validated ``CircuitIR`` (see the module docstring)."""

    return _Frontend(text, include_resolver).run()
