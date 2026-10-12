"""Per-parse state shared by the netlist line parsers (diagnostics, model lookup, locations)."""

from __future__ import annotations

from operator import length_hint
from typing import NoReturn

from ..errors import NetlistError
from .model_cards import KIND_NAMES, MOS_LEVELS, ModelCard, card_diagnostics


def located_error(message: str, code: str = "NETLIST_FATAL") -> NetlistError:
    """A ``NetlistError`` whose message already names its line (not prefixed again)."""

    exc = NetlistError(message, code=code)
    exc.located = True  # type: ignore[attr-defined]
    return exc


class ParseContext:
    """State of one ``parse_netlist_text`` call.

    The plain netlist loop uses this class as is (no ``.model`` cards exist there); the
    SPICE front end subclasses it to look cards up in ``.subckt`` scopes and to name
    the file, line and subcircuit instance in messages.
    """

    __slots__ = ("diagnostics", "_messages", "_cards", "line_number", "lines", "needs_defaults", "deferred", "ctrl_refs")

    #: False when no ``.model`` card can exist (the plain netlist loop): device lines skip the lookups.
    has_models = False

    def __init__(self) -> None:
        self.diagnostics: list[str] = []
        self._messages: set[str] = set()
        self._cards: set[object] = set()
        self.line_number = 0
        # ``(line count, line iterator)`` of the plain loop: the current line number is
        # computed only when a message needs it (``count - length_hint(iterator)``).
        self.lines: tuple[int, object] | None = None
        # Set by the source parser when a SPICE waveform needs the .tran step/stop defaults.
        self.needs_defaults = False
        self.deferred: list[object] = []  # records waiting for those defaults
        self.ctrl_refs = False  # a CCCS/CCVS was read: controlling names are resolved at the end

    def where(self) -> str:
        """Location of the line being parsed, for messages (``line 12``)."""

        number = self.line_number
        if self.lines is not None:
            count, iterator = self.lines
            number = count - length_hint(iterator)
        return f"line {number}" if number else "netlist"

    def lookup_model(self, token: str) -> ModelCard | None:
        """The ``.model`` card named ``token`` in scope (none in the plain netlist dialect)."""

        return None

    def diagnose(self, message: str) -> None:
        """Add a diagnostic once (identical messages are reported once per parse)."""

        if message not in self._messages:
            self._messages.add(message)
            self.diagnostics.append(message)

    def error(self, message: str) -> NoReturn:
        """Raise ``NETLIST_FATAL`` naming the current line."""

        raise located_error(f"{self.where()}: {message}")

    def require_model(self, token: str, component: str, allowed: tuple[str, ...]) -> ModelCard:
        """The card ``token`` referenced by ``component``; errors for undefined/unsupported/mismatched cards."""

        card = self.lookup_model(token)
        if card is None:
            self.error(f"model '{token}' is not defined (referenced by '{component}')")
        canonical = card.canonical_type
        if canonical is None:
            self.error(f"model '{card.name}' has type {card.type}, which is not supported (referenced by '{component}')")
        if canonical not in allowed:
            kind = KIND_NAMES[allowed[0]]
            self.error(f"'{component}' needs a {kind} model, but model '{card.name}' has type {card.type}")
        if canonical in {"NMOS", "PMOS"}:
            level = card.mos_level()
            if level not in MOS_LEVELS:
                shown = card.params.get("level", "?")
                self.error(
                    f"model '{card.name}': MOS LEVEL={shown} is not supported (LEVEL 1, 2 and 3 cards are "
                    f"mapped to the Level-1 equations; BSIM and other levels are not available)"
                )
        if card.key is not None and card.key not in self._cards:
            self._cards.add(card.key)
            for message in card_diagnostics(card):
                self.diagnose(message)
        return card
