"""Shared exception types for the simulator."""

from __future__ import annotations


class CircuitSimulatorError(Exception):
    """Base exception for all simulator failures."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or "UNKNOWN"
        self.message = message


class NetlistError(CircuitSimulatorError):
    """Raised when the netlist is malformed or unsupported."""

    def __init__(self, message: str, code: str = "NETLIST_FATAL") -> None:
        super().__init__(message, code)


class SolverConvergenceError(CircuitSimulatorError):
    """Raised when a numerical solve fails to converge."""

    def __init__(self, message: str, code: str = "SOLVER_RETRY") -> None:
        super().__init__(message, code)


def error_handler(message: str) -> None:
    """Parse a coded message and raise the matching exception."""

    if ":" in message:
        code, msg = message.split(":", 1)
        code = code.strip()
        msg = msg.strip()
    else:
        code = "UNKNOWN"
        msg = message

    if code in {"NETLIST_FATAL", "PARSE_ERROR", "COMPONENT_ERROR", "MODEL_ERROR", "MOR_CONFIG"}:
        raise NetlistError(msg, code=code)

    if code in {
        "NR_CONVERGENCE",
        "JACOBIAN_SINGULAR",
        "SOLVER_RETRY",
        "LU_CONVERGENCE",
        "HB_ERROR",
        "AC_ERROR",
    }:
        raise SolverConvergenceError(msg, code=code)

    raise CircuitSimulatorError(msg, code=code)
