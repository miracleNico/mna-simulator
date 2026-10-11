"""Python backend adapter for solver execution."""

from __future__ import annotations

import math
import numbers
from pathlib import Path

import numpy as np

from ..api.contracts import AnalysisMode, AnalysisOptions, BackendCapabilities, CircuitIR, DirectiveRecord, SimulationResult
from ..core_advanced import build_and_run_advanced
from ..core_basic import build_and_run_basic
from ..errors import NetlistError
from ..library.registry import DeviceRegistry, create_default_registry
from ..netlist import parse_netlist_file, parse_netlist_text, select_analysis
from ..native.threads import coerce_num_threads
from ..utils import DEFAULT_GMIN_STEPS, parse_value


# Strict request-option coercion (CPP_MIGRATION_PLAN.md section 2.4.1). Every failure is a
# plain ValueError: the service tags it with a PEP 678 note and the HTTP layer answers 400.
_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})
_INIT_CONDITION_STRINGS = frozenset({"DEFAULT", "RANDOM"})


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _coerce_bool(value: object, name: str, default: bool) -> bool:
    """Return a boolean option; ``None`` means unset.

    Accepts bools, the integers 0/1, and ``true/false/yes/no/on/off/1/0``
    (case-insensitive, trimmed). ``bool("false")`` is no longer ``True``.
    """

    if value is None:
        return default
    if _is_bool(value):
        return bool(value)
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in _TRUE_STRINGS:
            return True
        if cleaned in _FALSE_STRINGS:
            return False
    raise ValueError(f"{name} must be a boolean")


def _coerce_int(value: object, name: str) -> int:
    """Return an integer option from an int or an integral float/string (``3.0``, ``"3"``)."""

    message = f"{name} must be an integer"
    if _is_bool(value):
        raise ValueError(message)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        try:
            return int(cleaned)
        except ValueError:
            pass
        try:
            number = float(cleaned)
        except ValueError:
            raise ValueError(message) from None
    else:
        raise ValueError(message)  # noqa: TRY004 - option errors stay ValueError (HTTP 400 contract)
    if math.isfinite(number) and number.is_integer():
        return int(number)
    raise ValueError(message)


def _coerce_optional_int(value: object, name: str, default: int | None) -> int | None:
    return default if value is None else _coerce_int(value, name)


def _coerce_float(value: object, name: str) -> float:
    """Return a float option from a number or a metric-suffixed string (``"1m"`` is 1e-3)."""

    message = f"{name} must be a number (metric suffixes allowed)"
    if _is_bool(value):
        raise ValueError(message)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return parse_value(value)
        except NetlistError:
            raise ValueError(message) from None
    raise ValueError(message)


def _coerce_optional_float(value: object, name: str, default: float | None) -> float | None:
    return default if value is None else _coerce_float(value, name)


def _coerce_init_condition(value: object) -> np.ndarray | str | None:
    """Return ``None``, ``"DEFAULT"``/``"RANDOM"`` (upper case), or a float array.

    Lists/tuples of numbers (e.g. a JSON array) become a float column array. A
    NumPy array is passed through unchanged. Length is validated by the solvers.
    """

    message = "init_condition must be 'DEFAULT', 'RANDOM', or a numeric array"
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in _INIT_CONDITION_STRINGS:
            return cleaned
        raise ValueError(message)
    if isinstance(value, (list, tuple)):
        try:
            entries = np.asarray(value, dtype=object)
        except ValueError:  # ragged nesting
            raise ValueError(message) from None
        if not all(isinstance(entry, numbers.Real) and not _is_bool(entry) for entry in entries.flat):
            raise ValueError(message)
        return entries.astype(float).reshape(-1, 1)
    raise ValueError(message)


def _coerce_krylov_rank(value: object) -> int | str:
    """Return ``auto`` or a positive integer Krylov rank request."""

    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    rank = _coerce_int(value, "krylov_rank")
    if rank < 1:
        raise ValueError("krylov_rank must be 'auto' or a positive integer.")
    return rank


def _coerce_krylov_method(value: object) -> str:
    """Return a normalized Krylov method selector."""

    if value is None:
        return "auto"
    cleaned = str(value).strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "arnoldi": "arnoldi_gmres",
        "gmres": "arnoldi_gmres",
        "arnoldi_gmres": "arnoldi_gmres",
        "general": "arnoldi_gmres",
        "cr": "conjugate_residual",
        "minres": "conjugate_residual",
        "conjugate_residual": "conjugate_residual",
        "symmetric": "conjugate_residual",
        "cg": "conjugate_gradient",
        "conjugate_gradient": "conjugate_gradient",
        "positive_definite": "conjugate_gradient",
        "spd": "conjugate_gradient",
    }
    if cleaned not in aliases:
        raise ValueError(f"Unknown Krylov method '{value}'.")
    return aliases[cleaned]


def _coerce_mor_method(value: object) -> str:
    """Return a normalized MOR method selector."""

    if value is None:
        return "auto"
    cleaned = str(value).strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "linear": "linear_krylov",
        "linear_krylov": "linear_krylov",
        "krylov": "linear_krylov",
        "rational_krylov": "linear_krylov",
        "prima": "linear_krylov",
        "tpwl": "tpwl",
        "pod": "tpwl",
        "nonlinear": "tpwl",
    }
    if cleaned not in aliases:
        raise ValueError(f"Unknown MOR method '{value}'.")
    return aliases[cleaned]


def _coerce_mor_order(value: object) -> int | str:
    """Return ``auto`` or a positive integer reduced order."""

    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() in ("auto", ""):
        return "auto"
    order = _coerce_int(value, "mor_order")
    if order < 1:
        raise ValueError("mor_order must be 'auto' or a positive integer.")
    return order


def _coerce_label_list(value: object) -> list[str]:
    """Normalize a UI/API label payload into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


class PythonBackend:
    """Default backend using the extracted Python numerical stack."""

    def __init__(self, registry: DeviceRegistry | None = None) -> None:
        self.registry = registry or create_default_registry(Path(__file__).resolve().parent.parent / "library" / "models")
        self.capabilities = BackendCapabilities(
            supports_behavioral_sources=True,
            supports_harmonic_balance=True,
            supports_sparse_linear_solver=True,
            supports_krylov_linear_solver=True,
            supports_model_order_reduction=True,
            supports_cpp_acceleration=False,
        )

    def parse_text(self, netlist_text: str) -> CircuitIR:
        return parse_netlist_text(netlist_text)

    def parse_file(self, path: str | Path) -> CircuitIR:
        return parse_netlist_file(path)

    def build_options(self, circuit: CircuitIR, overrides: dict | None = None, mode: AnalysisMode | None = None) -> AnalysisOptions:
        # ``.dyn`` is a UI-only mode that is executed as a ``.tran`` on the backend;
        # the realtime streaming pace is controlled by the ``/ws/dyn`` endpoint.
        effective_mode = mode
        if effective_mode == AnalysisMode.DYN:
            effective_mode = AnalysisMode.TRAN
        directive: DirectiveRecord = select_analysis(circuit, fallback=effective_mode or AnalysisMode.SHOW_MATRIX)
        option_values = dict(directive.params)
        if overrides:
            option_values.update(overrides)

        # Booleans: the first alias present wins (``use_krylov`` over ``krylov``), None is unset.
        krylov_key = "use_krylov" if "use_krylov" in option_values else "krylov"
        use_krylov = _coerce_bool(option_values.get(krylov_key), krylov_key, False)
        mor_key = "use_mor" if "use_mor" in option_values else "mor"
        use_mor = _coerce_bool(option_values.get(mor_key), mor_key, False)
        mor_validate = _coerce_bool(option_values.get("mor_validate"), "mor_validate", True)

        krylov_restart = _coerce_optional_int(option_values.get("krylov_restart"), "krylov_restart", 80)
        if "krylov_rank" in option_values:
            krylov_rank = _coerce_krylov_rank(option_values.get("krylov_rank"))
        elif "krylov_restart" in option_values:
            krylov_rank = _coerce_krylov_rank(option_values.get("krylov_restart"))
        else:
            krylov_rank = "auto"
        krylov_method = _coerce_krylov_method(option_values.get("krylov_method", option_values.get("krylov_algorithm", "auto")))
        mor_method = _coerce_mor_method(option_values.get("mor_method", "auto"))
        mor_order = _coerce_mor_order(option_values.get("mor_order", "auto"))
        num_threads = coerce_num_threads(option_values.get("num_threads"))

        # Time/frequency values accept metric suffixes like the netlist directives ("1m" is
        # 1e-3). A value of 0 is passed through: the solvers still treat it as unset (P3/P4).
        return AnalysisOptions(
            mode=effective_mode or directive.mode,
            gmin_steps=option_values.get("gmin_steps", DEFAULT_GMIN_STEPS),
            max_iter=_coerce_optional_int(option_values.get("max_iter"), "max_iter", 10000),
            v_tol=_coerce_optional_float(option_values.get("v_tol"), "v_tol", 1e-9),
            f_tol=_coerce_optional_float(option_values.get("f_tol"), "f_tol", 1e-9),
            tran_stop=_coerce_optional_float(option_values.get("t_stop"), "t_stop", None),
            tran_step=_coerce_optional_float(option_values.get("t_step"), "t_step", None),
            ac_start=_coerce_optional_float(option_values.get("f_start"), "f_start", None),
            ac_stop=_coerce_optional_float(option_values.get("f_stop"), "f_stop", None),
            ac_points=_coerce_optional_int(option_values.get("points"), "points", None),
            hb_harmonics=_coerce_optional_int(option_values.get("harmonics"), "harmonics", None),
            hb_time_window=_coerce_optional_float(option_values.get("time_window"), "time_window", None),
            init_condition=_coerce_init_condition(option_values.get("init_condition")),
            probe_nodes=_coerce_label_list(option_values.get("probe_nodes")),
            use_krylov=use_krylov,
            krylov_tol=_coerce_optional_float(option_values.get("krylov_tol"), "krylov_tol", 1e-9),
            krylov_max_iter=_coerce_optional_int(option_values.get("krylov_max_iter"), "krylov_max_iter", 2000),
            krylov_restart=krylov_restart,
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            use_mor=use_mor,
            mor_method=mor_method,
            mor_order=mor_order,
            mor_output_nodes=_coerce_label_list(option_values.get("mor_output_nodes")),
            mor_validate=mor_validate,
            num_threads=num_threads,
        )

    def run(self, circuit: CircuitIR, options: AnalysisOptions) -> SimulationResult:
        """Execute a simulation with Gmin stepping."""

        last_error: Exception | None = None
        for gmin in options.gmin_steps:
            try:
                if options.mode == AnalysisMode.HB:
                    result = build_and_run_advanced(circuit, options, gmin=gmin)
                else:
                    result = build_and_run_basic(circuit, options, gmin=gmin)
                result.metadata = dict(result.metadata)
                result.metadata["gmin"] = gmin
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Simulation failed without producing a result.")
