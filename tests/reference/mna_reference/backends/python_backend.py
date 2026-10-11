"""Python backend adapter for solver execution."""

from __future__ import annotations

from pathlib import Path

from ..api.contracts import AnalysisMode, AnalysisOptions, BackendCapabilities, CircuitIR, DirectiveRecord, SimulationResult
from ..core_advanced import build_and_run_advanced
from ..core_basic import build_and_run_basic
from ..library.registry import DeviceRegistry, create_default_registry
from ..netlist import parse_netlist_file, parse_netlist_text, select_analysis
from ..utils import DEFAULT_GMIN_STEPS


def _coerce_krylov_rank(value: object) -> int | str:
    """Return ``auto`` or a positive integer Krylov rank request."""

    if value is None:
        return "auto"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "auto":
            return "auto"
        value = cleaned
    rank = int(value)
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
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "auto" or cleaned == "":
            return "auto"
        value = cleaned
    order = int(value)
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

        use_krylov = bool(option_values.get("use_krylov", option_values.get("krylov", False)))
        if "krylov_rank" in option_values:
            krylov_rank = _coerce_krylov_rank(option_values.get("krylov_rank"))
        elif "krylov_restart" in option_values:
            krylov_rank = _coerce_krylov_rank(option_values.get("krylov_restart"))
        else:
            krylov_rank = "auto"
        krylov_method = _coerce_krylov_method(option_values.get("krylov_method", option_values.get("krylov_algorithm", "auto")))
        use_mor = bool(option_values.get("use_mor", option_values.get("mor", False)))
        mor_method = _coerce_mor_method(option_values.get("mor_method", "auto"))
        mor_order = _coerce_mor_order(option_values.get("mor_order", "auto"))

        return AnalysisOptions(
            mode=effective_mode or directive.mode,
            gmin_steps=option_values.get("gmin_steps", DEFAULT_GMIN_STEPS),
            max_iter=int(option_values.get("max_iter", 10000)),
            v_tol=float(option_values.get("v_tol", 1e-9)),
            f_tol=float(option_values.get("f_tol", 1e-9)),
            tran_stop=option_values.get("t_stop"),
            tran_step=option_values.get("t_step"),
            ac_start=option_values.get("f_start"),
            ac_stop=option_values.get("f_stop"),
            ac_points=option_values.get("points"),
            hb_harmonics=option_values.get("harmonics"),
            hb_time_window=option_values.get("time_window"),
            init_condition=option_values.get("init_condition"),
            probe_nodes=_coerce_label_list(option_values.get("probe_nodes")),
            use_krylov=use_krylov,
            krylov_tol=float(option_values.get("krylov_tol", 1e-9)),
            krylov_max_iter=int(option_values.get("krylov_max_iter", 2000)),
            krylov_restart=int(option_values.get("krylov_restart", 80)),
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            use_mor=use_mor,
            mor_method=mor_method,
            mor_order=mor_order,
            mor_output_nodes=_coerce_label_list(option_values.get("mor_output_nodes")),
            mor_validate=bool(option_values.get("mor_validate", True)),
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
