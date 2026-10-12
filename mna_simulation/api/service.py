"""Frontend-facing orchestration service."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from functools import lru_cache

import numpy as np

from .contracts import AnalysisRequest, SimulationResponse
from ..backends.python_backend import PythonBackend
from ..errors import CircuitSimulatorError
from ..native import build_info, core_version
from ..native.threads import ThreadBudget, default_budget, pin_blas_threads, useful_threads
from ..schematic_pipeline import schematic_to_circuit_ir
from ..utils import ndarray_to_list


# PEP 678 note attached to ValueErrors raised while validating request options. The
# exception stays a plain ValueError (a pinned contract, see tests/EDGE_CASES.md); the note
# only lets the HTTP layer answer 400 for bad options without catching every ValueError.
INVALID_OPTION_NOTE = "mna_simulation: invalid request option"


_DEVICE_TYPES = frozenset({"D", "B", "QNPN", "QPNP", "NMOS", "PMOS"})


def _circuit_size(circuit: object) -> dict[str, int]:
    """Device and node counts of a parsed ``CircuitIR`` (for the ``"auto"`` thread cap)."""

    devices = 0
    nodes: set[str] = set()
    for component in getattr(circuit, "components", ()):
        if component.type in _DEVICE_TYPES:
            devices += 1
        for node in (component.node1, component.node2, component.ctrl_node1, component.ctrl_node2):
            if node is not None and node != "0":
                nodes.add(node)
    return {"devices": devices, "nodes": len(nodes)}


def is_invalid_option_error(exc: BaseException) -> bool:
    """True for a ValueError raised by request-option validation in ``run_request``."""

    return isinstance(exc, ValueError) and INVALID_OPTION_NOTE in getattr(exc, "__notes__", ())


@lru_cache(maxsize=1)
def _core_version() -> str:
    return core_version()


class SimulationService:
    """Single orchestration entry point shared by CLI and HTTP API."""

    # Which implementation produces the numbers: every analysis runs in the native core
    # since the P5 cut-over (CPP_MIGRATION_PLAN.md section 6).
    core_backend = "cpp"

    def __init__(self, backend: PythonBackend | None = None, budget: ThreadBudget | None = None) -> None:
        self.backend = backend or PythonBackend()
        self._budget = budget

    @property
    def budget(self) -> ThreadBudget:
        """Thread budget leased by every run (the process-wide default unless one was given)."""

        return self._budget if self._budget is not None else default_budget()

    def run_request(self, request: AnalysisRequest) -> SimulationResponse:
        """Run a request and convert the result to a JSON-safe response."""

        generated_netlist: str | None = None
        if request.schematic is not None:
            converted = schematic_to_circuit_ir(
                schematic=request.schematic,
                mode_override=request.mode,
                option_overrides=request.options,
            )
            circuit = converted.circuit
            generated_netlist = converted.netlist_text
        else:
            circuit = self.backend.parse_text(request.netlist_text)
        try:
            options = self.backend.build_options(circuit, overrides=request.options, mode=request.mode)
        except ValueError as exc:
            if not is_invalid_option_error(exc):
                exc.add_note(INVALID_OPTION_NOTE)
            raise

        # Lease worker threads for the run and keep NumPy/SciPy BLAS single-threaded, so
        # concurrent requests divide the machine instead of oversubscribing it. The granted
        # count replaces the raw request so the backend reads exactly what it may use.
        budget = self.budget
        threads_requested: int | str = options.num_threads if options.num_threads is not None else "auto"
        # "auto" takes at most what this analysis can use (native.threads.useful_threads), so
        # concurrent small requests run side by side instead of queueing behind one lease.
        cap = useful_threads(options.mode, ac_points=options.ac_points, **_circuit_size(circuit))
        with budget.lease(threads_requested, cap=cap) as threads_used, pin_blas_threads():
            options.num_threads = threads_used
            started = time.perf_counter()
            result = self.backend.run(circuit, options)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.metadata = dict(result.metadata)
        result.metadata.update(
            {
                "threads_requested": threads_requested,
                "threads_used": threads_used,
                "thread_budget_total": budget.total,
                "core_backend": self.core_backend,
                "core_version": _core_version(),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
        if generated_netlist is not None:
            result.metadata = dict(result.metadata)
            result.metadata["generated_netlist"] = generated_netlist
        result.metadata = dict(result.metadata)

        output_max_points = int(request.options.get("output_max_points", 0) or 0)

        waveform = None
        if result.waveform is not None:
            time_values, waveform_values = _sample_waveform_for_output(
                result.waveform.time,
                result.waveform.values,
                output_max_points,
                result.metadata,
            )
            waveform = {
                "time": ndarray_to_list(time_values),
                "values": ndarray_to_list(waveform_values),
                "labels": result.waveform.labels,
            }

        spectrum = None
        if result.spectrum is not None:
            spectrum = {
                "frequencies": ndarray_to_list(result.spectrum.frequencies),
                "magnitudes": ndarray_to_list(result.spectrum.magnitudes),
                "labels": result.spectrum.labels,
            }

        matrices = None
        if result.matrices is not None:
            matrices = {name: ndarray_to_list(value) for name, value in result.matrices.items()}

        dc_solution = None
        if result.dc_solution is not None:
            dc_solution = np.asarray(result.dc_solution).flatten().tolist()

        return SimulationResponse(
            mode=result.mode,
            status=result.status,
            labels=result.labels,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
            dc_solution=dc_solution,
            waveform=waveform,
            spectrum=spectrum,
            matrices=matrices,
        )

    def healthcheck(self) -> dict[str, object]:
        """Return backend information for the UI."""

        return {
            "status": "ok",
            "capabilities": asdict(self.backend.capabilities),
            "models": [model.name for model in self.backend.registry.list_models()],
            "core_backend": self.core_backend,
            "core_version": _core_version(),
            "build_info": build_info(),
            "thread_budget": self.budget.snapshot(),
        }


CORE_ENV_VAR = "MNA_CORE"
CORES = ("cpp", "python")


def resolve_core(core: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """``core``, else ``MNA_CORE``, else ``"cpp"``; anything but ``cpp``/``python`` is a ValueError."""

    if core is None:
        core = (os.environ if environ is None else environ).get(CORE_ENV_VAR) or "cpp"
    cleaned = core.strip().lower()
    if cleaned not in CORES:
        raise ValueError(f"{CORE_ENV_VAR} must be one of {', '.join(CORES)} (got {core!r})")
    return cleaned


def create_service(core: str | None = None, budget: ThreadBudget | None = None):  # noqa: ANN201
    """The service for ``core`` (see :func:`resolve_core`).

    ``"cpp"`` (the default) is :class:`SimulationService` on the native core;
    ``"python"`` is :class:`~mna_simulation.api.reference_service.ReferenceSimulationService`,
    the frozen pre-migration pure-Python core (source checkouts only).
    """

    if resolve_core(core) == "python":
        from .reference_service import ReferenceSimulationService

        return ReferenceSimulationService(budget=budget)
    return SimulationService(budget=budget)


def _sample_indexes(length: int, max_points: int) -> np.ndarray | None:
    if max_points <= 0 or length <= max_points:
        return None
    stride = max(1, int(np.ceil(length / max_points)))
    indexes = np.arange(0, length, stride, dtype=int)
    if indexes[-1] != length - 1:
        indexes = np.append(indexes, length - 1)
    return indexes


def _sample_waveform_for_output(
    time_values: np.ndarray,
    waveform_values: np.ndarray,
    max_points: int,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = _sample_indexes(len(time_values), max_points)
    if indexes is None:
        return time_values, waveform_values

    sampled_time = time_values[indexes]
    values = np.asarray(waveform_values)
    if values.ndim == 2 and values.shape[1] == len(time_values):
        sampled_values = values[:, indexes]
    elif values.ndim == 2 and values.shape[0] == len(time_values):
        sampled_values = values[indexes, :]
    else:
        sampled_values = values

    metadata["output_decimation"] = {
        "original_points": int(len(time_values)),
        "returned_points": int(len(sampled_time)),
        "max_points": int(max_points),
    }
    return sampled_time, sampled_values
