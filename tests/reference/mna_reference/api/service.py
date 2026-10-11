"""Frontend-facing orchestration service."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import AnalysisRequest, SimulationResponse
from ..backends.python_backend import PythonBackend
from ..errors import CircuitSimulatorError
from ..schematic_pipeline import schematic_to_circuit_ir
from ..utils import ndarray_to_list


class SimulationService:
    """Single orchestration entry point shared by CLI and HTTP API."""

    def __init__(self, backend: PythonBackend | None = None) -> None:
        self.backend = backend or PythonBackend()

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
        options = self.backend.build_options(circuit, overrides=request.options, mode=request.mode)
        result = self.backend.run(circuit, options)
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
        }


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
