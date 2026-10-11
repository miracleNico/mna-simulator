"""Prototype component and model library registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from ..errors import error_handler


@dataclass(slots=True)
class ModelDefinition:
    """Reusable model-card definition for a device family."""

    name: str
    symbol: str
    pins: list[str]
    parameter_schema: dict[str, str] = field(default_factory=dict)
    backend_model_type: str = "python"
    defaults: dict[str, float | str] = field(default_factory=dict)
    stamp_hook: str | None = None


class DeviceRegistry:
    """Registry for built-in and user-defined device models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelDefinition] = {}

    def register(self, definition: ModelDefinition) -> None:
        self._models[definition.name] = definition

    def get(self, name: str) -> ModelDefinition:
        if name not in self._models:
            error_handler(f"MODEL_ERROR: Unknown model '{name}'.")
        return self._models[name]

    def list_models(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def load_directory(self, directory: str | Path) -> None:
        path = Path(directory)
        if not path.exists():
            return
        for model_file in sorted(path.glob("*.json")):
            payload = json.loads(model_file.read_text(encoding="utf-8"))
            self.register(ModelDefinition(**payload))


def create_default_registry(model_dir: str | Path | None = None) -> DeviceRegistry:
    """Create the default registry with built-in prototype models."""

    registry = DeviceRegistry()
    registry.register(
        ModelDefinition(
            name="ideal_diode",
            symbol="D",
            pins=["anode", "cathode"],
            parameter_schema={"is": "Saturation current"},
            defaults={"is": "1e-15"},
            stamp_hook="diode_current_expression",
        )
    )
    registry.register(
        ModelDefinition(
            name="behavioral_current",
            symbol="B",
            pins=["p", "n"],
            parameter_schema={"expression": "Current expression as a function of x[] or t"},
            defaults={"expression": "0"},
            stamp_hook="behavioral_source",
        )
    )
    registry.register(
        ModelDefinition(
            name="bjt_proto",
            symbol="Q",
            pins=["collector", "base", "emitter"],
            parameter_schema={"beta": "Forward current gain", "is": "Saturation current"},
            backend_model_type="prototype",
            defaults={"beta": 100, "is": "1e-15"},
        )
    )
    registry.register(
        ModelDefinition(
            name="mosfet_proto",
            symbol="M",
            pins=["drain", "gate", "source", "bulk"],
            parameter_schema={"kp": "Transconductance parameter", "vto": "Threshold voltage"},
            backend_model_type="prototype",
            defaults={"kp": 1e-3, "vto": 1.0},
        )
    )
    if model_dir is not None:
        registry.load_directory(model_dir)
    return registry
