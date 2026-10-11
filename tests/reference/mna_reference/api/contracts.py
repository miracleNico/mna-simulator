"""Stable data contracts shared by parser, builder, solvers, API, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class AnalysisMode(str, Enum):
    SHOW_MATRIX = "show_matrix"
    OP = "op"
    TRAN = "tran"
    AC = "ac"
    HB = "hb"
    DYN = "dyn"


@dataclass(slots=True)
class ComponentRecord:
    """Normalized component description independent of execution backend."""

    name: str
    type: str
    node1: str | None
    node2: str | None
    value: str | None
    subtype: str | None = None
    ctrl_node1: str | None = None
    ctrl_node2: str | None = None
    ctrl_source: str | None = None
    value2: str | None = None
    value3: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CircuitIR:
    """Validated circuit representation used across all layers."""

    components: list[ComponentRecord]
    directives: list["DirectiveRecord"] = field(default_factory=list)
    source_text: str = ""


@dataclass(slots=True)
class DirectiveRecord:
    """Structured simulation directive parsed from netlist text or CLI flags."""

    mode: AnalysisMode
    params: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


@dataclass(slots=True)
class AnalysisOptions:
    """Numerical options supplied by CLI/API/UI."""

    mode: AnalysisMode
    gmin_steps: list[float] = field(default_factory=lambda: [0.0])
    max_iter: int = 10000
    v_tol: float = 1e-9
    f_tol: float = 1e-9
    tran_stop: float | None = None
    tran_step: float | None = None
    ac_start: float | None = None
    ac_stop: float | None = None
    ac_points: int | None = None
    hb_harmonics: int | None = None
    hb_time_window: float | None = None
    init_condition: np.ndarray | str | None = None
    probe_nodes: list[str] = field(default_factory=list)
    use_krylov: bool = False
    krylov_tol: float = 1e-9
    krylov_max_iter: int = 2000
    krylov_restart: int = 80
    krylov_rank: int | str | None = "auto"
    krylov_method: str = "auto"
    use_mor: bool = False
    mor_method: str = "auto"
    mor_order: int | str | None = "auto"
    mor_output_nodes: list[str] = field(default_factory=list)
    mor_validate: bool = True


@dataclass(slots=True)
class IndexMap:
    """Stable name/index map returned from MNA assembly."""

    node_names: list[str]
    branch_names: list[str]
    node_to_index: dict[str, int]
    branch_to_index: dict[str, int]


@dataclass(slots=True)
class MnaProblem:
    """Matrix-form circuit problem consumed by solver backends."""

    G: np.ndarray
    C: np.ndarray
    G_sparse: Any | None
    C_sparse: Any | None
    f_str: np.ndarray
    solver_f_str: np.ndarray | None
    b_dc: np.ndarray
    b_ac: np.ndarray
    b_time_str: np.ndarray
    index_map: IndexMap
    components: list[ComponentRecord]
    level1_mos_devices: list[dict[str, Any]] = field(default_factory=list)
    level1_bjt_devices: list[dict[str, Any]] = field(default_factory=list)
    gmin: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WaveformResult:
    """Time-domain result series."""

    time: np.ndarray
    values: np.ndarray
    labels: list[str]


@dataclass(slots=True)
class SpectrumResult:
    """Frequency-domain result series."""

    frequencies: np.ndarray
    magnitudes: np.ndarray
    labels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    """Backend-neutral simulation result envelope."""

    mode: AnalysisMode
    status: str
    dc_solution: np.ndarray | None = None
    waveform: WaveformResult | None = None
    spectrum: SpectrumResult | None = None
    harmonic_state: np.ndarray | None = None
    matrices: dict[str, np.ndarray] | None = None
    labels: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendCapabilities:
    """Feature flags surfaced by a backend adapter."""

    supports_behavioral_sources: bool = True
    supports_harmonic_balance: bool = True
    supports_sparse_linear_solver: bool = False
    supports_krylov_linear_solver: bool = False
    supports_model_order_reduction: bool = False
    supports_cpp_acceleration: bool = False


class AnalysisRequest(BaseModel):
    """HTTP/API request payload."""

    netlist_text: str = Field(default="", description="Raw netlist text.")
    mode: AnalysisMode | None = Field(default=None)
    options: dict[str, Any] = Field(default_factory=dict)
    schematic: "SchematicDocument | None" = None


class SimulationResponse(BaseModel):
    """HTTP/API response payload."""

    mode: AnalysisMode
    status: str
    labels: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dc_solution: list[float] | None = None
    waveform: dict[str, Any] | None = None
    spectrum: dict[str, Any] | None = None
    matrices: dict[str, object] | None = None


class SchematicPosition(BaseModel):
    """Canvas position for components or junctions."""

    x: float
    y: float


class SchematicComponent(BaseModel):
    """Schematic component node sent from the frontend graph."""

    id: str
    type: Literal[
        "R",
        "C",
        "L",
        "V",
        "I",
        "D",
        "GND",
        "VCVS",
        "VCCS",
        "CCCS",
        "CCVS",
        "QNPN",
        "QPNP",
        "NMOS",
        "PMOS",
        "SUBCKT",
    ]
    name: str | None = None
    value: str | None = None
    subtype: str | None = None
    value2: str | None = None
    value3: str | None = None
    position: SchematicPosition | None = None
    # Controlled source extras.
    ctrl_node1: str | None = None
    ctrl_node2: str | None = None
    ctrl_source: str | None = None
    # Hierarchy: when type == SUBCKT, subcircuit_id refers to a SchematicDocument in the library.
    subcircuit_id: str | None = None
    # Pin names for SUBCKT instances (order-preserving). For primitive types this is ignored.
    pins: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_defaults(self) -> "SchematicComponent":
        defaults = {
            "R": "1k",
            "C": "1u",
            "L": "1m",
            "V": "1",
            "I": "1m",
            "D": "1e-15",
            "QNPN": "1e-15",
            "QPNP": "1e-15",
            "NMOS": "1m",
            "PMOS": "1m",
        }
        if self.type in defaults and (self.value is None or str(self.value).strip() == ""):
            self.value = defaults[self.type]
        if self.type in {"QNPN", "QPNP"}:
            model = self.metadata.get("model", "level1")
            self.metadata["model"] = model
            if model == "level1":
                self.value2 = self.value2 or "150"
                self.value3 = self.value3 or "3"
                self.metadata.setdefault("vaf", "100")
                self.metadata.setdefault("var", "25")
                self.metadata.setdefault("cje", "4p")
                self.metadata.setdefault("cjc", "2p")
                self.metadata.setdefault("rb", "50")
                self.metadata.setdefault("re", "0.5")
                self.metadata.setdefault("rc", "5")
        if self.type in {"NMOS", "PMOS"}:
            model = self.metadata.get("model", "level1")
            self.metadata["model"] = model
            if model == "level1":
                self.value2 = self.value2 or "0.7"
                self.value3 = self.value3 or "0.02"
                self.metadata.setdefault("cgs", "2p")
                self.metadata.setdefault("cgd", "1p")
        if self.type in {"V", "I"}:
            self.subtype = (self.subtype or "DC").upper()
        else:
            self.subtype = None
        return self


class SchematicEndpoint(BaseModel):
    """Wire endpoint reference."""

    kind: Literal["component_pin", "junction"]
    component_id: str | None = None
    pin: str | None = None
    junction_id: str | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> "SchematicEndpoint":
        if self.kind == "component_pin":
            if not self.component_id or not self.pin:
                raise ValueError("component_pin endpoint requires component_id and pin.")
            self.junction_id = None
        else:
            if not self.junction_id:
                raise ValueError("junction endpoint requires junction_id.")
            self.component_id = None
            self.pin = None
        return self


class SchematicWire(BaseModel):
    """Connection segment in the schematic graph."""

    id: str
    start: SchematicEndpoint
    end: SchematicEndpoint


class SchematicJunction(BaseModel):
    """Optional explicit junction node."""

    id: str
    position: SchematicPosition | None = None


class SchematicAnalysis(BaseModel):
    """Analysis command stored on the schematic document."""

    mode: AnalysisMode = AnalysisMode.OP
    params: dict[str, Any] = Field(default_factory=dict)


class SchematicDocument(BaseModel):
    """Backend-owned schematic graph format."""

    components: list[SchematicComponent]
    wires: list[SchematicWire] = Field(default_factory=list)
    junctions: list[SchematicJunction] = Field(default_factory=list)
    analysis: SchematicAnalysis | None = None
    title: str | None = None
    # Library of sub-schematics keyed by subcircuit_id (for SUBCKT components above).
    subcircuits: dict[str, "SchematicDocument"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SchematicDocument":
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Schematic component ids must be unique.")
        junction_ids = [junction.id for junction in self.junctions]
        if len(junction_ids) != len(set(junction_ids)):
            raise ValueError("Schematic junction ids must be unique.")
        wire_ids = [wire.id for wire in self.wires]
        if len(wire_ids) != len(set(wire_ids)):
            raise ValueError("Schematic wire ids must be unique.")
        return self


class SimulationDemoPreset(BaseModel):
    """Serializable preset schematic used by demo clients."""

    id: str
    title: str
    description: str
    schematic: SchematicDocument
