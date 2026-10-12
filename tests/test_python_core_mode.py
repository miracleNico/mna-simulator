"""Pure-Python core mode (``MNA_CORE=python``): the frozen pre-migration core behind the live contract."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.server.main as server
from mna_simulation import cli
from mna_simulation.api import reference_service
from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest
from mna_simulation.api.reference_service import ReferenceCoreUnavailable, ReferenceSimulationService
from mna_simulation.api.service import (
    CORE_ENV_VAR,
    SimulationService,
    create_service,
    is_invalid_option_error,
    resolve_core,
)
from mna_simulation.errors import NetlistError
from mna_simulation.native.threads import ThreadBudget

RC_NETLIST = """
V1 n1 0 DC 5
R1 n1 n2 1k
R2 n2 0 2k
C1 n2 0 1u
.end
"""

DIODE_NETLIST = """
V1 n1 0 DC 2
R1 n1 n2 1k
D1 n2 0 1e-14
.op
.end
"""


@pytest.fixture(scope="module")
def python_service() -> ReferenceSimulationService:
    return ReferenceSimulationService(budget=ThreadBudget(4))


@pytest.fixture(scope="module")
def cpp_service() -> SimulationService:
    return SimulationService(budget=ThreadBudget(4))


def _request(netlist: str, mode: str, **options: object) -> AnalysisRequest:
    return AnalysisRequest(netlist_text=netlist, mode=AnalysisMode(mode), options=dict(options))


def test_resolve_core_defaults_to_cpp_and_reads_the_environment() -> None:
    assert resolve_core(environ={}) == "cpp"
    assert resolve_core(environ={CORE_ENV_VAR: ""}) == "cpp"
    assert resolve_core(environ={CORE_ENV_VAR: " Python "}) == "python"
    assert resolve_core("cpp", environ={CORE_ENV_VAR: "python"}) == "cpp"
    with pytest.raises(ValueError, match="MNA_CORE must be one of cpp, python"):
        resolve_core("fortran")


def test_create_service_picks_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CORE_ENV_VAR, raising=False)
    assert type(create_service()) is SimulationService
    assert isinstance(create_service("python"), ReferenceSimulationService)
    monkeypatch.setenv(CORE_ENV_VAR, "python")
    assert isinstance(create_service(), ReferenceSimulationService)


def test_python_core_runs_the_frozen_package(python_service: ReferenceSimulationService) -> None:
    assert python_service.reference.__name__ == "mna_reference"
    assert type(python_service.backend).__module__ == "mna_simulation.api.reference_service"
    assert type(python_service.backend).__mro__[1].__module__ == "mna_reference.backends.python_backend"


def test_missing_reference_package_is_a_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(reference_service, "REFERENCE_PACKAGE", "mna_reference_not_there")
    monkeypatch.setattr(reference_service, "REFERENCE_DIR", tmp_path)
    with pytest.raises(ReferenceCoreUnavailable, match="source checkout"):
        ReferenceSimulationService()


@pytest.mark.parametrize(
    ("mode", "options"),
    [
        ("op", {}),
        ("tran", {"t_stop": 1e-3, "t_step": 1e-5}),
        ("ac", {"f_start": 1.0, "f_stop": 1e5, "points": 20}),
    ],
)
def test_python_core_agrees_with_the_cpp_core(
    python_service: ReferenceSimulationService, cpp_service: SimulationService, mode: str, options: dict
) -> None:
    py = python_service.run_request(_request(RC_NETLIST, mode, **options))
    cc = cpp_service.run_request(_request(RC_NETLIST, mode, **options))
    assert py.status == cc.status == "ok"
    assert py.mode == cc.mode == AnalysisMode(mode)
    assert py.labels == cc.labels
    if mode == "op":
        np.testing.assert_allclose(py.dc_solution, cc.dc_solution, rtol=1e-9, atol=1e-12)
    elif mode == "tran":
        np.testing.assert_allclose(py.waveform["time"], cc.waveform["time"], rtol=1e-12)
        np.testing.assert_allclose(py.waveform["values"], cc.waveform["values"], rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_allclose(py.spectrum["frequencies"], cc.spectrum["frequencies"], rtol=1e-12)
        np.testing.assert_allclose(py.spectrum["magnitudes"], cc.spectrum["magnitudes"], rtol=1e-9, atol=1e-15)


def test_python_core_metadata(python_service: ReferenceSimulationService) -> None:
    metadata = python_service.run_request(_request(DIODE_NETLIST, "op", num_threads=3)).metadata
    assert metadata["core_backend"] == "python"
    assert metadata["core_version"].startswith("python-reference@")
    assert metadata["core_version"] != "python-reference@unknown"
    assert metadata["threads_requested"] == 3
    assert metadata["threads_used"] == 1  # the pure-Python core is serial
    assert metadata["thread_budget_total"] == 4
    assert metadata["elapsed_ms"] >= 0
    assert python_service.budget.snapshot() == {"total": 4, "free": 4, "active": 0}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"num_threads": 0}, "num_threads must be 'auto' or a positive integer"),
        ({"use_krylov": "maybe"}, "use_krylov must be a boolean"),
        ({"use_krylov": True, "krylov_rank": 0}, "krylov_rank must be 'auto' or a positive integer."),
    ],
)
def test_python_core_rejects_invalid_options_like_the_cpp_core(
    python_service: ReferenceSimulationService, options: dict, message: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        python_service.run_request(_request(RC_NETLIST, "op", **options))
    assert str(excinfo.value) == message
    assert is_invalid_option_error(excinfo.value)


def test_python_core_errors_use_the_live_error_classes(python_service: ReferenceSimulationService) -> None:
    with pytest.raises(NetlistError) as excinfo:
        python_service.run_request(_request("R1 a b\n.op\n.end\n", "op"))
    assert excinfo.value.code == "NETLIST_FATAL"
    assert type(excinfo.value.__cause__).__module__ == "mna_reference.errors"


def test_python_core_healthcheck(python_service: ReferenceSimulationService) -> None:
    health = python_service.healthcheck()
    assert health["status"] == "ok"
    assert health["core_backend"] == "python"
    assert health["core_version"].startswith("python-reference@")
    assert health["build_info"] is None
    assert health["capabilities"]["supports_cpp_acceleration"] is False
    assert health["models"]


def _rc_schematic() -> dict:
    def pin(component_id: str, name: str) -> dict:
        return {"kind": "component_pin", "component_id": component_id, "pin": name}

    return {
        "components": [
            {"id": "gnd1", "type": "GND", "name": "GND1"},
            {"id": "v1", "type": "V", "name": "V1", "subtype": "DC", "value": "1"},
            {"id": "r1", "type": "R", "name": "R1", "value": "1k"},
            {"id": "c1", "type": "C", "name": "C1", "value": "1u"},
        ],
        "wires": [
            {"id": "w1", "start": pin("v1", "n"), "end": pin("gnd1", "g")},
            {"id": "w2", "start": pin("v1", "p"), "end": pin("r1", "p")},
            {"id": "w3", "start": pin("r1", "n"), "end": pin("c1", "p")},
            {"id": "w4", "start": pin("c1", "n"), "end": pin("gnd1", "g")},
        ],
        "analysis": {"mode": "tran", "params": {"t_stop": "1m", "t_step": "0.1m"}},
    }


@pytest.fixture()
def python_client(monkeypatch: pytest.MonkeyPatch, python_service: ReferenceSimulationService):
    monkeypatch.setattr(server, "service", python_service)
    with TestClient(server.app) as test_client:
        yield test_client


def test_http_health_and_simulate_on_the_python_core(python_client: TestClient) -> None:
    health = python_client.get("/health").json()
    assert health["core_backend"] == "python"

    body = python_client.post("/api/simulate", json={"schematic": _rc_schematic(), "mode": "tran"}).json()
    assert body["status"] == "ok"
    assert body["metadata"]["core_backend"] == "python"
    assert "generated_netlist" in body["metadata"]
    assert body["waveform"]["labels"]

    bad = python_client.post("/api/simulate", json={"netlist_text": DIODE_NETLIST, "options": {"num_threads": "lots"}})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "INVALID_OPTIONS"

    broken = python_client.post("/api/simulate", json={"netlist_text": "R1 a b\n.op\n.end\n"})
    assert broken.status_code == 400
    assert broken.json()["detail"]["code"] == "NETLIST_FATAL"


def test_http_dyn_stream_on_the_python_core(python_client: TestClient) -> None:
    with python_client.websocket_connect("/ws/dyn") as ws:
        ws.send_json({"schematic": _rc_schematic(), "speed": 1.0, "t_stop": "1m", "t_step": "0.1m", "max_frame_rate": 1000})
        meta = ws.receive_json()
        assert meta["type"] == "meta"
        assert meta["threads_used"] == 1
        kinds = set()
        while True:
            message = ws.receive_json()
            kinds.add(message["type"])
            if message["type"] in ("done", "error"):
                break
        assert kinds == {"frame", "done"}


def test_cpp_health_reports_its_core() -> None:
    assert SimulationService(budget=ThreadBudget(1)).healthcheck()["core_backend"] == "cpp"


def test_cli_core_option(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CORE_ENV_VAR, raising=False)
    assert cli.main(["run-text", DIODE_NETLIST, "--core", "python"]) == 0
    assert "Core: python (python-reference@" in capsys.readouterr().out
    assert cli.main(["run-text", DIODE_NETLIST]) == 0
    assert "Core: cpp (" in capsys.readouterr().out
