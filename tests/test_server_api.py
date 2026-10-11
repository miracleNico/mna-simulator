"""FastAPI app: /health, /api/simulate on the bounded executor, thread budget, and /ws/dyn options."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app.server.main as server
from mna_simulation.native import core_version
from mna_simulation.native.threads import ThreadBudget

OP_NETLIST = """
V1 n1 0 DC 5
R1 n1 n2 1k
R2 n2 0 2k
.op
.end
"""


@pytest.fixture()
def client():
    with TestClient(server.app) as test_client:
        yield test_client


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


def test_lifespan_creates_and_shuts_down_executor() -> None:
    with TestClient(server.app):
        executor = server.app.state.executor
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor._max_workers == server.budget_total()  # noqa: SLF001
    assert server.app.state.executor is None


def test_health_reports_core_and_thread_budget(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {"capabilities", "models", "core_version", "build_info", "thread_budget"} <= set(body)
    assert body["core_version"] == core_version()
    assert isinstance(body["build_info"], dict)
    budget = body["thread_budget"]
    assert set(budget) == {"total", "free", "active"}
    assert budget["total"] >= 1
    assert 0 <= budget["free"] <= budget["total"]


def test_simulate_op_reports_threads(client: TestClient) -> None:
    response = client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"num_threads": 1}})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dc_solution"] is not None
    metadata = body["metadata"]
    assert metadata["threads_requested"] == 1
    assert metadata["threads_used"] == 1
    assert metadata["thread_budget_total"] >= 1
    assert metadata["core_backend"] == "python"  # flips to "cpp" at the C++ cut-overs
    assert metadata["core_version"] == core_version()
    assert metadata["elapsed_ms"] >= 0

    auto = client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op"}).json()["metadata"]
    assert auto["threads_requested"] == "auto"
    assert 1 <= auto["threads_used"] <= auto["thread_budget_total"]


@pytest.mark.parametrize("bad", [0, -2, "lots", 1.5])
def test_simulate_rejects_invalid_num_threads(client: TestClient, bad: object) -> None:
    response = client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"num_threads": bad}})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_OPTIONS"
    assert detail["message"] == "num_threads must be 'auto' or a positive integer"


def test_simulate_maps_other_option_errors_to_400(client: TestClient) -> None:
    response = client.post(
        "/api/simulate",
        json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"use_krylov": True, "krylov_rank": 0}},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "krylov_rank must be 'auto' or a positive integer."


def test_solver_value_error_is_not_reported_as_bad_options(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_run(circuit, options):  # noqa: ANN001
        raise ValueError("internal solver failure")

    monkeypatch.setattr(server.service.backend, "run", failing_run)
    with TestClient(server.app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op"})
    assert response.status_code == 500


def test_simulate_keeps_netlist_error_mapping(client: TestClient) -> None:
    response = client.post("/api/simulate", json={"netlist_text": "R1 a 0 1k\nX1 bad\n.op\n.end", "mode": "op"})
    assert response.status_code == 400
    assert {"code", "message"} <= set(response.json()["detail"])


def test_concurrent_requests_share_the_budget(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    total = 2
    budget = ThreadBudget(total)
    monkeypatch.setattr(server.service, "_budget", budget)

    lock = threading.Lock()
    in_use = 0
    peak = 0
    overlaps = 0
    original_run = server.service.backend.run

    def tracking_run(circuit, options):  # noqa: ANN001
        nonlocal in_use, peak, overlaps
        with lock:
            in_use += options.num_threads
            peak = max(peak, in_use)
            if in_use > options.num_threads:
                overlaps += 1
        try:
            time.sleep(0.05)  # hold the lease long enough for the requests to overlap
            return original_run(circuit, options)
        finally:
            with lock:
                in_use -= options.num_threads

    monkeypatch.setattr(server.service.backend, "run", tracking_run)

    requests = [1, 1, "auto", 2]

    def post(num_threads: object):
        return client.post(
            "/api/simulate",
            json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"num_threads": num_threads}},
        )

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        responses = list(pool.map(post, requests))

    assert [r.status_code for r in responses] == [200] * len(requests)
    used = [r.json()["metadata"]["threads_used"] for r in responses]
    for value in used:
        assert 1 <= value <= total
    assert all(r.json()["metadata"]["thread_budget_total"] == total for r in responses)
    assert peak <= total, f"threads in use peaked at {peak} with a budget of {total}"
    assert budget.snapshot() == {"total": total, "free": total, "active": 0}


def test_ws_dyn_passes_num_threads(client: TestClient) -> None:
    payload = {
        "schematic": _rc_schematic(),
        "speed": 1000.0,  # 1 ms of simulated time streams in about 1 us
        "t_stop": "1m",
        "t_step": "0.1m",
        "continuous": False,
        "num_threads": 1,
    }
    with client.websocket_connect("/ws/dyn") as ws:
        ws.send_json(payload)
        meta = ws.receive_json()
        assert meta["type"] == "meta", meta
        assert meta["threads_used"] == 1
        assert meta["elapsed_ms"] >= 0
        kinds = set()
        while True:
            message = ws.receive_json()
            kinds.add(message["type"])
            if message["type"] in {"done", "error"}:
                break
    assert "frame" in kinds and "done" in kinds


def test_ws_dyn_rejects_invalid_num_threads(client: TestClient) -> None:
    payload = {"schematic": _rc_schematic(), "speed": 1000.0, "t_stop": "1m", "t_step": "0.1m", "num_threads": "x"}
    with client.websocket_connect("/ws/dyn") as ws:
        ws.send_json(payload)
        message = ws.receive_json()
    assert message["type"] == "error"
    assert "num_threads must be 'auto' or a positive integer" in message["message"]
