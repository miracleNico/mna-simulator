"""num_threads request option: backend coercion, service lease, and result metadata."""

from __future__ import annotations

import pytest

from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions, AnalysisRequest
from mna_simulation.api.service import SimulationService, is_invalid_option_error
from mna_simulation.backends.python_backend import PythonBackend
from mna_simulation.native import core_version
from mna_simulation.native.threads import ThreadBudget, default_budget

OP_NETLIST = """
V1 n1 0 DC 5
R1 n1 n2 1k
R2 n2 0 1k
.op
.end
"""


@pytest.fixture(scope="module")
def backend() -> PythonBackend:
    return PythonBackend()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (None, None),
        ({}, None),
        ({"num_threads": None}, None),
        ({"num_threads": "auto"}, "auto"),
        ({"num_threads": "AUTO"}, "auto"),
        ({"num_threads": 3}, 3),
        ({"num_threads": "2"}, 2),
    ],
)
def test_build_options_coerces_num_threads(backend: PythonBackend, overrides: dict | None, expected: object) -> None:
    circuit = backend.parse_text(OP_NETLIST)
    options = backend.build_options(circuit, overrides=overrides)
    assert isinstance(options, AnalysisOptions)
    assert options.num_threads == expected


@pytest.mark.parametrize("bad", [0, -1, "zero", "1.5", 2.5, True])
def test_build_options_rejects_invalid_num_threads(backend: PythonBackend, bad: object) -> None:
    circuit = backend.parse_text(OP_NETLIST)
    with pytest.raises(ValueError, match="num_threads must be 'auto' or a positive integer"):
        backend.build_options(circuit, overrides={"num_threads": bad})


def test_analysis_options_default_num_threads_is_none() -> None:
    assert AnalysisOptions(mode=AnalysisMode.OP).num_threads is None


class _RecordingBackend(PythonBackend):
    """Captures the options the service hands to ``run``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[AnalysisOptions] = []

    def run(self, circuit, options):  # noqa: ANN001
        self.seen.append(options)
        return super().run(circuit, options)


def _run(service: SimulationService, options: dict) -> dict:
    response = service.run_request(AnalysisRequest(netlist_text=OP_NETLIST, mode=AnalysisMode.OP, options=options))
    assert response.status == "ok"
    assert response.dc_solution is not None
    return response.metadata


def test_service_metadata_for_explicit_threads() -> None:
    backend = _RecordingBackend()
    service = SimulationService(backend, budget=ThreadBudget(4))
    metadata = _run(service, {"num_threads": 2})
    assert metadata["threads_requested"] == 2
    assert metadata["threads_used"] == 2
    assert metadata["thread_budget_total"] == 4
    assert metadata["core_backend"] == "cpp"  # every analysis runs in the native core since P5
    assert metadata["core_version"] == core_version()
    assert isinstance(metadata["elapsed_ms"], float) and metadata["elapsed_ms"] >= 0.0
    # The backend sees the grant, not the raw request.
    assert backend.seen[-1].num_threads == 2
    # Existing keys are untouched.
    assert "gmin" in metadata
    # The lease is returned after the run.
    assert service.budget.snapshot() == {"total": 4, "free": 4, "active": 0}


def test_service_metadata_for_auto_and_absent() -> None:
    backend = _RecordingBackend()
    service = SimulationService(backend, budget=ThreadBudget(3))
    for options in ({"num_threads": "auto"}, {}):
        metadata = _run(service, options)
        assert metadata["threads_requested"] == "auto"
        # "auto" takes what the analysis can use: a small .op runs its devices on one
        # thread, so the lease is 1 and the other two threads stay free for other requests.
        assert metadata["threads_used"] == 1
        assert metadata["thread_budget_total"] == 3
        assert backend.seen[-1].num_threads == 1


def test_service_caps_request_at_budget() -> None:
    service = SimulationService(budget=ThreadBudget(2))
    metadata = _run(service, {"num_threads": "64"})
    assert metadata["threads_requested"] == 64
    assert metadata["threads_used"] == 2


AC_NETLIST = "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n"


def _run_ac(service: SimulationService, options: dict, points: int = 100) -> dict:
    request = AnalysisRequest(
        netlist_text=AC_NETLIST,
        mode=AnalysisMode.AC,
        options={"f_start": 1.0, "f_stop": 1e6, "points": points, **options},
    )
    response = service.run_request(request)
    assert response.status == "ok"
    return response.metadata


def test_service_auto_shares_with_active_run() -> None:
    budget = ThreadBudget(4)
    service = SimulationService(budget=budget)
    with budget.lease(1):
        # An AC sweep of 100 points can use up to 50 threads; one active run leaves the
        # fair share 4 // 2 = 2.
        metadata = _run_ac(service, {"num_threads": "auto"})
    assert metadata["threads_used"] == 2


def test_service_auto_takes_the_whole_budget_when_the_analysis_can_use_it() -> None:
    service = SimulationService(budget=ThreadBudget(3))
    assert _run_ac(service, {"num_threads": "auto"})["threads_used"] == 3
    # Two frequency points spread over at most one thread.
    assert _run_ac(service, {"num_threads": "auto"}, points=2)["threads_used"] == 1
    # An explicit count is the caller's upper limit and is never capped by the policy.
    assert _run(service, {"num_threads": 3})["threads_used"] == 3


@pytest.mark.parametrize(
    ("mode", "kwargs", "expected"),
    [
        ("op", {"devices": 622, "nodes": 248}, 1),
        ("tran", {"devices": 2048, "nodes": 900}, None),
        ("ac", {"devices": 0, "nodes": 1035, "ac_points": 100}, 50),
        ("ac", {"devices": 0, "nodes": 3, "ac_points": None}, 50),
        ("hb", {"devices": 1, "nodes": 3}, 1),
        ("hb", {"devices": 40, "nodes": 64}, None),
        (AnalysisMode.OP, {"devices": 1, "nodes": 2}, 1),
    ],
)
def test_useful_threads_policy(mode, kwargs, expected) -> None:
    from mna_simulation.native.threads import useful_threads

    assert useful_threads(mode, **kwargs) == expected


def test_service_uses_process_default_budget() -> None:
    metadata = _run(SimulationService(), {})
    assert metadata["thread_budget_total"] == default_budget().total
    assert 1 <= metadata["threads_used"] <= default_budget().total


def test_service_rejects_invalid_num_threads() -> None:
    budget = ThreadBudget(2)
    service = SimulationService(budget=budget)
    with pytest.raises(ValueError, match="num_threads must be 'auto' or a positive integer") as excinfo:
        service.run_request(AnalysisRequest(netlist_text=OP_NETLIST, mode=AnalysisMode.OP, options={"num_threads": 0}))
    # Stays a plain ValueError (tests/EDGE_CASES.md contract); a PEP 678 note marks it for HTTP 400.
    assert type(excinfo.value) is ValueError
    assert is_invalid_option_error(excinfo.value)
    assert budget.snapshot() == {"total": 2, "free": 2, "active": 0}


def test_service_releases_lease_when_run_fails() -> None:
    class _FailingBackend(PythonBackend):
        def run(self, circuit, options):  # noqa: ANN001
            raise RuntimeError("boom")

    budget = ThreadBudget(2)
    service = SimulationService(_FailingBackend(), budget=budget)
    with pytest.raises(RuntimeError, match="boom"):
        service.run_request(AnalysisRequest(netlist_text=OP_NETLIST, mode=AnalysisMode.OP))
    assert budget.snapshot() == {"total": 2, "free": 2, "active": 0}


def test_healthcheck_reports_core_and_budget() -> None:
    service = SimulationService(budget=ThreadBudget(2))
    health = service.healthcheck()
    assert health["status"] == "ok"
    assert health["core_version"] == core_version()
    assert isinstance(health["build_info"], dict) and health["build_info"]
    assert health["thread_budget"] == {"total": 2, "free": 2, "active": 0}
    assert {"capabilities", "models"} <= set(health)


def test_cli_threads_option(capsys: pytest.CaptureFixture[str]) -> None:
    from mna_simulation.cli import main

    assert main(["run-text", OP_NETLIST, "--threads", "1"]) == 0
    out = capsys.readouterr().out
    assert "Threads used: 1" in out

    assert main(["run-text", OP_NETLIST, "--threads", "auto"]) == 0
    assert "Threads used:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        main(["run-text", OP_NETLIST, "--threads", "0"])
    assert excinfo.value.code == 2
    assert "num_threads must be 'auto' or a positive integer" in capsys.readouterr().err
