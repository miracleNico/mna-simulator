"""Resource check: repeated native solves through the service do not grow the process memory.

200 SRAM 10x10 ``.op`` solves and 50 SRAM ``.tran 6n 0.25n`` transients (622
Level-1 MOS, 248 unknowns) run through ``SimulationService.run_request`` with
``num_threads="auto"``, so every run builds a compiled circuit, KLU symbolic
and numeric objects, per-call worker threads and the NumPy result arrays.
After a warm-up of 20 runs of each kind (interpreter caches, allocator pools,
the expression and value caches reach their steady state) the resident set
may grow by less than ``RSS_GROWTH_LIMIT`` (20 MB). A leak of one SRAM
circuit or one transient result per run would exceed that several times over.

Memory is read with ``resource.getrusage`` (peak RSS, ``ru_maxrss``) and, on
Linux, also from ``/proc/self/statm`` (current RSS): the peak alone can hide a
leak below an earlier peak of the same process (another test's large HB
system, for example), the current value cannot. Both must stay under the
limit. Windows has no ``resource`` module: the test runs there only when
``psutil`` is importable.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest

from corpus import sram_netlist
from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest
from mna_simulation.api.service import SimulationService

RSS_GROWTH_LIMIT = 20 * 1024 * 1024
WARMUP_RUNS = 20
OP_RUNS = 200
TRAN_RUNS = 50

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    resource is None and psutil is None,
    reason="needs the resource module (POSIX) or psutil to read the process RSS",
)


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _current_rss_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    return None


def _snapshot() -> dict[str, int | None]:
    gc.collect()
    return {"peak": _peak_rss_bytes(), "current": _current_rss_bytes()}


def _mb(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MB"


def test_repeated_sram_op_and_transient_runs_do_not_grow_rss() -> None:
    service = SimulationService()
    netlist = sram_netlist()

    def op() -> None:
        response = service.run_request(AnalysisRequest(netlist_text=netlist, mode=AnalysisMode.OP, options={"num_threads": "auto"}))
        assert response.status == "ok" and len(response.dc_solution) == 248

    def tran() -> None:
        response = service.run_request(AnalysisRequest(netlist_text=netlist, mode=AnalysisMode.TRAN, options={"num_threads": "auto"}))
        assert response.status == "ok" and len(response.waveform["time"]) == 25

    for _ in range(WARMUP_RUNS):
        op()
        tran()
    before = _snapshot()
    for index in range(OP_RUNS):
        op()
        if index % (OP_RUNS // TRAN_RUNS) == 0:
            tran()
    after = _snapshot()

    measured = {kind: (before[kind], after[kind]) for kind in before if before[kind] is not None and after[kind] is not None}
    assert measured, "no RSS reading available"
    report = ", ".join(f"{kind} RSS {_mb(start)} -> {_mb(end)}" for kind, (start, end) in measured.items())
    for kind, (start, end) in measured.items():
        assert end - start < RSS_GROWTH_LIMIT, (
            f"{kind} RSS grew by {_mb(end - start)} over {OP_RUNS} SRAM .op and {TRAN_RUNS} .tran runs "
            f"(limit {_mb(RSS_GROWTH_LIMIT)}; {report})"
        )
