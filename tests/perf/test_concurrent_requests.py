"""P6 exit criterion for concurrent requests (restated in PERFORMANCE_PLAN.md 10.2).

``benchmarks/concurrent_requests.py`` sends N identical requests to the FastAPI app
in this process (``httpx.ASGITransport``, app lifespan entered, so the requests go
through the bounded simulation executor, the thread budget and the native core)
and compares the best batch wall time of N = 4 with the best single request.

Two workloads, two assertions (marker ``perf``, ``pytest --run-perf``):

* numerics-bound: the close-tone diode mixer ``.hb`` (H = 205, about 0.7 s of
  GIL-free native work per request) must finish four concurrent requests within
  1.5x of one. ``"auto"`` leases one thread per request for a small HB circuit
  (``native.threads.useful_threads``), so the four run side by side.
* glue-bound: the SRAM 10x10 ``.tran 6n 0.25n`` request takes about 30 ms, of
  which only ~3 ms is native; parsing, building, the operating-point table and
  the response conversion hold the GIL, so four concurrent requests take about
  4x one (a process pool is the P7 remedy). The criterion for this workload is
  the absolute batch time: four requests within 0.5 s.

Taking the best of three batches keeps both robust to a briefly loaded machine.
The smoke test always runs: it drives the same code path on a 2x2 SRAM with a
single batch and checks the report, not the timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import concurrent_requests  # noqa: E402

MAX_RATIO = concurrent_requests.EXIT_CRITERION_RATIO
CONCURRENT = 4
BATCHES = 3


def test_concurrent_requests_script_smoke() -> None:
    report = concurrent_requests.measure([2], batches=1, rows=2, cols=2, num_threads=1)
    assert [entry["n"] for entry in report["series"]] == [1, 2]
    assert report["ratios"]["1"] == 1.0
    assert report["ratios"]["2"] > 0
    for entry in report["series"]:
        assert len(entry["wall_s"]) == 1 and entry["best_s"] > 0
        assert entry["threads_used"] == [1]
        assert len(entry["batches"][0]["elapsed_ms"]) == entry["n"]
    table = concurrent_requests.render(report)
    assert "| 2 |" in table and "ratio to 1" in table


SRAM_BATCH_LIMIT_S = 0.5


@pytest.mark.perf
def test_four_concurrent_hb_requests_finish_within_1_5x_of_one() -> None:
    report = concurrent_requests.measure([CONCURRENT], batches=2, num_threads="auto", workload="hb_mixer")
    ratio = report["ratios"][str(CONCURRENT)]
    one, many = report["series"][0], report["series"][1]
    assert many["threads_used"] == [1], many["threads_used"]  # auto leases one thread per small HB run
    assert ratio <= MAX_RATIO, (
        f"{CONCURRENT} concurrent HB mixer requests took {ratio:.2f}x the wall time of one "
        f"(criterion <= {MAX_RATIO}x): best batch {many['best_s']:.3f} s vs {one['best_s']:.3f} s; "
        f"per-request elapsed_ms median {many['request_elapsed_ms_median']:.0f} vs {one['request_elapsed_ms_median']:.0f}; "
        f"load average {report['host']['loadavg_after']}"
    )


@pytest.mark.perf
def test_four_concurrent_sram_requests_finish_within_half_a_second() -> None:
    report = concurrent_requests.measure([CONCURRENT], batches=BATCHES, num_threads="auto")
    one, many = report["series"][0], report["series"][1]
    ratio = report["ratios"][str(CONCURRENT)]
    assert many["threads_used"] == [1], many["threads_used"]  # auto leases one thread per small .tran
    assert many["best_s"] <= SRAM_BATCH_LIMIT_S, (
        f"{CONCURRENT} concurrent SRAM .tran requests took {many['best_s'] * 1e3:.0f} ms "
        f"(limit {SRAM_BATCH_LIMIT_S * 1e3:.0f} ms; {ratio:.2f}x one request at {one['best_s'] * 1e3:.0f} ms); "
        f"load average {report['host']['loadavg_after']}"
    )
