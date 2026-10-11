"""P6 exit criterion: four concurrent SRAM requests finish within 1.5x of one.

``benchmarks/concurrent_requests.py`` sends N identical SRAM 10x10
``.tran 6n 0.25n`` requests (``num_threads="auto"``) to the FastAPI app in this
process (``httpx.ASGITransport``, app lifespan entered, so the requests go
through the bounded simulation executor, the thread budget and the native
core). The perf test (marker ``perf``, ``pytest --run-perf``) asserts
``best(4 concurrent) / best(1) <= 1.5`` over three interleaved batches per
level; taking the best batch keeps it robust to a briefly loaded machine.
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


@pytest.mark.perf
def test_four_concurrent_sram_requests_finish_within_1_5x_of_one() -> None:
    report = concurrent_requests.measure([CONCURRENT], batches=BATCHES, num_threads="auto")
    ratio = report["ratios"][str(CONCURRENT)]
    one, many = report["series"][0], report["series"][1]
    assert ratio <= MAX_RATIO, (
        f"{CONCURRENT} concurrent SRAM .tran requests took {ratio:.2f}x the wall time of one "
        f"(criterion <= {MAX_RATIO}x): best batch {many['best_s'] * 1e3:.1f} ms vs {one['best_s'] * 1e3:.1f} ms; "
        f"per-request elapsed_ms median {many['request_elapsed_ms_median']:.1f} vs {one['request_elapsed_ms_median']:.1f}; "
        f"threads_used {many['threads_used']} vs {one['threads_used']}; load average {report['host']['loadavg_after']}"
    )
