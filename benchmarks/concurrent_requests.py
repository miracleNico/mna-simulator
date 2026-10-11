"""Concurrent ``POST /api/simulate`` requests against the FastAPI app.

Sends ``N`` identical SRAM 10x10 ``.tran 6n 0.25n`` requests (the UI demo
workload of ``benchmarks/workload_netlists.py``) at the same time and reports
the wall time of each batch and its ratio to the single-request batch. The
P6 exit criterion of ``CPP_MIGRATION_PLAN.md`` section 6 is "four concurrent
SRAM requests finish within 1.5x of one" (``tests/perf/test_concurrent_requests.py``
asserts it with ``--run-perf``).

Usage::

    python benchmarks/concurrent_requests.py                     # N = 1, 4; best of 3; in-process ASGI
    python benchmarks/concurrent_requests.py --server uvicorn    # real HTTP to a uvicorn subprocess
    python benchmarks/concurrent_requests.py --n 1 2 4 --batches 5 --num-threads 1
    python benchmarks/concurrent_requests.py --json out.json --max-ratio 1.5

Servers:

* ``asgi`` (default): the app runs in this process (``httpx.ASGITransport``,
  with the app lifespan entered so the bounded simulation executor exists).
  Requests take the same path as in production from the route handler on:
  ``/api/simulate`` -> executor (one worker per budget thread) -> service
  (thread lease, BLAS pinning) -> native core.
* ``uvicorn``: ``python -m uvicorn app.server.main:app`` in a subprocess on a
  free local port (``OPENBLAS_NUM_THREADS=1``, ``OMP_NUM_THREADS=1``), with
  real HTTP and JSON over the loopback interface.

Method: one untimed warm-up request (imports, expression and value caches),
then ``--batches`` batches per ``N``; each batch fires ``N`` requests with
``asyncio.gather`` and is timed from the first send to the last response.
The best batch per ``N`` is reported (robust to a loaded machine) together
with the median, and ``ratio(N) = best(N) / best(1)``. Every response must be
HTTP 200 with ``status == "ok"``; its ``threads_used`` and ``elapsed_ms``
metadata are reported. The script runs on whatever machine it is given and
records the load average: numbers from a busy machine are noisy.

Exit status: 0, or 1 when ``--max-ratio`` is given and ``ratio(max N)``
exceeds it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
for _path in (BENCH_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from workload_netlists import HB_CLOSE_TONE_MIXER, sram_netlist  # noqa: E402

DEFAULT_N = (1, 4)
DEFAULT_BATCHES = 3
EXIT_CRITERION_RATIO = 1.5
REQUEST_TIMEOUT_S = 600.0
SERVER_START_TIMEOUT_S = 120.0


WORKLOADS = ("sram", "hb_mixer")


def sram_request_body(
    rows: int = 10,
    cols: int = 10,
    step: str = "0.25n",
    num_threads: int | str = "auto",
) -> dict[str, Any]:
    """The JSON body of one SRAM ``.tran`` request (``.tran 6n <step>``)."""

    return {
        "netlist_text": sram_netlist(rows=rows, cols=cols, step=step),
        "mode": "tran",
        "options": {"num_threads": num_threads},
    }


def hb_mixer_request_body(harmonics: int = 205, num_threads: int | str = "auto") -> dict[str, Any]:
    """The JSON body of one close-tone diode mixer ``.hb`` request (numerics-bound: ~0.7 s native)."""

    return {
        "netlist_text": HB_CLOSE_TONE_MIXER,
        "mode": "hb",
        "options": {"num_threads": num_threads, "harmonics": harmonics, "max_iter": 100, "f_tol": 1e-6},
    }


def request_body(
    workload: str, *, num_threads: int | str = "auto", **params: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(body, description)`` of one request of ``workload`` (``"sram"`` or ``"hb_mixer"``)."""

    if workload == "sram":
        rows, cols, step = int(params.get("rows", 10)), int(params.get("cols", 10)), str(params.get("step", "0.25n"))
        request = sram_request_body(rows=rows, cols=cols, step=step, num_threads=num_threads)
        return request, {"circuit": f"SRAM {rows}x{cols} Level-1 MOS", "analysis": f".tran 6n {step}", "num_threads": num_threads}
    if workload == "hb_mixer":
        harmonics = int(params.get("harmonics", 205))
        request = hb_mixer_request_body(harmonics=harmonics, num_threads=num_threads)
        return request, {"circuit": "close-tone diode mixer", "analysis": f".hb H={harmonics}", "num_threads": num_threads}
    raise ValueError(f"unknown workload {workload!r} (expected one of {WORKLOADS})")


@dataclass
class Batch:
    """One batch of ``n`` concurrent requests."""

    n: int
    wall_s: float
    threads_used: list[int]
    elapsed_ms: list[float]


@dataclass
class Series:
    """All batches for one concurrency level."""

    n: int
    batches: list[Batch] = field(default_factory=list)

    @property
    def best_s(self) -> float:
        return min(batch.wall_s for batch in self.batches)

    @property
    def median_s(self) -> float:
        return statistics.median(batch.wall_s for batch in self.batches)

    def summary(self) -> dict[str, Any]:
        elapsed = [value for batch in self.batches for value in batch.elapsed_ms]
        return {
            "n": self.n,
            "best_s": self.best_s,
            "median_s": self.median_s,
            "wall_s": [batch.wall_s for batch in self.batches],
            "threads_used": sorted({value for batch in self.batches for value in batch.threads_used}),
            "request_elapsed_ms_median": statistics.median(elapsed) if elapsed else None,
            "batches": [asdict(batch) for batch in self.batches],
        }


def _load_average() -> list[float] | None:
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return [round(value, 2) for value in getloadavg()]
    except OSError:
        return None


async def _post(client: Any, body: dict[str, Any]) -> tuple[int, float]:
    response = await client.post("/api/simulate", json=body)
    if response.status_code != 200:
        raise RuntimeError(f"/api/simulate returned HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"/api/simulate returned status {payload.get('status')!r}")
    metadata = payload.get("metadata") or {}
    return int(metadata.get("threads_used", 0)), float(metadata.get("elapsed_ms", float("nan")))


async def run_batch(client: Any, body: dict[str, Any], n: int) -> Batch:
    """Fire ``n`` identical requests at once and time the whole batch."""

    started = time.perf_counter()
    results = await asyncio.gather(*(_post(client, body) for _ in range(n)))
    wall = time.perf_counter() - started
    return Batch(n=n, wall_s=wall, threads_used=[r[0] for r in results], elapsed_ms=[r[1] for r in results])


async def _measure_with_client(
    client: Any, body: dict[str, Any], n_values: Sequence[int], batches: int
) -> list[Series]:
    await _post(client, body)  # warm-up (untimed)
    series = [Series(n) for n in n_values]
    # Interleave the concurrency levels so slow drifts in machine load hit every level alike.
    for _ in range(batches):
        for entry in series:
            entry.batches.append(await run_batch(client, body, entry.n))
    return series


async def _measure_asgi(body: dict[str, Any], n_values: Sequence[int], batches: int) -> list[Series]:
    import httpx

    from app.server.main import app

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mna.test", timeout=REQUEST_TIMEOUT_S) as client:
            return await _measure_with_client(client, body, n_values, batches)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_uvicorn(port: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(command, cwd=REPO_ROOT, env=env)


async def _wait_for_health(client: Any, process: subprocess.Popen[bytes]) -> None:
    import httpx

    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"uvicorn exited with status {process.returncode} before it became healthy")
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError(f"uvicorn did not answer /health within {SERVER_START_TIMEOUT_S:g} s")


async def _measure_uvicorn(body: dict[str, Any], n_values: Sequence[int], batches: int) -> list[Series]:
    import httpx

    port = _free_port()
    process = _start_uvicorn(port)
    try:
        # One connection per concurrent request, so HTTP keep-alive never serializes them.
        limits = httpx.Limits(max_connections=max(n_values) + 1, max_keepalive_connections=max(n_values) + 1)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=REQUEST_TIMEOUT_S, limits=limits
        ) as client:
            await _wait_for_health(client, process)
            return await _measure_with_client(client, body, n_values, batches)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def measure(
    n_values: Sequence[int] = DEFAULT_N,
    batches: int = DEFAULT_BATCHES,
    *,
    server: str = "asgi",
    num_threads: int | str = "auto",
    rows: int = 10,
    cols: int = 10,
    step: str = "0.25n",
    workload: str = "sram",
    harmonics: int = 205,
) -> dict[str, Any]:
    """Run the batches and return the JSON-ready report (``ratios`` keyed by ``str(n)``)."""

    n_values = sorted({int(n) for n in n_values})
    if not n_values or n_values[0] < 1:
        raise ValueError("concurrency levels must be positive integers")
    if batches < 1:
        raise ValueError("batches must be >= 1")
    if 1 not in n_values:
        n_values = [1, *n_values]
    body, description = request_body(
        workload, num_threads=num_threads, rows=rows, cols=cols, step=step, harmonics=harmonics
    )
    runner = {"asgi": _measure_asgi, "uvicorn": _measure_uvicorn}.get(server)
    if runner is None:
        raise ValueError(f"unknown server {server!r} (expected 'asgi' or 'uvicorn')")
    load_before = _load_average()
    series = asyncio.run(runner(body, n_values, batches))
    one = next(entry for entry in series if entry.n == 1).best_s
    from mna_simulation.native.threads import budget_total

    return {
        "workload": {"name": workload, **description},
        "server": server,
        "batches": batches,
        "host": {
            "cpu_count": os.cpu_count(),
            "thread_budget_total": budget_total(),
            "python": sys.version.split()[0],
            "loadavg_before": load_before,
            "loadavg_after": _load_average(),
        },
        "series": [entry.summary() for entry in series],
        "ratios": {str(entry.n): entry.best_s / one for entry in series},
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"{report['workload']['circuit']} {report['workload']['analysis']}, num_threads={report['workload']['num_threads']!r}, "
        f"server={report['server']}, best of {report['batches']} batch(es)",
        f"cpus={report['host']['cpu_count']} thread budget={report['host']['thread_budget_total']} "
        f"load average before={report['host']['loadavg_before']} after={report['host']['loadavg_after']}",
        "",
        "| concurrent requests | best batch wall [s] | median batch wall [s] | ratio to 1 (best) | threads_used | request elapsed_ms (median) |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for entry in report["series"]:
        elapsed = entry["request_elapsed_ms_median"]
        lines.append(
            f"| {entry['n']} | {entry['best_s']:.3f} | {entry['median_s']:.3f} | {report['ratios'][str(entry['n'])]:.2f}x "
            f"| {', '.join(map(str, entry['threads_used']))} | {elapsed:.1f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N), help="concurrency levels (1 is always added)")
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES, help="batches per level; the best is reported")
    parser.add_argument("--server", choices=("asgi", "uvicorn"), default="asgi", help="in-process ASGI or a uvicorn subprocess")
    parser.add_argument("--num-threads", default="auto", help="num_threads request option ('auto' or a positive integer)")
    parser.add_argument("--step", default="0.25n", help="SRAM .tran step (.tran 6n <step>)")
    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        default="sram",
        help="sram: SRAM 10x10 .tran (GIL-bound Python glue, ~30 ms); hb_mixer: diode mixer .hb H=205 (numerics-bound, ~0.7 s)",
    )
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument(
        "--max-ratio",
        type=float,
        help=f"exit 1 when the ratio of the largest level exceeds this (the P6 criterion is {EXIT_CRITERION_RATIO})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    num_threads: int | str = args.num_threads
    if str(num_threads).strip().lower() != "auto":
        try:
            num_threads = int(num_threads)
        except ValueError:
            build_parser().error("--num-threads must be 'auto' or a positive integer")
    report = measure(
        args.n, args.batches, server=args.server, num_threads=num_threads, step=args.step, workload=args.workload
    )
    print(render(report))
    largest = str(max(entry["n"] for entry in report["series"]))
    ratio = report["ratios"][largest]
    verdict = "meets" if ratio <= EXIT_CRITERION_RATIO else "misses"
    print(f"\n{largest} concurrent requests: {ratio:.2f}x of one ({verdict} the P6 criterion <= {EXIT_CRITERION_RATIO}x)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.max_ratio is not None and ratio > args.max_ratio:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
