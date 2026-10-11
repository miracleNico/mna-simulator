"""Show how concurrent simulations interact with multithreaded OpenBLAS.

Runs ``N`` identical SRAM transients at the same time in separate processes
for several ``OPENBLAS_NUM_THREADS`` settings and prints the wall-clock time
of the whole batch. On a 4-core machine the default (one BLAS thread per
core, per process) collapses by more than an order of magnitude as soon as
two simulations overlap, which is exactly what happens when two API requests
or a request and a websocket stream run together.

Usage::

    python benchmarks/blas_contention.py            # N in (1, 2, 4), threads in (default, 1, 2)
    python benchmarks/blas_contention.py --n 2 8    # custom concurrency levels
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_WORKER = """
import os, sys
sys.path.insert(0, {benchmarks!r})
from workload_netlists import sram_netlist
from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions
from mna_simulation.core_basic import build_and_run_basic
from mna_simulation.netlist import parse_netlist_text
circuit = parse_netlist_text(sram_netlist(rows=10, cols=10, step="1n"))
build_and_run_basic(circuit, AnalysisOptions(mode=AnalysisMode.TRAN, tran_stop=6e-9, tran_step=1e-9, max_iter=200))
"""


def run_batch(concurrency: int, threads: str) -> float:
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    if threads != "default":
        env["OPENBLAS_NUM_THREADS"] = threads
    code = _WORKER.format(benchmarks=os.path.dirname(os.path.abspath(__file__)))
    start = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(concurrency)]
    for proc in procs:
        proc.wait()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 4], help="concurrency levels to test")
    parser.add_argument("--threads", nargs="+", default=["default", "1", "2"], help="OPENBLAS_NUM_THREADS values")
    args = parser.parse_args()
    print(f"cpus={os.cpu_count()}  (each run includes ~1 s of interpreter start-up and imports)")
    print(f"{'concurrent':>10s} {'OPENBLAS_NUM_THREADS':>22s} {'batch wall [s]':>15s}")
    for concurrency in args.n:
        for threads in args.threads:
            print(f"{concurrency:10d} {threads:>22s} {run_batch(concurrency, threads):15.2f}")


if __name__ == "__main__":
    main()
