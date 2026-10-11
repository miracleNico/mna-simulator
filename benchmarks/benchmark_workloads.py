"""Time the representative analysis workloads described in PERFORMANCE_PLAN.md.

Usage::

    python benchmarks/benchmark_workloads.py            # wall-clock table
    python benchmarks/benchmark_workloads.py --profile  # also print cProfile hot spots

Run it with nothing else competing for the CPU: multithreaded OpenBLAS calls
slow down dramatically under contention (see ``blas_contention.py``).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload_netlists import BJT_THREE_STAGE, HB_CLOSE_TONE_MIXER, rlc_mesh_netlist, sram_netlist  # noqa: E402

from mna_simulation.api.contracts import AnalysisMode, AnalysisOptions  # noqa: E402
from mna_simulation.core_advanced import run_advanced_analysis  # noqa: E402
from mna_simulation.core_basic import run_basic_analysis  # noqa: E402
from mna_simulation.mna_builder import build_mna_problem  # noqa: E402
from mna_simulation.netlist import parse_netlist_text  # noqa: E402


def _environment() -> str:
    parts = [f"python {sys.version.split()[0]}", f"cpus {os.cpu_count()}"]
    try:
        import numpy

        parts.append(f"numpy {numpy.__version__}")
        import scipy

        parts.append(f"scipy {scipy.__version__}")
    except Exception:  # noqa: BLE001
        pass
    try:
        from threadpoolctl import threadpool_info

        pools = {(p.get("internal_api"), p.get("num_threads")) for p in threadpool_info()}
        parts.append("blas " + ", ".join(f"{api}x{n}" for api, n in sorted(pools, key=str)))
    except Exception:  # noqa: BLE001
        parts.append("blas threads unknown (pip install threadpoolctl)")
    return "  ".join(parts)


def _workloads() -> list[tuple[str, Callable[[], object]]]:
    sram = build_mna_problem(parse_netlist_text(sram_netlist(rows=10, cols=10, step="1n")))
    rlc = build_mna_problem(parse_netlist_text(rlc_mesh_netlist(23, 23)))
    bjt = build_mna_problem(parse_netlist_text(BJT_THREE_STAGE))
    mixer = build_mna_problem(parse_netlist_text(HB_CLOSE_TONE_MIXER))
    tran = AnalysisMode.TRAN
    return [
        (f"SRAM 10x10 .op (n={sram.G.shape[0]}, {len(sram.level1_mos_devices)} MOS)", lambda: run_basic_analysis(sram, AnalysisOptions(mode=AnalysisMode.OP, max_iter=200))),
        ("SRAM 10x10 .tran 6n/1n (7 pts)", lambda: run_basic_analysis(sram, AnalysisOptions(mode=tran, tran_stop=6e-9, tran_step=1e-9, max_iter=200))),
        ("SRAM 10x10 .tran 6n/0.25n (25 pts, UI demo)", lambda: run_basic_analysis(sram, AnalysisOptions(mode=tran, tran_stop=6e-9, tran_step=0.25e-9, max_iter=200))),
        (f"RLC 23x23 .tran 40 steps dense direct (n={rlc.G.shape[0]})", lambda: run_basic_analysis(rlc, AnalysisOptions(mode=tran, tran_stop=40e-9, tran_step=1e-9))),
        ("RLC 23x23 .tran 40 steps sparse Krylov", lambda: run_basic_analysis(rlc, AnalysisOptions(mode=tran, tran_stop=40e-9, tran_step=1e-9, use_krylov=True))),
        ("RLC 23x23 .ac 100 pts dense direct", lambda: run_basic_analysis(rlc, AnalysisOptions(mode=AnalysisMode.AC, ac_start=1e6, ac_stop=1e9, ac_points=100))),
        (f"BJT 3-stage .tran 10m/5u (2001 pts, n={bjt.G.shape[0]})", lambda: run_basic_analysis(bjt, AnalysisOptions(mode=tran, tran_stop=10e-3, tran_step=5e-6))),
        ("BJT 3-stage .ac 400 pts (Level-1 op + sweep)", lambda: run_basic_analysis(bjt, AnalysisOptions(mode=AnalysisMode.AC, ac_start=10, ac_stop=1e6, ac_points=400))),
        (f"HB close-tone mixer H=205 (n={mixer.G.shape[0]}, dim={mixer.G.shape[0] * 411})", lambda: run_advanced_analysis(mixer, AnalysisOptions(mode=AnalysisMode.HB, hb_harmonics=205, max_iter=100, f_tol=1e-6))),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", action="store_true", help="print the top cProfile entries per workload")
    parser.add_argument("--top", type=int, default=10, help="number of profile rows to print")
    args = parser.parse_args()

    print(_environment())
    print(f"{'workload':62s} {'wall [s]':>9s}")
    for label, fn in _workloads():
        fn()  # warm up (SymPy/lambdify caches, BLAS thread pool)
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        print(f"{label:62s} {elapsed:9.3f}")
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            fn()
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(args.top)
            rows = [line for line in stream.getvalue().splitlines() if line.strip() and ("ncalls" in line or "(" in line)]
            print("\n".join("    " + row[:140] for row in rows[: args.top + 1]))


if __name__ == "__main__":
    main()
