"""Measure the linear-algebra alternatives discussed in PERFORMANCE_PLAN.md.

* Backward-Euler step on the 1035-unknown RLC mesh: dense LAPACK solve per step
  versus one cached SuperLU factorization plus a triangular solve per step.
* AC sweep parallelism: 32 complex solves run serially, on a 4-thread pool
  (BLAS pinned to one thread per worker) and on a 4-process pool.

    python -m pip install threadpoolctl
    python benchmarks/prototypes/sparse_lu_bench.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from workload_netlists import rlc_mesh_netlist  # noqa: E402

from mna_simulation.mna_builder import build_mna_problem  # noqa: E402
from mna_simulation.netlist import parse_netlist_text  # noqa: E402
from mna_simulation.solvers import solve_transient  # noqa: E402

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional
    threadpool_limits = None

PROBLEM = build_mna_problem(parse_netlist_text(rlc_mesh_netlist(23, 23)))
RNG = np.random.default_rng(0)
B_AC = RNG.standard_normal((PROBLEM.G.shape[0], 1)) + 0j


def _time(fn, reps: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps


def dense_solve(frequency: float) -> np.ndarray:
    system = PROBLEM.G + 1j * 2 * np.pi * frequency * PROBLEM.C
    if threadpool_limits is not None:
        with threadpool_limits(1):
            return np.linalg.solve(system, B_AC)
    return np.linalg.solve(system, B_AC)


def sparse_solve(frequency: float) -> np.ndarray:
    system = (PROBLEM.G_sparse + 1j * 2 * np.pi * frequency * PROBLEM.C_sparse).tocsc()
    return spla.splu(system).solve(B_AC)


def main() -> None:
    n = PROBLEM.G.shape[0]
    h = 1e-9
    rhs = RNG.standard_normal((n, 1))
    dense_operator = PROBLEM.G + PROBLEM.C / h
    sparse_operator = (PROBLEM.G_sparse + PROBLEM.C_sparse / h).tocsc()
    t_dense = _time(lambda: np.linalg.solve(dense_operator, rhs), 3)
    t_factor = _time(lambda: spla.splu(sparse_operator), 5)
    lu = spla.splu(sparse_operator)
    t_solve = _time(lambda: lu.solve(rhs), 50)
    steps = 40
    print(f"RLC mesh n={n}, nnz(G+C/h)={sparse_operator.nnz}, nnz(L+U)={lu.L.nnz + lu.U.nnz}")
    print(f"  dense LAPACK solve per step : {t_dense * 1e3:8.2f} ms")
    print(f"  SuperLU factor (once)       : {t_factor * 1e3:8.2f} ms")
    print(f"  SuperLU solve per step      : {t_solve * 1e3:8.3f} ms")
    print(f"  {steps} steps: dense {steps * t_dense:.3f} s vs cached LU {t_factor + steps * t_solve:.4f} s ({steps * t_dense / (t_factor + steps * t_solve):.0f}x)")
    start = time.perf_counter()
    solve_transient(PROBLEM, steps * h, h, use_krylov=True, krylov_stats=[])
    print(f"  current sparse Krylov (MINRES) path, {steps} steps: {time.perf_counter() - start:.3f} s")

    frequencies = np.linspace(1e6, 1e9, 32)
    print(f"\nAC sweep, {len(frequencies)} complex solves, n={n}")
    for label, fn in (("dense LAPACK", dense_solve), ("sparse SuperLU", sparse_solve)):
        start = time.perf_counter()
        for frequency in frequencies:
            fn(frequency)
        serial = time.perf_counter() - start
        with ThreadPoolExecutor(4) as pool:
            start = time.perf_counter()
            list(pool.map(fn, frequencies))
            threads = time.perf_counter() - start
        with ProcessPoolExecutor(4) as pool:
            start = time.perf_counter()
            list(pool.map(fn, frequencies))
            procs = time.perf_counter() - start
        print(f"  {label:15s}: serial {serial:.2f} s | 4 threads {threads:.2f} s ({serial / threads:.1f}x) | 4 processes {procs:.2f} s ({serial / procs:.1f}x)")


if __name__ == "__main__":
    main()
