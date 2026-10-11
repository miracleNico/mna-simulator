"""Compare Level-1 MOS evaluation kernels: Python loop, NumPy, C++ (pybind11), Numba.

Research prototype backing the C++ migration section of PERFORMANCE_PLAN.md.
It is not used by the package. Optional pieces are skipped when missing:

    python -m pip install pybind11 numba
    cd benchmarks/prototypes && g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 \
        $(python -m pybind11 --includes) mos_kernel.cpp \
        -o mos_kernel$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    python benchmarks/prototypes/kernel_bench.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from vec_mos import MosTable  # noqa: E402
from workload_netlists import sram_netlist  # noqa: E402

from mna_simulation.mna_builder import build_mna_problem  # noqa: E402
from mna_simulation.netlist import parse_netlist_text  # noqa: E402
from mna_simulation.solvers import _evaluate_level1_mos_devices  # noqa: E402


def _time(fn, reps: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps


def main() -> None:
    problem = build_mna_problem(parse_netlist_text(sram_netlist(rows=10, cols=10, step="1n")))
    n = problem.G.shape[0]
    devices = problem.level1_mos_devices
    table = MosTable(devices, n)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.2, size=n)
    f_ref, J_ref = _evaluate_level1_mos_devices(devices, n, x)
    args = (table.drain, table.gate, table.source, table.beta, table.vth, table.lam, table.sign, n)

    def coo_to_dense(vals: np.ndarray) -> np.ndarray:
        J = np.zeros((n, n))
        ordered = np.concatenate([vals[k::6] for k in range(6)])
        np.add.at(J.ravel(), table.flat, ordered[table.valid])
        return J

    print(f"{len(devices)} Level-1 MOS devices, n={n}")
    kernels: dict[str, object] = {"python loop (package)": lambda xv: _evaluate_level1_mos_devices(devices, n, xv)}
    f_vec, J_vec = table.evaluate(x)
    print(f"numpy vectorized matches: f={np.allclose(f_ref, f_vec)} J={np.allclose(J_ref, J_vec)}")
    kernels["numpy vectorized (dense J)"] = lambda xv: table.evaluate(xv)
    try:
        import mos_kernel

        f_c, v_c = mos_kernel.eval_mos_coo(x, *args)
        print(f"C++ COO kernel matches:   f={np.allclose(f_ref, f_c)} J={np.allclose(J_ref, coo_to_dense(v_c))}")
        kernels["C++ pybind11 COO (GIL released)"] = lambda xv: mos_kernel.eval_mos_coo(xv, *args)
    except ImportError:
        print("C++ kernel not built (see module docstring); skipping")
    try:
        from numba_mos import eval_mos_coo_numba, eval_mos_coo_numba_gil

        start = time.perf_counter()
        f_n, v_n = eval_mos_coo_numba(x, *args)
        print(f"numba COO kernel matches: f={np.allclose(f_ref, f_n)} J={np.allclose(J_ref, coo_to_dense(v_n))}  (JIT compile {time.perf_counter() - start:.2f}s)")
        eval_mos_coo_numba_gil(x, *args)
        kernels["numba COO nogil=True"] = lambda xv: eval_mos_coo_numba(xv, *args)
        kernels["numba COO (GIL held)"] = lambda xv: eval_mos_coo_numba_gil(xv, *args)
    except ImportError:
        print("numba not installed; skipping")

    print(f"\n{'kernel':34s} {'per call':>10s}")
    for name, fn in kernels.items():
        print(f"{name:34s} {_time(lambda: fn(x), 200) * 1e6:8.0f} us")

    xs = [rng.uniform(0.0, 1.2, size=n) for _ in range(400)]
    chunks = [xs[i::4] for i in range(4)]
    print(f"\nthread scaling: 400 evaluations as 4 chunks of 100 on a 4-thread pool")
    print(f"{'kernel':34s} {'serial':>9s} {'4 threads':>10s} {'speedup':>8s}")
    for name, fn in kernels.items():
        def run_chunk(chunk, fn=fn):
            for xv in chunk:
                fn(xv)
        start = time.perf_counter()
        for chunk in chunks:
            run_chunk(chunk)
        serial = time.perf_counter() - start
        with ThreadPoolExecutor(4) as pool:
            start = time.perf_counter()
            list(pool.map(run_chunk, chunks))
            threaded = time.perf_counter() - start
        print(f"{name:34s} {serial * 1e3:7.1f}ms {threaded * 1e3:8.1f}ms {serial / threaded:7.2f}x")

    print("\ncoarse-grained: each task is ONE native call over a (100, n) batch (GIL released once per call)")
    print(f"{'kernel':34s} {'serial':>9s} {'4 threads':>10s} {'speedup':>8s}")
    batches = [np.vstack(chunk) for chunk in chunks]
    coarse: dict[str, object] = {}
    try:
        import mos_kernel

        coarse["C++ batch call (GIL released)"] = lambda B: mos_kernel.eval_mos_coo_batch(B, *args[:-1], False)
        coarse["C++ batch call + OpenMP inside"] = lambda B: mos_kernel.eval_mos_coo_batch(B, *args[:-1], True)
        buffers = {id(B): (np.zeros((B.shape[0], n)), np.empty((B.shape[0], 6 * len(devices)))) for B in batches}
        coarse["C++ batch into preallocated bufs"] = lambda B: mos_kernel.eval_mos_coo_batch_into(B, *args[:-1], *buffers[id(B)], False)
    except ImportError:
        pass
    try:
        from numba_mos import eval_mos_coo_batch_numba

        eval_mos_coo_batch_numba(batches[0][:2], *args)
        coarse["numba batch call nogil=True"] = lambda B: eval_mos_coo_batch_numba(B, *args)
    except ImportError:
        pass
    for name, fn in coarse.items():
        fn(batches[0])
        start = time.perf_counter()
        for batch in batches:
            fn(batch)
        serial = time.perf_counter() - start
        with ThreadPoolExecutor(4) as pool:
            start = time.perf_counter()
            list(pool.map(fn, batches))
            threaded = time.perf_counter() - start
        print(f"{name:34s} {serial * 1e3:7.1f}ms {threaded * 1e3:8.1f}ms {serial / threaded:7.2f}x")


if __name__ == "__main__":
    main()
