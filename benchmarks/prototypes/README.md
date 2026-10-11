# Performance prototypes

Research artifacts behind `PERFORMANCE_PLAN.md`. Nothing here is imported by
the `mna_simulation` package; the scripts exist so the numbers in the plan can
be reproduced and re-checked as the code evolves.

| File | Purpose |
| --- | --- |
| `vec_mos.py` | Structure-of-arrays, vectorized NumPy version of the Level-1 MOS evaluator (`MosTable`). |
| `mos_kernel.cpp` | pybind11 C++ kernel for the same device model, dense-Jacobian and COO-triplet variants, GIL released inside the loop. |
| `numba_mos.py` | Numba `njit` versions of the same loop, with and without `nogil=True`. |
| `kernel_bench.py` | Correctness check plus per-call timing and 4-thread scaling of all kernels on the 10x10 SRAM (622 devices). |
| `sparse_lu_bench.py` | Cached SuperLU factorization versus dense LAPACK per Backward-Euler step, and AC-sweep parallelism on the 1035-unknown RLC mesh. |

Build the C++ kernel in place (needs a C++17 compiler with OpenMP and the Python headers):

```bash
python -m pip install pybind11 numba threadpoolctl
cd benchmarks/prototypes
g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 $(python -m pybind11 --includes) mos_kernel.cpp \
    -o mos_kernel$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cd ../..
python benchmarks/prototypes/kernel_bench.py
python benchmarks/prototypes/sparse_lu_bench.py
```

Run the benchmarks with nothing else competing for the CPU; see
`benchmarks/blas_contention.py` for why.
