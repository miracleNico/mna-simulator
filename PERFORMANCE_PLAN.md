# Performance Review, Multithreading Plan, and C++ Migration Research

Status: sections 1-9 are the proposal, based on measurements taken on 2026-10-11 against
commit `3754bb6` (v0.60), and are kept as written; the scripts that produced every number
live under `benchmarks/` so they can be re-run as the code evolves. Option C of section 7
was carried out as `CPP_MIGRATION_PLAN.md`: section 10 records what landed, the measured
speedups, the concurrent-request finding, the known limitations, and what is left.

## 1. Summary

- **The single biggest performance problem is not CPU count, it is OpenBLAS
  oversubscription.** NumPy and SciPy ship OpenBLAS with one thread per core.
  When two or more simulations overlap (two API requests, a request plus a
  `.dyn` websocket, or the test suite next to a benchmark) every small LAPACK
  solve becomes 20-700x slower. Four concurrent 10x10 SRAM transients take
  36 s instead of 1.7 s; the test suite takes 54 s instead of 11 s. The fix is
  cheap: pin BLAS to one thread per simulation and get parallelism from a
  process pool instead (section 5, tier 0).
- **Inside one simulation the time goes to Python glue, not to arithmetic.**
  Per-device Python loops, SymPy parsing of hundreds of `"0"` strings on every
  run, dense n x n allocations per Newton iteration, and dense LAPACK on
  matrices that are 99.6 % zeros account for 50-90 % of every measured
  workload. These are algorithmic and data-layout fixes that yield 10-100x
  without any threads: a cached sparse LU makes the 1035-unknown RLC step
  120x faster, and a structure-of-arrays device evaluator is 14x faster in
  plain NumPy and 60-300x faster in C++ or Numba (section 6).
- **Multithreading pays off in exactly three places:** independent requests
  (process pool), AC-sweep frequencies (3.3x measured on 4 cores for the dense
  1035-unknown case), and future batch/sweep features. A single transient on a
  1k-unknown circuit has almost no exploitable parallelism; do not expect
  threads to speed it up (section 5).
- **A C++ migration is feasible and the toolchain is in place, but it should be
  a kernel migration, not a rewrite.** The measured 60-300x on device
  evaluation is real, and only native code can hold a long GIL-free region.
  However dense LAPACK is already native, the fine-grained GIL hand-off costs
  more than a 10 us kernel saves, and the parser, schematic pipeline, API, and
  UI are not hot. Recommendation: do the Python-side fixes first (they define
  the array-based kernel boundary anyway), then move device evaluation,
  sparse assembly, KLU solves, and the transient/Newton driver behind the
  existing `CppBackend` seam if the targets in section 8 demand it (section 7).

## 2. Scope and method

Measured on a 4-core Linux container with Python 3.13.16 (standard GIL build),
NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, and the OpenBLAS 0.3.34 pthreads build
bundled with the NumPy/SciPy wheels (4 threads each). Timings are wall-clock
with nothing else running unless stated. Profiles use `cProfile`, which
inflates Python-heavy code roughly 2x, so shares are quoted, not absolute
profiled seconds.

Workloads (generators in `benchmarks/workload_netlists.py`, identical to the
test-suite and UI demo circuits):

| Workload | Unknowns | Nonlinear devices | What it stresses |
| --- | --- | --- | --- |
| SRAM 10x10 Level-1 MOS, `.tran 6n` at 1n and 0.25n | 248 | 622 MOS | Newton loop, device evaluation, small dense solves |
| RLC mesh 23x23, `.tran` 40 steps, dense and sparse-Krylov paths | 1035 | 0 | Linear transient, large sparse operator |
| RLC mesh 23x23, `.ac` 100 points | 1035 | 0 | Repeated complex solves |
| Three-stage BJT amplifier, `.tran 10m/5u` (2001 steps) | 14 | 4 BJT | Many small steps, per-step overhead |
| Close-tone diode mixer, `.hb 205` | 3 (1233 in HB) | 1 diode | Harmonic balance dense structure |

Reproduce with `python benchmarks/benchmark_workloads.py --profile`,
`python benchmarks/blas_contention.py`, and the scripts in
`benchmarks/prototypes/`.

## 3. Architecture review from a performance angle

The pipeline (`STRUCT.md`) is clean and well layered; the performance issues
are all inside the numerical core and its data layout.

**`mna_builder.py`.** Every stamp writes a dense `G`/`C` entry *and* a
`scipy.sparse.dok_matrix` entry (`add_entry`). `dok_matrix.__setitem__` costs
tens of microseconds per call in SciPy 1.18, so building the 1035-unknown mesh takes
0.17 s and the 248-unknown SRAM also 0.17 s, most of it in sparse index
validation. Dense `G` and `C` are always materialized (8.6 MB each at n=1035).
Nonlinear devices are kept as a list of dicts; sources and diodes are kept as
Python expression strings (`f_str`, `b_time_str`) with one string per unknown,
almost all of them `"0"`.

**`solvers.py`, symbolic path.** `compile_time_source_func` sympifies every
entry of `b_time_str`, including the `"0"` ones: 81 ms for the SRAM (248
entries, 4 distinct non-zero expressions) and 316 ms for the RLC mesh (1035
entries, one non-zero expression). It is called twice per Level-1 transient
(once in `core_basic.run_basic_analysis` for the operating point, once in
`solve_transient`), and nothing is cached between runs, so a `.dyn` websocket
reconnect or a repeated UI run pays it again. `compile_nl_functions` has the
same shape for `f_str`; it is skipped for Level-1 devices because
`solver_f_str` stays `"0"` for them, which is why the SRAM does not pay it.

**`solvers.py`, device evaluation.** `_evaluate_level1_mos_devices` and
`_evaluate_level1_bjt_devices` loop over devices in Python, call three nested
Python functions per device, index the state vector element by element, and
stamp into a freshly allocated dense `n x n` Jacobian. Cost is about 2.9 us per
MOS device per evaluation (1.8 ms for 622 devices) and about 20 us per BJT
device, dominated by `np.clip` and `np.exp` on scalars. Each Newton iteration
evaluates `f` at least twice (residual plus damping trials) and `J` once.

**`solvers.py`, linear algebra.** `solve_linear` always factors from scratch
with `numpy.linalg.solve`. For the linear transient the operator `G + C/h` is
constant whenever the step is constant, yet it is rebuilt densely
(`problem.G + C_h`, an n x n allocation) and re-factored every step. The
sparse Krylov path avoids the dense rebuild but classifies the matrix on every
call (`classify_krylov_matrix`, including a dense Cholesky for n <= 256) and
`_result_metadata` runs a dense symmetry check plus Cholesky on `G` just to
report metadata. On the 1035-unknown mesh MINRES needs about 9 iterations per
step; a cached SuperLU factorization needs 0.08 ms per step.

**Harmonic balance.** `Y_bar` is built with dense `np.kron`, so memory is
`(n(2H+1))^2`: 12 MB at n=3, H=205 but 3.4 GB at n=50. `create_fourier_matrices`
is a pure-Python double loop (190 ms at H=205). The nonlinear Jacobian is
assembled block by block in a Python `for r, c` loop with a dense K x K
product per block, and `f`/`J` are evaluated sample by sample (15 000 Python
calls for the mixer). This caps HB at a handful of unknowns regardless of
threads.

**Server (`app/server/main.py`).** `POST /api/simulate` is a synchronous
endpoint, so Starlette runs it in its worker thread pool (default capacity 40);
the websocket path uses `run_in_executor(None, ...)`. Concurrent requests
therefore already run concurrently, each with a 4-thread OpenBLAS, with no cap
on concurrency and no isolation. This is the configuration that produces the
collapse in section 4.2.

**Backends.** `PythonBackend.run` retries the whole build-and-solve for up to
nine `gmin` values on any exception; a convergence failure therefore costs up
to nine full runs. `CppBackend` is a placeholder with the right capability
flags, which is a good seam for section 7.

## 4. Measured baselines

### 4.1 Wall-clock and where the time goes (idle machine)

| Workload | Wall | Dominant costs (unprofiled estimates) |
| --- | ---: | --- |
| SRAM 10x10 `.op` | 0.05 s | device loop |
| SRAM 10x10 `.tran 6n/1n` (7 points) | 0.50 s | device loop ~45 %, SymPy compile x2 ~30 %, LAPACK ~5 % |
| SRAM 10x10 `.tran 6n/0.25n` (25 points, UI demo) | 0.70 s | device loop ~55 % (200 evaluations x 1.8 ms), SymPy ~23 %, LAPACK 62 x 1 ms ~9 %, glue ~13 % |
| RLC 23x23 `.tran` 40 steps, dense direct | 1.36 s | 41 dense solves x 15-17 ms ~50 %, SymPy 0.32 s ~23 %, dense `G + C/h` rebuilds ~10 % |
| RLC 23x23 `.tran` 40 steps, sparse Krylov (MINRES) | 0.50 s | SymPy 0.32 s ~63 %, matrix classification ~15 %, MINRES itself ~10 % |
| RLC 23x23 `.ac` 100 points, dense | 3.79 s | 100 complex dense solves x 34 ms ~90 % |
| BJT 3-stage `.tran` 2001 steps | 0.88 s | BJT device loop ~70 % (scalar `np.clip`/`np.exp`), LAPACK 2500 x 33 us ~9 % |
| BJT 3-stage `.ac` 400 points | 0.015 s | negligible (n=14) |
| HB mixer H=205 (dim 1233) | 2.39 s | Python HB loops ~45 %, 36 dense 1233x1233 solves x 28 ms ~40 %, Fourier matrices ~8 % |
| `pytest` (82 tests) | 11.2 s | HB tests 6.2 s |

### 4.2 OpenBLAS oversubscription (the contention collapse)

`benchmarks/blas_contention.py` runs N identical SRAM transients at once in
separate processes (each about 0.7 s of compute plus about 1 s of interpreter
start-up):

| Concurrent runs | `OPENBLAS_NUM_THREADS` default (4) | `=1` | `=2` |
| ---: | ---: | ---: | ---: |
| 1 | 1.97 s | 1.69 s | 1.62 s |
| 2 | 8.95 s | 1.59 s | 1.77 s |
| 4 | 36.3 s | 1.73 s | 3.19 s |

With one BLAS thread per process, four simultaneous simulations finish in the
time of one. With the default, throughput drops 21x. The mechanism is the
pthreads OpenBLAS busy-wait barrier: when any of the 4 threads is descheduled
by a competing process, the other three spin until it returns. The same thing
happens inside one process when the FastAPI thread pool runs two requests, and
it was visible in this review in two more ways: a 248x248 `np.linalg.solve`
measured 558 ms while the test suite ran in the background (1 ms alone), and
the suite itself took 54 s next to a benchmark versus 11 s alone.

On an idle machine the 4 BLAS threads buy little at these sizes: n=248 solve
0.77 ms (4 threads) vs 1.27 ms (1 thread); n=1035 17.6 ms vs 45 ms. Giving
that up in exchange for robust behaviour under load is clearly right.

### 4.3 Prototype measurements (`benchmarks/prototypes/`)

Level-1 MOS evaluation, 622 devices, n=248, one full `f` and Jacobian
(`kernel_bench.py`; results verified element-for-element against the package):

| Kernel | Per evaluation | Speedup |
| --- | ---: | ---: |
| Current Python loop (`_evaluate_level1_mos_devices`) | 1.8-2.4 ms | 1x |
| NumPy structure-of-arrays, dense J output (`vec_mos.py`) | 0.13-0.19 ms | 10-14x |
| C++ pybind11, COO triplet output, GIL released | 8-30 us | 60-230x |
| Numba `njit`, COO triplet output | 5-27 us | 70-350x |

The spread for the native kernels is allocation noise, not arithmetic: the
device math itself is about 10-15 ns per device. A dense `n x n` Jacobian
output (492 KB) costs more than the arithmetic at this size, which is itself an
argument for writing Jacobian values straight into a fixed CSR value array.

Linear algebra on the 1035-unknown RLC mesh (`sparse_lu_bench.py`):

| Operation | Time |
| --- | ---: |
| Dense LAPACK solve per Backward-Euler step | 14.8-17 ms |
| SuperLU (`scipy.sparse.linalg.splu`) factorization, once | 1.6 ms |
| SuperLU triangular solve per step | 0.083 ms |
| 40 steps: dense vs cached LU | 0.59 s vs 0.005 s (120x) |
| 32 complex AC solves, dense: serial / 4 threads / 4 processes | 3.1 s / 0.94 s (3.3x) / 1.0 s (3.1x) |
| 32 complex AC solves, sparse LU per frequency, serial | 0.10 s (31x over dense serial) |

Fixed overheads: SymPy compile 81 ms (SRAM) and 316 ms (RLC) per call; HB
Fourier matrices 190 ms at H=205; Numba JIT 0.8-0.9 s on first call per process.

Thread scaling of the kernels (4-thread `ThreadPoolExecutor`, same work split
four ways): every fine-grained variant, including the GIL-releasing C++ and
`nogil=True` Numba kernels, runs *slower* on four threads (0.3-0.9x). A
10-30 us native call is far below Python's 5 ms GIL switch interval, so
threads spend their time handing the GIL back and forth. Making each task one
native call over a batch of 100 state vectors and parallelizing *inside* the
call with OpenMP gives 4.4x (3.9 ms -> 0.9 ms) with a single Python thread.
Four Python threads each making one GIL-released batch call scale only when
the call does not allocate: 0.7-0.9x when every call allocates its 3 MB output
(the kernel's memory-map lock serializes the threads), 2.6-2.8x when each
thread writes into preallocated buffers. Nesting Python threads over calls that
also run OpenMP inside collapses to 0.1x, the same oversubscription effect as
the BLAS case in section 4.2.
The lesson for both the multithreading plan and the C++ research: GIL-free
regions must be coarse (a whole Newton step, a whole sweep, a whole batch) and
must not allocate per call; per-device or per-evaluation calls from Python
threads cannot scale.

## 5. Multithreading: what can and cannot be parallelized

### 5.1 Parallelism inventory

| Level | Independent work | Today | Potential on 4 cores | Mechanism |
| --- | --- | --- | --- | --- |
| Across requests (UI runs, `.dyn` streams, several users) | Whole simulations | Thread pool, 4-thread BLAS each, collapses under load | ~4x throughput, no latency loss | Process pool, BLAS pinned to 1 |
| Across runs of a sweep / corners / Monte Carlo | Whole simulations | Feature does not exist | ~4x | Same process pool, new batch endpoint |
| `.ac` sweep | One complex solve per frequency | Serial | 3.3x measured (dense n=1035); not worth it once sparse LU makes a point cost 3 ms | Threads (LAPACK releases the GIL) with BLAS=1, or processes |
| Harmonic balance | Device evaluation per time sample; block assembly | Serial Python loops | 2-4x after vectorization/native; structure fix matters more | Vectorize over samples first |
| Newton iteration | Device evaluation across devices | Serial Python loop | Only native (OpenMP/`prange`); 4.4x measured inside one call | C++/Numba kernel |
| Transient time stepping | None (sequential by nature) | Serial | ~1x | Only via faster per-step work |
| Sparse LU | Limited at n ~ 1e3 | Dense LAPACK | ~1x | Algorithmic switch to sparse LU is the win |

### 5.2 Why threads alone will not fix single-run latency

In every workload in section 4.1, the majority of time is Python bytecode
(device loops, SymPy, glue) that holds the GIL. `ThreadPoolExecutor` cannot
speed that up, and the measurements in 4.3 show that releasing the GIL around
short native calls makes it worse. The free-threaded CPython 3.13t/3.14t build
removes the GIL, but SymPy and parts of SciPy are not yet validated on it and
single-thread throughput is lower; it is worth re-evaluating in 2027, not
adopting now. Process-level parallelism sidesteps all of this and is the
recommended mechanism for everything except AC-sweep threads.

### 5.3 Planned modifications

**Tier 0: stop the bleeding (1-2 days, no numerical changes).**

1. Pin BLAS threads. Add a small runtime module that wraps every backend run
   in `threadpoolctl.threadpool_limits(limits=MNA_BLAS_THREADS)`, default 1,
   with an environment override. `threadpoolctl` is a 50 kB pure-Python
   dependency and works for OpenBLAS, MKL, and Accelerate alike. Also set
   `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` in the documented uvicorn
   start command as belt and braces.
2. Run simulations in a bounded `ProcessPoolExecutor(max_workers=os.cpu_count())`
   owned by the FastAPI app (initializer pins BLAS to one thread and warms the
   imports). Make `/api/simulate` an `async def` that awaits the pool; route the
   websocket solve through the same pool. Requests beyond the pool queue
   instead of oversubscribing. A thread pool plus a semaphore is the fallback
   if pickling results proves costly, but it keeps Python-heavy phases
   serialized by the GIL and buys only ~1.5x instead of ~4x.
3. Add `benchmarks/blas_contention.py` to the release checklist. Expected
   result: the 4-concurrent case drops from 36 s to about 2 s.

**Tier 1: remove fixed overheads and Python loops (1-2 weeks). This is also
the prerequisite for any native kernel, because it defines the array-based
data layout the kernel will consume.**

1. SymPy: skip `"0"` entries, cache lambdified callables keyed by the tuple of
   expression strings (per process, LRU), compile once per run instead of
   twice, and evaluate `b_time` for the entire time grid in one vectorized
   call. Expected: -0.16 s on the SRAM transient, -0.32 s on every RLC run (the
   sparse-Krylov RLC run becomes about 0.15 s).
2. Device tables as structure-of-arrays (`benchmarks/prototypes/vec_mos.py`
   is a drop-in, bit-identical prototype for Level-1 MOS; do the same for
   Level-1 BJT and diodes with vectorized `exp`/`clip`). Expected: 10-14x on
   device evaluation for circuits with hundreds of devices; SRAM demo from
   0.70 s to about 0.35 s before any other change. The four-device BJT
   amplifier gains less (roughly 0.88 s to 0.4-0.5 s) because at that size the
   per-call NumPy overhead dominates; the many-small-steps regime is where a
   native kernel matters most.
3. Builder emits COO triplets once and converts to CSR; dense `G`/`C` are
   materialized only when n <= ~300 or for `.show_matrix`. Precompute, per
   nonlinear device, the positions of its Jacobian entries in the CSR value
   array so an evaluation writes into a fixed buffer with no allocation.
   Removes the `dok_matrix` cost from `build_mna_problem` (0.17 s -> ms).
4. Linear-solver policy in `solve_linear`: for n above ~300 use
   `scipy.sparse.linalg.splu` (SuperLU, already a SciPy dependency); cache the
   factorization of `G + C/h` across constant-step linear transients (120x on
   the RLC mesh); build AC systems sparsely and factor per frequency (31x on
   the RLC AC sweep); classify a Krylov operator once per operator, not per
   solve; stop running dense Cholesky in `_result_metadata`. Keep dense LAPACK
   for small n where it is faster than any sparse code.
5. Harmonic balance: vectorize `create_fourier_matrices` (190 ms -> <1 ms);
   evaluate `f`/`J` for all K samples in one vectorized call; assemble the
   nonlinear Jacobian with `einsum` over blocks or, better, exploit that each
   block is `Gamma diag(g) Gamma_inv` and move to a matrix-free GMRES with a
   block-diagonal (per-harmonic) preconditioner, which also removes the dense
   `kron` memory wall. The first two items are small; the matrix-free solver
   is a separate project.

Expected single-run results after tier 1, estimated from the measured
component costs: SRAM demo 0.10-0.15 s (from 0.70), RLC transient 0.02-0.05 s
(from 1.36), RLC AC 0.3-0.4 s (from 3.8), BJT transient 0.3-0.5 s (from
0.88), HB mixer 1.2-1.5 s (from 2.4; the dense 1233x1233 solve remains until
the matrix-free HB lands).

**Tier 2: parallelism inside and around one simulation (1-2 weeks, after tier 1).**

1. AC sweep: parallelize over frequencies with a thread pool when the
   per-point cost exceeds ~5 ms (dense path, large n), with BLAS pinned to one
   thread per worker. LAPACK releases the GIL, hence the measured 3.3x. Below
   that threshold the sparse path is already fast and the pool overhead
   dominates.
2. Batch endpoint (`POST /api/simulate/batch`) for parameter sweeps, corners,
   and Monte Carlo over the same schematic: the most valuable new use of the
   process pool and nearly embarrassingly parallel. The UI currently issues
   one request at a time; a sweep tile is a natural follow-up feature.
3. `.dyn` streaming: solve in a worker process and stream frames as chunks of
   time steps are produced instead of solving everything and then replaying,
   so playback overlaps computation and long runs become interruptible.
4. HB device evaluation across samples in parallel only once the kernel is
   native (section 7); the vectorized NumPy version from tier 1 is the
   intermediate step.

**Tier 3: native kernels.** See section 7.

### 5.4 Things that are explicitly not worth doing

- Threading the Python device loop or the Newton iteration (GIL-bound).
- Multithreaded dense LAPACK for n < ~2000 (measured gain <= 2.5x on an idle
  machine, catastrophic under contention).
- Parallel sparse LU (SuperLU_MT and friends) at n ~ 1e3: factorization is
  1.6 ms; the win is caching it, not threading it.
- Replacing SciPy's Krylov kernels with hand-written ones: the pure-Python
  fallbacks in `solvers.py` are fine as fallbacks; the SciPy path already runs.

## 6. Algorithmic levers beyond threading (for completeness)

These are out of scope for this plan but they change the picture more than any
parallelization and should be weighed when prioritizing:

- Adaptive time stepping with a predictor (local truncation error control)
  typically cuts step counts 3-10x on circuits like the SRAM demo and is the
  standard way SPICE-class simulators earn their speed; `_transient_time_points`
  is fixed-step today.
- Trapezoidal or Gear-2 integration for accuracy per step.
- Newton warm starts from the predictor, limiting voltage steps per iteration
  for `exp` devices (instead of the generic halving search in
  `_damped_newton_step`), and `gmin` continuation inside one run instead of
  nine restarts.
- A proper KLU-style refactorization (same pivot order, no symbolic re-analysis)
  for Newton iterations; this is a C++ item (section 7).

## 7. C++ migration research

### 7.1 What a C++ core would and would not buy

Measured or directly implied by the measurements above:

| Area | Python today | C++ (or Numba) | Comment |
| --- | --- | --- | --- |
| Device evaluation + Jacobian stamping | 2.9 us/MOS, 20 us/BJT | 10-15 ns/MOS | 60-300x measured (section 4.3); the largest single win for nonlinear circuits |
| Sparse LU | SuperLU via SciPy, 1.6 ms factor at n=1035 | KLU: typically 2-5x faster on circuit matrices, plus cheap numeric refactorization with a fixed pivot order | Benefits every Newton iteration; SciPy exposes no refactor API |
| Dense LAPACK | Already native (OpenBLAS) | Same | No gain from C++; Eigen would even be slower unless linked to BLAS |
| Per-step/iteration glue, allocations | ~0.1-0.3 ms per step | ~us | Matters for the 2001-step BJT case and for `.dyn` |
| Multithreading | GIL-bound except LAPACK | OpenMP over devices (4.4x measured in one call), over AC frequencies, over HB samples; one GIL release per run | Only native code can hold a coarse GIL-free region |
| SymPy / behavioral sources | Flexible `FUNC` and `B` expressions | Needs an expression evaluator or precomputed vectors | The main feature-parity risk |
| Memory | Dense n x n everywhere | CSR + device arrays | Enables 1e4-1e5 unknowns |
| Packaging | pip install, pure Python | Compiled wheels per OS/Python version | The main operational cost |

Where native code does **not** help: the parser, schematic pipeline,
validation, JSON serialization, the MOR projections (small dense algebra), and
anything already inside LAPACK. A full rewrite would spend most of its effort
re-implementing code that is not on the critical path.

### 7.2 Options

| Option | Effort | Expected effect on the section 4 workloads | Risk / cost |
| --- | --- | --- | --- |
| **A. Python only: tiers 0-2** | 3-4 weeks | 5-30x on current demos; robust under load | None beyond ordinary refactoring; also defines the kernel boundary |
| **B. Numba kernels at the tier-1 array boundary** | +1 week after A | Device evaluation 60-300x, `nogil`/`prange` for device loops; no gain on sparse LU | +100 MB `llvmlite` dependency, 0.8-0.9 s JIT on first use per process (`cache=True` mitigates), tight NumPy version coupling |
| **C. C++ kernel extension behind `CppBackend`** | 6-10 weeks to parity on the current device set | Same kernel speedups as B plus KLU refactorization, OpenMP, deterministic memory, no runtime JIT; the only route to real-time `.dyn` or 1e5-device circuits | Build/CI/wheels for Linux, macOS, Windows; two implementations during transition; behavioral sources need a plan; harder debugging |
| **D. Full C++ core (parser to API)** | 4-6 months | No additional speed over C | Rewrites non-hot code, loses the Python/React velocity; not recommended |

### 7.3 Recommendation

1. Do option A now. Its tier-1 data layout (SoA device tables, CSR with
   precomputed stamp positions, precomputed source vectors) is exactly the
   `CompiledCircuit` contract a native kernel needs, so none of the work is
   wasted whichever way the later decision goes.
2. Decide between B and C after A against concrete targets. Choose B if the
   targets stay at <= ~5 k devices and interactive latency, and the team wants
   to avoid a compiled build. Choose C if any of these are targets: 50 k+
   devices, transients of 1e5+ steps, real-time `.dyn` at UI frame rates,
   distribution without an LLVM dependency, or OpenMP-level parallel device
   evaluation. The user's question was specifically about C++; the measured
   data say it is feasible and worthwhile for the kernel, and unnecessary for
   everything else.
3. Never option D.

### 7.4 Design sketch for the C++ kernel (option C)

**Boundary.** Python builds a `CompiledCircuit` of plain arrays: node/branch
counts; CSR row pointers and column indices for the union pattern of `G`, `C`,
and all device Jacobians; `G` and `C` values in that pattern; per-model device
tables (SoA: terminal indices, parameters, and the precomputed CSR positions of
each device's stamps); source vectors. Time-dependent sources are evaluated in
Python (vectorized over the time grid, SymPy stays where it is) and passed as a
`(n, steps)` array; behavioral `B` sources, which are rare, either stay on the
Python backend (capability flag) or get a small expression evaluator such as
`exprtk` later.

**Kernel API (first version).**

- `evaluate(x, out_f, out_jvals)`: all nonlinear models, OpenMP over devices
  with per-thread COO buffers reduced into the CSR values (or a precomputed
  coloring to stamp without atomics).
- `factor(jvals)` / `refactor(jvals)` / `solve(rhs)`: KLU with symbolic
  analysis cached per pattern; dense Eigen `PartialPivLU` fallback for n < ~200.
- `transient(circuit, x0, t_grid, sources, options) -> values`: Backward Euler
  now, trapezoidal/Gear-2 and adaptive stepping later; the whole loop runs with
  the GIL released and reports progress through a callback every N steps so
  `.dyn` can stream.
- `ac_sweep(circuit, frequencies, b_ac)`: complex KLU per frequency, OpenMP
  across frequencies.
- Harmonic balance stays in Python until its matrix-free formulation exists;
  the per-sample `evaluate` is already enough to remove the Python loop.

**Toolchain and libraries (verified available in this environment).**
g++ 13.3 and clang 18 with OpenMP, CMake 3.28, Ninja, Python 3.13 headers;
`pybind11` 3.1 and `nanobind` 3.1 install from pip (nanobind has lower call
overhead and smaller binaries; pybind11 is more widely known; either is fine);
`scikit-build-core` for the build backend and `cibuildwheel` for wheels. Eigen
3.4 (MPL-2.0, header-only) is fetchable; SuiteSparse is not installed as
headers here and its GitHub tarball was blocked by this container's proxy, so
it would be vendored into the repo or fetched in CI. Licensing: KLU and BTF are
LGPL-2.1+, AMD and COLAMD BSD-3, SuperLU BSD-3, Eigen MPL-2.0, pybind11 and
nanobind BSD-3. KLU is the industry choice for circuit matrices; if LGPL is a
concern, Eigen's `SparseLU` (slower, no refactor API) or SciPy's SuperLU from
the Python side are the alternatives.

**Phases.**

| Phase | Weeks | Deliverable | Exit criterion |
| --- | ---: | --- | --- |
| C0 scaffolding | 1 | `cpp/` with CMake + nanobind + scikit-build-core, CI matrix, `CppBackend` capability flags, golden-test harness running every test netlist through both backends | Extension builds on Linux/macOS/Windows CI; harness compares results |
| C1 device kernels | 2 | Level-1 MOS/BJT, diode, static stamps into CSR; OpenMP | Bit-level agreement with Python within 1e-12 relative on all test netlists; 60x+ on SRAM evaluation |
| C2 linear solve | 2 | KLU (vendored) with cached symbolic analysis, dense fallback | RLC step < 0.1 ms; SRAM Newton iteration < 0.3 ms |
| C3 drivers | 2 | Transient + Newton driver, AC sweep, progress callback | Section 8 targets met; `.dyn` streams from the kernel |
| C4 packaging | 1-2 | Wheels, automatic fallback to the Python backend when the extension is missing, docs | `pip install mna-simulation` works without a compiler |

**Validation.** Golden comparison against the Python backend on every test
netlist (all 82 tests run through both backends), the existing energy-convergence
test on random RLC meshes, and a per-release benchmark run of
`benchmarks/benchmark_workloads.py` with both backends.

**Risks and mitigations.** Numerical divergence from the Python path (match the
`exp` clipping and damping heuristics exactly, test at tight tolerance);
behavioral sources (capability flag and Python fallback); maintenance burden
for a small team (keep the kernel surface to the five calls above; keep MOR,
HB, parsing, and schematics in Python); contributor toolchain on Windows
(MSVC in CI, wheels for users); LGPL compliance for KLU (dynamic linking or
documented static-link obligations).

## 8. Targets and acceptance criteria

| Metric | Today | After tier 0 | After tier 1 | With native kernel |
| --- | ---: | ---: | ---: | ---: |
| 4 concurrent SRAM runs, batch wall | 36 s | <= 2.5 s | <= 1.5 s | <= 1.2 s |
| SRAM 10x10 `.tran 6n/0.25n` | 0.70 s | 0.70 s | <= 0.15 s | <= 0.03 s |
| RLC 23x23 `.tran` 40 steps (default path) | 1.36 s | 1.36 s | <= 0.05 s | <= 0.02 s |
| RLC 23x23 `.ac` 100 points | 3.8 s | 3.8 s | <= 0.4 s | <= 0.1 s |
| BJT 3-stage `.tran` 2001 steps | 0.88 s | 0.88 s | <= 0.5 s | <= 0.05 s |
| HB mixer H=205 | 2.4 s | 2.4 s | <= 1.5 s | <= 0.5 s (matrix-free) |
| `pytest` | 11 s | 11 s | <= 6 s | <= 4 s |

Every target is checked with `python benchmarks/benchmark_workloads.py` and
`python benchmarks/blas_contention.py` on an otherwise idle machine, and the
existing test suite must stay green at every step.

## 9. Suggested order of work

1. Tier 0 (BLAS pinning, process pool, benchmark scripts in CI).
2. Tier 1 items 1-2 (SymPy caching, SoA device tables): small, isolated, and
   they remove most of the single-run latency of the UI demos.
3. Tier 1 items 3-4 (CSR builder, sparse LU policy): touch `mna_builder.py`
   and `solve_linear`; land behind the existing `use_krylov`-style options
   first, then make them the default.
4. Tier 1 item 5 and tier 2 (HB vectorization, AC threads, batch endpoint,
   streaming `.dyn`).
5. Decide B versus C with the section 8 targets in hand; if C, start with C0
   and C1, which already deliver the largest measured win.

## 10. Status after the C++ migration (2026-10-11)

Option C of section 7 was chosen and carried out, with a wider scope than the five-call kernel
of 7.4: every analysis, the Krylov solvers, HB, and MOR run in the mandatory extension
`mna_simulation._core` (`CPP_MIGRATION_PLAN.md`, status at its top). The pure-Python core is
kept only as the test oracle `tests/reference/mna_reference`.

All numbers in this section are **shared machine, interim**: a 4-CPU Linux VM (Python 3.13.16,
NumPy 2.5.4, SciPy 1.18.1, GCC 13.3 Release build) that another worker was using at the same
time (load average 4-5 during the baseline run, 0.7-4 during the request measurements).
`benchmarks/results/baseline-shared-machine.md` / `.json` hold the full comparison
(`python benchmarks/compare_backends.py --threads 1,4`, one warm-up plus 7 repetitions, median,
BLAS pinned to one thread), measured at commit `04c46a3` (P5 and P6a, before P3b). An
idle-machine baseline will replace them; ratios are more stable than absolute times.

### 10.1 What landed, per phase

Reference = the frozen Python core on the same inputs; core = the native path at one thread
unless stated.

| Phase | What moved | Measured, reference -> core (speedup) |
| --- | --- | --- |
| P0 | frozen oracle, corpus, differential and edge-case tests, `cpp/` scaffold with vendored Eigen/KLU, CI, `klu_solve` | RLC 1035 direct solve 34.1 -> 0.88 ms (39x); SRAM 248 direct solve 1.13 -> 0.26 ms (4.4x) |
| P1 | expression bytecode VM, compiled circuit (union CSC pattern, SoA device tables), device kernels with deterministic two-phase assembly | SRAM 622-MOS device evaluation 2.57 -> 0.088 ms (29x); SRAM Jacobian assembly 3.09 -> 0.11 ms (28x); BJT stage 7.1x; one-diode mixer 2.2x |
| P2 | real and complex KLU, dense LU, GMRES/MINRES/CR/CG, damped Newton, `.op` | SRAM `.op` 52.0 -> 3.6 ms (14x); complex RLC direct solve 84.7 -> 1.06 ms (80x); Krylov on the RLC operator 3.4-3.8x (SciPy's kernels were already compiled) |
| P3 | Backward-Euler transient (factorization reuse), AC sweep parallel over frequencies, operating points, the 2.4.1 rulings | SRAM `.tran 6n 0.25n` 841 -> 6.8 ms (123x); BJT `.tran` 2000 steps 1494 -> 4.2 ms (354x); RLC `.tran` 40 steps 2224 -> 8.2 ms (272x), Krylov path 656 -> 13.4 ms (49x); RLC `.ac` 100 points 9592 -> 55 ms (174x) and 18.3 ms at 4 threads (3.0x scaling); BJT `.ac` 400 points 11.9x |
| P4 | harmonic balance with block-sparse `Y_bar` and dense device blocks | close-tone mixer H = 205 4427 -> 706 ms (6.3x; 693 ms at 4 threads); mixer H = 64 312 -> 33 ms (9.6x) |
| P5 | rational Krylov basis, projections, reduced AC/transient loops, TPWL training and run | MOR basis 232 -> 7.8 ms (30x); TPWL training 894 -> 6.8 ms (131x); reduced AC 336 -> 7.2 ms (47x); reduced transient 778 -> 9.7 ms (80x) |
| P6a | parser and builder fast path (still Python) | SRAM circuit build 173 -> 8.7 ms (20x); RLC build 147 -> 7.4 ms (20x); SRAM parse 11.4 -> 3.3 ms (3.5x) |
| P6 | thread budget, `num_threads`, UI control, metadata, CLI `--threads`, comparison harness and perf gate, determinism and leak tests | every perf-gate floor met on the interim baseline; every corpus analysis bitwise identical at 1 and 4 threads; RSS flat within 1 MB over 1000 SRAM `.op` and over 1000 SRAM `.tran` runs after the first 100 |
| P3b | junction limiting (pnjlim), gmin continuation in `.op`, node-to-ground gmin, BJT exponent extrapolation, damped HB Newton | convergence, not speed: the strategies engage only where the reference iteration fails or stalls, so runs that converged before are unchanged (not part of the interim baseline) |

Against the "with native kernel" column of section 8:

| Metric | Target | Measured (shared machine, interim) |
| --- | ---: | --- |
| SRAM 10x10 `.tran 6n/0.25n` | <= 0.03 s | 6.8 ms in the core; about 26 ms for the whole service request (10.2) |
| RLC 23x23 `.tran` 40 steps | <= 0.02 s | 8.2 ms |
| RLC 23x23 `.ac` 100 points | <= 0.1 s | 55 ms; 18 ms at 4 threads |
| BJT 3-stage `.tran` 2001 steps | <= 0.05 s | 4.2 ms |
| HB mixer H = 205 | <= 0.5 s (matrix-free) | 0.69-0.71 s with dense device blocks; matrix-free HB not done (10.4) |
| 4 concurrent SRAM runs, batch wall | <= 1.2 s | 0.12-0.18 s, but 3.8-5.0x the time of one run (10.2) |
| `pytest` | <= 4 s | not comparable: the default suite now also runs the differential corpus and the binding tests: 2909 passed and 62 skipped in 156 s after P3b |

### 10.2 Concurrent requests in one server process

The P6 exit criterion "four concurrent SRAM requests finish within 1.5x of one" is not met.
`benchmarks/concurrent_requests.py` sends N identical SRAM 10x10 `.tran 6n 0.25n` requests at
once and compares the best batch wall time with the best single request:

| Setup | 1 request | 4 concurrent | Ratio |
| --- | ---: | ---: | ---: |
| in-process ASGI, `num_threads="auto"`, best of 3 | 36 ms | 150 ms | 4.1x |
| in-process ASGI, `num_threads=1`, best of 5 | 30 ms | 138 ms | 4.6x |
| uvicorn subprocess (real HTTP), `"auto"`, best of 5 | 31 ms | 119 ms | 3.8x |
| `tests/perf/test_concurrent_requests.py` (ASGI, `"auto"`, best of 3), load 0.7 | 30 ms | 143 ms | 4.8x |
| the same test after P3b, load 1.2 | 37 ms | 154 ms | 4.2x |
| the script after P3b (ASGI, `"auto"`, best of 3), load 0.7 | 35 ms | 175 ms | 5.0x |

Where one SRAM `.tran` request spends its time (medians of 15 runs): parse 3.2 ms, MNA build
about 9-11 ms (Python, including the native circuit compile), the native `.op` and transient
(`solvers.solve_transient`) 3.3 ms, the device operating-point table 4.0 ms (622 Python dicts,
one native call per transistor), and response conversion about 7 ms; about 26 ms in total, of
which roughly an eighth runs without the GIL. Two mechanisms follow:

1. **The GIL.** `CPP_MIGRATION_PLAN.md` 2.5 chose a thread pool because "the heavy work no longer
   holds the GIL". For a circuit this size the native part is now the small part, so concurrent
   requests take turns at the Python part: with `num_threads=1` four requests overlap their
   native calls but each one's backend time (`elapsed_ms`) grows from 17 ms to 93 ms. The native transient alone
   overlaps (`solvers.solve_transient` four times in threads: 3.25 -> 6.6 ms, 2.0x on the loaded
   machine), and the same request in a 4-process pool runs 21.8 -> 47.6 ms (2.2x).
2. **The `auto` lease.** `ThreadBudget` gives an `auto` request that finds no active run
   `total // 1` threads, i.e. the whole budget, so simultaneous `auto` requests hold their leases
   one after another (every response reports `threads_used = 4`).

Options, roughly in order of effort: batch the operating-point table through the existing
`_core.operating_points` binding and build the response arrays without per-element Python work
(about 11 ms of the 26); give `auto` a smaller share when the circuit has no parallel work (below
the 2048-device, 4096-row, and 2048-unknown thresholds the threads are idle anyway); run
`/api/simulate` in a process pool (the tier-0 proposal of 5.3, measured 2.2x here, but the thread
budget would have to become machine-wide across processes and responses would be pickled); or
restate the criterion as the absolute batch time, which is 7-10x under the 1.2 s target of
section 8.

### 10.3 Known limitations

- **gmin in the builder versus the service.** Since P3b gmin stepping also stamps gmin from every
  node unknown to ground. The node stamps live in the compiled circuit's gmin pattern and reach
  `G_sparse` only through `problem_at_gmin` (`backends/python_backend.py`), which the service's
  gmin loop uses. `build_mna_problem(circuit, gmin > 0)` called directly still puts only the
  device-terminal gmin into `G`/`G_sparse`, while the native analyses of that problem form
  `G + gmin * stamps` from the compiled pattern, node stamps included; the fallback for a circuit
  whose native compile was deferred rebuilds without them. A caller of the builder therefore sees
  a `G` that differs from the one the native analyses use at the same gmin.
- **Concurrent requests** in one process serialize (10.2).
- **HB mixer time.** 0.69-0.71 s on the shared machine against the 0.5 s target of P4; it does
  not scale with threads (706 vs 693 ms) because its largest BTF diagonal block is a dense
  411 x 411 block, factored on one thread in each of the 38 Newton iterations.
- **Damped HB is off by default.** `hb_solve(damping="always")` cuts the mixer from 38 to 11
  Newton iterations but stops at a different iterate inside the 1e-6 tolerance, so the default is
  `"on_failure"` (damping only after the undamped reference iteration breaks down).
- **Small circuits gain nothing from threads** and pay for them where a unit forces them: each
  parallel region starts its own `std::thread` workers, so SRAM device evaluation takes 0.95 ms
  at 4 threads against 0.088 ms at 1 (the analyses avoid this through their thresholds).
- **Leftovers.** `solvers._nmos_level1_forward` has no callers; a comment in
  `api/service.py` above `core_backend` still says "Still the Python solvers"; the in-app Help
  (`app/web/src/lib/userGuide.ts`) still carries the pre-migration Threads paragraph and lacks the
  "What Changed With The Native Core" section of `USER_GUIDE.md`.

### 10.4 Follow-ups (P7 candidates)

- **FFT-based, matrix-free HB** (phase B of `CPP_MIGRATION_PLAN.md` 2.6): GMRES with a
  block-diagonal per-harmonic preconditioner instead of dense `K x K` device blocks. P3b removed
  the blocker: with damping the O(K^2) Fourier-sum Jacobian converges on the mixer.
- **Always-damped HB behind a retuned tolerance**: make `damping="always"` the default once the HB
  tolerance (or the oracle comparison of the `hb` corpus cases) is revisited, for the 3.5x fewer
  mixer iterations.
- **Parallel dense LU for the HB blocks**: a blocked, threaded factorization of the large dense
  BTF diagonal blocks (411 x 411 on the mixer), which today take one thread each.
- **Lazy `f_str` in the builder**: the Level-1 device terms of the display vector `f_str` are
  rendered on every build although only `.show_matrix` reads them (the solvers use
  `solver_f_str` and the compiled device tables); in a cProfile of five SRAM requests the 3110 MOS
  template renders took 17 of the 123 ms spent in the builder.
- **Persistent thread pool**: reuse worker threads across calls to remove the per-call thread
  start, then lower the parallel thresholds (2048 devices, 4096 SpMV rows, 2048 MOR unknowns) so
  mid-size circuits use their threads; the chunking by item count keeps results deterministic.
- **Request overheads** from 10.2: the operating-point table through `_core.operating_points`,
  leaner response conversion, a different `auto` share, or a process pool.
- From the original P7 list: dense-vs-KLU threshold tuning (`kDenseMaxN` is 0, so every circuit
  solve uses KLU) and adaptive time stepping (section 6).
