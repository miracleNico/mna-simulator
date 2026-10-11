# C++ Core Migration Plan

Status: approved scope, execution in progress on branch `claude/trusting-hopper-lnnt8g`.
Companion documents: `PERFORMANCE_PLAN.md` (measurements that motivate this
migration), `tests/EDGE_CASES.md` (behaviour specification of the current core),
`cpp/third_party/README.md` (vendored components and licenses).

## 1. Decisions (binding)

| Topic | Decision |
| --- | --- |
| Scope | `.op`, `.tran`, `.ac`, `.hb`, all Krylov solvers, model-order reduction (linear Krylov basis, TPWL training), HB reconstruction, and any other compute-heavy module move to C++. Parser, schematic pipeline, FastAPI app, CLI, and the React app stay in Python/TypeScript. |
| Extension | Mandatory. Once a unit reaches parity its Python implementation leaves the runtime package. There is no Python numerical fallback. |
| Sparse direct solver | SuiteSparse KLU (with AMD, COLAMD, BTF, SuiteSparse_config) vendored and statically linked. Eigen for dense linear algebra, complex arithmetic, and SVD. |
| Test oracle | Today's Python numerical core is frozen as `tests/reference/mna_reference` (test-only, not shipped) and used for differential tests forever, including randomized circuits. |
| Platforms | Linux and Windows (the web app is used on Windows). CI builds and tests both; wheels for CPython 3.12+ (stable ABI). |
| Thread control | Explicit end-user control: a `num_threads` request option, a UI control in the simulation panel, and `threads_used` in result metadata. The service enforces a machine-wide budget (`MNA_NUM_THREADS`, default = CPU count) so concurrent requests never oversubscribe; "auto" lets the service pick. |
| Performance comparison | `benchmarks/compare_backends.py` (frozen Python vs C++ at 1, 2, 4, N threads, per migrated unit) plus a `perf`-marked pytest gate with generous floors, skipped unless `--run-perf`. |
| Bindings and build | nanobind (stable ABI, static libnanobind), scikit-build-core, CMake >= 3.26, C++17, OpenMP (MSVC `/openmp` is OpenMP 2.0, so loops use signed indices and per-call `num_threads(k)` clauses). |
| Orchestration | Coordinator and auditor reviews every change against the differential suite before it is committed; workers implement one unit at a time behind existing facades (strangler pattern). |

## 2. Target architecture

```
Python                                   C++ (mna_simulation._core, nanobind)
---------------------------------------  --------------------------------------------
netlist.py / schematic_pipeline.py       (unchanged)
mna_builder.py  -> flat device tables -> build_circuit()  -> Circuit (CSR pattern, SoA
                                                             device tables, source tables,
                                                             compiled expressions)
core_basic.py   -> thin dispatch      -> op(), transient(), ac_sweep()
core_advanced.py-> thin dispatch      -> hb_solve(), hb_reconstruct()
mor/*.py        -> cache + metadata   -> mor_linear_basis(), mor_project(), tpwl_train(), tpwl_run()
solvers.py      -> facades only       -> krylov_solve(), linear_solve(), newton()
runtime.py      -> thread budget      -> every call takes num_threads; GIL released for the
                                         whole call; results returned as NumPy arrays
api/service.py, app/server, cli.py       (unchanged contracts; new options/metadata)
```

### 2.1 Data contract: `Circuit`

Python keeps everything that involves strings: node naming and ordering
(`_build_index_map`), labels, error messages, and the `f_str`/`b_time_str`
display strings used by `.show_matrix`. Python converts `CircuitIR` into flat
integer/float tables, one per device class, and calls `build_circuit`:

| Table | Columns |
| --- | --- |
| `static_g`, `static_c` | COO triplets (row, col, value) for R, C, L, controlled sources, small-signal transistors, gmin stamps, voltage-source branch rows |
| `b_dc`, `b_ac` | dense vectors (`b_ac` complex) |
| `time_sources` | (row, sign, expression id) for SIN/COS/STEP/FUNC contributions |
| `expressions` | compiled bytecode programs (see 2.2) for time sources and behavioral sources, each with its derivative programs |
| `diodes` | (anode, cathode, Is) |
| `mos_level1` | (drain, gate, source, beta, vth, lambda, is_pmos) |
| `bjt_level1` | (collector, base, emitter, is, bf, br, vaf, var, is_pnp) |
| `behavioral` | (node_p, node_n, expression id, list of referenced unknowns) |

Ground is index -1 everywhere. `build_circuit` forms the union CSR pattern of
the static matrices and every device Jacobian footprint, precomputes for each
device the CSR slot of each of its Jacobian entries, and returns a `Circuit`
handle exposing `n`, `nnz`, `pattern()`, `to_dense(which)`, `metadata()`. The
Python `MnaProblem` keeps its public fields (dense `G`/`C` only when
n <= 300 or on demand, SciPy CSR views of the pattern, `b_dc`, `b_ac`,
`index_map`, `components`, device lists, `metadata`) and gains `native`
(the handle). Existing tests that inspect `G`, `C`, `b_ac`, `index_map`
keep working.

### 2.2 Expression sources without a Python fallback

FUNC time sources and `B` behavioral sources are arbitrary expressions today
(SymPy `sympify` + `lambdify`). SymPy stays as a compile-time tool on the
Python side: `mna_simulation/exprc.py` sympifies the expression with the same
locals as `solvers._sympy_locals`, differentiates behavioral expressions with
respect to each referenced unknown, and lowers both to a postfix bytecode with
the operator set the current core accepts: `+ - * / **`, unary minus, `sin cos
exp sqrt log abs`, `Heaviside(x, h0)`, `Piecewise`, `Mod`, `Min`, `Max`,
comparisons, `And/Or/Not`, constants, symbols `t` and `x_i`. The C++ evaluator
(`cpp/src/expr/`) executes the bytecode with a small value stack. Parity rules:
`Heaviside(0, h0) = h0` (the current core passes `1`), `Piecewise` takes the
first true branch, `Mod` follows SymPy sign semantics, division by zero yields
inf as NumPy does. Time-source programs are evaluated once per time point for
the whole grid (vectorized in C++), behavioral programs inside every Newton
evaluation. Diodes stop being expressions and become a native model
(Shockley with `V_T = 0.025`, no clipping, exactly as today).

### 2.3 Kernel API (nanobind surface)

All entry points release the GIL for the whole native region and take
`num_threads`. Arrays cross as NumPy (float64/complex128/int64), results come
back as plain structs of arrays.

| Call | Purpose |
| --- | --- |
| `build_circuit(tables, gmin) -> Circuit` | 2.1 |
| `Circuit.to_dense(name)` | dense `G`, `C`, `J(x)` for `.show_matrix`, `.op` response matrices, and tests |
| `evaluate(circuit, x, t, out_f, out_jvals, num_threads)` | nonlinear currents and Jacobian values into caller buffers (device-parallel, deterministic) |
| `LinearSolver(pattern, kind="klu"|"dense"|"complex_klu")` with `factor(values)`, `refactor(values)`, `solve(rhs)` | KLU symbolic analysis cached per pattern; numeric refactor per Newton iteration; dense Eigen `PartialPivLU` for n <= 64 |
| `krylov_solve(A_csr, b, method, rank, tol, max_iter, x0, num_threads) -> (x, info)` | GMRES(restart), MINRES/CR, CG, matrix classification and the same `info` keys as `solvers.krylov_solve_linear` |
| `newton_op(circuit, b, options) -> (x, stats)` | damped Newton with the current heuristics (2.4) |
| `transient(circuit, x0, t_grid, options, progress) -> (values, stats)` | Backward Euler; cached factorization for linear constant-step runs; optional progress callback every N steps for `.dyn` streaming |
| `ac_sweep(circuit, x_op, frequencies, options) -> (magnitudes, stats)` | linearize at the operating point, complex KLU per frequency, frequency-parallel |
| `hb_solve(circuit, omega0, H, options) -> (x_bar, stats)` and `hb_reconstruct(x_bar, n, H, omega0, t)` | 2.6 |
| `mor_linear_basis(...)`, `mor_project(...)`, `mor_reduced_ac/transient(...)`, `tpwl_train(...)`, `tpwl_run(...)` | 2.7 |
| `version()`, `build_info()` | diagnostics (compiler, OpenMP, KLU, Eigen versions) |

Errors are raised as `mna_simulation._core.CoreError` with a `.code`
attribute using the existing code set (`NETLIST_FATAL`, `NR_CONVERGENCE`,
`JACOBIAN_SINGULAR`, `TRAN_ERROR`, `AC_ERROR`, `HB_ERROR`, `MOR_CONFIG`);
`mna_simulation.errors.error_handler` maps them to the existing Python
exception classes so `pytest.raises(SolverConvergenceError)` keeps working.

### 2.4 Semantics preserved (parity rules)

- Newton: residual `F = G x + f(x) - b`, Jacobian `G + J(x)`, damping by halving
  up to 28 trials accepting the first scale that does not increase the finite
  residual norm, otherwise the best trial; convergence when `||F|| < f_tol` or
  `||dx|| < v_tol`; `NR_CONVERGENCE` when no scale reduces the residual or
  `max_iter` is exhausted; `JACOBIAN_SINGULAR` on a singular factorization.
- Gmin stepping stays in `PythonBackend.run` (rebuild and retry over
  `DEFAULT_GMIN_STEPS` on any exception); `build_circuit` is cheap enough.
- Time grid: `_transient_time_points` semantics (fixed step, snap to `t_stop`
  with the 1e-9 relative tolerance, `TRAN_ERROR` codes).
- Level-1 MOS/BJT equations, `vt = 0.02585`, exp clipping at +/-80 for BJT,
  `V_T = 0.025` and no clipping for diodes, PMOS/PNP sign conventions.
- Krylov auto-rank `ceil(0.5 n)`, method aliases, classification rule (dense
  Cholesky attempt only when n <= 256 on the sparse path), fallback to the
  direct solver when not converged, and the metadata fields consumed by tests.
- HB base frequency (`find_gcd_frequency`), harmonic expansion rule, Fourier
  matrix conventions, reconstruction grid of 2000 points.
- Output layout label-major `(n_labels, n_points)`; `dc_solution` flat.

Behaviour changes are allowed only when listed in `tests/EDGE_CASES.md` with a
decision; candidates already known: `R = 0` raises `NETLIST_FATAL` instead of
`ZeroDivisionError`; `TRAN_ERROR` maps to `SolverConvergenceError` instead of
the base class. Each change updates the differential expectation explicitly.

### 2.5 Multithreading design

- Mechanism: OpenMP with an explicit `num_threads(k)` clause on every parallel
  region, k from the call argument; never `omp_set_num_threads` or other global
  state, so concurrent Python callers do not interfere (verified).
- Device evaluation: devices are partitioned statically across threads; each
  thread writes its devices' Jacobian values into a per-device triplet buffer
  (fixed slots, no atomics); a final pass sums triplets into CSR slots in
  device order. Result is bitwise identical for any thread count.
- AC sweep and HB sample evaluation: one output slot per frequency/sample,
  trivially deterministic; each thread owns its own KLU numeric object.
- Transient stepping is sequential; parallelism inside a step is device
  evaluation only. Linear constant-step transients reuse one factorization.
- MOR: shift systems factor in parallel; Gram-Schmidt is sequential.
- GIL: released once per kernel call. Python-side concurrency across requests
  uses a `ThreadPoolExecutor` (no pickling) because the heavy work no longer
  holds the GIL; BLAS is pinned to one thread with `threadpoolctl` for the few
  remaining NumPy operations.
- Budget: `mna_simulation/runtime.py` holds `ThreadBudget(total=MNA_NUM_THREADS
  or cpu_count)`; a run acquires `min(requested, available)` tokens (at least
  1) and releases them at the end; `num_threads="auto"` means "as many as are
  free". Metadata reports `threads_requested`, `threads_used`, `core_backend`,
  `core_version`, and per-phase timings.
- UI: a "Threads" control (auto, 1..N where N comes from `/health`) in the
  simulation panel; the request carries `num_threads`; the result tile shows
  `threads_used` and wall time.
- Determinism tests: every analysis on every corpus case at `num_threads` 1
  and `max(2, cpu_count)` must be `np.array_equal`.

### 2.6 Harmonic balance

Phase A (parity): same algorithm as `solve_harmonic_balance` with the dense
Kronecker products replaced by block-structured assembly: `Y_bar` as a sparse
matrix of `K x K` blocks (`G_ij I + C_ij Omega`), the nonlinear Jacobian as
dense `K x K` blocks `Gamma diag(g_ij(t)) Gamma_inv` only for nonzero `(i, j)`,
solved with KLU on the block-sparse matrix. Device evaluation at the K samples
is one batched, parallel call. Memory is `nnz(G + J) * K^2` doubles instead of
`(nK)^2`. Phase B (optional, after parity): FFT-based Gamma products and a
matrix-free GMRES with block-diagonal preconditioner.

### 2.7 Model-order reduction

`mor/linear.py` keeps option handling, metadata, and validation; the basis
construction (shifted solves through real/complex KLU, block Gram-Schmidt
with the same tolerances), projections, and the reduced AC/transient loops move
to C++. `mor/tpwl.py` keeps the LRU cache (keyed by the existing digest) and
stores the C++-produced basis and local models as NumPy arrays; training (full
transient, `BDCSVD`, local Jacobians) and the reduced run move to C++.

## 3. Repository layout after the migration

```
cpp/CMakeLists.txt            targets: klu_vendored, mna_core, _core, mna_core_tests
cpp/include/mna/...           public headers (circuit, devices, expr, linear, krylov, analyses, threads, error)
cpp/src/...                   implementation, one directory per unit
cpp/bindings/module.cpp       nanobind module
cpp/tests/...                 doctest unit tests (ctest)
cpp/third_party/              eigen/, suitesparse/{SuiteSparse_config,AMD,COLAMD,BTF,KLU}, doctest/, LICENSES/
mna_simulation/_core*.so|pyd  built extension (mandatory)
mna_simulation/native.py      import guard and build info
mna_simulation/exprc.py       SymPy -> bytecode compiler
mna_simulation/runtime.py     thread budget, BLAS pinning
mna_simulation/{mna_builder,solvers,core_basic,core_advanced,mor}/  facades over _core
tests/reference/mna_reference frozen oracle (not shipped)
tests/corpus.py               differential corpus and random generators
tests/differential/           parity tests and comparison helpers
tests/test_edge_cases.py      behaviour spec
tests/perf/                   perf gate (marker perf)
benchmarks/compare_backends.py, benchmarks/results/
.github/workflows/ci.yml, wheels.yml
```

## 4. Test harness

| Layer | Location | What it guarantees |
| --- | --- | --- |
| Frozen oracle | `tests/reference/mna_reference` | an unchanging implementation of every analysis to compare against |
| Differential parity | `tests/differential/test_reference_parity.py` over `tests/corpus.py` | every corpus case x analysis gives the same status, labels, values (tolerances per class: linear 1e-9/1e-12, nonlinear 1e-6/1e-9, HB 1e-6/1e-9, MOR 5e-2/1e-6), contractual metadata, and the same exception class and code on error cases; randomized generators with fixed seeds cover shapes the hand-written tests miss |
| Edge-case spec | `tests/test_edge_cases.py` + `tests/EDGE_CASES.md` | the catalogue in section 4.1 asserted against the current core, with each deliberate change recorded |
| C++ unit tests | `cpp/tests/*.cpp` (doctest, ctest) | device models at region boundaries, KLU wrapper (singular, refactor, 1x1, n=0), Krylov methods on known matrices, Newton damping cases, time grid, expression evaluator against SymPy-generated vectors, `parallel_for` coverage and determinism |
| Binding unit tests | `tests/test_native_*.py` | each exported call: shapes, dtypes, invalid inputs (wrong shapes, NaN, negative steps), error codes, GIL release (concurrent calls overlap), thread determinism |
| Existing suite | `tests/test_*.py` (82 tests) | unchanged behaviour of parser, builder, service, MOR, HB, SRAM; they keep calling the public functions, which become facades |
| Determinism | `tests/test_threads_determinism.py` | bitwise equality across thread counts |
| Resource tests | `tests/test_resources.py` | 2000 small runs without RSS growth (Linux), no handle leaks across KLU objects |
| Sanitizers and coverage | CI jobs | ASan/UBSan clean on ctest and on a pytest subset; line coverage >= 85 % for `cpp/src` (gcovr) and `mna_simulation` (pytest-cov) once facades are thin |

### 4.1 Edge-case catalogue

Empty netlist; single node; floating node / no DC path with gmin 0 and with
gmin stepping; zero-valued R, C, L; negative values; parallel voltage sources;
voltage-source loop; inductor-only loop; capacitor-only node at DC; magnitude
spread 1e-15..1e9; diode exp overflow; BJT clipping at |vbe/vt| > 80; MOS at
`vgs == vth`, `vds == vgs - vth`, reverse mode, PMOS polarity; `t_stop == 0`,
negative `t_stop`, `t_step > t_stop`, non-dividing steps, snapping tolerance;
`.tran` option defaults (`or 0.1`); `NR_CONVERGENCE` and `JACOBIAN_SINGULAR`
paths; `init_condition` DEFAULT/RANDOM/array/invalid; NaN/Inf parameter text;
duplicate names; ground aliases; unknown probe labels; decimation boundaries;
Krylov non-convergence fallback and invalid rank/method; MOR invalid outputs
and probe subset; HB without sinusoidal source and with too few harmonics;
`.ac` with one point and `f_start = 0`; `.show_matrix`; large-n parity
(1035 unknowns); thread count 1 vs N; repeated runs for leaks.

## 5. Performance comparison harness

`benchmarks/compare_backends.py` measures, per migrated unit, the frozen
Python reference against the C++ core at `num_threads` 1, 2, 4 and
`cpu_count`: circuit build; device evaluation (622 MOS, BJT stage, diode net);
Jacobian assembly; factor and solve (n = 248 and 1035, real and complex);
Newton `.op` (SRAM); transient (SRAM 0.25n, RLC 40 steps, BJT 2001 steps);
AC (RLC 100 points); HB mixer; each Krylov method on the RLC operator; MOR
basis build and TPWL training. Method: warm-up run, 5 repetitions, median,
idle-machine note, BLAS pinned to 1 thread for the reference. Output: a
Markdown table on stdout and JSON (`env`, git commit, timestamp, per-unit
`reference_ms`, `cpp_ms` by thread count, speedups) written to
`benchmarks/results/`; one committed baseline per platform
(`baseline-linux.json`, `baseline-windows.json`) updated deliberately.
`tests/perf/test_perf_gate.py` (marker `perf`, needs `--run-perf`) asserts
floors at roughly one third of the measured speedups (device evaluation >= 20x,
RLC transient >= 10x, SRAM transient >= 5x, AC >= 5x, HB >= 2x, AC 4-thread
scaling >= 2x on machines with >= 4 cores). CI runs the script on Linux and
uploads the JSON as an artifact; it never blocks a merge.

## 6. Phases, deliverables, exit criteria

| Phase | Deliverables | Exit criteria |
| --- | --- | --- |
| P0 Foundations | frozen reference, `conftest.py`, corpus, differential tests (reference vs current), edge-case spec, `.gitignore` and index cleanup, `cpp/` skeleton with vendored Eigen/KLU/doctest, `_core` with `version/build_info/klu_solve/parallel_row_sums`, `pyproject` on scikit-build-core, CI (Linux + Windows, sanitizers, coverage, wheels), smoke tests, this document | `pip install -e .` builds on Linux; CI green on both platforms; all existing tests plus differential and smoke tests pass; `ctest` passes |
| P1 Circuit and devices | `exprc.py`, `build_circuit`, device tables, static stamps, CSR pattern with slots, `evaluate`, `to_dense`, native diode/MOS/BJT/behavioral models, `mna_builder` facade with `native` handle | dense `G`, `C`, `b_dc`, `b_ac` equal to the reference to 1e-15 relative on the whole corpus; `f` and `J` equal to the reference's `compile_nl_functions` at random states; doctest coverage of every model boundary |
| P2 Linear algebra and basic analyses | KLU/Eigen `LinearSolver`, complex KLU, Krylov methods and classification, `newton_op`, `transient` (cached factorization, progress callback), `ac_sweep`; facades `solve_dc_nr`, `solve_transient`, `solve_ac`, `solve_linear`, `krylov_solve_linear`, `classify_krylov_matrix`; `.dyn` streaming from the progress callback | differential suite passes for `.op`, `.tran`, `.ac` on the whole corpus; metadata parity for Krylov fields; `perf` floors met for these units |
| P3 HB and MOR, runtime removal | `hb_solve`/`hb_reconstruct` (block-structured), MOR basis/projection/reduced runs, TPWL training/run; `core_advanced` and `mor/*` facades; Python numerical implementations deleted from `mna_simulation` (kept only in the frozen reference) | differential suite passes for `.hb` and MOR cases; `mna_simulation` no longer imports SymPy at solve time (only in `exprc.py`) nor NumPy linear-algebra solvers; wheel contains no dead code |
| P4 Threads, control, performance | parallel device evaluation with deterministic reduction, frequency/sample/shift parallelism, `runtime.ThreadBudget`, `num_threads` request option, UI control and metadata, determinism and resource tests, `compare_backends.py`, perf gate, committed baselines, Windows wheel validation, README/STRUCT/USER_GUIDE updates | determinism tests pass; four concurrent SRAM requests finish within 1.5x of one; perf gate passes on Linux CI; Windows wheel passes the full suite |
| P5 Stretch (not required for completion) | HB phase B (FFT, matrix-free GMRES), adaptive time stepping, trapezoidal/Gear-2 | separate plan |

Order of work inside each phase: C++ unit with doctest tests, binding and
binding tests, facade switch behind the public function, differential run,
audit, commit. The suite is green at every commit.

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Numerical drift versus the oracle (pivoting order, summation order) | tolerances per class; bitwise checks only where the algorithm is identical (device evaluation, time grid); investigate every tolerance widening before accepting it |
| Behavioural quirks of the current core that are accidental | `tests/EDGE_CASES.md` decides preserve vs change case by case before P2 |
| Windows/MSVC differences (OpenMP 2.0, `long` size, `/fp` model, DLL loading) | signed loop indices, `int64_t`/`double` only in interfaces, `/fp:precise`, static linking, CI on `windows-latest` from P0 |
| KLU and BTF are LGPL-2.1+ and statically linked | license texts shipped; the project is open source so relinking from source is possible; switch to a shared library inside the wheel (`delvewheel`/`auditwheel`) if closed-source distribution is ever needed |
| HB memory and time at large n | block-structured assembly (2.6); document the limit (`nnz(G+J) * K^2` doubles); phase B if needed |
| Expression evaluator parity with SymPy | the compiler and evaluator are tested against SymPy-evaluated vectors for every function in the allowed set, including `Heaviside(0)` and `Piecewise` ordering |
| SymPy remains a dependency | only `exprc.py` imports it, at circuit-build time; acceptable |
| Throughput regressions hidden by faster hardware | committed baselines and the perf gate compare speedup ratios, not absolute times |
| Removing the Python solvers breaks external users of `mna_simulation.solvers` | the public names stay as facades with identical signatures; private `_`-prefixed helpers are not preserved |
