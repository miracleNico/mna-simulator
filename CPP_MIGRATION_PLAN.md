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
| Sparse direct solver | SuiteSparse KLU (with AMD, COLAMD, BTF, SuiteSparse_config) vendored as copied sources. Statically linked in phase 0; from phase 2 (when it becomes load-bearing) built as a shared library inside the wheel (`auditwheel`/`delvewheel`) so the LGPL relinking requirement is met cleanly. Eigen for dense linear algebra, complex arithmetic, and SVD. |
| Test oracle | Today's Python numerical core is frozen as `tests/reference/mna_reference` (test-only, not shipped) and used for differential tests forever, including randomized circuits. |
| Platforms | Linux and Windows (the web app is used on Windows). CI builds and tests both; wheels for CPython 3.12+ (stable ABI). |
| Thread control | Explicit end-user control: a `num_threads` request option, a UI control in the simulation panel, and `threads_used` in result metadata. The service enforces a machine-wide budget (`MNA_NUM_THREADS`, default = CPU count) so concurrent requests never oversubscribe; "auto" lets the service pick. |
| Performance comparison | `benchmarks/compare_backends.py` (frozen Python vs C++ at 1, 2, 4, N threads, per migrated unit) plus a `perf`-marked pytest gate with generous floors, skipped unless `--run-perf`. |
| Bindings and build | nanobind (stable ABI, static libnanobind), scikit-build-core, CMake >= 3.26, C++17. Threads come from an internal `std::thread` pool created per call, not OpenMP: MSVC only offers OpenMP 2.0 and every OpenMP runtime would have to be bundled into the wheel. |
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
Python side: `mna_simulation/native/expressions.py` sympifies the expression
with the same locals as `solvers._sympy_locals` and lowers the tree to a postfix
bytecode with the operator set the current core accepts: `+ - * / **`, unary minus, `sin cos
exp sqrt log abs`, `Heaviside(x, h0)`, `Piecewise`, `Mod`, `Min`, `Max`,
comparisons, `And/Or/Not`, constants, symbols `t` and `x_i`. The C++ evaluator
(`cpp/src/expr/`) executes the bytecode with a small value stack. Parity rules:
`Heaviside(0, h0) = h0` (the current core passes `1`), `Piecewise` takes the
first true branch, `Mod` follows SymPy sign semantics, division by zero yields
inf as NumPy does; `Piecewise` with no true branch yields NaN like
`numpy.select`. Derivatives of behavioral sources are not generated as strings:
the evaluator runs the same program on forward-mode dual numbers over the set
of unknowns the expression references (discovered at parse time), which also
removes the latent lambdify failure for `Heaviside` derivatives. Time-source
programs are evaluated per grid point inside the core (so the same path serves
transient, HB samples and future adaptive stepping); behavioral programs run
inside every Newton evaluation. Unsupported SymPy nodes fail at build time
with `NETLIST_FATAL` naming the component. Programs are cached per expression
string. Diodes stop being expressions and become a native model (Shockley
with `V_T = 0.025`, no clipping, exactly as today).

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
| `version()`, `build_info()` | diagnostics (compiler, threads implementation and hardware concurrency, KLU, Eigen versions) |

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
- Gmin stepping stays in `PythonBackend.run` over `DEFAULT_GMIN_STEPS`, but
  the circuit is compiled once: the builder records a unit `gmin` stamp pattern
  and the core forms `G_eff = G + gmin * gmin_pattern`. Retries happen only for
  numerical codes (`NR_CONVERGENCE`, `JACOBIAN_SINGULAR`, `AC_ERROR`, numerical
  `HB_ERROR`); structural errors such as a zero-valued resistor are raised once.
  The final error after exhausting the steps is unchanged.
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

### 2.4.1 Edge-case decisions

`tests/EDGE_CASES.md` pins the current behaviour and flags what looks
accidental. Rulings (a "change" is applied at the cut-over of the unit that
owns the behaviour, with the pinned test updated in the same commit and, when
the oracle must follow, a CHANGELOG entry in `tests/reference/README.md`):

| Behaviour | Ruling | When |
| --- | --- | --- |
| `0` option values treated as unset (`tran_stop`, `tran_step`, `f_start`, `f_stop`, `points`, `harmonics`) | change: only `None` means unset; `t_stop = 0` returns the single initial sample, `t_step <= 0` is `TRAN_ERROR`, `points <= 0` is `AC_ERROR`, `harmonics <= 0` is `HB_ERROR` | P3 (`.tran`/`.ac`), P4 (`.hb`) |
| Diode across an ideal source never converges; CMOS chains of 4+ stages fail at gmin 0 | improve, not parity: junction voltage-step limiting for `exp` devices and SPICE-style node-to-ground gmin during gmin stepping; corpus expectations for the affected cases updated when it lands | P3 |
| `.ac` ignores `max_iter`, `v_tol`, `f_tol` for its operating point | change: pass the options through | P3 |
| `R = 0` raises `ZeroDivisionError` | change: `NETLIST_FATAL: '<name>' has zero resistance` from the builder, raised once (no gmin retries) | P2 cut-over |
| Singular linear transient step raises raw `LinAlgError` | change: `NR_CONVERGENCE` with the step time in the message (gmin stepping still applies) | P3 |
| String `t_stop` / float `points` raise `TypeError`; JSON-list `init_condition` raises `AttributeError` | change: option coercion in `build_options` (metric suffix strings accepted, lists accepted); wrong-length arrays become `NETLIST_FATAL` in P3 | applied (coercion); P3 (length check) |
| Non-finite results returned with `status = "ok"` | change: `NR_CONVERGENCE: non-finite solution` | P3 |
| `init_condition` strings ignored on linear `.op` and on `.tran` | change: `DEFAULT`/`RANDOM` (case-insensitive) set the Newton starting point everywhere nonlinear and are ignored with a diagnostic on linear paths; arrays are validated | P3 |
| `use_krylov = "false"` enables Krylov; `krylov_rank = 2.7` truncated | change: strict boolean and integer coercion, `ValueError` otherwise (HTTP 400) | applied |
| Independent current source injects into `node1` (opposite of SPICE) | preserve: project convention used by the UI, demos and every test; documented | - |
| Floating island fails at every gmin (gmin is stamped only across C, D and transistor terminals) | change together with node-to-ground gmin above; the pinned expectation moves from "fails" to "converges at the first gmin step that regularizes it" | P3 |
| Inductor-only loops and C-only nodes are singular at DC; `C = 0` / `L = 0` silently skipped; negative C/L accepted; negative `vth` taken as magnitude; duplicate non-branch names stamped twice | preserve (physically or conventionally defensible); a parser diagnostic for duplicate names may be added later outside the core | - |
| Level-1 BJT Jacobian disagrees with the clipped current beyond `|vbe/vt| > 80` | change in P3 together with junction voltage-step limiting: beyond the clip the exponential continues linearly (`exp(80) * (1 + x - 80)`), which keeps function and derivative consistent and the Newton matrix nonsingular. A plain zero slope was tried and rejected: it made the HB Newton Jacobian singular when an iterate crossed the clip (random BJT CE case). Until then the verbatim clipped slope is kept | P3 |
| `TRAN_ERROR` raises the base `CircuitSimulatorError` | change: map `TRAN_ERROR` to `NetlistError` (it is an input error, like `MOR_CONFIG`) | applied |
| Time grid by repeated addition; snapping tolerance behaviour | preserve bit-for-bit (parity rule) | - |
| Unknown `probe_nodes` ignored without MOR; decimation keeps the last sample so it can return `max_points + 1` | preserve (service layer, harmless) | - |
| Invalid Krylov/MOR option values raise plain `ValueError` even when the feature is off | preserve the validation; the server maps tagged option `ValueError`s to HTTP 400 | applied |
| `.op` ignores time-dependent sources at `t = 0` while `.tran`'s initial operating point includes them | preserve (parity quirk 5 in 6.1); documented in the user guide | - |

### 2.5 Multithreading design

- Mechanism: an internal `std::thread` pool (`cpp/include/mna/threads.hpp`)
  created per analysis call with `k - 1` workers plus the caller and joined at
  the end; `k = 1` runs inline with zero overhead. No global state, so
  concurrent Python callers never interfere. Chunk boundaries of `parallel_for`
  depend only on the item count, never on `k`. No OpenMP anywhere (MSVC OpenMP
  2.0 limits, and bundling libgomp/vcomp into wheels is a known hazard).
  Parallel device evaluation is enabled above ~2048 devices; below that the
  serial loop (about 12 ns per device) is faster than any dispatch.
- Device evaluation is two-phase: phase 1 writes each device's contributions
  into per-device scratch slots in parallel over device chunks; phase 2 gathers
  into CSR slots and residual rows through a precomputed assembly map in fixed
  device order (parallel over slots/rows, no atomics). Result is bitwise
  identical for any thread count. Flags: no fast-math, `-ffp-contract=off`,
  `/fp:precise`; the serial path calls the same chunk function.
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
mna_simulation/native/        import guard and build info (__init__), circuit.py (tables -> build_circuit),
                              expressions.py (SymPy -> bytecode), analyses.py (wrappers, error translation,
                              timings), linalg.py (Krylov/direct facades), threads.py (ThreadBudget, MNA_NUM_THREADS)
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

| Phase | Weeks | Deliverables | Exit criteria |
| --- | ---: | --- | --- |
| P0 Freeze and scaffold | 1 | frozen reference, `conftest.py`, corpus, differential tests (reference vs current), edge-case spec and `EDGE_CASES.md`, `.gitignore` and index cleanup, `cpp/` skeleton with vendored Eigen/KLU/doctest, `_core` with `version/build_info/klu_solve/parallel_row_sums`, `pyproject` on scikit-build-core, CI (Linux + Windows, sanitizers, coverage, wheels), smoke tests, this document, `dev` integration branch | `pip install -e .` builds on Linux; CI green on both platforms; all existing tests plus differential and smoke tests pass; `ctest` passes |
| P1 Expressions and devices | 2 | `native/expressions.py` + expression VM with dual numbers, `build_circuit` (COO triplets, union pattern, per-device slots, gmin pattern), MOS/BJT/diode/behavioral kernels with deterministic gather, `to_dense`, facades `compile_nl_functions`, `_level1_*`, `DeviceSet` bindings | dense `G`, `C`, `b_dc`, `b_ac` equal to the reference to 1e-15 relative on the whole corpus; `f` and `J` equal to the reference at 1000 random states to 1e-13; `test_physical_sram.py` and `test_netlist_extensions.py` pass through the facades; doctest coverage of every model boundary |
| P2 Linear algebra and `.op` | 2 | `KluSolver` real/complex with cached symbolic analysis and refactor, dense Eigen solver, GMRES/CR/MINRES/CG with classification and the legacy info keys (including the `minres` naming rule at n >= 64), Newton driver ported verbatim, `.op` cut over, `solve_linear`/`krylov_solve_linear`/`classify_krylov_matrix` facades, KLU as a shared library in the wheel | `.op` differential green on the whole corpus; `test_krylov_solvers.py` passes unedited; SRAM `.op` at least 5x faster than the reference |
| P3 Transient, AC, operating points | 2 | transient driver (Backward Euler, cached factorization for constant-step linear runs, Krylov path, progress callback for `.dyn`), AC sweep parallel over frequencies, `OperatingPointTable`, lazy dense `G`/`C` for `.show_matrix`, gmin scalar passthrough with retry only on numerical codes, metadata additions | `.tran`/`.ac` differential green including the 1035-unknown mesh; thread-determinism test green; `test_service.py` and `test_mor.py` still green |
| P4 Harmonic balance | 2 | block-sparse `Y_bar`, dense `K x K` device-Jacobian blocks only on the device pattern, Fourier matrices, Newton, reconstruction, memory policy (`HB_ERROR` with sizes), `core_advanced` facade | HB differential green (mixer H=205 within 1e-6); `test_harmonic_balance.py` green; mixer under 0.5 s |
| P5 MOR | 2 | rational Krylov basis through KLU/complex KLU shifted solves and block Gram-Schmidt, projections, reduced AC/transient, validation path, TPWL training (full transient, `BDCSVD`, reduced local models) and simulation; Python LRU cache holds handles | MOR differential green; `test_mor.py` green unedited |
| P6 Cut-over and hardening | 1.5 | Python numerics deleted from the runtime package (reference only), `CoreBackend`, `ThreadBudget` + `num_threads` option + UI control + metadata, `compare_backends.py`, perf gate and committed baselines, leak and determinism tests, sanitizer/coverage/wheels workflows final, README/STRUCT/USER_GUIDE/PERFORMANCE_PLAN updates, CLI `--threads` | wheels for Linux and Windows pass the full suite on 3.12 and 3.13; perf floors met; ASan/UBSan clean; four concurrent SRAM requests finish within 1.5x of one |
| P7 Optional | 2-3 | matrix-free HB (FFT products, GMRES with block-diagonal per-harmonic preconditioner), dense-vs-KLU threshold tuning, adaptive time stepping | separate plan |

Each cut-over (P2 to P5) lands behind an environment toggle `MNA_CORE_UNIT`
for one merge window so the differential suite can run both paths in CI,
then the toggle is removed. Order of work inside each phase: C++ unit with
doctest tests, binding and binding tests, facade switch behind the public
function, differential run, audit, commit. The suite is green at every commit.

### 6.1 Load-bearing quirks of the current core (parity checklist)

| Quirk | Where | Port decision |
| --- | --- | --- |
| Symmetric Krylov route reports `minres` when n >= 64 (SciPy engine) and `conjugate_residual` below 64 or for complex matrices | `solvers.py:425-511` | port both and keep the threshold in the routing table |
| Sparse classification: symmetry if `max abs(A - A^H) <= 1e-10`, Cholesky attempted only when n <= 256; a `1x1 [[1e-3]]` is `positive_definite` | `solvers.py:88-120` | use the sparse rule everywhere |
| Non-converged Krylov answer accepted when `relative_residual <= max(10 tol, 1e-8)`, otherwise direct fallback with `used_direct_fallback = True` | `solvers.py:593-599` | keep exactly |
| Damped Newton accepts the first trial with `trial_norm <= current_norm`, 28 halvings, best strictly smaller trial tracked, `step_scale == 0` raises, `norm(delta) < v_tol` exits, residual checked before the first solve | `solvers.py:1022-1051, 1108-1151` | port verbatim |
| Transient initial op uses `b_dc + b_time(0)`; AC linearizes with `b_dc` only; `op_used` is True only with Level-1 devices and no `init_condition` | `core_basic.py:435-471` | keep both call shapes, do not "fix" `op_used` |
| Time grid accumulates `current + t_step` then snaps the last point | `solvers.py:992-1013` | identical operation order, bitwise grids |
| `init_condition == "RANDOM"` uses NumPy's global RNG | `solvers.py:1102` | RNG stays in Python; the core always receives `x0` |
| `PythonBackend.run` retries any exception nine times and rebuilds everything | `backends/python_backend.py:180-197` | retry numerical codes only, compile once |
| HB metadata allocates a dense `(nK)^2` matrix only to classify it | `core_advanced.py:83-89` | classify from dimension instead |
| Tests use `problem.G[c, c]`, `len(problem.G)`, `G_sparse + C_sparse / h`, `.nnz`, `matrix_storage == "dense+csr"`, `krylov_operator_storage == "csr"` | `test_netlist_extensions.py:459`, `test_krylov_solvers.py:237-255, 337` | keep SciPy CSR fields, lazy dense `G`/`C`, report `csr` for the row-permuted operator view |
| `_level1_*`, `compile_nl_functions`, `_transient_time_points`, `solve_linear`, `krylov_solve_linear`, `classify_krylov_matrix`, `resolve_krylov_rank` are imported by tests or by `core_basic`/`mor` | `tests/*`, `core_basic.py` | keep as facades with identical signatures |
| Expression grammar in use: `Piecewise`, `sin cos exp sqrt`, `Max/Min`, `np.heaviside(x, 1)`, `np.mod(t, T)`, `pi`, relations, `t`, `x[i]`; `np.`/`sp.` prefixes stripped | `device_models.py`, `solvers.py:645-693` | defines the VM opcode set |

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Numerical drift versus the oracle (pivoting order, summation order) | tolerances per class; bitwise checks only where the algorithm is identical (device evaluation, time grid); investigate every tolerance widening before accepting it |
| Behavioural quirks of the current core that are accidental | `tests/EDGE_CASES.md` decides preserve vs change case by case before P2 |
| Windows/MSVC differences (`long` size, `/fp` model, libm last-bit differences, DLL loading) | `int64_t`/`double` only in interfaces, `/fp:precise`, `std::thread` instead of OpenMP, `delvewheel` for the KLU shared library, tolerances and determinism asserted per platform, CI on `windows-latest` from P0 |
| KLU and BTF are LGPL-2.1+ | license texts shipped; static link only during phase 0-1; shared `mna_suitesparse` library inside the wheel from phase 2 so relinking is possible without rebuilding the extension |
| HB memory and time at large n | block-structured assembly (2.6); document the limit (`nnz(G+J) * K^2` doubles); phase B if needed |
| Expression evaluator parity with SymPy | the compiler and evaluator are tested against SymPy-evaluated vectors for every function in the allowed set, including `Heaviside(0)` and `Piecewise` ordering |
| SymPy remains a dependency | only `exprc.py` imports it, at circuit-build time; acceptable |
| Throughput regressions hidden by faster hardware | committed baselines and the perf gate compare speedup ratios, not absolute times |
| Removing the Python solvers breaks external users of `mna_simulation.solvers` | the public names stay as facades with identical signatures; private `_`-prefixed helpers are not preserved |
