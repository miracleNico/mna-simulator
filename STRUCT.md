# Project Structure and File Functions

This document describes the project-authored source, config, docs, and tests. It intentionally ignores dependency trees such as `app/web/node_modules`.

## End-to-End Flow

1. A request enters from CLI (`mna_simulation/cli.py`), HTTP (`app/server/main.py`), or the React schematic editor.
2. `SimulationService` (`mna_simulation/api/service.py`) normalizes request handling, leases worker threads from the machine-wide thread budget (`mna_simulation/native/threads.py`), and pins NumPy/SciPy BLAS to one thread for the run.
3. Raw netlists go directly to the parser. Schematic JSON goes through `mna_simulation/schematic_pipeline.py`.
4. The schematic pipeline validates hierarchy, flattens `SUBCKT` instances, forms nets, emits canonical netlist text, and parses it back into `CircuitIR`.
5. `mna_simulation/backends/python_backend.py` builds `AnalysisOptions` (strict option coercion) and runs the gmin-stepping loop.
6. `mna_simulation/netlist.py` and `mna_simulation/mna_builder.py` produce the MNA problem; the builder also compiles it once into the native `Circuit` (`mna_simulation/native/circuit.py`, `_core.build_circuit`) that every gmin step reuses.
7. Optional selected-output MOR in `mna_simulation/mor/` reduces `.ac` and `.tran` problems when requested (numerics in `_core.mor_*`).
8. `mna_simulation/core_basic.py` / `core_advanced.py` dispatch to the facades in `mna_simulation/solvers.py`, which call the native core (`_core.solve_op`, `solve_transient`, `ac_sweep`, `hb_solve`, `direct_solve`, `krylov_solve`) with the GIL released and the granted thread count.
9. Results return through the API service (with `threads_used`, `core_backend`, `core_version`, and `elapsed_ms` metadata) and are plotted by the frontend.

Every analysis runs in the mandatory C++17 extension `mna_simulation._core` (`cpp/`, nanobind, vendored Eigen and SuiteSparse KLU); there is no Python numerical fallback (`CPP_MIGRATION_PLAN.md`, status at its top). Python keeps everything that involves strings: parsing, node naming, labels, error messages, the expression text, option handling, metadata, and the response. The pure-Python core as of the migration start is frozen as the test oracle `tests/reference/mna_reference`, against which `tests/differential/` compares every corpus analysis.

## Root Files

### `README.md`
- Project-level usage guide.
- Covers setup, CLI/API usage, hierarchy behavior, supported devices/modes, tests, and frontend startup.
- "Native Core" section: what `mna_simulation._core` is (C++17, nanobind, vendored Eigen and KLU, deterministic per-call threading), build requirements, wheels, and the C++ unit tests.
- "Threads and Concurrency" section: the `MNA_NUM_THREADS` budget, the `num_threads` request option and its `auto` rule, `--threads`, BLAS pinning, what concurrent requests in one process can and cannot overlap, and the recommended `OPENBLAS_NUM_THREADS=1` server start; "Environment Variables" adds `MNA_HB_MAX_NNZ`.
- "Testing" and "Benchmark": the default suite, the differential subset, `pytest tests/perf --run-perf`, `compare_backends.py`, and `concurrent_requests.py`.

### `USER_GUIDE.md`
- User-facing guide for the web app.
- Mirrors the main topics exposed by the in-app Help dialog: demos, schematic editing, hierarchy, analyses, threads, result display, Krylov subspace solver controls, raw netlist mode, and troubleshooting.
- "What Changed With The Native Core" lists every user-visible behaviour change with the old behaviour (strict options, zero values, `init_condition`, zero-ohm resistors, the AC operating point, the MOR basis, threads, error reports, and the P3b convergence improvements: junction limiting, gmin continuation, node-to-ground gmin, BJT exponent extrapolation, damped HB Newton). The in-app copy (`app/web/src/lib/userGuide.ts`) has not been updated for this section yet.

### `STRUCT.md`
- This file-by-file architecture reference.

### `DESIGN_AUDIT.md`
- Front end design audit: findings from driving the app in headless Chromium, what was changed, and an LTspice feature comparison with suggested next steps.

### `.gitignore`
- Ignores Python bytecode and tool caches, packaging outputs (anchored to the root: `/build/`, `/dist/`, `/wheelhouse/`, `*.whl`, `*.egg-info/`), native extension artefacts (`*.so`, `*.pyd`, `cpp/build*/`, `/build-cpp/`), coverage outputs, and Vite caches.
- Ignores perf comparison runs (`benchmarks/results/*.json`) except the committed baselines (`!benchmarks/results/baseline-*.json`, e.g. `baseline-linux.json`); the Markdown tables next to them are not ignored.
- `app/web/node_modules` and `app/web/dist` remain tracked.

### `PERFORMANCE_PLAN.md`
- Measured performance review of the Python core (profiles of the SRAM, RLC mesh, BJT amplifier, and harmonic-balance workloads).
- Prioritized plan for multithreading and solver changes, and the research on a C++ kernel migration.
- Section 4.1 (wall-clock per workload) and 4.3 (prototype kernels) fix the workload sizes used by `benchmarks/perf_units.py`.
- Section 10 (status): what landed per migration phase with the measured speedups, the concurrent-request finding, and what is left (P7).

### `CPP_MIGRATION_PLAN.md`
- Status table at the top: P0-P6 and P3b with one line of evidence each (tests and gates), the open P6 exit criteria, and P7 (optional).
- Approved plan for moving the whole numerical core (`.op`, `.tran`, `.ac`, `.hb`, Krylov, MOR, TPWL) into the mandatory nanobind extension `mna_simulation._core`; parser, schematic pipeline, FastAPI app, CLI, and UI stay in Python/TypeScript.
- Binding decisions (KLU + Eigen, frozen Python oracle, Linux + Windows, explicit thread control, `std::thread` pool), target architecture and `Circuit` data contract, kernel API, parity rules and edge-case rulings (2.4.1), multithreading design, HB and MOR designs.
- Repository layout, test harness layers, the performance comparison harness (section 5), phases P0-P7 with exit criteria (section 6), the load-bearing quirks parity checklist (6.1), and risks.

### `pyproject.toml`
- Python package metadata (version 0.70.0).
- Build backend is `scikit-build-core` + `nanobind`: `pip install -e .` configures `cpp/` with CMake >= 3.26 (Release, C++ tests off, build dir `build/{wheel_tag}`) and installs `_core` into `mna_simulation/`; one abi3 wheel (`cp312`) serves CPython 3.12+, and the wheel packs only `mna_simulation` (never `tests/reference`).
- Ships the third-party notices of `cpp/third_party/LICENSES/` via `license-files`.
- Declares runtime dependencies (`numpy`, `scipy`, `sympy`, `fastapi`, `uvicorn[standard]` for websocket support, `threadpoolctl`, etc.).
- Declares dev dependencies (`pytest`, `httpx`, `pytest-cov`, `threadpoolctl`).
- Registers CLI entrypoint: `mna-sim = mna_simulation.cli:main`.
- Pytest config: `testpaths = ["tests"]` and the `perf`, `slow`, `differential` markers.

## Python Package: `mna_simulation/`

### `mna_simulation/__init__.py`
- Package exports for top-level use.

### `mna_simulation/errors.py`
- Central exception hierarchy.
- `error_handler(message)` maps coded failures such as `NETLIST_FATAL` into typed exceptions.

### `mna_simulation/utils.py`
- Numeric and serialization helpers.
- Includes suffix parsing (`k`, `u`, `meg`, etc.) and JSON-safe ndarray conversion.

### `mna_simulation/device_models.py`
- Symbolic source and nonlinear device expression builders.
- Builds time-domain source expressions and diode current expressions.
- Supports FUNC source expression normalization used by transient and harmonic-balance paths.

### `mna_simulation/netlist.py`
- Text netlist tokenizer, parser, and validator.
- Parses directives (`.op`, `.tran`, `.ac`, `.hb`) and supported device lines.
- Preserves FUNC expression tails with spaces instead of splitting them into separate tokens.
- Supports extended BJT/MOS small-signal parameter forms.
- Supports physical Level-1 BJT/MOS syntax alongside small-signal compatibility syntax.
- Emits backend-neutral `CircuitIR` (with `diagnostics` for ignored constructs).
- SPICE dialect (phase M1): device lines that name a `.model` card produce the `spice` / `gummel_poon` / `spice_level1` records of `HIERARCHY_AND_MODELS_PLAN.md` 3.2, SPICE source forms (`DC`/`AC` with phase, `SIN(...)`, `PULSE`, `PWL`, `EXP`), unit letters, case-insensitive node names; the single-pass loop hands netlists with structural SPICE constructs to `mna_simulation/spice/frontend.py`. `parse_netlist_file(path, search_dirs=...)` resolves `.include`/`.lib`.

### `mna_simulation/spice/`
- `values.py` (SPICE literals, token helpers), `expressions.py` (restricted `.param`/`{...}` evaluator, no `eval`), `model_cards.py` (card parsing, supported/modelled parameter lists), `sources.py` (source specifications, PULSE/PWL/EXP/SIN waveform expressions, `.tran` defaults), `context.py` (per-parse diagnostics, model lookup, line locations), `frontend.py` (lexing, `.include`/`.lib`, `.subckt`/`X` flattening, parameter scopes, `FileIncludeResolver`).

### `mna_simulation/mna_builder.py`
- Converts `CircuitIR` to `MnaProblem`.
- Stamps COO triplets once (P6a fast path) and forms the SciPy CSR `G`/`C` with duplicates summed in stamping order (bit-identical to the reference builder), RHS vectors, the symbolic `f_str`/`b_time_str` display vectors, labels, and branch maps; dense `G`/`C` stay available for `.show_matrix`, small circuits, and tests.
- Stamps passive devices, sources, controlled sources, diodes, BJTs, and MOSFETs, plus Level-1 device metadata and junction/gate capacitances for transient analysis.
- Raises `NETLIST_FATAL` "'<name>' has zero resistance" for `R = 0` and a small-signal BJT with `rpi = 0` (2.4.1).
- SPICE-form `V`/`I` records (non-empty `metadata`: `DC x AC mag phase`, `PULSE`, `PWL`, `EXP`, `SIN(...)`) stamp their DC value into `b_dc`, the AC phasor (with phase) into `b_ac` and the waveform minus the DC value into `b_time_str` (`device_models.spice_source_terms`).
- `_compile_native` builds `problem.native`, the compiled `_core.Circuit` (gmin-free `G`, `C`, the unit gmin stamps, device tables, behavioral programs) shared by every gmin step.
- Model-card devices (phase M2, `card_devices.py`): `_build_index_map` numbers their internal nodes (`D1#a`, `Q1#b`, `Q1#c`, `Q1#e`) after the sorted external nodes and before the branch currents and ignores a Gummel-Poon substrate node; the builder stamps the series-resistance conductances, the constant junction / overlap capacitors and gmin, fills the native `spice_diodes`, `gummel_poon` and `mos4` tables, puts a readable placeholder (`I_spice_diode(D1)`, `Ic_gummel_poon(Q1)`, `Id_spice_level1(M1)`) into the display `f_str` (never into `solver_f_str`), and lists the devices in `metadata["card_devices"]`.

### `mna_simulation/card_devices.py`
- Model-card device preparation for the builder (`HIERARCHY_AND_MODELS_PLAN.md` 3.2 and 3.4): `card_kind` (which component selects `spice`, `gummel_poon` or `spice_level1`), `parse_card_value` (SPICE numbers: `T G MEG K M MIL U N P F`, trailing units ignored, `inf`), the SPICE defaults of each equation set, the `area` / `m` rules, validation (`NETLIST_FATAL` naming component, card and parameter) and the internal-node names; `prepare_card_devices` returns `CardDevice`s with the kernel parameters. `CARD_VT` is `k T / q` at 300.15 K.

### `mna_simulation/solvers.py`
- Facades over the native core with the reference signatures, return shapes, metadata keys, and exceptions: `solve_dc_nr` / `solve_operating_point` (`_core.solve_op`), `solve_transient` (`_core.solve_transient`, optional progress callback), `solve_ac` (`_core.ac_sweep`), `solve_harmonic_balance` / `create_fourier_matrices` / `reconstruct_time_domain` (`_core.hb_*`), `compile_nl_functions` / `compile_time_source_func` (compiled programs), `_transient_time_points`, and the Level-1 device facades.
- Re-exports the linear-algebra facades of `native/linalg.py` (`solve_linear`, `krylov_solve_linear`, `classify_krylov_matrix`, `resolve_krylov_rank`, `describe_krylov_choice`).
- `newton_start` turns `init_condition` (`DEFAULT`/`RANDOM` in any case, or an array validated for length) into the Newton starting point; `RANDOM` draws from NumPy's global RNG in Python.
- `hb_max_nnz()` reads `MNA_HB_MAX_NNZ` (default `_core.HB_DEFAULT_MAX_NNZ`, 3e7).
- `has_card_devices(problem)` (model-card devices live only in `problem.native`): part of `_problem_is_nonlinear`; `.ac` then computes its operating point on `problem.native` (`solve_operating_point`) instead of the `solve_dc_nr` matrix path.
- `_nmos_level1_forward` is a leftover Python copy of the Level-1 MOS equations with no callers.

### `mna_simulation/core_basic.py`
- Facade for `.op`, `.tran`, `.ac`, and matrix inspection.
- Builds the `.op`/`.tran` operating-point tables in Python (`_device_operating_points`, one native device call per transistor); model-card devices come from one `_core.operating_points` call on `problem.native` (`_card_operating_points`: region names, currents, conductances, internal nodes, non-finite values as `None`).
- Adds solver metadata for Krylov method, requested method, matrix dimension, sparse density, iteration count, rank mode, and fallback status.
- Routes selected-output MOR for `.ac`, linear `.tran`, and nonlinear TPWL `.tran`; `.op` and matrix inspection report MOR as intentionally unused.

### `mna_simulation/core_advanced.py`
- Facade for `.hb`.
- Infers base frequency from sinusoidal sources and optionally reconstructs time-domain waveforms from harmonic-balance coefficients.
- Passes Krylov rank/method options, `num_threads`, and the `MNA_HB_MAX_NNZ` bound through to `_core.hb_solve`; only an unset `harmonics` means 8, `harmonics <= 0` is `HB_ERROR` (2.4.1); solver statistics are reported as `metadata["hb_solver"]`.
- Reports MOR as intentionally unused for `.hb` v1.

### `mna_simulation/mor/`
- Decoupled model-order reduction package: option handling, metadata, validation, and the TPWL cache in Python; the numerics in `_core.mor_*` through `native/mor.py`.
- `selectors.py` maps labels such as `V(out)` and `I(L1)` to selector matrices and validates display-node subsets.
- `metadata.py` resolves automatic MOR order and emits compression/cache metadata.
- `linear.py` implements output-aware rational Krylov/PRIMA-like projection for linear and `.op`-linearized `.ac` plus linear transient (basis built with classical Gram-Schmidt twice, 2.4.1). Large sparse linear transients can validate selected outputs against the sparse full-order Krylov path and fall back when the reduced model underfits or becomes unstable.
- `tpwl.py` implements TPWL/POD nonlinear transient MOR with an in-memory LRU cache (8 entries) keyed by a digest of the problem and options.

### `mna_simulation/schematic_pipeline.py`
- Backend-owned schematic graph to netlist conversion.
- Validates component pins, disconnected pins, ground presence, and hierarchy ports.
- Flattens `SUBCKT` instances using logical paths such as `DiffAmp.in`.
- Requires child schematics to expose matching local `port_<pin>` junctions.
- Keeps top-level net names stable while naming hierarchy-internal nets separately.

### `mna_simulation/cli.py`
- Command-line interface.
- Commands: `run <path>` and `run-text <netlist_text>`.
- Supports optional mode overrides and JSON output.
- `--threads N|auto` (validated by `coerce_num_threads`) is passed as the `num_threads` request option; text output prints the threads used, requested, and the budget.
- `--core cpp|python` picks the numerical core through `create_service` (default: `MNA_CORE`, else `cpp`); text output prints the core and its version.

## Native Package: `mna_simulation/native/`

### `mna_simulation/native/__init__.py`
- Import guard for the mandatory extension `mna_simulation._core` (built from `cpp/`; there is no Python fallback).
- `require_core()` returns the module or raises an `ImportError` that explains how to build it (`pip install -e ".[dev]"`, CMake >= 3.26, a C++17 compiler).
- `core_version()` and `build_info()` (compiler, Eigen/SuiteSparse/KLU versions, build flags) for health checks and result metadata.

### `mna_simulation/native/circuit.py`
- Plain-table interface to `_core.build_circuit` (the tables of `CPP_MIGRATION_PLAN.md` 2.1): `triplets`, `diode_table`, `mos_table`, `bjt_table`, `behavioral_table`, the model-card tables `spice_diode_table`, `gummel_poon_table`, `mos4_table` (phase M2), and `build_circuit` (0-based node indices, `-1` is ground; `node_count`, the node unknowns that get a node-to-ground gmin stamp, is inferred from the tables when not given and is 0 for device-only circuits).
- `compile_device_functions` is the engine of `solvers.compile_nl_functions`: it compiles the nonlinear rows of `solver_f_str` and the Level-1 device dicts into a temporary circuit and returns native `f(*x)` / `J(*x)` callables.

### `mna_simulation/native/expressions.py`
- Compiles FUNC time sources, `B` behavioral sources, and nonlinear current rows to the native bytecode VM: SymPy `sympify` with the reference locals, the NumPy code printer configured as `lambdify` does, then the printed AST lowered to postfix `_core.Program` bytecode, so evaluation reproduces the reference operation by operation.
- `compile_expression(text, time_dependent=..., n_unknowns=...)` is cached per `(text, time_dependent, n_unknowns)` (LRU, 4096); unsupported constructs raise `NETLIST_FATAL`, unknown symbols `TypeError` (the reference's exception).
- Also home of `_sympy_locals` and `_prepare_sympy_expression`, which `solvers.py` re-exports.

### `mna_simulation/native/linalg.py`
- Drop-in facades for the reference linear algebra: `classify_krylov_matrix` (the sparse rule for every input), `resolve_krylov_rank`, `describe_krylov_choice`, `krylov_solve_linear` (`_core.krylov_solve`, `info["engine"] == "mna_core"`), `solve_linear` and `direct_solve` (`_core.direct_solve`: KLU or dense LU, `np.linalg.LinAlgError("Singular matrix")` on a singular system), and `LinearSolver`.
- Converts dense arrays and SciPy sparse matrices to CSC with int32 indices once per call (`to_csc_arrays`).

### `mna_simulation/native/mor.py`
- Facades over `_core.mor_*` used by `mor/linear.py` and `mor/tpwl.py`: `circuit_operands` / `matrix_operands` (`MorOperands`: `G` and `C` values in one CSC pattern), `krylov_basis`, `project`, `reduced_ac`, `reduced_transient`, `pod_basis`, `training_indexes`, `tpwl_local_models`, `tpwl_simulate`, and `safe_solve`.
- `resolve_threads` maps `None`/`"auto"` to 1; `PARALLEL_MIN_UNKNOWNS` (2048) is the size from which the per-circuit MOR work uses more than one thread. Core errors are translated to the reference exceptions.

### `mna_simulation/native/threads.py`
- Machine-wide thread budget: `budget_total()` reads `MNA_NUM_THREADS` (capped at the CPU count, invalid values warn and fall back to the CPU count).
- `coerce_num_threads()` validates the `num_threads` option (`None`, `"auto"`, positive integers or integer strings; anything else raises `ValueError`).
- `ThreadBudget` is a counting semaphore of worker threads: `lease(requested)` grants `min(requested, free, total)` for an integer or a fair share for `auto`, blocks while nothing is free (optional timeout), and reports `snapshot()` for `/health`. `default_budget()` / `reset_default_budget()` manage the process-wide instance.
- `pin_blas_threads()` limits NumPy/SciPy BLAS to one thread during a run (reference-counted `threadpoolctl` limiter with a cached controller; warns once when threadpoolctl is missing).

## API Layer: `mna_simulation/api/`

### `mna_simulation/api/contracts.py`
- Shared contracts for parser, builder, solver, service, and UI.
- Contains analysis enums, IR dataclasses, result dataclasses, API Pydantic models, and schematic graph models.
- `SchematicComponent` includes hierarchy metadata, arbitrary device metadata, and transistor defaults.
- `AnalysisOptions` carries Krylov options plus selected-output MOR options (`use_mor`, `mor_method`, `mor_order`, `mor_output_nodes`, and `mor_validate`).
- `AnalysisOptions.num_threads` holds the request (`None`, `"auto"`, or an integer) until the service replaces it with the granted thread count.

### `mna_simulation/api/service.py`
- Orchestration entrypoint used by CLI and FastAPI.
- Routes schematic payloads through `schematic_pipeline`.
- Adds generated canonical netlist text to response metadata.
- Applies optional output decimation to large waveform responses.
- Leases threads from a `ThreadBudget` (the process default unless one is injected) and runs the backend inside `pin_blas_threads()`; adds `threads_requested`, `threads_used`, `thread_budget_total`, `core_backend` (`SimulationService.core_backend`, `"cpp"`: every analysis runs in the native core), `core_version`, and `elapsed_ms` (wall time of `backend.run`) to result metadata.
- Tags option-validation `ValueError`s with a PEP 678 note (`is_invalid_option_error`) so the HTTP layer can answer 400 without catching solver `ValueError`s.
- `healthcheck()` also reports `core_backend`, `core_version`, `build_info`, and the thread budget snapshot.
- `resolve_core` / `create_service`: the core selection shared by the server and the CLI. `MNA_CORE` (`CORE_ENV_VAR`) is `cpp` (default, `SimulationService`) or `python` (`ReferenceSimulationService`); any other value is a `ValueError`.

### `mna_simulation/api/reference_service.py`
- Python mode (`MNA_CORE=python`): `ReferenceSimulationService` serves the `SimulationService` contract (`run_request`, `healthcheck`, `budget`) with the frozen pre-migration pure-Python core `tests/reference/mna_reference`.
- `load_reference_package` imports `mna_reference`, appending `tests/reference` to `sys.path` when it is not importable yet; without a source checkout (wheels) it raises `ReferenceCoreUnavailable`. The oracle is only imported, never changed.
- Validates request options with the live `PythonBackend.build_options` rules first (same HTTP 400s as the C++ mode), then runs the request through the oracle's own `SimulationService` (its backend subclassed only to tag option `ValueError`s for HTTP 400).
- Re-raises oracle simulator errors as the matching `mna_simulation.errors` classes (same `code`/`message`), leases one thread per run and pins BLAS, and reports `core_backend: "python"`, `core_version: "python-reference@<frozen commit>"` (from `FROZEN_AT.txt`), the thread fields and `elapsed_ms`; `healthcheck()` reports `build_info: None`.

## Backends: `mna_simulation/backends/`

### `mna_simulation/backends/python_backend.py`
- Active backend adapter (the name is historical: its analyses run in the native core).
- Parses netlists, builds options from directives, runs analysis, and exposes capability flags.
- `build_options` coerces request options strictly (booleans, integers, metric-suffix numbers, `init_condition`); rejections are plain `ValueError`s that the service tags for HTTP 400 (`tests/test_option_coercion.py`).
- `run` builds the problem and its native circuit once and retries over `DEFAULT_GMIN_STEPS` only for numerical failures (`is_numerical_failure`: `NR_CONVERGENCE`, `JACOBIAN_SINGULAR`, `AC_ERROR`, `LU_CONVERGENCE`, numerical `HB_ERROR`, `LinAlgError`); structural errors are raised at once.
- Coerces Krylov rank/method request options, including aliases such as `arnoldi`, `gmres`, `minres`, and `cg`.
- Coerces MOR method/order/output options and advertises model-order reduction capability.
- Coerces `num_threads` with `coerce_num_threads` into `AnalysisOptions.num_threads`.

### `mna_simulation/backends/cpp_backend.py`
- Placeholder adapter from before the migration (raises `NotImplementedError`); unused, since the native core is reached through `PythonBackend` and the solver facades.

## Model Library: `mna_simulation/library/`

### `mna_simulation/library/registry.py`
- Registry abstraction for built-in and user-defined models.
- Loads optional JSON model cards.

### `mna_simulation/library/models/behavioral_resistor.json`
- Example registry model-card metadata.

## Native Core: `cpp/`

The C++17 sources of `mna_simulation._core`, built by scikit-build-core (wheel and editable installs) or directly with CMake (C++ unit tests). One directory per unit under `include/mna/` (public headers) and `src/` (implementation), one nanobind file per unit under `bindings/`, and doctest files under `tests/`; CMake globs each directory, so a new file needs no CMake edit. Every header opens with a comment that states the algorithm, the parity rules against the Python reference, the threading, and the error codes; read those comments before the code.

### `cpp/CMakeLists.txt`
- CMake >= 3.26 project `mna_core` (C++17, C11, hidden symbol visibility, PIC).
- Targets: `klu_vendored` (SuiteSparse_config + AMD + COLAMD + BTF + KLU compiled as C with `NTIMER` into a static library, guarded by a source-count check), `mna_core` (static library from every `src/**/*.cpp`, Eigen with `EIGEN_MPL2_ONLY` and `EIGEN_DONT_PARALLELIZE`), `_core` (nanobind module, `STABLE_ABI NB_STATIC`, from every `bindings/*.cpp`, installed into `mna_simulation/`), and `mna_core_tests` (doctest, every `tests/*.cpp`, registered with ctest).
- Options: `MNA_BUILD_TESTS`, `MNA_BUILD_PYTHON_MODULE` (`AUTO`/`ON`/`OFF`; forced on under scikit-build), `MNA_ENABLE_SANITIZERS` (ASan + UBSan), `MNA_ENABLE_COVERAGE`, `MNA_NATIVE_ARCH` (never for wheels). Warnings `/W4` or `-Wall -Wextra -Wpedantic` on project targets only; no fast-math anywhere.

### `cpp/include/mna/` and `cpp/src/`

Headers are listed relative to `include/mna/`; each `x.hpp` with an implementation has a matching `src/x.cpp` (the assembly lives in `src/devices/assembly.cpp`).

- **Errors** (`error.hpp`, `src/error.cpp`): `mna::Error` (a `std::runtime_error`) with an `ErrorCode` whose names mirror the Python codes (`NETLIST_FATAL`, `NR_CONVERGENCE`, `JACOBIAN_SINGULAR`, `TRAN_ERROR`, `AC_ERROR`, `HB_ERROR`, ...), so errors translate 1:1 to Python.
- **Version** (`version.hpp`, `src/build_info.cpp`): version `0.70.0-dev` and `BuildInfo` (compiler, C++ standard, build type, thread implementation, hardware concurrency, Eigen/SuiteSparse/KLU versions, platform, sanitizer/coverage/native-arch/debug-assert flags).
- **Threads** (`threads.hpp`, header-only): `parallel_for` / `parallel_for_chunks` on per-call `std::thread` workers over at most 256 chunks whose boundaries depend only on the item count (bitwise-identical reductions for every thread count); `num_threads <= 1` runs inline; the first worker exception is rethrown after all threads join; `effective_threads(requested)` clamps to `[1, hardware_threads()]`. No global state.
- **Circuit** (`circuit/circuit.hpp`): the compiled circuit: union CSC pattern of `G`, `C`, the unit gmin stamps, and every device Jacobian footprint; static values in pattern order; structure-of-arrays device tables (diodes, Level-1 MOS and BJT, behavioral programs, and the phase-M2 model-card tables: SPICE diodes with their effective breakdown voltage, Gummel-Poon BJTs, four-terminal card MOS) with their pattern slots; the assembly map (four summation groups: diodes + behavioral, MOS, BJT, model-card devices). `g_effective(gmin)` forms `G + gmin * stamps` for any gmin without a rebuild; since P3b the gmin pattern also holds one node-to-ground stamp per node unknown (`node_count()`, never branch rows), and `junctions()` lists the exponential junctions (diodes, BJT base-emitter and base-collector, then the SPICE diodes and Gummel-Poon junctions) for junction limiting.
- **Devices** (`devices/diode.hpp`, `devices/mos_level1.hpp`, `devices/bjt_level1.hpp`, `devices/fp_contract.hpp`, header-only): Shockley diode (`V_T = 0.025`, no clipping), Level-1 MOS (bit-identical to the Python equations), Level-1 BJT (`vt = 0.02585`; since P3b the exponential continues linearly beyond `|v/vt| = 80`, value and slope continuous); `MNA_NO_FP_CONTRACT` turns off fused multiply-add on Clang so the kernels round like the reference. Model-card kernels (phase M2, `vt = kCardVt` = k T / q at 300.15 K, every exponential through `bjt_exp`): `devices/spice_diode.hpp` (SPICE3f5 diode: forward law with `n`, smooth reverse region, breakdown with the `diotemp.c` breakdown-voltage shift), `devices/gummel_poon.hpp` (Gummel-Poon DC: `q1`/`q2` base charge, leakage diodes, NPN/PNP), `devices/mos_card.hpp` (four-terminal Level-1 MOS with body effect, `vth0 = type * VTO`, bit-identical to `mos_level1` when `gamma = 0`).
- **Assembly** (`assembly.hpp`): device currents `f(x, t)` and Jacobian values in two phases (per-device scratch in parallel, then a gather in fixed device order through the assembly map); bitwise identical for every thread count.
- **Expressions** (`expr/program.hpp`): postfix bytecode of FUNC, `B`, and nonlinear-row expressions (produced by `native/expressions.py`) with scalar, batch, and forward-mode dual-number evaluators; NumPy float64 semantics (`mod`, `minimum`/`maximum`, `select`, division by zero).
- **Linear algebra** (`linear/`): `linear_solver.hpp` (CSC pattern, the `LinearSolver` interface, and the dense-vs-KLU policy; `kDenseMaxN = 0`, so every circuit solve uses KLU), `klu_solver.hpp` (`KluSolver` real and `KluSolverZ` complex: `analyze` once per pattern, `factor`, `refactor` reusing the pivot order with a fallback, `solve`), `dense_solver.hpp` (Eigen `PartialPivLU`, real and complex, with an explicit singularity rule), `sparse.hpp` (CSR view and SpMV, row-parallel from 4096 rows), `classify.hpp` (general, symmetric, or positive definite by the sparse rule; Cholesky only for n <= 256), `krylov.hpp` (GMRES(restart), MINRES/CR, and CG, real and complex, with the reference routing, the kernel regimes below and above n = 64, the acceptance rule, and the `info` keys).
- **Analyses** (`analyses/`): `time_grid.hpp` (bitwise port of the fixed-step grid and its `TRAN_ERROR`s), `newton.hpp` (damped Newton shared by `.op` and `.tran`; device evaluation threaded only from 2048 devices; since P3b the junction strategy after a failed or stalled damping loop), `limiting.hpp` (SPICE3 `pnjlim` and `limit_junctions`, P3b), `op.hpp` (`.op` on `G + gmin * stamps`: one linear solve or Newton, then gmin continuation from 1e-2 after a failed solve, P3b), `transient.hpp` (Backward Euler, factorization reused while the step is unchanged, Krylov path, progress callback), `ac.hpp` (complex sweep with frequencies split over the threads; the lowest singular frequency is reported), `hb.hpp` (harmonic balance: block-sparse `Y_bar`, dense `K x K` device blocks, `BtfSolver`, the memory policy `N <= 1e6` and stored entries <= `max_nnz` (default 3e7) checked before allocating, reconstruction; since P3b a damped Newton (`HbDamping`: `OnFailure` default, `Always`, `Off`) and the experimental, default-off Fourier-sum Jacobian).
- **MOR** (`mor/`): `krylov_basis.hpp` (rational Krylov basis on sparse operands, KLU per shift, Gram-Schmidt twice), `projection.hpp` (`V^T A V` without a dense `A`), `reduced_ac.hpp` and `reduced_transient.hpp` (reduced frequency and step loops), `safe_solve.hpp` (LU, then least squares, then regularized least squares), `pod.hpp` (POD basis with `BDCSVD`), `tpwl.hpp` (training indexes, local models, reduced TPWL loop).

### `cpp/bindings/`
- `module.cpp`: `NB_MODULE(_core)`: `CoreError` (a `RuntimeError` with `.code` and `.message`, translated from `mna::Error`), `version()`, `build_info()`, `effective_threads()`, `klu_solve(Ap, Ai, Ax, b)`, and `parallel_row_sums(a, num_threads)` (deterministic threaded smoke kernel); then calls every unit registered through `registry.hpp`.
- `registry.hpp`: self-registration of binding units: each `bind_<unit>.cpp` defines `bind_<unit>(nanobind::module_&)` and a static `Registrar{"<unit>", &bind_<unit>}`, so new units never edit `module.cpp` or CMake.
- One file per unit. Every heavy call releases the GIL for its whole native region; arrays cross as NumPy (float64, complex128, int32/int64) and 2-D results are Fortran-ordered.

| File | Exports |
| --- | --- |
| `bind_meta.cpp` | `binding_units()` (names of the registered units) |
| `bind_circuit.cpp` | `Circuit` (`n`, `nnz`, `pattern()`, `values()`, `g_effective()`, `to_dense()`, `device_counts`, `card_device_counts`, `device_slots()`, `junctions()`, `spice_diode_breakdown_voltages()`, ...), `build_circuit(spec)` |
| `bind_expr.cpp` | `Program`, `expr_opcodes`, `compile_program`, `evaluate_program`, `evaluate_program_batch`, `evaluate_program_gradient` |
| `bind_devices.cpp` | `evaluate_devices` (into caller buffers), `device_f`, `device_jacobian_dense`, `device_f_jacobian_dense`, `eval_mos_level1`, `eval_bjt_level1`, the model-card kernels `eval_spice_diode`, `spice_diode_breakdown_voltage`, `eval_gummel_poon`, `eval_mos_card`, `operating_points`, region-name tables, `card_vt`, `exp_clip` |
| `bind_linear.cpp` | `classify_matrix`, `direct_solve`, `krylov_solve`, `LinearSolver`; constants `DENSE_MAX_N`, `PARALLEL_SPMV_MIN_N`, `SCIPY_ENGINE_MIN_N`, `CHOLESKY_MAX_N`, `REFACTOR_RCOND_FLOOR` |
| `bind_analyses_basic.cpp` | `transient_time_points`, `time_source_vector`, `solve_op`, `solve_transient` (the progress callback re-acquires the GIL only for the call); `PARALLEL_DEVICE_MIN_COUNT` |
| `bind_ac.cpp` | `ac_sweep`, `linearized_g` |
| `bind_hb.cpp` | `hb_solve` (with `damping="on_failure"`/`"always"`/`"off"`), `hb_reconstruct`, `hb_fourier_matrices`, `hb_time_points`, test hooks `hb_jacobian_block` and `btf_solve`; `HB_MAX_DIMENSION`, `HB_DEFAULT_MAX_NNZ`, `BTF_DENSE_MAX_SMALL`, `BTF_DENSE_FRACTION` |
| `bind_mor.cpp` | `mor_krylov_basis`, `mor_project`, `mor_reduced_ac`, `mor_reduced_transient`, `mor_pod_basis`, `mor_training_indexes`, `mor_tpwl_models`, `mor_tpwl_simulate`, `mor_dense_operands`, `mor_safe_solve`; `MOR_PARALLEL_MIN_UNKNOWNS` |

### `cpp/tests/`
- doctest unit tests built as `mna_core_tests` and run by ctest (`main.cpp` is the doctest main), one file per unit: `test_build_info.cpp`, `test_threads.cpp` (chunk tiling, clamping, determinism, exception propagation), `test_circuit.cpp`, `test_devices.cpp` (model region boundaries; for the model-card kernels also central-difference derivatives, the limits that reduce to simpler forms, continuity at the clips, NPN/PNP and NMOS/PMOS symmetry, and a compiled circuit with every card table), `test_assembly.cpp`, `test_expr.cpp`, `test_klu_solver.cpp` and `test_klu_z.cpp` (known systems, refactor and its fallback, singular matrices, 1x1, n = 0, malformed input), `test_dense_solver.cpp`, `test_spmv.cpp`, `test_classify.cpp`, `test_krylov.cpp`, `test_time_grid.cpp`, `test_newton.cpp`, `test_pnjlim.cpp` (junction limiting and the P3b strategies), `test_transient.cpp`, `test_ac.cpp`, `test_hb.cpp`, `test_mor_basis.cpp`, `test_mor_reduced.cpp`, `test_mor_tpwl.cpp`.
- CI runs them on Linux (Ninja + GCC) and Windows (Visual Studio), under ASan + UBSan, and with coverage (`.github/workflows/ci.yml`).

### `cpp/third_party/`
- Vendored, unmodified upstream code so a build needs no downloads: `eigen/` (Eigen 3.4.0 headers), `suitesparse/` (SuiteSparse 7.8.3: SuiteSparse_config, AMD, COLAMD, BTF, KLU sources and headers), `doctest/` (2.5.0, tests only).
- `LICENSES/` holds every license text (Eigen MPL-2.0 and the other headers' licenses, SuiteSparse BSD-3-Clause and LGPL-2.1+ for KLU/BTF, doctest, nanobind, robin-map); the wheel ships the folder in `.dist-info/licenses/`.
- `README.md` records versions, hashes and commits, what was dropped, the LGPL note for the statically linked KLU/BTF, the Eigen license guard, and the update procedure.

## Server App: `app/server/`

### `app/server/main.py`
- FastAPI entrypoint.
- Exposes:
  - `GET /health`
  - `POST /api/simulate`
  - websocket dynamic simulation route used by frontend display/probe flows
- Translates simulator errors into HTTP 400 payloads; option-validation errors become HTTP 400 `INVALID_OPTIONS`.
- Dynamic websocket requests accept Krylov options and cap display frame rate while continuing looped playback.
- Runs `/api/simulate` (now async) and `/ws/dyn` simulations on a bounded `ThreadPoolExecutor` with one worker per budget thread (`budget_total()`), created in the app lifespan (or lazily) and shut down with it.
- Builds its service with `create_service()`, so `MNA_CORE=python` boots the server on the frozen pre-migration Python core (`mna_simulation/api/reference_service.py`); the lifespan logs `MNA core: <core_backend> (<core_version>)` at startup.
- `GET /health` adds `core_backend`, `core_version`, `build_info`, and the thread budget; `/ws/dyn` accepts `num_threads` and reports `threads_used` and `elapsed_ms` in its `meta` message.

## Web App: `app/web/`

### `app/web/README.md`
- Frontend startup and usage notes.

### `app/web/package.json`
- Frontend scripts and dependencies.
- Main scripts: `dev`, `build`, `preview`.

### `app/web/vite.config.ts`
- Vite + React config.
- Proxies `/api`, `/health`, and websocket traffic to the backend.

### `app/web/src/main.tsx`
- React bootstrap.

### `app/web/src/styles.css`
- Global styling for shell, controls, canvas, hierarchy tree, plot tiles, and dialogs.

### `app/web/src/lib/schematic.ts`
- Frontend schematic model and payload conversion.
- Defines device types, component/wire/level types, pin layouts, geometry helpers, net-label computation, and `buildSchematicPayload()`.
- Editor helpers: connection census (junction dots, wired/open pins), schematic bounds for zoom-to-fit, clipboard extraction, wire splitting for T-junctions, and the JSON workspace file format (`serializeWorkspace` / `parseWorkspace`).
- Defines analysis state for Krylov enablement plus MOR enablement, method, auto/manual order, and separate MOR outputs.
- `ThreadSetting` (`"auto"` or a positive integer) and `AnalysisState.numThreads`.
- Expands frontend `NODE` markers into backend `port_<name>` junctions.
- Repairs stale `SUBCKT` wire endpoint names before sending payloads.

### `app/web/src/lib/userGuide.ts`
- Structured content for the in-app Help dialog.
- Covers the same user-facing topics as the root `USER_GUIDE.md`, including the Threads section.

### `app/web/src/lib/demoPresets.ts`
- Hidden demo schematics.
- Includes CE amplifier, BJT 3-stage amplifier, GHz close-tone HB diode mixer, hierarchical 10x10 Level-1 MOS SRAM, and 23x23 RLC mesh demos.
- 3-stage amplifier uses real `SUBCKT` child levels and editable `NODE` ports rather than hardcoded fixed port junctions.
- Close-tone HB mixer uses 1 GHz and 1.005 GHz current tones into a nonlinear diode load plus a high-Q 5 MHz envelope tank, so HB exposes the steady-state beat while transient would need 40+ envelope periods to settle.
- Large SRAM uses a reusable `SRAM 6T Cell` child level with physical Level-1 MOS devices, while the root canvas keeps only array cell instances plus FUNC-driven row/column drivers.
- Large RLC mesh is sized for roughly 1000 MNA unknowns and sparse Krylov benchmarking.
- Default analysis state requests `numThreads: "auto"`.

### `app/web/src/lib/plot.ts`
- Canvas plotting helpers for line, stem, and bar visualizations.
- Used by `.op`, `.tran`, `.ac`, `.hb`, and dynamic plots.

### `app/web/src/lib/symbols.ts`
- Canvas symbol drawing for schematic components.
- Includes passives, sources, controlled sources, transistor symbols, labels, nodes, and subcircuits (with pin names drawn upright inside the body).
- Dedicated palette glyphs for LABEL, NODE, and SUBCKT toolbar buttons.

### `app/web/src/components/Toolbar.tsx`
- Component toolbar that wraps on narrow viewports.
- Device palette (primitives, transistors, controlled sources, `SUBCKT`, `NODE`, `LABEL`), an edit group (wire tool, rotate, mirror, delete, undo, redo, zoom to fit), and a simulate group (probe, dynamic display, Run, Help).

### `app/web/src/components/LeftPane.tsx`
- Simulation controls, hierarchy tree, library list, and property editor.
- Supports `SUBCKT` displayed/entity name, node count, editable pin names, transistor small-signal fields, and display-node controls.
- Primary controls (mode, parameters, display nodes, Run) come first; dynamic display, Krylov, and MOR live in collapsible "Advanced" sub-panels.
- Exposes Krylov subspace solver algorithm selection, method-specific manual value wording, display-node clear, and simulation-control reset.
- Exposes MOR controls with method selection including Linear Krylov MOR, auto/manual order, and output-node selection separate from display nodes.
- `ThreadsControl` ("Threads" in the simulation controls): `auto` or 1..N, where N is the server budget from `/health`; a saved request above the budget stays selectable and is labelled as capped.
- Properties expands and focuses its first field when a part is double-clicked on the canvas.
- Contains the editable netlist panel. Editing asks for confirmation, clears schematic hierarchy, and switches runs to raw `netlist_text`.

### `app/web/src/components/SchematicCanvas.tsx`
- Schematic graph editor.
- Handles hit-testing, selection, dragging, wiring, marquee selection, copy/paste, deletion, rotation, hover badges, and drawing.
- LTspice-style editing: wire tool with multi-segment routing and T-junction splits, sticky placement with rotate/mirror on the cursor part, right-drag pan, wheel zoom, zoom-to-fit, arrow-key nudge, double-click to open Properties.
- Draws junction dots, open-pin squares, dangling-end rings, de-cluttered automatic net labels, and reports cursor position and zoom through callbacks. Exposes an imperative zoom handle.

### `app/web/src/components/PlotTile.tsx`
- Movable/resizable plot tile wrapper.
- Supports static and dynamic streaming plots.

### `app/web/src/pages/App.tsx`
- Main frontend orchestration component.
- Owns hierarchy levels, active level, analysis settings, selection, tiles, generated netlist preview, and API calls.
- Owns the Help dialog state and renders the in-app user guide.
- Owns the File / Edit / View / Simulate / Demo menus, the undo/redo stacks, workspace save/open, netlist export, the wire tool and view toggles, and the window-level keyboard shortcuts.
- Synchronizes child `NODE` port renames to only the matching `SUBCKT` instances.
- Rewrites stale `SUBCKT` wire endpoints when pins are renamed.
- Tracks schematic mode versus raw netlist mode and sends the matching API payload.
- Passes Krylov method/rank options through static simulation and dynamic websocket requests.
- Passes MOR options through static simulation requests and keeps display-node filtering independent from MOR output selection.
- Reads the thread budget from `GET /health`, sends `num_threads` with simulation requests, keeps the Threads choice across demo loads, resets and workspace opens (validating saved values), and shows threads used and elapsed time in the status line.

### `app/web/src/components/AnalysisPanel.tsx`
- Older/legacy panel component retained in the tree.
- The current UI path is driven primarily through `LeftPane.tsx` and `App.tsx`.

## Tests: `tests/`

### `tests/conftest.py`
- Puts `tests/reference` (the oracle `mna_reference`) and `tests/` (corpus and differential helpers) on `sys.path`.
- `reference_pkg` / `current_pkg` session fixtures; registers the `perf`, `slow`, and `differential` markers.
- `--run-perf`: tests carrying the `perf` marker are skipped unless it is given.

### `tests/corpus.py`
- Declarative differential corpus: `Case` (netlist, analyses with explicit parameters, tolerance class, tags, expected error) built from every netlist in the pre-existing tests, frozen copies of the four benchmark workloads, seeded random generators (RLC ladders and meshes, controlled sources, MOS inverter chains, BJT stages, linear sources, diode nets), and error/gmin cases.
- Size budget per analysis (transient ~60 steps, AC 40 points, HB 8 harmonics except the GHz mixer); `iter_cases`, `iter_analyses`, `get_case`.
- `Deviation`: a deliberate behaviour change of one analysis (a `CPP_MIGRATION_PLAN.md` 2.4.1 ruling, e.g. the P3b convergence changes), recorded on its `Case` with the oracle's expected outcome and an equivalent netlist or analytic answer for the live result (`Case.deviation_for`).

### `tests/differential/`
- `compare.py`: runs one request through a package's own `SimulationService` and compares responses (status, labels, `dc_solution`, waveform, spectrum, matrices, contractual metadata, operating points) with per-class tolerances `linear` 1e-9/1e-12, `nonlinear` and `hb` 1e-6/1e-9, `mor` 5e-2/1e-6; error outcomes compare exception class and `.code`.
- `compare.check_deviation`: for an analysis with a `Deviation`, the oracle must still behave as recorded, the live package must succeed and match the oracle on the equivalent netlist (or the analytic operating point).
- `test_reference_parity.py`: one parametrized test per corpus `(case, analysis)`, live `mna_simulation` against the oracle (marker `differential`, large circuits also `slow`); analyses with a `Deviation` go through `check_deviation`.
- `test_harness_selftest.py`: oracle integrity (tree digest, distinct packages, no imports of `mna_simulation`), comparator sensitivity in every tolerance class, and corpus rules.

### `tests/reference/`
- `mna_reference/`: byte-for-byte frozen copy of the pure-Python `mna_simulation` at commit `e5172cb` (relative imports only), the test-only oracle for the C++ migration; never shipped.
- `mna_reference/FROZEN_AT.txt`: source commit and the `tree_sha256` checked by the self-test.
- `README.md`: how the oracle is used, the never-edit rule, the confirmed-oracle-bug exception, and the CHANGELOG.

### `tests/test_edge_cases.py`, `tests/EDGE_CASES.md`
- Behaviour specification of the current core on edge cases (empty and single-node circuits, singular G, zero/negative values, parser conflicts, conditioning, time grid, non-convergence, `init_condition`, probes and decimation, Krylov, MOR, HB, AC, `show_matrix`), pinning exception classes, codes, and numbers.
- `EDGE_CASES.md` lists every pinned case with a "looks intentional?" verdict and the highlights flagged as likely unintended; the rulings live in `CPP_MIGRATION_PLAN.md` section 2.4.1.

### Native binding and facade tests: `tests/test_native_*.py`
Each module exercises one native unit through its binding and its Python facade, against the frozen oracle where one exists, including shapes, dtypes, invalid input, error codes, GIL release, and 1 vs 4 thread bitwise determinism:
- `test_native_smoke.py`: version and build info, `klu_solve` against NumPy, `CoreError` codes for singular and malformed input, deterministic `parallel_row_sums`.
- `test_native_expressions.py`: the expression compiler and bytecode VM against SymPy `lambdify` on every expression form used in the repository (10 000 random points at 1e-13), dual-number gradients against SymPy derivatives.
- `test_native_circuit.py`: the COO-triplet builder bit for bit against the reference builder (dense and CSR `G`/`C`, sources, expression strings, labels, metadata) and the compiled circuit (gmin-free `G`, `C`, unit gmin stamps, node-to-ground stamps).
- `test_native_devices.py`: compiled `f`/`J` against `compile_nl_functions` of the oracle at 1000 random states per nonlinear corpus case (Level-1 MOS bit-identical, exponentials 1e-13, BJT inside the clip), the `_level1_*` facades, thread determinism of `evaluate_devices`, and the opt-in (`perf`) SRAM evaluation speedup.
- `test_native_linalg.py`: `classify_krylov_matrix`, `krylov_solve_linear`, `solve_linear` on the Krylov test matrices and on `G + C/h` / `G + j omega C` of every linear corpus case, the `direct_solve`/`krylov_solve`/`classify_matrix`/`LinearSolver` bindings, parallel SpMV determinism.
- `test_native_newton.py`: every nonlinear corpus `.op` against the oracle, error parity of `solve_dc_nr`, `init_condition` handling, Krylov stats, compile-once gmin stepping with numerical-only retries.
- `test_native_transient.py`: the bitwise time grid, every corpus `.tran` waveform, the cached factorization of linear constant-step runs, the progress callback, the P3 rulings at the facade level.
- `test_native_ac.py`: every corpus full `.ac` sweep, `points` 1 and 2, `f_start = 0`, singular systems (`AC_ERROR`, lowest frequency for every thread count), Krylov stats.
- `test_native_hb.py`: Fourier helpers bitwise, every corpus `hb` analysis within 1e-6 (the 205-harmonic mixer included), analytic RC phasor, the `MNA_HB_MAX_NNZ` memory policy, the `harmonics <= 0` ruling, Krylov metadata.
- `test_native_mor.py`: every corpus MOR analysis (basis sizes and reduced outputs), `_safe_solve` stages, the POD rank rule, training indexes, the TPWL loop and LRU cache, orthonormal bases on the former noise cases.
- `test_native_convergence.py`: the P3b convergence improvements (`CPP_MIGRATION_PLAN.md` 2.4.1): junction limiting on a diode across an ideal source, unchanged reference iterations on the 1 ohm series versions, the transient junction strategy, gmin continuation on 4/6/8-stage CMOS chains, node-to-ground gmin on a floating island, BJT exponent extrapolation continuity, damped HB iteration counts and restart, and 1 vs 4 thread determinism on these paths.
- `test_native_leaks.py`: 200 SRAM 10x10 `.op` solves and 50 SRAM transients through the service after a warm-up of 20 each; peak RSS (`resource.getrusage`) and, on Linux, current RSS (`/proc/self/statm`) may grow by less than 20 MB. Skipped on Windows unless `psutil` is importable.

### `tests/device_reference/`, `tests/test_spice_devices.py`
- `device_reference/`: independent NumPy implementations of the phase-M2 equations written from SPICE3f5 (`spice_diode.py` with `breakdown_voltage`, `gummel_poon.py`, `mos_card.py`, the clip in `common.py`); `test_reference_jacobians.py` checks their Jacobians by central differences over bias grids in every region, `test_native_vs_reference.py` compares the native kernels and the compiled-circuit path with them to 1e-12 (boundaries and clip regions included) and checks `gamma = 0` against `eval_mos_level1` bit for bit.
- `test_spice_devices.py`: model-card circuits built from `ComponentRecord`s: internal nodes and labels, defaults and validation, the 1N4148 law with series resistance, zener breakdown, junction limiting, a 2N2222 common-emitter stage (`.op` beta against the reference, `.ac`, `.tran`, `.hb`, TPWL, linear MOR), a 2N2907 mirror, the NMOS body-effect threshold shift, a CMOS inverter `.tran`, a diode rectifier `.hb`, JSON-safe operating points, and thread-count determinism.

### `tests/test_spice_dialect.py`, `tests/vendor/`, `tests/test_vendor_models.py`
- `test_spice_dialect.py`: the SPICE front end construct by construct (continuations, comments, case folding, `.model` cards and their diagnostics, nested `.subckt`/`X` flattening and names, `.param`/`{...}` evaluation, `.include`/`.lib` through a temporary directory, the SPICE source forms and their `.op`/`.ac`/`.tran` split, the current-source sign inside subcircuits, every error text) and end-to-end `.model` cards on the phase-M2 equations.
- `vendor/vendor_parts.lib`: cards and macromodels in the formats manufacturers publish (1N4148 with string-valued bookkeeping fields, 1N4007, a zener, 2N2222A/2N2907A/2N3904, a LEVEL=3 MOS card, a card with UO/TOX and no KP, a VDMOS card, a parameterised op-amp macromodel, a current mirror with an internal bias source).
- `test_vendor_models.py`: that library end to end through `SimulationService` and the CLI (`--lib`): diode law, rectifier `.tran`, zener regulation, BJT bias and PNP stage, LEVEL=3 switch, derived KP, VDMOS error, op-amp gain and bandwidth, mirror polarity.

### `tests/test_determinism.py`
- Every corpus analysis through `SimulationService.run_request` at `num_threads` 1 and 4 (a private 4-thread `ThreadBudget`): the whole response must be bitwise identical (floats compared by bit pattern), except the metadata that reports the thread grant, timings, or memory (`threads_requested`, `threads_used`, `elapsed_ms`, `memory_bytes`, keys ending in `_ms`/`_seconds`). Analyses that raise on both runs are skipped after checking that the errors agree; the NumPy RNG is reseeded and the TPWL cache cleared before every run.

### `tests/test_front_end_fast_path.py`
- The P6a parser/builder front end against the oracle: equal `CircuitIR` for every corpus case and workload, bitwise `G`/`C`/`b_dc`/`b_ac` at gmin 0 and 1e-9, the native circuit's pattern and tables, `parse_value` on ~200 literals, and malformed netlists (same exception class, code, and message).

### `tests/test_option_coercion.py`
- Strict request-option coercion in `PythonBackend.build_options` (2.4.1): booleans, integers, metric-suffix numbers, `init_condition` forms; every rejection is a tagged plain `ValueError` (HTTP 400); `TRAN_ERROR` is a `NetlistError`.

### `tests/test_thread_budget.py`
- `ThreadBudget` grants (explicit, `auto` fair share), blocking and release, timeouts, `MNA_NUM_THREADS` parsing, `coerce_num_threads` validation, and reference-counted BLAS pinning.

### `tests/test_thread_options.py`
- `num_threads` through the backend (coercion and rejection), the service lease (caps, `auto` sharing, release on failure), result metadata, `healthcheck()`, and the CLI `--threads` option.

### `tests/test_server_api.py`
- FastAPI app: lifespan executor, `/health` core and budget fields, `/api/simulate` thread metadata and 400 mappings (`INVALID_OPTIONS` versus netlist and solver errors), concurrent requests sharing the budget, and `num_threads` on `/ws/dyn`.

### `tests/test_parser_and_builder.py`
- Parser and MNA builder regression tests.

### `tests/test_netlist_extensions.py`
- Extended netlist/device tests.
- Covers BJT/MOS parsing and small-signal stamping.
- Covers FUNC parsing/evaluation and Level-1 MOS behavior.

### `tests/test_harmonic_balance.py`
- Harmonic balance regression tests, including frequency handling, close-tone beat/difference behavior, and reconstruction behavior.

### `tests/test_schematic_pipeline.py`
- Schematic conversion and hierarchy tests.
- Covers top-level net naming, 3-level hierarchy flattening, logical port names, and port mismatch failures.

### `tests/test_service.py`
- Service-level API response tests.

### `tests/test_krylov_solvers.py`
- Krylov rank/method option tests.
- Verifies general, symmetric, and positive-definite routing plus explicit method override.
- Verifies sparse RLC MNA structure, sparse transient Krylov metadata, and energy convergence checks.

### `tests/test_physical_sram.py`
- Level-1 MOS and SRAM behavior tests.
- Verifies inverter convergence, selected 6T cell write/hold/read behavior, and full 10x10 SRAM transient execution.

### `tests/test_python_core_mode.py`
- Python mode (`MNA_CORE=python`): `resolve_core`/`create_service` selection, the frozen package actually running, the error when the oracle is missing, `.op`/`.tran`/`.ac` agreement with the C++ core, metadata and health fields, the same option 400s as the C++ mode, live error classes, `/health`, `/api/simulate` and `/ws/dyn` on a Python-mode server, and the CLI `--core` option.

### `tests/test_mor.py`
- Selected-output MOR tests.
- Covers output selectors, auto/manual MOR order, backend option parsing, linear AC/transient accuracy, Level-1 `.op` linearization before MOR, TPWL cache reuse, display subset validation, and `.op`/`.hb` fallback metadata.

### `tests/perf/`
- `test_perf_harness.py` (always runs): the registry lists every unit of `CPP_MIGRATION_PLAN.md` section 5, every reference callable runs once on smoke inputs, available native units match the reference, the JSON schema and Markdown table, `--strict`, and `compare_backends.py --list` in a subprocess.
- `test_perf_gate.py` (marker `perf`, needs `--run-perf`): speedup floors per unit id at one core thread with the `--quick` methodology (`FLOORS`: device evaluation 20x for the SRAM, 3x and 1x for the tiny BJT and diode nets; Jacobian assembly 10x; circuit build 7.5x/9x and SRAM parse 1.7x; RLC direct solves 10x, SRAM direct solves 1.5x; Krylov 1x; `.op` SRAM 5x; SRAM and BJT transients 5x, RLC transients 10x; RLC AC 5x; HB 2x; MOR basis 15x, TPWL training 75x, reduced AC 19x, reduced transient 37x) plus RLC AC 4-thread scaling >= 1.5x on machines with >= 4 CPUs; failures print both timings.
- `test_concurrent_requests.py`: the P6 exit criterion through `benchmarks/concurrent_requests.py`. The `perf` test asserts that 4 concurrent SRAM 10x10 `.tran` requests (`num_threads="auto"`, in-process ASGI, best of 3 interleaved batches) take at most 1.5x the wall time of one; it currently fails (about 4x, `PERFORMANCE_PLAN.md` section 10). An always-on smoke test drives the same path on a 2x2 SRAM.

## Benchmarks: `benchmarks/`

### `benchmarks/benchmark_basic.py`
- Lightweight benchmark for the basic solver path.

### `benchmarks/workload_netlists.py`
- Netlist generators for the representative workloads (10x10 Level-1 SRAM, 23x23 RLC mesh, three-stage BJT amplifier, close-tone HB mixer).

### `benchmarks/benchmark_workloads.py`
- Wall-clock table (optionally with cProfile hot spots) for the workloads referenced by `PERFORMANCE_PLAN.md`.

### `benchmarks/blas_contention.py`
- Runs concurrent simulations in separate processes under different `OPENBLAS_NUM_THREADS` settings to show the multithreaded-BLAS oversubscription collapse.

### `benchmarks/concurrent_requests.py`
- Sends N identical SRAM 10x10 `.tran 6n 0.25n` requests to `POST /api/simulate` at once (`asyncio.gather` over `httpx`) and reports the best and median batch wall time per N, `ratio(N) = best(N) / best(1)`, and each response's `threads_used` and `elapsed_ms`.
- `--server asgi` (default: the app in this process through `httpx.ASGITransport`, lifespan entered) or `--server uvicorn` (a uvicorn subprocess on a free port, real HTTP); `--n`, `--batches` (3), `--num-threads` (`auto`), `--step`, `--json`, `--max-ratio` (exit 1 above it). `measure()` and `render()` are used by `tests/perf/test_concurrent_requests.py`.

### `benchmarks/perf_units.py`
- Registry of Python-vs-C++ comparison units (27, categories `device_eval`, `assembly`, `linear_solve`, `krylov`, `newton`, `transient`, `ac`, `hb`, `mor`) with the workload sizes of `PERFORMANCE_PLAN.md` 4.1.
- Each `Unit` has `id`, `category`, `size`, `setup(smoke=False)`, `reference(inputs)` (a frozen `mna_reference` callable), optional `core_setup(inputs)`, `core(inputs, num_threads)`, `parity(ref_out, core_out)` (max relative error or MOR subspace distance), `phase`, and `tolerance`.
- Every unit has a native side (the live facades or `_core` calls); `UnitUnavailable` remains for units added before their implementation.
- The HB H=64 unit uses the mixer with tones on a 20 MHz grid, because the close-tone mixer forces H >= 201.

### `benchmarks/compare_backends.py`
- CLI: `--list`, `--units`, `--category`, `--threads 1,2,4,N` (default 1, 2, 4 plus the CPU count when larger), `--reps` (7) / `--quick` (3), `--json` (default `benchmarks/results/<UTC timestamp>_<hostname>.json`), `--markdown`, `--skip-unavailable` (default) / `--strict`, `--smoke` (tiny inputs for tests).
- Method: one warm-up, GC disabled while timing, `perf_counter_ns`, median and MAD; reference with BLAS pinned to one thread, core at each thread count; parity from the warm-up outputs.
- Writes JSON `schema_version` 1 (host with platform, CPUs, versions, `core_version`/`build_info`, git commit, load average and a warning above 0.5; methodology; per-unit timings, speedups, parity) and prints a Markdown table (`n/a (P<n>)` for units without a native side).

### `benchmarks/results/`
- Output directory of `compare_backends.py`; run files are git-ignored. Committed: `baseline-linux.json` and `.md`, a full run (`--threads 1,4`, 7 repetitions) on an idle 4-CPU Linux VM at commit `057478f` (after P3b), and `baseline-windows.json` and `.md`, the same run on an idle 16-thread Windows 11 laptop (MSVC Release) at commit `4b0d65a`. `README.md` documents the format and the policy.

### `benchmarks/prototypes/`
- Research prototypes behind the plan: vectorized NumPy, Numba, and pybind11 C++ Level-1 MOS kernels, a kernel comparison harness, and a cached sparse-LU versus dense solve benchmark. Not imported by the package.

## CI: `.github/workflows/`

### `.github/workflows/ci.yml`
- On push, pull request, and manual dispatch: `pip install -e ".[dev]"` (builds the extension) and `pytest` on Ubuntu and Windows with Python 3.12 and 3.13, then the C++ doctest suite through CMake + ctest (Ninja + GCC on Linux, Visual Studio on Windows).
- ASan + UBSan job and a C++ coverage job (gcovr XML uploaded as an artifact) on Linux; both cover the ctest suite only.

### `.github/workflows/wheels.yml`
- On `v*` tags, manual dispatch, and pushes to `dev`/`main` that change `cpp/**`, `pyproject.toml`, or the workflow: cibuildwheel builds one abi3 `cp312` wheel per platform (64-bit, no musllinux), smoke-imports `_core`, repairs Windows wheels with `delvewheel`, and uploads them. The test suite is not run against the wheels.

### `.github/workflows/perf.yml`
- Weekly schedule and manual dispatch, Ubuntu and Windows, Python 3.13: runs `benchmarks/compare_backends.py --quick`, appends the Markdown table to the job summary, and uploads `perf-results.json` and `perf-table.md`; `continue-on-error` so it never blocks.

## Legacy: `legacy/`

- Historical simulator snapshots.
- Useful for comparisons, but not part of the active runtime path.

## Practical Reading Order

1. `mna_simulation/api/contracts.py`
2. `mna_simulation/netlist.py`
3. `mna_simulation/mna_builder.py`
4. `mna_simulation/solvers.py`
5. `mna_simulation/schematic_pipeline.py`
6. `mna_simulation/api/service.py`
7. `app/server/main.py`
8. `app/web/src/lib/schematic.ts`
9. `app/web/src/pages/App.tsx`
10. `mna_simulation/mor/linear.py`
11. `mna_simulation/mor/tpwl.py`
12. `app/web/src/components/SchematicCanvas.tsx`
13. `CPP_MIGRATION_PLAN.md`
14. `mna_simulation/native/threads.py`
15. `cpp/bindings/module.cpp`
16. `benchmarks/perf_units.py`
