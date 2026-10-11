# Project Structure and File Functions

This document describes the project-authored source, config, docs, and tests. It intentionally ignores dependency trees such as `app/web/node_modules`.

## End-to-End Flow

1. A request enters from CLI (`mna_simulation/cli.py`), HTTP (`app/server/main.py`), or the React schematic editor.
2. `SimulationService` (`mna_simulation/api/service.py`) normalizes request handling, leases worker threads from the machine-wide thread budget (`mna_simulation/native/threads.py`), and pins NumPy/SciPy BLAS to one thread for the run.
3. Raw netlists go directly to the parser. Schematic JSON goes through `mna_simulation/schematic_pipeline.py`.
4. The schematic pipeline validates hierarchy, flattens `SUBCKT` instances, forms nets, emits canonical netlist text, and parses it back into `CircuitIR`.
5. `mna_simulation/backends/python_backend.py` builds `AnalysisOptions`.
6. `mna_simulation/netlist.py` and `mna_simulation/mna_builder.py` produce the MNA problem.
7. Optional selected-output MOR in `mna_simulation/mor/` reduces `.ac` and `.tran` problems when requested.
8. `mna_simulation/solvers.py` executes `.op`, `.tran`, `.ac`, or `.hb`.
9. Results return through the API service (with `threads_used`, `core_version`, and `elapsed_ms` metadata) and are plotted by the frontend.

The numerical core is being moved into the mandatory C++ extension `mna_simulation._core` (`cpp/`), one unit at a time behind the existing Python functions (`CPP_MIGRATION_PLAN.md`). Every analysis still runs on the Python solvers today; the extension currently provides `version`, `build_info`, `klu_solve`, `parallel_row_sums`, `effective_threads`, and `binding_units`. The pure-Python core as of the migration start is frozen as the test oracle `tests/reference/mna_reference`.

## Root Files

### `README.md`
- Project-level usage guide.
- Covers setup, CLI/API usage, hierarchy behavior, supported devices/modes, tests, and frontend startup.
- "Threads and Concurrency" section: the `MNA_NUM_THREADS` budget, the `num_threads` request option, `--threads`, BLAS pinning, and the recommended `OPENBLAS_NUM_THREADS=1` server start.

### `USER_GUIDE.md`
- User-facing guide for the web app.
- Mirrors the main topics exposed by the in-app Help dialog: demos, schematic editing, hierarchy, analyses, threads, result display, Krylov subspace solver controls, raw netlist mode, and troubleshooting.

### `STRUCT.md`
- This file-by-file architecture reference.

### `DESIGN_AUDIT.md`
- Front end design audit: findings from driving the app in headless Chromium, what was changed, and an LTspice feature comparison with suggested next steps.

### `.gitignore`
- Ignores Python bytecode and tool caches, packaging outputs (anchored to the root: `/build/`, `/dist/`, `/wheelhouse/`, `*.whl`, `*.egg-info/`), native extension artefacts (`*.so`, `*.pyd`, `cpp/build*/`, `/build-cpp/`), coverage outputs, and Vite caches.
- Ignores perf comparison runs (`benchmarks/results/*.json`) except the committed per-platform baselines (`!benchmarks/results/baseline-*.json`).
- `app/web/node_modules` and `app/web/dist` remain tracked.

### `PERFORMANCE_PLAN.md`
- Measured performance review of the Python core (profiles of the SRAM, RLC mesh, BJT amplifier, and harmonic-balance workloads).
- Prioritized plan for multithreading and solver changes, and the research on a C++ kernel migration.
- Section 4.1 (wall-clock per workload) and 4.3 (prototype kernels) fix the workload sizes used by `benchmarks/perf_units.py`.

### `CPP_MIGRATION_PLAN.md`
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
- Emits backend-neutral `CircuitIR`.

### `mna_simulation/mna_builder.py`
- Converts `CircuitIR` to `MnaProblem`.
- Builds dense and sparse conductance/capacitance matrices, RHS vectors, nonlinear vectors, labels, and branch maps.
- Stamps passive devices, sources, controlled sources, diodes, BJTs, and MOSFETs.
- Stamps Level-1 BJT/MOS nonlinear device metadata and junction/gate capacitances for transient analysis.

### `mna_simulation/solvers.py`
- Numerical solver library.
- Implements Newton-Raphson DC, Backward-Euler transient, linear AC sweep, harmonic balance, and time reconstruction helpers.
- Solves physical Level-1 BJT/MOS operating points, linearizes them for `.ac`, and keeps nonlinear device I-V equations active in `.tran`.
- Implements Krylov routing for Arnoldi/GMRES, MINRES/CR, and Conjugate Gradient.
- Allows explicit Krylov method overrides even when matrix classification would choose another route.
- Uses cached sparse CSR Backward-Euler operators for linear transient Krylov runs when sparse MNA matrices are available.

### `mna_simulation/core_basic.py`
- Facade for `.op`, `.tran`, `.ac`, and matrix inspection.
- Adds solver metadata for Krylov method, requested method, matrix dimension, sparse density, iteration count, rank mode, and fallback status.
- Routes selected-output MOR for `.ac`, linear `.tran`, and nonlinear TPWL `.tran`; `.op` and matrix inspection report MOR as intentionally unused.

### `mna_simulation/core_advanced.py`
- Facade for `.hb`.
- Infers base frequency from sinusoidal sources and optionally reconstructs time-domain waveforms from harmonic-balance coefficients.
- Passes Krylov rank/method options through harmonic-balance linear solves.
- Reports MOR as intentionally unused for `.hb` v1.

### `mna_simulation/mor/`
- Decoupled model-order reduction package.
- `selectors.py` maps labels such as `V(out)` and `I(L1)` to selector matrices and validates display-node subsets.
- `metadata.py` resolves automatic MOR order and emits compression/cache metadata.
- `linear.py` implements output-aware rational Krylov/PRIMA-like projection for linear and `.op`-linearized `.ac` plus linear transient. Large sparse linear transients can validate selected outputs against the sparse full-order Krylov path and fall back when the reduced model underfits or becomes unstable.
- `tpwl.py` implements TPWL/POD nonlinear transient MOR with an in-memory LRU cache.

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

## Native Package: `mna_simulation/native/`

### `mna_simulation/native/__init__.py`
- Import guard for the mandatory extension `mna_simulation._core` (built from `cpp/`; there is no Python fallback).
- `require_core()` returns the module or raises an `ImportError` that explains how to build it (`pip install -e ".[dev]"`, CMake >= 3.26, a C++17 compiler).
- `core_version()` and `build_info()` (compiler, Eigen/SuiteSparse/KLU versions, build flags) for health checks and result metadata.

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
- Leases threads from a `ThreadBudget` (the process default unless one is injected) and runs the backend inside `pin_blas_threads()`; adds `threads_requested`, `threads_used`, `thread_budget_total`, `core_backend` (`"python"` until the cut-overs), `core_version`, and `elapsed_ms` to result metadata.
- Tags option-validation `ValueError`s with a PEP 678 note (`is_invalid_option_error`) so the HTTP layer can answer 400 without catching solver `ValueError`s.
- `healthcheck()` also reports `core_version`, `build_info`, and the thread budget snapshot.

## Backends: `mna_simulation/backends/`

### `mna_simulation/backends/python_backend.py`
- Active backend adapter.
- Parses netlists, builds options from directives, runs analysis, and exposes capability flags.
- Coerces Krylov rank/method request options, including aliases such as `arnoldi`, `gmres`, `minres`, and `cg`.
- Coerces MOR method/order/output options and advertises model-order reduction capability.
- Coerces `num_threads` with `coerce_num_threads` into `AnalysisOptions.num_threads`.

### `mna_simulation/backends/cpp_backend.py`
- Placeholder adapter for a future C++ core.

## Model Library: `mna_simulation/library/`

### `mna_simulation/library/registry.py`
- Registry abstraction for built-in and user-defined models.
- Loads optional JSON model cards.

### `mna_simulation/library/models/behavioral_resistor.json`
- Example registry model-card metadata.

## Native Core: `cpp/`

The C++17 sources of `mna_simulation._core`, built by scikit-build-core (wheel and editable installs) or directly with CMake (C++ tests). Only the phase-0 scaffold is committed; later units add files under `include/mna/`, `src/`, `bindings/`, and `tests/` without editing CMake.

### `cpp/CMakeLists.txt`
- CMake >= 3.26 project `mna_core` (C++17, C11, hidden symbol visibility, PIC).
- Targets: `klu_vendored` (SuiteSparse_config + AMD + COLAMD + BTF + KLU compiled as C with `NTIMER`, guarded by a source-count check), `mna_core` (static library from every `src/**/*.cpp`, Eigen with `EIGEN_MPL2_ONLY` and `EIGEN_DONT_PARALLELIZE`), `_core` (nanobind module, `STABLE_ABI NB_STATIC`, from every `bindings/*.cpp`, installed into `mna_simulation/`), and `mna_core_tests` (doctest, every `tests/*.cpp`, registered with ctest).
- Options: `MNA_BUILD_TESTS`, `MNA_BUILD_PYTHON_MODULE` (`AUTO`/`ON`/`OFF`; forced on under scikit-build), `MNA_ENABLE_SANITIZERS` (ASan + UBSan), `MNA_ENABLE_COVERAGE`, `MNA_NATIVE_ARCH` (never for wheels). Warnings `/W4` or `-Wall -Wextra -Wpedantic` on project targets only.

### `cpp/include/mna/error.hpp`, `cpp/src/error.cpp`
- `mna::Error` (a `std::runtime_error`) with an `ErrorCode` whose names mirror the Python core's codes (`NETLIST_FATAL`, `NR_CONVERGENCE`, `JACOBIAN_SINGULAR`, `HB_ERROR`, ...), so errors translate 1:1 to Python.

### `cpp/include/mna/version.hpp`, `cpp/src/build_info.cpp`
- Version `0.70.0-dev` and `BuildInfo` (compiler and version, C++ standard, build type, thread implementation, hardware concurrency, Eigen/SuiteSparse/KLU versions, platform, sanitizer/coverage/native-arch/debug-assert flags).

### `cpp/include/mna/threads.hpp`
- Header-only `std::thread` parallel loop: `parallel_for` / `parallel_for_chunks` over at most 256 chunks whose boundaries depend only on the item count (bitwise-identical reductions for every thread count), `num_threads <= 1` runs inline, the first worker exception is rethrown after all threads join.
- `effective_threads(requested)` clamps a request to `[1, hardware_threads()]`. No global state: the thread count is an argument of every call.

### `cpp/include/mna/linear/klu_solver.hpp`, `cpp/src/linear/klu_solver.cpp`
- RAII `KluSolver` over real KLU with 32-bit indices: `analyze(n, Ap, Ai)` once per pattern, `factor(Ax)`, `refactor(Ax)` reusing the pivot order with a full-factor fallback (`FactorPath`), in-place `solve(b)`; validated `CscPattern`.
- Errors: `NETLIST_FATAL` for malformed input or calls out of order, `JACOBIAN_SINGULAR` for a zero pivot, `std::bad_alloc` on memory exhaustion.

### `cpp/bindings/module.cpp`
- Defines `NB_MODULE(_core)`: `CoreError` (a `RuntimeError` with `.code` and `.message`, translated from `mna::Error`), `version()`, `build_info()`, `effective_threads()`, `klu_solve(Ap, Ai, Ax, b)` (CSC int32, GIL released), and `parallel_row_sums(a, num_threads)` (deterministic threaded smoke kernel).
- Then calls every unit registered through `registry.hpp`.

### `cpp/bindings/registry.hpp`
- Self-registration scheme for binding units: each `bindings/bind_<unit>.cpp` defines `bind_<unit>(nanobind::module_&)` and a static `Registrar{"<unit>", &bind_<unit>}`; the registry is a function-local static, so new units never edit `module.cpp` or CMake.

### `cpp/bindings/bind_meta.cpp`
- Example registered unit; exposes `binding_units()` (names of the registered units in registration order).

### `cpp/tests/`
- doctest unit tests built as `mna_core_tests` and run by ctest: `main.cpp` (doctest main), `test_build_info.cpp` (version and library versions), `test_klu_solver.cpp` (known systems, several right-hand sides, refactor and its fallback, singular matrices, 1x1, n = 0, malformed input, move semantics), `test_threads.cpp` (chunk tiling, clamping, coverage, inline execution, bitwise-identical reductions, exception propagation).

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
- `GET /health` adds `core_version`, `build_info`, and the thread budget; `/ws/dyn` accepts `num_threads` and reports `threads_used` and `elapsed_ms` in its `meta` message.

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

### `tests/test_native_convergence.py`
- P3b convergence improvements (`CPP_MIGRATION_PLAN.md` 2.4.1): junction limiting on a diode across an ideal source, unchanged reference iterations on the 1 ohm series versions, the transient junction strategy, gmin continuation on 4/6/8-stage CMOS chains, node-to-ground gmin on a floating island, BJT exponent extrapolation continuity, damped HB iteration counts and restart, and 1 vs 4 thread determinism on these paths.

### `tests/test_native_smoke.py`
- Smoke tests of `_core`: version and build info, `klu_solve` against NumPy, `CoreError` codes for singular and malformed input, deterministic `parallel_row_sums`.

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

### `tests/test_mor.py`
- Selected-output MOR tests.
- Covers output selectors, auto/manual MOR order, backend option parsing, linear AC/transient accuracy, Level-1 `.op` linearization before MOR, TPWL cache reuse, display subset validation, and `.op`/`.hb` fallback metadata.

### `tests/perf/`
- `test_perf_harness.py` (always runs): the registry lists every unit of `CPP_MIGRATION_PLAN.md` section 5, every reference callable runs once on smoke inputs, available native units match the reference (direct solves within 1e-9 at full size), the JSON schema and Markdown table, `--strict`, and `compare_backends.py --list` in a subprocess.
- `test_perf_gate.py` (marker `perf`, needs `--run-perf`): speedup floors per unit id at one core thread with the `--quick` methodology (device evaluation 20x, assembly 10x, RLC direct solve 10x, `.op` SRAM 5x, SRAM/BJT transients 5x, RLC transients 10x, RLC AC 5x, HB 2x, Krylov 1x, MOR basis and TPWL 2x) plus RLC AC 4-thread scaling >= 1.5x on machines with >= 4 CPUs; units without a native side are skipped without timing the reference; failures print both timings.

## Benchmarks: `benchmarks/`

### `benchmarks/benchmark_basic.py`
- Lightweight benchmark for the basic solver path.

### `benchmarks/workload_netlists.py`
- Netlist generators for the representative workloads (10x10 Level-1 SRAM, 23x23 RLC mesh, three-stage BJT amplifier, close-tone HB mixer).

### `benchmarks/benchmark_workloads.py`
- Wall-clock table (optionally with cProfile hot spots) for the workloads referenced by `PERFORMANCE_PLAN.md`.

### `benchmarks/blas_contention.py`
- Runs concurrent simulations under different `OPENBLAS_NUM_THREADS` settings to show the multithreaded-BLAS oversubscription collapse.

### `benchmarks/perf_units.py`
- Registry of Python-vs-C++ comparison units (24, categories `device_eval`, `assembly`, `linear_solve`, `krylov`, `newton`, `transient`, `ac`, `hb`, `mor`) with the workload sizes of `PERFORMANCE_PLAN.md` 4.1.
- Each `Unit` has `id`, `category`, `size`, `setup(smoke=False)`, `reference(inputs)` (a frozen `mna_reference` callable), optional `core_setup(inputs)`, `core(inputs, num_threads)`, `parity(ref_out, core_out)` (max relative error or MOR subspace distance), `phase`, and `tolerance`.
- Native side today: `direct_solve_rlc_1035` and `direct_solve_sram_248` through `_core.klu_solve` (CSC int32 of `G + C/h`); every other `_core_<unit id>` function raises `UnitUnavailable("lands in P<n>: ...")` until its phase replaces the body.
- The HB H=64 unit uses the mixer with tones on a 20 MHz grid, because the close-tone mixer forces H >= 201.

### `benchmarks/compare_backends.py`
- CLI: `--list`, `--units`, `--category`, `--threads 1,2,4,N` (default 1, 2, 4 plus the CPU count when larger), `--reps` (7) / `--quick` (3), `--json` (default `benchmarks/results/<UTC timestamp>_<hostname>.json`), `--markdown`, `--skip-unavailable` (default) / `--strict`, `--smoke` (tiny inputs for tests).
- Method: one warm-up, GC disabled while timing, `perf_counter_ns`, median and MAD; reference with BLAS pinned to one thread, core at each thread count; parity from the warm-up outputs.
- Writes JSON `schema_version` 1 (host with platform, CPUs, versions, `core_version`/`build_info`, git commit, load average and a warning above 0.5; methodology; per-unit timings, speedups, parity) and prints a Markdown table (`n/a (P<n>)` for units without a native side).

### `benchmarks/results/`
- Output directory of `compare_backends.py`; run files are git-ignored, only `baseline-linux.json` and `baseline-windows.json` are meant to be committed (refreshed deliberately from the Perf workflow artifacts at release time, idle machines only). `README.md` documents the format and the policy.

### `benchmarks/prototypes/`
- Research prototypes behind the plan: vectorized NumPy, Numba, and pybind11 C++ Level-1 MOS kernels, a kernel comparison harness, and a cached sparse-LU versus dense solve benchmark. Not imported by the package.

## CI: `.github/workflows/`

### `.github/workflows/ci.yml`
- On push, pull request, and manual dispatch: `pip install -e ".[dev]"` (builds the extension) and `pytest` on Ubuntu and Windows with Python 3.12 and 3.13, then the C++ doctest suite through CMake + ctest (Ninja + GCC on Linux, Visual Studio on Windows).
- ASan + UBSan job and a C++ coverage job (gcovr XML uploaded as an artifact) on Linux.

### `.github/workflows/wheels.yml`
- On `v*` tags and manual dispatch: cibuildwheel builds one abi3 `cp312` wheel per platform (64-bit, no musllinux), smoke-imports `_core`, repairs Windows wheels with `delvewheel`, and uploads them.

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
