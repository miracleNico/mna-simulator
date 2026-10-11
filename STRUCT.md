# Project Structure and File Functions

This document describes the project-authored source, config, docs, and tests. It intentionally ignores dependency trees such as `app/web/node_modules`.

## End-to-End Flow

1. A request enters from CLI (`mna_simulation/cli.py`), HTTP (`app/server/main.py`), or the React schematic editor.
2. `SimulationService` (`mna_simulation/api/service.py`) normalizes request handling.
3. Raw netlists go directly to the parser. Schematic JSON goes through `mna_simulation/schematic_pipeline.py`.
4. The schematic pipeline validates hierarchy, flattens `SUBCKT` instances, forms nets, emits canonical netlist text, and parses it back into `CircuitIR`.
5. `mna_simulation/backends/python_backend.py` builds `AnalysisOptions`.
6. `mna_simulation/netlist.py` and `mna_simulation/mna_builder.py` produce the MNA problem.
7. Optional selected-output MOR in `mna_simulation/mor/` reduces `.ac` and `.tran` problems when requested.
8. `mna_simulation/solvers.py` executes `.op`, `.tran`, `.ac`, or `.hb`.
9. Results return through the API service and are plotted by the frontend.

## Root Files

### `README.md`
- Project-level usage guide.
- Covers setup, CLI/API usage, hierarchy behavior, supported devices/modes, tests, and frontend startup.

### `USER_GUIDE.md`
- User-facing guide for the web app.
- Mirrors the main topics exposed by the in-app Help dialog: demos, schematic editing, hierarchy, analyses, result display, Krylov subspace solver controls, raw netlist mode, and troubleshooting.

### `STRUCT.md`
- This file-by-file architecture reference.

### `PERFORMANCE_PLAN.md`
- Measured performance review of the Python core (profiles of the SRAM, RLC mesh, BJT amplifier, and harmonic-balance workloads).
- Prioritized plan for multithreading and solver changes, and the research on a C++ kernel migration.

### `pyproject.toml`
- Python package metadata.
- Declares runtime dependencies (`numpy`, `scipy`, `sympy`, `fastapi`, `uvicorn`, etc.).
- Declares dev dependencies (`pytest`, `httpx`).
- Registers CLI entrypoint: `mna-sim = mna_simulation.cli:main`.

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

## API Layer: `mna_simulation/api/`

### `mna_simulation/api/contracts.py`
- Shared contracts for parser, builder, solver, service, and UI.
- Contains analysis enums, IR dataclasses, result dataclasses, API Pydantic models, and schematic graph models.
- `SchematicComponent` includes hierarchy metadata, arbitrary device metadata, and transistor defaults.
- `AnalysisOptions` carries Krylov options plus selected-output MOR options (`use_mor`, `mor_method`, `mor_order`, `mor_output_nodes`, and `mor_validate`).

### `mna_simulation/api/service.py`
- Orchestration entrypoint used by CLI and FastAPI.
- Routes schematic payloads through `schematic_pipeline`.
- Adds generated canonical netlist text to response metadata.
- Applies optional output decimation to large waveform responses.

## Backends: `mna_simulation/backends/`

### `mna_simulation/backends/python_backend.py`
- Active backend adapter.
- Parses netlists, builds options from directives, runs analysis, and exposes capability flags.
- Coerces Krylov rank/method request options, including aliases such as `arnoldi`, `gmres`, `minres`, and `cg`.
- Coerces MOR method/order/output options and advertises model-order reduction capability.

### `mna_simulation/backends/cpp_backend.py`
- Placeholder adapter for a future C++ core.

## Model Library: `mna_simulation/library/`

### `mna_simulation/library/registry.py`
- Registry abstraction for built-in and user-defined models.
- Loads optional JSON model cards.

### `mna_simulation/library/models/behavioral_resistor.json`
- Example registry model-card metadata.

## Server App: `app/server/`

### `app/server/main.py`
- FastAPI entrypoint.
- Exposes:
  - `GET /health`
  - `POST /api/simulate`
  - websocket dynamic simulation route used by frontend display/probe flows
- Translates simulator errors into HTTP 400 payloads.
- Dynamic websocket requests accept Krylov options and cap display frame rate while continuing looped playback.

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
- Defines analysis state for Krylov enablement plus MOR enablement, method, auto/manual order, and separate MOR outputs.
- Expands frontend `NODE` markers into backend `port_<name>` junctions.
- Repairs stale `SUBCKT` wire endpoint names before sending payloads.

### `app/web/src/lib/userGuide.ts`
- Structured content for the in-app Help dialog.
- Covers the same user-facing topics as the root `USER_GUIDE.md`.

### `app/web/src/lib/demoPresets.ts`
- Hidden demo schematics.
- Includes CE amplifier, BJT 3-stage amplifier, GHz close-tone HB diode mixer, hierarchical 10x10 Level-1 MOS SRAM, and 23x23 RLC mesh demos.
- 3-stage amplifier uses real `SUBCKT` child levels and editable `NODE` ports rather than hardcoded fixed port junctions.
- Close-tone HB mixer uses 1 GHz and 1.005 GHz current tones into a nonlinear diode load plus a high-Q 5 MHz envelope tank, so HB exposes the steady-state beat while transient would need 40+ envelope periods to settle.
- Large SRAM uses a reusable `SRAM 6T Cell` child level with physical Level-1 MOS devices, while the root canvas keeps only array cell instances plus FUNC-driven row/column drivers.
- Large RLC mesh is sized for roughly 1000 MNA unknowns and sparse Krylov benchmarking.

### `app/web/src/lib/plot.ts`
- Canvas plotting helpers for line, stem, and bar visualizations.
- Used by `.op`, `.tran`, `.ac`, `.hb`, and dynamic plots.

### `app/web/src/lib/symbols.ts`
- Canvas symbol drawing for schematic components.
- Includes passives, sources, controlled sources, transistor symbols, labels, nodes, and subcircuits.

### `app/web/src/components/Toolbar.tsx`
- Component toolbar.
- Exposes primitives, controlled sources, transistors, `SUBCKT`, `LABEL`, `NODE`, probe, and run controls.
- Exposes Help, which opens the detailed user-guide dialog from the toolbar.

### `app/web/src/components/LeftPane.tsx`
- Simulation controls, hierarchy tree, library list, and property editor.
- Supports `SUBCKT` displayed/entity name, node count, editable pin names, transistor small-signal fields, and display-node controls.
- Exposes Krylov subspace solver algorithm selection, method-specific manual value wording, display-node clear, and simulation-control reset.
- Exposes MOR controls with method selection including Linear Krylov MOR, auto/manual order, and output-node selection separate from display nodes.
- Contains the editable netlist panel. Editing asks for confirmation, clears schematic hierarchy, and switches runs to raw `netlist_text`.

### `app/web/src/components/SchematicCanvas.tsx`
- Schematic graph editor.
- Handles hit-testing, selection, dragging, wiring, marquee selection, copy/paste, deletion, rotation, hover badges, and drawing.
- Allows `NODE` markers to be selected and edited while still acting as wire endpoints.

### `app/web/src/components/PlotTile.tsx`
- Movable/resizable plot tile wrapper.
- Supports static and dynamic streaming plots.

### `app/web/src/pages/App.tsx`
- Main frontend orchestration component.
- Owns hierarchy levels, active level, analysis settings, selection, tiles, generated netlist preview, and API calls.
- Owns the Help dialog state and renders the in-app user guide.
- Synchronizes child `NODE` port renames to only the matching `SUBCKT` instances.
- Rewrites stale `SUBCKT` wire endpoints when pins are renamed.
- Tracks schematic mode versus raw netlist mode and sends the matching API payload.
- Passes Krylov method/rank options through static simulation and dynamic websocket requests.
- Passes MOR options through static simulation requests and keeps display-node filtering independent from MOR output selection.

### `app/web/src/components/AnalysisPanel.tsx`
- Older/legacy panel component retained in the tree.
- The current UI path is driven primarily through `LeftPane.tsx` and `App.tsx`.

## Tests: `tests/`

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

## Benchmarks: `benchmarks/`

### `benchmarks/benchmark_basic.py`
- Lightweight benchmark for the basic solver path.

### `benchmarks/workload_netlists.py`
- Netlist generators for the representative workloads (10x10 Level-1 SRAM, 23x23 RLC mesh, three-stage BJT amplifier, close-tone HB mixer).

### `benchmarks/benchmark_workloads.py`
- Wall-clock table (optionally with cProfile hot spots) for the workloads referenced by `PERFORMANCE_PLAN.md`.

### `benchmarks/blas_contention.py`
- Runs concurrent simulations under different `OPENBLAS_NUM_THREADS` settings to show the multithreaded-BLAS oversubscription collapse.

### `benchmarks/prototypes/`
- Research prototypes behind the plan: vectorized NumPy, Numba, and pybind11 C++ Level-1 MOS kernels, a kernel comparison harness, and a cached sparse-LU versus dense solve benchmark. Not imported by the package.

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
