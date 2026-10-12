# MNA Simulation

Layered circuit-simulation workbench built around Modified Nodal Analysis (MNA).

The project includes:

- A native C++17 numerical core (`mna_simulation._core`) for `.op`, `.tran`, `.ac`, `.hb`, the Krylov solvers, and model-order reduction, driven by a Python front end (parser, MNA builder, schematic pipeline, service).
- FastAPI service for schematic and netlist simulation.
- React + TypeScript schematic editor.
- Backend-owned schematic-to-netlist pipeline with hierarchy validation.
- Physical Level-1 BJT/MOSFET defaults, plus opt-in small-signal compatibility models.
- Sparse Krylov-capable solver path for large linear transient circuits.

## Highlights

- **Single validation path:** schematic JSON is flattened to canonical netlist text, then parsed by the same parser used for raw netlists.
- **Hierarchy-aware schematics:** `SUBCKT` instances connect to child ports through editable `NODE` markers. A child node named `in` exports as `port_in` and flattens as logical instance names such as `DiffAmp.in`.
- **Scoped child ports:** each child entity owns its own port names. Renaming ports in one child syncs only instances of that entity and repairs connected wire endpoints.
- **SPICE-style transistor flow:** `QNPN`, `QPNP`, `NMOS`, and `PMOS` default to physical Level-1 nonlinear models. `.op` solves bias, `.ac` linearizes around that bias, and `.tran` keeps solving nonlinear I-V equations.
- **Harmonic balance workflow:** `.hb <harmonics> [time_window]` supports harmonic-domain results and optional reconstructed time-domain waveforms.
- **Close-tone HB demo:** Frequency Domain Demos includes a nonlinear diode mixer driven by 1 GHz and 1.005 GHz tones, showing the 5 MHz beat/difference product in a high-Q envelope tank that would need 40+ beat periods of picosecond-step transient settling.
- **Krylov subspace solver controls:** API/UI requests can use automatic method selection or explicitly choose Arnoldi/GMRES, MINRES/CR, or Conjugate Gradient for full-system iterative solves. Auto Krylov rank resolves after MNA assembly as `ceil(0.5 * matrix_dimension)`.
- **Selected-output MOR:** optional model-order reduction keeps only cared outputs. Linear `.ac` and linearized `.tran` use output-aware rational Krylov projection; nonlinear `.tran` can use TPWL/POD with an in-memory ROM cache.
- **Sparse large-circuit path:** every direct solve runs through SuiteSparse KLU on the circuit's sparse pattern; linear transients reuse one factorization of `G + C/h` while the step size is unchanged, and linear transient Krylov runs use the CSR Backward-Euler operator.
- **Functional MOS SRAM demo:** the Large Scale Circuits menu includes a hierarchical 10x10 Level-1 MOS 6T SRAM demo with FUNC-driven write/hold/read timing; the root canvas shows cell instances while the reusable child level holds the transistor cell body.
- **Large RLC benchmark demo:** the 23x23 distributed RLC mesh produces a 1035-unknown sparse MNA system for Krylov/MINRES benchmarking.
- **Editable netlist mode:** the web app can switch from schematic mode to raw netlist editing after a confirmation that clears the schematic hierarchy.
- **Interactive visualization:** `.op`, `.tran`, `.ac`, `.hb`, and `.dyn` results render in movable plot tiles, with selectable display nodes.
- **LTspice-style schematic editing:** wire tool with corners and T-junction taps, junction dots, open-pin markers, sticky part placement with rotate/mirror on the cursor, undo/redo, duplicate, zoom-to-fit, right-drag pan, and working File / Edit / View / Simulate menus with keyboard shortcuts.
- **Workspace files:** save and reopen the whole hierarchy plus analysis settings as JSON, and export the generated SPICE netlist.
- **In-app user guide:** toolbar Help (F1) opens a detailed guide for demos, schematic editing, hierarchy, analysis modes, visualization, Krylov controls, and raw netlist mode.

## Repository Layout

- `mna_simulation/`: Python parser, MNA builder, solver facades, schematic pipeline, API service, and backend adapters.
- `mna_simulation/native/`: the Python side of the native core: import guard, circuit tables, expression compiler, linear-algebra and MOR facades, and the thread budget.
- `cpp/`: C++17 sources of the extension `mna_simulation._core` (headers, sources, nanobind bindings, doctest unit tests, vendored Eigen/SuiteSparse/doctest).
- `app/server/`: FastAPI app exposing `/health`, `/api/simulate`, and dynamic simulation websocket routes.
- `app/web/`: React + TypeScript schematic editor and demo UI.
- `tests/`: parser, builder, service, harmonic-balance, transistor, and hierarchy regression tests; native binding tests (`test_native_*.py`); the frozen Python oracle (`tests/reference/`) with the differential corpus (`tests/corpus.py`, `tests/differential/`); thread-determinism and leak tests; opt-in performance tests (`tests/perf/`).
- `benchmarks/`: benchmark entry points, the Python-vs-C++ comparison harness, the concurrent-request benchmark, workload netlists, committed results, and performance research prototypes.
- `CPP_MIGRATION_PLAN.md`: plan, edge-case rulings, and status of the move of the numerical core to C++.
- `PERFORMANCE_PLAN.md`: measured performance review, multithreading plan, C++ migration research, and the status of what landed.
- `legacy/`: older simulator snapshots kept for comparison.
- `USER_GUIDE.md`: user-facing web-app guide mirrored by the in-app Help dialog.
- `DESIGN_AUDIT.md`: front end design audit, changes made, and LTspice feature comparison.
- `STRUCT.md`: file-by-file architecture reference.

## Native Core

Every analysis runs in the compiled extension `mna_simulation._core`, built from `cpp/`:

- C++17, bound to Python with nanobind (stable ABI: one wheel serves CPython 3.12 and later) and built by scikit-build-core.
- Sparse direct solves with SuiteSparse KLU (with AMD, COLAMD, and BTF); dense linear algebra, complex arithmetic, and SVD with Eigen 3.4. Both are vendored, unmodified, under `cpp/third_party/` (versions, licenses, and the LGPL note for the statically linked KLU/BTF are in `cpp/third_party/README.md`), so a build downloads nothing.
- Deterministic internal threading: each call starts its own `std::thread` workers for the threads it was granted (no OpenMP, no process-wide pool) and splits work into chunks that depend only on the item count, so results are bitwise identical for every thread count (`tests/test_determinism.py`). Parallel work: device evaluation from 2048 devices, AC sweep frequencies, harmonic-balance samples and block factorizations, the reduced MOR AC sweep, MOR shift factorizations from 2048 unknowns, and Krylov matrix-vector products from 4096 unknowns. A transient steps sequentially; below those sizes a call runs on one thread whatever it was granted.
- The GIL is released for the whole native region of a call. FUNC time sources and `B` behavioral sources are parsed by SymPy once in Python and compiled to a bytecode that the core evaluates.
- The extension is mandatory: there is no Python numerical fallback. The pure-Python core as of the migration start survives only as the test oracle `tests/reference/mna_reference`, which never ships.

Build requirements (Linux and Windows are the supported, CI-tested platforms):

- Python >= 3.12, CMake >= 3.26, and a C++17 compiler (GCC or Clang on Linux, Visual Studio 2022 Build Tools on Windows). Ninja is optional (`pip install ninja`); CI uses it for the C++ unit tests on Linux.
- `python -m pip install -e ".[dev]"` compiles the extension (Release build into `build/<wheel tag>/`) and installs it into `mna_simulation/`. Editable installs do not rebuild on import: run `python -m pip install -e .` again after changing C++ sources or after adding or moving Python modules.
- An `ImportError` that names `mna_simulation._core` means the extension was not built for this interpreter; the message repeats the build command.

Wheels: the Wheels workflow (`.github/workflows/wheels.yml`; runs on `v*` tags, on manual dispatch, and on pushes to `dev`/`main` that change `cpp/`, `pyproject.toml`, or the workflow) builds one abi3 `cp312` wheel per platform for 64-bit Linux (manylinux) and Windows (repaired with `delvewheel`), smoke-imports `_core`, and uploads the wheels as artifacts. Installing a wheel needs no compiler.

C++ unit tests (doctest, registered with ctest) build without Python:

```bash
cmake -S cpp -B build-cpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNA_BUILD_TESTS=ON -DMNA_BUILD_PYTHON_MODULE=OFF
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
```

`-DMNA_ENABLE_SANITIZERS=ON` builds them with ASan + UBSan and `-DMNA_ENABLE_COVERAGE=ON` with coverage instrumentation; CI runs both on Linux.

## Quick Start

Install the Python package with dev dependencies (this compiles the native core, see [Native Core](#native-core)):

```bash
python -m pip install -e ".[dev]"
```

Start the backend API:

```bash
python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8010
```

The server runs every analysis in the native C++ core. To compare with the simulator as it was before the C++ migration, boot it on the frozen pure-Python core instead by setting `MNA_CORE=python` (`MNA_CORE=cpp` or unset is the default):

```bash
MNA_CORE=python python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8010
```

In PowerShell, run `$env:MNA_CORE="python"` first. Python mode runs the parser, MNA builder and solvers of `tests/reference/mna_reference`, the pre-migration package kept as the test oracle, behind the same API. Requests are validated by the same option rules as in the default mode (HTTP 400 for the same inputs), each run uses one thread, and results report `core_backend: "python"` and `core_version: "python-reference@e5172cb"`. Pre-migration behaviour is kept as it was, including what `USER_GUIDE.md` lists as changed with the native core. The oracle is not shipped in wheels, so this mode needs a source checkout (`pip install -e .`). Expect analyses to run roughly 5x to 300x slower, depending on the analysis (`benchmarks/results/baseline-*.md`). The startup log line `MNA core: ...` and `GET /health` (`core_backend`) show which core is running.

Start the frontend:

```bash
cd app/web
npm install
npm run dev
```

The frontend proxies `/api`, `/health`, and websocket simulation traffic to `http://127.0.0.1:8010` by default. Set `VITE_API_PORT` before starting Vite to use a different backend port.

## CLI Usage

Run a netlist file:

```bash
python -m mna_simulation.cli run path/to/circuit.cir
```

Run inline netlist text:

```bash
python -m mna_simulation.cli run-text "V1 n1 0 DC 5\nR1 n1 0 1k\n.op\n.end"
```

Return JSON:

```bash
python -m mna_simulation.cli run path/to/circuit.cir --json
```

Run on the frozen pre-migration Python core (`--core python`; the default is `--core cpp`, or `MNA_CORE` when set):

```bash
python -m mna_simulation.cli run path/to/circuit.cir --core python
```

## API

- `GET /health`: service health, backend capabilities, and model registry metadata.
- `POST /api/simulate`: accepts either `netlist_text` or structured `schematic` JSON.
- `WS /ws/dyn`: dynamic transient stream used by the frontend display/probe tools.

Schematic runs return the generated canonical netlist in response metadata. Simulation options include:

- `use_krylov`: enable the full-system Krylov subspace iterative solver path.
- `krylov_rank`: `"auto"` or any positive integer.
- `krylov_method`: `"auto"`, `"arnoldi_gmres"`, `"conjugate_residual"`, or `"conjugate_gradient"`.
- `use_mor`: enable selected-output model-order reduction.
- `mor_method`: `"auto"`, `"linear_krylov"`, or `"tpwl"`.
- `mor_order`: `"auto"` or any positive integer. Auto uses `min(n, max(10, min(120, 4 * (inputs + outputs))))`.
- `mor_output_nodes`: labels such as `V(out)` or `I(L1)` retained by the reduced model.
- `mor_validate`: when true, requires `probe_nodes` to be a subset of `mor_output_nodes`.
- `probe_nodes`: optional labels such as `V(n1)` or `I(L1)` for result filtering.
- `output_max_points`: optional waveform decimation limit for API responses.
- Option values are type-checked: booleans take `true`/`false` (also `yes/no/on/off/1/0`, any case), counts take whole numbers (`3`, `"3"`, `3.0`), time/frequency values (`t_stop`, `t_step`, `f_start`, `f_stop`, `time_window`) and tolerances take numbers or metric strings such as `"1m"`, and `init_condition` takes `DEFAULT`, `RANDOM` (any case) or a list of numbers. Anything else returns HTTP 400 (`INVALID_OPTIONS`).
- Only an absent (or `null`) option falls back to its default; zero is a value: `t_stop = 0` returns the single initial sample, `f_start = 0` / `f_stop = 0` are DC points, and `t_step <= 0`, `points <= 0`, or `harmonics <= 0` are errors (`TRAN_ERROR`, `AC_ERROR`, `HB_ERROR`; HTTP 400 with the code). `USER_GUIDE.md` lists every behaviour that changed with the native core.

## Threads and Concurrency

Every run leases worker threads from one machine-wide budget, so concurrent API requests and websocket streams divide the cores instead of oversubscribing them.

- `MNA_NUM_THREADS` caps the server (default: `os.cpu_count()`, never more). `/api/simulate` and `/ws/dyn` run on a thread pool of that size, and a request waits while every thread is leased. `GET /health` reports `thread_budget` (`total`, `free`, `active`), `core_backend`, `core_version`, and `build_info`.
- Requests pass `options.num_threads`: `"auto"` (the default) or a positive integer, which acts as an upper limit (`min(requested, free, total)`). `"auto"` takes a fair share of what is free, `max(1, total // (active runs + 1))`, capped at what the analysis can use (`mna_simulation.native.threads.useful_threads`): one thread for `.op`/`.tran` below 2048 devices and for harmonic balance below 64 nodes (the core runs those on one thread anyway), `points // 2` for `.ac`; a large analysis that finds the server idle gets the whole budget. Anything else returns HTTP 400. The CLI takes `--threads N|auto` and the UI has a Threads control. Result metadata reports `threads_requested`, `threads_used`, `thread_budget_total`, `core_backend` (`"cpp"`, or `"python"` in Python mode), `core_version`, and `elapsed_ms` (wall time of the backend run).
- The native core uses the granted threads where an analysis has parallel work (see [Native Core](#native-core)); the thread count changes speed only, never results. Small circuits gain nothing from more threads, because every parallel region starts its own workers.
- Concurrent requests in one server process overlap in their native parts, which run without the GIL, and `"auto"` leases only what each analysis can use, so simultaneous small requests run side by side instead of queueing behind one lease. Four concurrent harmonic-balance mixer requests (about 0.7 s of native work each) finish in 1.10x the time of one. Parsing, MNA building, the device operating-point table and response conversion still run in Python under the GIL and are most of a short request: four simultaneous SRAM 10x10 transients (about 30 ms each) take about 4.9x as long as one, 153 ms in total (`python benchmarks/concurrent_requests.py [--workload hb_mixer]`; `PERFORMANCE_PLAN.md` section 10.2; a process pool is the follow-up).
- NumPy/SciPy BLAS is pinned to one thread for the duration of every run (`threadpoolctl`). Before the native core, with four concurrent SRAM transients in separate processes, this took the batch from 45 s to 3.5 s on a 4-core machine (`PERFORMANCE_PLAN.md` section 4.2). Still start the server with `OPENBLAS_NUM_THREADS=1` as belt and braces, because it also covers BLAS calls made outside a run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MNA_NUM_THREADS=4 python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8010
```

## Environment Variables

- `MNA_NUM_THREADS`: size of the machine-wide thread budget (a positive integer, capped at `os.cpu_count()`; unset means the CPU count; an invalid value is ignored with a `RuntimeWarning`).
- `MNA_HB_MAX_NNZ`: upper bound on the matrix entries a harmonic-balance run may store (default `3e7`, the core's `HB_DEFAULT_MAX_NNZ`; any positive number such as `5e7` or `250000`; an invalid value is ignored with a `RuntimeWarning`). A larger system fails before anything is allocated with `HB_ERROR` "HB system too large (N=..., nnz=...); reduce harmonics". The HB dimension N = n (2H + 1) is also capped at 1e6.
- `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`: set them to 1 for servers (above).

## Supported Devices

- Passive: `R`, `C`, `L`.
- Sources: `V`, `I` with `DC`, `AC`, `SIN`, `COS`, `STEP`, and `FUNC`.
- Nonlinear/behavioral: `D`, behavioral source path.
- Controlled sources: `VCVS`, `VCCS`, `CCCS`, `CCVS`.
- Transistors: `QNPN`, `QPNP`, `NMOS`, and `PMOS` default to physical Level-1 models, with small-signal compatibility still available.
- Physical BJT syntax: `Q1 c b e QNPN LEVEL1 is bf br vaf var cje cjc rb re rc` and PNP equivalent.
- Physical MOS syntax: `M1 d g s NMOS LEVEL1 beta vth lambda cgs cgd` and PMOS equivalent.
- Schematic-only frontend helpers: `SUBCKT`, `NODE`, `LABEL`, `GND`.

## Supported Analysis Modes

- `.op` / `.dc`
- `.tran <t_stop> [t_step]`
- `.ac <f_start> <f_stop> <points>`
- `.hb [harmonics] [time_window]`
- `.dyn` frontend websocket stream, executed by the backend as transient analysis.

Convergence aids, each engaged only when the plain iteration fails, so circuits that converge without them give the same results: gmin stepping in the service (`0, 1e-12, ..., 1e-5`, with gmin across device terminals and from every node to ground), gmin continuation from 1e-2 inside a failed `.op`, SPICE-style junction voltage-step limiting when the damped Newton of `.op`/`.tran` stalls, and a damped harmonic-balance Newton after the undamped one breaks down. `USER_GUIDE.md` ("What Changed With The Native Core") has the details.

## MOR Method Notes

For few-output analog LSI work, v1 uses PRIMA-like output-aware rational Krylov projection for linear/passive and `.op`-linearized circuits, because it targets input-output behavior and keeps the projection layer decoupled from the linear-solve backend. Nonlinear transient MOR uses a TPWL/POD cache. TPWL/POD requested on a purely linear circuit is routed to Linear Krylov MOR and reported in metadata. Metadata distinguishes `mor_order` / `mor_resolved_order` (requested reduction budget) from `mor_basis_size` (actual independent basis after pruning). Balanced truncation and SVDMOR/RecMOR remain documented later candidates for heavier global reduction or many-terminal structured interconnects.

## Schematic Demo Notes

The frontend supports:

- Component placement, selection, copy/paste, rotation, and deletion.
- Pin-to-pin and point-based wiring with orthogonal routing.
- Multi-level hierarchy with editable `SUBCKT` pin names.
- Editable `NODE` markers that connect by matching name within the same level.
- Editable child ports: a `NODE` named `in` in a child entity exports as `port_in`.
- Top-menu demos, including a BJT 3-stage amplifier built from real subcircuits, a GHz close-tone HB mixer, and Large Scale Circuits presets.
- Toolbar Help opens the same practical guide stored in `USER_GUIDE.md`.
- Raw netlist editing in the Visualization panel. Starting an edit asks for confirmation and then clears schematic hierarchy so the request is sent as `netlist_text`.
- Krylov subspace solver controls with method override, auto/manual rank, and method-specific wording for restart versus iteration budget.
- MOR controls with method selection including Linear Krylov MOR, auto/manual order, and a separate MOR output-node selector. Display nodes must be empty or contained in the MOR output set.
- `.op`, `.tran`, `.ac`, `.hb`, and dynamic display/probe visualization.

## Testing

Run all Python tests: parser, builder, service, analyses, the native bindings (`tests/test_native_*.py`), the differential suite against the frozen Python oracle, thread determinism, and the leak check. Tests marked `perf` are skipped unless `--run-perf` is given.

```bash
python -m pytest
```

Subsets and performance checks:

```bash
python -m pytest -m differential                 # live core vs the frozen oracle on every corpus analysis
python -m pytest -m "differential and not slow"  # the same without the 1035-unknown circuits
python -m pytest tests/perf --run-perf           # speedup floors, thread scaling, concurrent-request criterion
python benchmarks/compare_backends.py            # Python reference vs C++ core per unit (Markdown table + JSON)
python benchmarks/concurrent_requests.py         # N concurrent SRAM requests against the FastAPI app
```

The perf tests and the benchmarks compare ratios rather than absolute times, and they need an otherwise idle machine (`compare_backends.py` warns when the 1-minute load average is above 0.5). Committed results and their policy are in `benchmarks/results/README.md`.

Build the frontend:

```bash
cd app/web
npm run build
```

## Benchmark

```bash
python benchmarks/compare_backends.py --list    # comparison units: device evaluation, assembly, solves, analyses, MOR
python benchmarks/compare_backends.py --quick --threads 1,4 --json out.json --markdown out.md
python benchmarks/concurrent_requests.py --server uvicorn   # concurrent requests over real HTTP
python benchmarks/benchmark_basic.py        # tiny RC transient
python benchmarks/benchmark_workloads.py    # SRAM, RLC mesh, BJT amplifier, and HB workloads (add --profile for hot spots)
python benchmarks/blas_contention.py        # concurrent runs vs OPENBLAS_NUM_THREADS
```

Run benchmarks with nothing else competing for the CPU. `PERFORMANCE_PLAN.md` documents the original measurements, the speedups of the native core, and what is left.

## Legacy Entry Points

Older simulator snapshots are under `legacy/`. They are retained for comparison and regression work, not as the primary runtime path.
