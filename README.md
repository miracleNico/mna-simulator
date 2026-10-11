# MNA Simulation

Layered circuit-simulation workbench built around Modified Nodal Analysis (MNA).

The project includes:

- Python simulation core for `.op`, `.tran`, `.ac`, and `.hb`.
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
- **Sparse large-circuit path:** linear transient Krylov runs can use cached SciPy CSR Backward-Euler operators instead of dense conversion on every step.
- **Functional MOS SRAM demo:** the Large Scale Circuits menu includes a hierarchical 10x10 Level-1 MOS 6T SRAM demo with FUNC-driven write/hold/read timing; the root canvas shows cell instances while the reusable child level holds the transistor cell body.
- **Large RLC benchmark demo:** the 23x23 distributed RLC mesh produces a 1035-unknown sparse MNA system for Krylov/MINRES benchmarking.
- **Editable netlist mode:** the web app can switch from schematic mode to raw netlist editing after a confirmation that clears the schematic hierarchy.
- **Interactive visualization:** `.op`, `.tran`, `.ac`, `.hb`, and `.dyn` results render in movable plot tiles, with selectable display nodes.
- **LTspice-style schematic editing:** wire tool with corners and T-junction taps, junction dots, open-pin markers, sticky part placement with rotate/mirror on the cursor, undo/redo, duplicate, zoom-to-fit, right-drag pan, and working File / Edit / View / Simulate menus with keyboard shortcuts.
- **Workspace files:** save and reopen the whole hierarchy plus analysis settings as JSON, and export the generated SPICE netlist.
- **In-app user guide:** toolbar Help (F1) opens a detailed guide for demos, schematic editing, hierarchy, analysis modes, visualization, Krylov controls, and raw netlist mode.

## Repository Layout

- `mna_simulation/`: Python parser, MNA builder, solvers, schematic pipeline, API service, and backend adapters.
- `app/server/`: FastAPI app exposing `/health`, `/api/simulate`, and dynamic simulation websocket routes.
- `app/web/`: React + TypeScript schematic editor and demo UI.
- `tests/`: parser, builder, service, harmonic-balance, transistor, and hierarchy regression tests.
- `benchmarks/`: benchmark entry points, workload netlists, and performance research prototypes.
- `PERFORMANCE_PLAN.md`: measured performance review, multithreading plan, and C++ migration research.
- `legacy/`: older simulator snapshots kept for comparison.
- `USER_GUIDE.md`: user-facing web-app guide mirrored by the in-app Help dialog.
- `DESIGN_AUDIT.md`: front end design audit, changes made, and LTspice feature comparison.
- `STRUCT.md`: file-by-file architecture reference.

## Quick Start

Install the Python package with dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

Start the backend API:

```bash
python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8010
```

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

Run all Python tests:

```bash
python -m pytest
```

Build the frontend:

```bash
cd app/web
npm run build
```

## Benchmark

```bash
python benchmarks/benchmark_basic.py        # tiny RC transient
python benchmarks/benchmark_workloads.py    # SRAM, RLC mesh, BJT amplifier, and HB workloads (add --profile for hot spots)
python benchmarks/blas_contention.py        # concurrent runs vs OPENBLAS_NUM_THREADS
```

Run benchmarks with nothing else competing for the CPU. Multithreaded OpenBLAS calls degrade sharply when several simulations overlap; `PERFORMANCE_PLAN.md` documents the measurements and the planned fixes.

## Legacy Entry Points

Older simulator snapshots are under `legacy/`. They are retained for comparison and regression work, not as the primary runtime path.
