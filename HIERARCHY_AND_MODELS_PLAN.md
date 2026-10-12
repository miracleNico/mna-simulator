# Hierarchy UX, Vendor Model Import and Mixed-Signal Plan

Status: **in progress** (phase H1 and phases M1/M2 running, see "Status" below).
Companion documents: `MIXED_SIGNAL.md` (feasibility assessment and design for Python-defined
digital stimulus), `CPP_MIGRATION_PLAN.md` (native core, rulings), `USER_GUIDE.md`.

The request behind this plan, in three parts:

1. Make multi-level schematic encapsulation (hierarchical subcircuits) smoother to use.
2. Import industrial-grade component parameters as published by the major manufacturers
   (vendor SPICE model cards and subcircuit macromodels).
3. Judge whether mixed-signal simulation with Python-defined digital signals (for example the
   output of a USB transmitter) as stimulus is feasible, and how.

## 1. Where we start

### 1.1 Hierarchy today

* A workspace is a list of `SchematicLevel`s (`app/web/src/lib/schematic.ts`); the root has
  `parentId: null`, every other level is an *entity* that can be instanced with a `SUBCKT`
  component (`subcircuitId`, `pins`).
* A child entity exports a port by containing a `NODE` marker named after the pin; the
  frontend turns it into a `port_<pin>` junction (`expandSchematic`) and the backend bridges
  outer pins to inner nets in `schematic_pipeline._flatten_subcircuits`. Instance pins are
  re-derived from the entity's `NODE` markers (`inferredLevelPins`, `syncInstancesForEntity`
  in `App.tsx`).
* Pain points: a subcircuit is built top-down only (create an empty level, draw inside, place
  an instance, pick the entity in a dropdown, fix the pin count by hand); there is no way to
  turn an existing selection into a subcircuit; there is no descend/ascend navigation (double
  click opens the property editor); pins have no side or order, so the symbol is a box with
  pins in creation order; port mismatches are only reported by the backend at run time; an
  instance can be linked to an entity that (indirectly) contains it.

### 1.2 Device models today

* Netlist dialect (`mna_simulation/netlist.py`): `R L C V I D Q M B E F G H GND`, sources
  `DC AC SIN COS STEP FUNC`, diode `D1 a k [Is]` (ideal Shockley, n = 1, vt = 0.025 V),
  BJT `Q1 c b e QNPN LEVEL1 is bf br vaf var cje cjc rb re rc` (Ebers-Moll-lite: transport
  currents, finite betas, `1 + vce/vaf` Early factor; `rb re rc` parsed but not stamped),
  MOS `M1 d g s NMOS LEVEL1 beta vth lambda cgs cgd` (Shichman-Hodges, no body effect, bulk
  not modelled), plus the small-signal forms. No `.model`, `.subckt`, `X`, `.include`,
  `.lib`, `.param`, `PULSE`, `PWL`, no `+` continuation lines.
* Native kernels: `cpp/include/mna/devices/{diode,bjt_level1,mos_level1}.hpp`, tables in
  `mna_simulation/native/circuit.py` (`diode_table`, `bjt_table`, `mos_table`), stamped by
  `mna_simulation/mna_builder.py`. Capacitances are constant entries of the `C` matrix: the
  core has no charge-based (voltage-dependent) capacitor yet.
* Vendor material comes in two shapes: a **model card** (`.model D1N4148 D(IS=2.52n ...)`,
  `.model Q2N2222 NPN(IS=14.34f BF=255.9 ...)`, `.model BS170 NMOS(LEVEL=3 ...)`) and a
  **subcircuit macromodel** (`.subckt LM358 1 2 3 4 5 ... .ends`, built from `.model` cards,
  passives, controlled sources, `.param` expressions, and often `PULSE`/`PWL`/`TABLE`
  constructs). Both must parse; the physical accuracy then depends on how many card parameters
  the device equations honour.

## 2. Part A: hierarchy UX (phase H1)

Target workflow, each step one gesture:

| Step | Gesture | Result |
| --- | --- | --- |
| Encapsulate | select parts and wires, **Make subcircuit** (Edit menu, context menu, `Ctrl+Shift+G`) | a new child entity holding the selection; every net that crosses the selection boundary becomes a port (`NODE` marker inside, `port_<pin>` junction, pin on the instance); ground stays global; a `SUBCKT` instance replaces the selection and is wired to the outside stubs; undoable |
| Descend / ascend | double-click a `SUBCKT` instance; breadcrumb bar above the canvas; `Alt+Up` | the entity opens; the breadcrumb shows the path root › … › entity and navigates back |
| Edit ports | add, rename, move or delete a `NODE` marker inside the entity | pins of every instance follow (name, order, side); wires on renamed pins are repaired, wires on deleted pins are detached and reported |
| Place pins | `NODE` marker property `side` (left/right/top/bottom, default left for `in*`, right otherwise) and position | the `SUBCKT` symbol draws pins on the chosen side, ordered by the marker's coordinate; the box grows with the pin count and shows the entity title |
| Reuse | **Instantiate** on an entity in the hierarchy tree; **Duplicate entity** | places an instance of that entity (pending device, pins preset); copies an entity with its contents |
| Validate | continuous | inline errors on an instance: unlinked entity, pin set differs from the entity's ports, recursive instantiation (entity contains itself directly or indirectly); the Run action is blocked with the same text the backend would produce |
| Dissolve (optional) | **Dissolve subcircuit** on an instance | the entity's contents are inlined into the parent in place of the instance |

Rules:

* The workspace file stays readable in both directions: version-1 files load unchanged; new
  data goes into optional fields (`CanvasComponent.metadata.side`, `SchematicLevel.description`);
  `WORKSPACE_FILE_VERSION` is bumped only if a required field is added.
* The backend contract (`buildSchematicPayload` → `schematic_pipeline`) does not change: the
  frontend keeps emitting `port_<pin>` junctions and `subcircuits[levelId]`. Any new frontend
  check mirrors a backend check (`_validate_subckt_ports`), it does not replace it.
* Pure logic (boundary-net detection, make/dissolve transforms, recursion check, breadcrumb
  path, pin side/order) lives in `app/web/src/lib/hierarchy.ts` with vitest unit tests;
  `App.tsx` and the components only wire state and events.
* `app/web/dist` is rebuilt and committed with the change (the server serves it).

Acceptance: the vitest suite and `tsc --noEmit` pass; `python -m pytest tests/test_schematic_pipeline.py`
passes with a payload produced by the make-subcircuit transform; the user guide (both
`USER_GUIDE.md` and `app/web/src/lib/userGuide.ts`) describes the new workflow.

## 3. Part B: vendor model import

### 3.1 Phases

| Phase | Content | Worker |
| --- | --- | --- |
| M1 | SPICE dialect front end: lexer (`+` continuations, `*`/`;`/`$` comments, case-insensitive, trailing unit letters, `T` and `MIL` suffixes), `.model` cards, `.subckt`/`.ends` with nesting, `X` instances with hierarchical flattening (`X1.R1`, nets `X1.n`), `.param` and `{...}` expressions (restricted evaluator), `.include`/`.lib ... .endl` with a resolver (file paths from the CLI; text input must be self-contained), `.global`, SPICE source forms (`DC`/`AC` combinations, `SIN(...)`, `PULSE(...)`, `PWL(...)`, `EXP(...)`), diagnostics for ignored constructs, errors for unsupported ones | B |
| M2 | Device equations that honour the card parameters: SPICE diode (IS N RS BV IBV CJO VJ M TT FC EG XTI, area), Gummel-Poon BJT DC (IS BF NF VAF IKF ISE NE BR NR VAR IKR ISC NC RB RE RC, area) with CJE/CJC as zero-bias linear capacitors, MOS Level 1-3 cards mapped to the Level-1 equations with body effect (VTO KP LAMBDA GAMMA PHI W L, overlap capacitances CGSO CGDO CGBO, `M` multiplier), parasitic resistances through internal nodes, `.model` defaults, Python reference implementations and C++ unit tests | C |
| M3 | Integration: parser records → builder, end-to-end tests with published cards (1N4148, 2N2222/2N2907, BS170/2N7000 Level-3 card, an op-amp macromodel), docs, CLI `--lib` | coordinator |
| M4 (next round) | Charge-based capacitances in the core (junction and MOS charges with analytic dq/dv, in `.op`/`.tran`/`.ac`/`.hb`), transit-time charge (TT, TF, TR), bulk diodes of the MOS, temperature (TNOM/TEMP, EG, XTI, XTB), BJT TF/XTF/VTF/ITF, Level-3 short-channel terms; schematic UI for model libraries (paste or upload a vendor file, pick a model for a device, library panel) | later |
| Out of scope | BSIM3/4 (LEVEL 8/49/54), VDMOS, JFET/MESFET, HICUM/VBIC, behavioural `TABLE`/`LAPLACE` controlled sources, encrypted libraries | report as errors with the level/type named |

### 3.2 Data contract between the parser (M1) and the builder (M2)

Everything travels in `ComponentRecord` (`mna_simulation/api/contracts.py`). Keys are
lowercase strings; values are the SPICE value **as text** (metric suffixes allowed, already
multiplied by nothing). The builder parses them with `parse_value`, applies the SPICE
defaults below, and derives kernel parameters. The parser never does device physics; the
builder never reads vendor syntax.

Common keys: `metadata["model"]` selects the equation set, `metadata["model_name"]` is the
card name (for messages), `metadata["area"]` (default `1`) and `metadata["m"]` (default `1`)
are the SPICE instance multipliers. Ruling (M2): on a diode `metadata["m"]` is always the
card's grading coefficient `M`, never an instance multiplier (a diode has `area` only; an
instance `m=` on a diode line is a parser diagnostic and is dropped). On a BJT `m` multiplies
the area, on a MOS it multiplies `beta` and the overlap capacitances. Card parameters the parser does not recognise are still
copied into `metadata` (lowercase) and listed once per card in `CircuitIR.diagnostics`
(`"model D1N4148: parameters ... are not used by the diode equations"`). The parser owns the
supported-parameter lists in `mna_simulation/spice/model_cards.py`; they must match the tables
below.

**Diode** (`type="D"`, `metadata["model"]="spice"`), card type `D`:

| key | default | used by M2 |
| --- | --- | --- |
| `is` | `1e-14` | yes (`value` also holds `is` for compatibility) |
| `n` | `1` | yes |
| `rs` | `0` | yes, internal node `<name>#a` when `rs/area > 0` |
| `bv`, `ibv` | `inf`, `1e-3` | yes (reverse breakdown branch `-ibv*exp(-(v+bv)/vt)` with the SPICE continuity shift) |
| `cjo`, `vj`, `m`, `fc` | `0`, `1`, `0.5`, `0.5` | M2: `cjo*area` as a constant capacitor; M4: charge |
| `tt` | `0` | M4 |
| `eg`, `xti`, `tnom` | `1.11`, `3`, `27` | M4 (temperature) |

**BJT** (`type="QNPN"`/`"QPNP"`, `metadata["model"]="gummel_poon"`), card types `NPN`/`PNP`;
`value`, `value2`, `value3` carry `is`, `bf`, `br` for compatibility; `ctrl_node2` carries the
optional substrate node (ignored by M2, diagnostic):

| key | default | used by M2 |
| --- | --- | --- |
| `is`, `bf`, `nf`, `br`, `nr` | `1e-16`, `100`, `1`, `1`, `1` | yes |
| `vaf`, `var` | `inf`, `inf` | yes (Gummel-Poon `q1`) |
| `ikf`, `ikr` | `inf`, `inf` | yes (`q2`, high injection) |
| `ise`, `ne`, `isc`, `nc` | `0`, `1.5`, `0`, `2` | yes (leakage diodes) |
| `rb`, `re`, `rc` | `0`, `0`, `0` | yes, internal nodes `<name>#b`, `<name>#e`, `<name>#c` when > 0 (`rbm`, `irb` ignored: constant `rb`) |
| `cje`, `cjc` | `0`, `0` | M2: zero-bias constant capacitors (`vje mje vjc mjc xcjc fc` recorded, M4) |
| `tf`, `tr`, `xtf`, `vtf`, `itf`, `ptf` | `0` ... | M4 |
| `xtb`, `eg`, `xti`, `tnom` | | M4 |

**MOS** (`type="NMOS"`/`"PMOS"`, `metadata["model"]="spice_level1"`), card types `NMOS`/`PMOS`
with `level` 1, 2 or 3 (2 and 3 are mapped to the Level-1 equations with a diagnostic; any other
level is an error naming it); `ctrl_node2` carries the bulk node when the instance has four
nodes (three-node instances: bulk = source); instance keys `w`, `l`, `m`, `ad`, `as`, `pd`,
`ps`, `nrd`, `nrs`:

| key | default | used by M2 |
| --- | --- | --- |
| `vto` | `0` | yes (`vth = |vto|` sign handled by the kernel as today) |
| `kp` | `2e-5` | yes: `beta = kp * w / l * m` (matches the existing `0.5 * beta * vov^2` form) |
| `w`, `l` | `100u`, `100u` (SPICE `DEFW`/`DEFL`) | yes |
| `lambda` | `0` | yes |
| `gamma`, `phi` | `0`, `0.6` | yes: `vth_eff = vto + gamma * (sqrt(phi - vbs) - sqrt(phi))` for `vbs <= 0`, held at `vbs = 0` above (M4: SPICE's forward-bias form); needs the 4-terminal kernel with `d/dVb` |
| `cgso`, `cgdo`, `cgbo` | `0` | M2: `cgso*w`, `cgdo*w`, `cgbo*l` as constant capacitors |
| `cbd`, `cbs`, `cj`, `cjsw`, `pb`, `mj`, `mjsw`, `js` | | M4 (bulk junction diodes and charges) |
| `tox`, `nsub`, `nss`, `uo`, `tpg`, `xj`, `ld`, `vmax`, `theta`, `eta`, `kappa`, `delta`, `nfs`, `rsh`, `rd`, `rs` | | M4 (Level 2/3 terms; `rd`/`rs` would use internal nodes); `rd`/`rs` > 0 is a diagnostic in M2 |

Passives from cards (`.model RMOD R`, `C`, `L` with `R=`, `C=`, `TC1` ...) are flattened by the
parser into plain `R`/`C`/`L` records at parse time (nominal value; temperature coefficients
are a diagnostic).

**Sources**: `V`/`I` records keep `subtype` in {`DC`, `AC`, `SIN`, `COS`, `STEP`, `FUNC`} for
the existing forms; the SPICE forms add `subtype="PULSE"` with `metadata["pulse"]` = the
seven values as text (`v1 v2 td tr tf pw per`, missing ones defaulted by SPICE rules,
`tr`/`tf` default to the analysis step by the builder), `subtype="PWL"` with
`metadata["points"]` = the flat `t v` list as text and `metadata["repeat"]` for `R=`,
`subtype="EXP"` with `metadata["exp"]`, and `subtype="SIN"` with `metadata["sin"]` =
`vo va freq td theta phase` when the parenthesised SPICE form was used (the two-token form
`SIN amp freq` keeps its current meaning). A `DC x AC mag [phase]` line stores `value` = the
DC value, `metadata["ac_mag"]`, `metadata["ac_phase"]`, and a transient form after it goes into
`subtype`/`metadata` as above; the builder picks the DC value for `.op`, the AC values for
`.ac`, and the transient waveform (DC value if none) for `.tran`/`.hb`.

`device_models.source_expression` renders PULSE/PWL/EXP/SPICE-SIN as expression text for the
existing time-source programs (`heaviside`, `min`, `max`, `mod` are available in the VM); a
PWL with more than 64 points gets the native table source of `MIXED_SIGNAL.md` when it lands
(until then it is rendered as nested `heaviside` terms, correct but O(points) per evaluation).

Flattened names: `X1.R1` for components, `X1.n` for internal nets, dotted deeper for nested
instances; ports bind to the parent's net names; `0` and `.global` nets are never prefixed.
Node-name validation (`_node_name_ok`) accepts the dot in flattened names; result labels show
them unchanged (`V(X1.out)`).

### 3.3 Test strategy

* The frozen oracle (`tests/reference/mna_reference`) knows none of this: it stays untouched,
  and the differential harness keeps proving that every existing corpus case is unchanged.
* New device equations get an independent Python reference in `tests/device_reference/`
  (plain NumPy, written from the SPICE equations, Jacobians checked by central finite
  differences) and are compared against the native kernels through
  `mna_simulation._core` (`evaluate_device_functions`) to 1e-12 relative, and through full
  `.op`/`.tran` runs of small circuits.
* C++ unit tests in `cpp/tests/test_devices.cpp` cover limits (`ise = 0`, `ikf = inf`, `rs = 0`
  reduce to the simpler forms), continuity at region boundaries, and derivative consistency.
* Parser tests in `tests/test_spice_dialect.py` cover every construct of M1 with vendor-style
  snippets (continuations, nested `.subckt`, `.param` arithmetic, `.lib` sections, PULSE/PWL,
  error texts), and M3 adds published cards end to end under `tests/vendor/`.
* `tests/EDGE_CASES.md` gets a section per new ruling.

### 3.4 Rulings (decided here so the workers do not have to ask)

* No SPICE title line: the first line of a netlist is parsed like every other line
  (comment it out with `*` when pasting a deck). The `.end` line ends parsing as today.
* Case: component names, node names, model names and parameter keys are matched
  case-insensitively as in SPICE, but node names keep the case of their first occurrence in
  labels; `in` and `IN` are the same net.
* `vt` for the SPICE-card equations is `k*T/q` at `T = 300.15 K` (27 °C): `0.02586 V`
  (`8.617333262e-5 * 300.15`). The existing `LEVEL1`/ideal-diode paths keep their constants
  (0.025 / 0.02585) so the oracle comparison holds.
* Unsupported but harmless constructs (unknown card parameters, `.options`, `.temp`,
  `.nodeset`, `.ic` for now, `.probe`, `.print`, `.plot`, `.backanno`, `.step`, `.meas`,
  `.four`, `.noise`, `.tf`, `.dc` sweeps) are diagnostics, not errors; unsupported constructs
  that would change the answer (unknown model types that are referenced, LEVEL ≥ 4, `TABLE`
  and `LAPLACE` controlled sources, `PWL FILE=`, `.func`, switches `S`/`W`, transmission
  lines `T`/`O`/`U`, `K` coupling, JFET `J`, MESFET `Z`) are `NETLIST_FATAL` errors naming the
  line and the construct.
* `.include`/`.lib` from text input (HTTP API) is an error that tells the user to paste the
  library into the netlist; from a file (`parse_netlist_file`, CLI) paths resolve relative to
  the including file; `--lib PATH` on the CLI adds search directories.
* Internal nodes are named `<component>#<terminal>` (`D1#a`, `Q1#b`, `Q1#c`, `Q1#e`, `M1#d`,
  `M1#s`), appended after the sorted external nodes and before the branch unknowns, labelled
  like other nodes, and counted in `node_count` (node-to-ground gmin applies to them).
* Area/multiplier: `area` scales `is`, `ise`, `isc`, `ikf`, `ikr`, `cje`, `cjc`, `cjo` up and
  `rs`, `rb`, `re`, `rc` down; `m` scales a MOS like `m` parallel devices.

## 4. Part C: mixed-signal

The assessment and design are in `MIXED_SIGNAL.md`. Summary of the verdict: one-directional
digital stimulus (a Python-generated bit stream driving analog nodes) is feasible now with a
native table-driven piecewise-linear source and breakpoint-aware time stepping; bidirectional
co-simulation (analog values fed back into Python logic every step) is feasible through the
existing transient progress callback with a per-step Python call, at the cost of the GIL
round trip; a USB full-speed packet is within reach, high-speed USB is not a transient
problem for this simulator.

## 5. Worker split and status

| Worker | Scope | Status |
| --- | --- | --- |
| A (frontend) | Part A, phase H1 | landed on `dev` (make subcircuit, navigation, entity-owned ports, pin sides, instantiate/duplicate/dissolve, inline checks, vitest suite); follow-up: Run from inside an entity still simulates that entity alone |
| B (parser) | Part B, phase M1 | running |
| C (devices) | Part B, phase M2 | landed on `dev` (`cpp/include/mna/devices/{spice_diode,gummel_poon,mos_card}.hpp`, `mna_simulation/card_devices.py`, `tests/device_reference/`, `tests/test_spice_devices.py`) |
| D (stimulus) | `MIXED_SIGNAL.md` phase D1 | running (parser-owned wiring staged until M1 lands) |
| coordinator | M3 integration, `MIXED_SIGNAL.md` | assessment written; M3 after M1 |

Each worker reports its branch and commits; the coordinator audits the diff, runs the full
suite (`python -m pytest -q`, the C++ tests, the differential harness) and merges into `dev`.
