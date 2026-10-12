# Mixed-Signal Simulation with Python-Defined Digital Stimulus

Status: assessment and design (phase D1 implementation is listed at the end; nothing in this
document has landed yet unless the status table at the end says so).
Companion: `HIERARCHY_AND_MODELS_PLAN.md` (part C), `CPP_MIGRATION_PLAN.md` (native core).

## 1. Question and verdict

Can the simulator run mixed-signal analyses in which the digital side is defined in Python,
for example the output of a USB transmitter driving an analog front end?

Verdict:

* **One-directional digital stimulus (Python → analog) is feasible now and cheap.** A Python
  object produces a piecewise-linear waveform (bits, levels, edge times); the analog core
  consumes it as a time source. Today this already works for short patterns through the `FUNC`
  source; the missing pieces are a native table-driven source (so long bit streams cost
  O(log N) per evaluation instead of O(N)) and a time grid that lands exactly on the edges.
  Both are small, Python-side-first changes (the core already accepts an arbitrary monotone
  time grid). This is phase D1 below.
* **Bidirectional co-simulation (analog ↔ Python logic every step) is feasible with a modest
  core hook.** The transient loop already calls back into Python after any step
  (`progress(step, t, x)`); the measured cost of doing so every step is about 0.6 µs per step
  on top of the step itself. Adding a writable "control slot" vector that the callback can
  set and that sources read on the next step turns this into a synchronous co-simulation with
  one-step latency. This is phase D2 (design here, not scheduled).
* **A USB full-speed or low-speed packet is within reach** of the one-directional mode (tens of
  thousands of steps, well under a second). **USB high-speed is not** a problem this simulator
  can model faithfully: the bit period is 2 ns and the signalling is dominated by transmission
  lines and package parasitics, which the netlist dialect does not have (`T`/`O`/`U` lines are
  rejected). The stimulus side would still work; the analog fidelity would not.

## 2. What exists today

* Time sources are expression programs evaluated by the expression VM at every time point
  (`device_models.source_expression` → `compile_time_source_func`; the VM has `heaviside`,
  `min`, `max`, `mod`, `exp`, `sin`, `cos` and arithmetic). `V1 a 0 FUNC period expr` already
  lets a user write any waveform as an expression in `t`.
* The transient grid is built in Python (`solvers._transient_time_points`, uniform
  `t_step`) and handed to `_core.solve_transient` as an array, so a non-uniform grid needs no
  core change. The integrator is Backward Euler with exact per-step `h`.
* `progress(step, t, x)` is an optional Python callback called every `progress_every` steps
  from inside the native loop (the GIL is re-acquired for the call only).
* The `.dyn` websocket endpoint runs the whole transient first and then plays it back at a
  chosen speed; it is a display feature, not a co-simulation loop.
* Harmonic balance (`.hb`) handles periodic stimulus directly; a repeating bit pattern (a
  clock, a 1010 idle pattern) can be analysed in the frequency domain as well.

## 3. Measurements

Diode clamp (`V1 SIN`, `R`, `D`, `C`), `.tran 20m 1u`, 20 000 Backward-Euler steps, one thread,
this cloud container (shared, not idle: treat the numbers as orders of magnitude). Script:
the probe used for this document builds the circuit through `parse_netlist_text` and
`build_mna_problem` and times `solvers.solve_transient`.

| Configuration | Wall time | Per step |
| --- | --- | --- |
| no callback | 9.2 ms | 0.46 µs |
| `progress_every=1` (Python callback every step) | 20.8 ms | 1.04 µs |
| `progress_every=100` | 9.1 ms | 0.45 µs |
| stimulus as a sum of 16 `heaviside` terms (`FUNC`) | 10.4 ms | 0.52 µs |
| 128 `heaviside` terms | 84 ms | 4.2 µs |
| 512 `heaviside` terms | 269 ms | 13.5 µs |
| 1024 `heaviside` terms | fails: `RecursionError` while lowering the expression tree | |

Readings:

* The analog step itself is sub-microsecond for a small circuit; a Python callback per step
  costs about as much again (0.6 µs: GIL, argument marshalling, the empty function). A user
  callback doing real work in Python will cost 2-20 µs per step; a one-million-step run then
  spends seconds in Python, which is acceptable for an interactive tool but is the price of
  bidirectional co-simulation.
* Expressing a bit stream as a `heaviside` sum scales linearly with the number of edges and
  breaks around a thousand edges (left-recursive expression tree). A USB packet has a few
  hundred edges; anything longer needs a table source.

## 4. Levels of support

### L0, available now: `FUNC` and `STEP` sources

A periodic square wave is `V1 a 0 STEP amplitude frequency duty`; an arbitrary short pattern
is a `FUNC` expression (`heaviside`, `min`/`max` for linear ramps). Limits: O(edges) per
evaluation, ~1000 edges at most, edges are not guaranteed to fall on grid points (a 83 ns bit
with a 1 ns step is fine, a 5 ns edge with a 1 ns step is resolved by only five points), and
the expression text is awkward to generate by hand.

### L1 (phase D1): table source and Python stimulus helper

1. **Native table source.** A `PWL` record (`subtype="PWL"`, `metadata["points"]`, as the
   parser of phase M1 produces for `V1 a 0 PWL(t1 v1 t2 v2 ...)`) compiles to a native
   `TableSource {times[], values[], repeat_period}` instead of an expression: evaluation is a
   binary search plus one linear interpolation (O(log N), N up to 10^6 points), with the
   last value held after the final point and optional periodic repetition (`R=`). Expression
   sources keep working unchanged. `PULSE` is lowered to a table too (seven numbers become
   four points per period plus the repeat), which also gives exact linear ramps. `.hb` samples
   the table at its collocation points like any other source.
2. **Breakpoint-aware grid.** `solvers.solve_transient` merges the uniform grid with the
   breakpoints of every table source (each corner time, and `k` extra points across each ramp
   when the ramp is shorter than `t_step`, `k = 4` by default) so every edge is hit exactly
   and the integrator never steps over a corner. The core already takes any monotone grid;
   `WaveformResult.time` reports the grid actually used; `output_max_points` decimation keeps
   responses bounded as today. No change to results for netlists without table sources.
3. **Python stimulus module `mna_simulation.stimulus`.** Pure Python, no core dependency:
   * `DigitalSignal(bit_period, v_low, v_high, t_rise, t_fall, t_start=0)` with
     `from_bits([...])`, `from_string("0b1011...")`, `from_events([(t, level), ...])`,
     `from_numpy(t, v)`, `nrzi()`, `bit_stuff(max_run=6)`, `invert()`, `delay(td)`,
     `differential(v_common, v_swing)` → `(positive, negative)`; every signal renders to
     `points()` (a `(t, v)` array) and `netlist_line(name, node_p, node_n)` (`PWL(...)`).
   * `usb.full_speed_packet(pid, payload, ...)` and `usb.low_speed_packet(...)`: SYNC,
     PID, payload, CRC5/CRC16, EOP (SE0 for two bit times then J), NRZI with bit stuffing,
     USB 2.0 full-speed timing (12 Mbit/s, 83.3 ns bit, 4-20 ns edges, 3.3 V single-ended
     levels on D+/D-, J/K polarity selectable for upstream/downstream). This is the worked
     example the request asked for; the same builder covers UART, SPI, I²C patterns with a
     few lines of Python.
   * Request integration: a `stimuli` option on `AnalysisRequest` (`[{"name": "Vdp",
     "node_p": "dp", "node_n": "0", "points": [[t, v], ...]}]`) so a client (Python or the
     web app) can send waveforms as arrays instead of text; the service turns them into table
     sources. Netlist text with `PWL(...)` keeps working for files and the raw-netlist mode.
4. **Tests and acceptance.** A 600-bit USB full-speed DATA0 packet into a receiver front end
   (termination, ESD diodes, series resistors, a 2N2222-style input stage once phase M2 has
   landed) runs in under 2 s with `t_step = 1 ns` on the CI machines; every edge time appears
   in `WaveformResult.time`; the table source matches the equivalent `FUNC` expression to
   1e-12 for a 16-edge pattern; results are identical for 1 and 4 threads; a 10^5-point PWL
   compiles and runs; CRC and bit stuffing are checked against known-good vectors.

### L2 (phase D2, design only): bidirectional co-simulation

Digital logic that reacts to analog values (a comparator feeding a state machine, a receiver
that decodes what the analog front end delivered, a digitally controlled regulator):

* Core hook: a `control` vector `u` owned by the transient driver; a new source kind
  `SlotSource {row, slot}` contributes `u[slot]` to `b(t)` (or scales a conductance for a
  switch); the Python callback receives a writable view of `u` together with `(step, t, x)`
  and may change it before the next step. One-step latency (the decision taken from `x_k`
  acts from `t_{k+1}`), which is the usual synchronous-cosim contract and exact for logic that
  samples on a clock much slower than the step.
* Scheduler in Python: an event queue of `(time, callable)`; events are merged into the grid as
  breakpoints, so a digital model written as a generator (`yield (t_next, outputs)`) is
  driven at exactly its own event times, and the analog side is sampled only when the model
  asks (`progress_every` is replaced by "call me at these times").
* Cost model from the measurements: 0.6 µs per Python round trip plus the user code; for a
  12 Mbit/s receiver sampling once per bit this is one call per ~80 steps, negligible.
* Risks: Backward Euler's numerical damping on fast edges (a trapezoidal option would be the
  cure, a core change); Newton restarts after a discontinuous control change (handled by the
  breakpoint mechanism: the change lands on a grid point and the next step starts from the
  previous state as today); determinism across thread counts holds because the callback
  runs between steps, never inside a parallel region.

### L3, not planned

A real event-driven digital kernel (HDL semantics, delta cycles), IBIS buffer models,
transmission lines for USB high-speed, co-simulation with an external HDL simulator. L2's
hook is the natural attachment point for an external simulator over a socket if that is ever
wanted.

## 5. USB case study

| | Low speed | Full speed | High speed |
| --- | --- | --- | --- |
| Bit rate | 1.5 Mbit/s | 12 Mbit/s | 480 Mbit/s |
| Bit period | 667 ns | 83.3 ns | 2.08 ns |
| Edge rate (spec) | 75-300 ns | 4-20 ns | ~500 ps |
| Steps for a 64-byte packet at `t_step` = bit/80 | ~50 000 | ~50 000 | ~50 000 |
| Analog-side modelling | lumped RC, diodes, termination: fine | fine (cable as a lumped Π section or two) | needs transmission lines, package models, 45 Ω terminations: out of scope |
| Verdict | feasible (D1) | feasible (D1) | stimulus yes, fidelity no |

For full speed with a ten-node front end the analog step is a few microseconds, so a packet
costs roughly 0.1-0.3 s; a whole frame (1 ms of bus time, ~12 000 bits) a few seconds.

## 6. Status

| Phase | Content | Status |
| --- | --- | --- |
| D1 | native table source, breakpoint-aware grid, `mna_simulation.stimulus` with the USB builder, `stimuli` request option, tests, docs | planned, runs after the parser phase M1 lands (it owns the `PWL`/`PULSE` records) |
| D2 | control-slot hook and Python scheduler for bidirectional co-simulation | designed, not scheduled |
| L3 | HDL kernel, IBIS, transmission lines | not planned |
