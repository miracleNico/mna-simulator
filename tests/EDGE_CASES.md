# Edge-case behaviour of the current core

Spec of what the pure-Python numerical core (frozen as `tests/reference/mna_reference`,
commit `e5172cb`) does on edge cases. Every row is pinned by a test in
`tests/test_edge_cases.py`, which runs against `mna_simulation` and is the acceptance
spec for the C++ port. "Looks intentional?" is a reviewer's guess from the code, not a
decision:

- **yes**: deliberate, documented, or physically correct. Keep it.
- **no**: most likely an accident (truthiness bug, missing validation, unwrapped error).
  Decide whether the C++ core keeps it for parity or fixes it. A fix means changing the
  pinned test and, if the oracle must follow, a CHANGELOG entry in
  `tests/reference/README.md`.
- **unclear**: a defensible design choice that differs from SPICE or from a neighbouring
  code path. The coordinator should decide and record the decision.
- **decided: ... (2.4.1)**: the ruling in `CPP_MIGRATION_PLAN.md` section 2.4.1 has been
  applied to `mna_simulation`. "Current behaviour" gives the new behaviour; "Was:" keeps
  what the frozen oracle still does. Rulings scheduled for a later phase keep their old
  verdict until they land.

Notation: *basic* = `core_basic.build_and_run_basic` at gmin=0. *service* =
`SimulationService.run_request`, which retries over `DEFAULT_GMIN_STEPS = [0, 1e-12, ..., 1e-5]`
and reports the gmin that worked in `metadata["gmin"]`. Since P3 the circuit is compiled once
and only numerical failures are retried (`NR_CONVERGENCE`, `JACOBIAN_SINGULAR`, `AC_ERROR`,
`LU_CONVERGENCE`, numerical `HB_ERROR`, raw `LinAlgError`); structural errors (`NETLIST_FATAL`,
`TRAN_ERROR`, `MOR_CONFIG`, a zero resistor, ...) are raised at once. The error after the last
step is unchanged (decided: 2.4 / 2.4.1). Was: retried on **any** exception, rebuilding everything.
`SCE` = `SolverConvergenceError`, `CSE` = `CircuitSimulatorError` (base class).
Test ids are functions in `tests/test_edge_cases.py`.

## Highlights: behaviour flagged "no" (likely unintended)

1. **Falsy option values are replaced by defaults.** `options.x or default` turns 0 into
   "unset": `tran_stop=0` runs to 0.1 s, `tran_step=0` uses 1e-5, `.ac` `f_start=0` uses 1 Hz,
   `f_stop=0` uses 1 MHz, `points=0` uses 100, HB `harmonics=0` uses 8.
   **Decided (2.4.1), applied for `.ac` and `.tran`:** only `None` means unset; `f_start=0` /
   `f_stop=0` are DC points and `points <= 0` raises `AC_ERROR` (full and MOR sweeps);
   `tran_stop=0` returns the single initial sample and `tran_step <= 0` is `TRAN_ERROR`. The HB
   part lands with the P4 cut-over.
2. **A diode clamped by an ideal voltage source never converges** (5 V or 20 V, even with
   10 000 iterations). Damped Newton creeps about Vt per iteration. With 1 Ω in series it
   converges even at 1000 V.
   **Decided (2.4.1), applied (P3b):** when the damped Newton stalls (4 iterations accepted at
   a damping scale <= 1/64) or finds no damping scale, circuits with exponential junctions
   switch to the junction strategy: SPICE `pnjlim` on every diode / BJT junction before the
   damping loop (falling back to the plain step) and the natural monotonicity test as a
   second acceptance rule. 1 V and 5 V now converge at gmin 0 to the exact answer (5 V:
   I = -7.2e72 A); 20 V (exp(800) = inf) still fails. Circuits the damped Newton solves
   without stalling are bit for bit unchanged.
3. **CMOS inverter chains of 4 or more Level-1 stages do not converge at gmin=0.** Four
   stages succeed only at gmin=1e-5 (rails visibly pulled to 1.179 V / 8.5 mV). Six stages
   fail at every gmin ("Newton step could not reduce residual at iteration 1."). The corpus
   adds 10 Meg keeper resistors to its random chains to stay robust.
   **Decided (2.4.1), applied (P3b):** a failed `.op` Newton solve retries inside the core with
   gmin continuation (node-to-ground and device-terminal gmin from 1e-2 down to the requested
   gmin, each level from the previous solution, sharing `max_iter`); 4, 6 and 8 stages now
   converge at gmin 0 with the rails in place. During gmin stepping gmin is also stamped from
   every node to ground.
4. **`.ac` ignores `max_iter`, `v_tol` and `f_tol` for its internal operating point.**
   `solve_ac` calls `solve_dc_nr` with the defaults (10 000 / 1e-9 / 1e-9).
   **Decided (2.4.1), applied:** the options are passed through, so the AC operating point is
   the `.op` solution for the same `max_iter`/`v_tol`/`f_tol` and fails where `.op` fails.
5. **Unwrapped exceptions.** R=0 raises `ZeroDivisionError`. A singular linear transient
   step raises raw `numpy.linalg.LinAlgError` at gmin 0, which the service then hides by
   succeeding at gmin=1e-12. String `t_stop` and float `points` raise `TypeError`. A JSON list
   `init_condition` raises `AttributeError`.
   **Decided (2.4.1), applied for the option part:** `build_options` parses time/frequency
   strings with metric suffixes (`"1m"` is 1e-3), coerces integral `points` (`"3"`, `3.0`),
   rejects non-integral ones with a plain `ValueError` (HTTP 400), and turns a numeric list
   `init_condition` into a float column array. R=0 and the singular transient step are
   P2/P3 items. **Decided (2.4.1), applied for the transient step (P3):** a singular linear
   transient step is `NR_CONVERGENCE` naming the step time; gmin stepping still applies.
   **Decided (2.4.1), applied for R=0:** the builder raises `NETLIST_FATAL` "'<name>' has zero
   resistance" (also for a small-signal BJT with `rpi = 0`), once.
6. **Non-finite results come back as `status="ok"`.** `V1 a 0 DC 1e999` returns
   `[nan, -inf]`, and `FUNC 0 nan` returns an all-NaN waveform.
   **Decided (2.4.1), applied for `.op` and `.tran` (P3):** `NR_CONVERGENCE` "non-finite solution"
   (`.tran` steps add " at t=... s").
7. **`init_condition` is handled inconsistently.** Strings are ignored on linear `.op` and on
   every `.tran`. They are validated only on nonlinear `.op` (case-sensitive). A wrong-length
   array raises `TypeError` from the lambdified function.
   **Decided (2.4.1), applied at the request level:** `build_options` upper-cases
   `DEFAULT`/`RANDOM` and rejects any other string with a plain `ValueError`, so requests no
   longer reach the core with an ignored or case-mismatched string. The core-level handling
   and the length check are P3 items. **Decided (2.4.1), applied in the core (P3):**
   `DEFAULT`/`RANDOM` (any case) set the Newton starting point of every nonlinear `.op` and of
   the initial operating point of every nonlinear `.tran` (`RANDOM` draws from NumPy's global RNG
   in Python; the core receives `x0`), other strings are `NETLIST_FATAL` there; on linear paths
   strings and arrays are ignored with a diagnostic in `SimulationResult.diagnostics`; arrays of
   the wrong length are `NETLIST_FATAL` everywhere.
8. **Options are not type-checked.** `use_krylov="false"` (or `krylov="no"`) turns Krylov ON
   because `bool("false")` is true, and `krylov_rank=2.7` is silently truncated to 2.
   **Decided (2.4.1), applied:** strict booleans (`true/false/yes/no/on/off/1/0`, any case)
   and strict integers (integral floats/strings accepted); anything else is a plain
   `ValueError` that the HTTP layer answers with 400.

## Table

### Empty, trivial and single-node circuits

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Empty netlist, whitespace, comments only, directives only | `NetlistError` `NETLIST_FATAL` "No components entered in netlist." (parser and service) | `test_empty_netlist_raises_netlist_fatal` | yes |
| Netlist containing only `GND vss` | Accepted. `status="ok"`, no labels, empty matrices, `dc_solution=[]` | `test_gnd_only_netlist_is_accepted_and_returns_empty_ok_result` | unclear (GND counts as a component) |
| `CircuitIR(components=[])` straight into *basic* | Zero-size problem. `.op` gives shape (0,1), `.tran` gives values shape (0, n_steps+1) | `test_empty_circuit_ir_builds_a_zero_size_problem` | yes |
| `I1 n1 0 DC 1m` + `R1 n1 0 1k` | V(n1) = **+1 V**: current is injected into node1. SPICE gives -1 V. VCCS/CCCS follow the SPICE sign, so the independent I source is the odd one out | `test_single_node_current_source_injects_into_node1` | unclear |
| V source only / R only | `[5, 0]` / `[0]` | `test_single_node_voltage_source_only_and_resistor_only` | yes |
| Capacitor-only single node | *basic*: `SCE` `NR_CONVERGENCE` "Linear system is singular (G matrix): Singular matrix". *service*: succeeds at gmin=1e-12, V=0 | `test_single_node_capacitor_only_singular_at_gmin0_solved_by_gmin_stepping` | yes |

### Singular G

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Floating resistor island, *basic* gmin 0 | `SCE` `NR_CONVERGENCE` "Linear system is singular (G matrix): Singular matrix" | `test_floating_island_is_singular_at_gmin0` | yes |
| Floating island, *service* | Converges at gmin=1e-12: since P3b gmin stepping also stamps gmin from every node unknown to ground (never on branch rows), the island sits at 0 V; `metadata["gmin"] == 1e-12` and the response `G` carries the node stamps. Was: fails at every gmin, because gmin was stamped only across C, D and Level-1 BJT/MOS terminals and never node-to-ground (unlike SPICE GMIN). Same error | `test_floating_island_converges_at_the_first_gmin_step_in_the_service` (was `test_floating_island_fails_every_gmin_step_in_the_service`) | decided: node-to-ground gmin during gmin stepping (2.4.1) |
| Capacitor-only internal node, *basic* | Same singular `NR_CONVERGENCE` | `test_capacitor_only_node_is_singular_at_gmin0` | yes |
| Capacitor-only internal node, *service* | Succeeds at gmin=1e-12. The node voltage is defined by gmin alone: with node-to-ground gmin (P3b) V(c) = V(b)/3 (1T across C1, 1T + 1T at c), I(V1) ≈ -2.67e-12, agreeing with the analytic value to only ~7 digits (ill-conditioned). Was: the 1k + 1T + 1T divider, V(c) = V(b)/2, I(V1) ≈ -5e-13 | `test_capacitor_only_node_is_solved_by_gmin_stepping_in_the_service` | decided: node-to-ground gmin moves a gmin-defined operating point (2.4.1) |
| Voltage-source loop (KVL-consistent or not) | Passes the parser (only *parallel* sources are checked). Singular `NR_CONVERGENCE`. The code says "NR" for a linear singular matrix | `test_voltage_source_loop_passes_the_parser_and_is_singular` | unclear (no topology check; misleading code) |
| Inductor-only loop / parallel inductors | Passes the parser. Singular `NR_CONVERGENCE` for `.op` and for `.tran` (its OP) at every gmin | `test_inductor_only_loop_is_singular_at_dc_for_op_and_tran` | unclear |

### Zero and negative values

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| R = `0`, `0.0`, `0k`, `1e-999` (underflows to 0), `-0`; small-signal BJT `rpi = 0` | `NetlistError` `NETLIST_FATAL` "'R1' has zero resistance" (BJT: "'Q1' has zero resistance (rpi)") from `build_mna_problem`, raised once by the service (structural, no gmin retries; `.op` and `show_matrix`). Was: raw `ZeroDivisionError` "float division by zero", no `.code` (the oracle still raises it; corpus case `zero_resistor` declares the new error in `current_error`) | `test_zero_resistor_raises_netlist_fatal_from_the_builder` (was `test_zero_resistor_raises_zero_division_error`) | decided: NETLIST_FATAL from the builder (2.4.1) |
| C = 0 | Silently skipped (`add_entry` drops 0.0). A node reached only through C=0 floats, giving singular `NR_CONVERGENCE` | `test_zero_capacitor_is_silently_skipped` | unclear |
| L = 0 | Branch row and `I(L1)` label kept, C entry dropped, so it acts as a 0 V source (a short) | `test_zero_inductor_keeps_its_branch_and_acts_as_a_short` | yes |
| Negative R | Accepted: divider 1k/-2k gives V(b) = 2 V | `test_negative_resistor_is_accepted` | unclear (no validation) |
| Negative C / L in `.tran` | Negative C: accepted and integrated (pinned values). Negative L with L/h making G + C/h exactly singular: native KLU reports the zero pivot as `SCE` `NR_CONVERGENCE` "Linear system is singular at t=... s (G + C/h matrix): Singular matrix" (the oracle returned garbage from a 1e-16 LAPACK pivot). Was: raw `LinAlgError` from the KLU facade | `test_negative_capacitor_and_inductor_are_accepted_in_transient` | decided: KLU singularity report kept, NR_CONVERGENCE wrapper (2.4.1) |
| Linear `.tran` step with G + C/h exactly singular (R=1k, C=-1u, h=1m) | *basic*: `SCE` `NR_CONVERGENCE` "Linear system is singular at t=1.00e-03 s (G + C/h matrix): Singular matrix". *service*: succeeds at gmin=1e-12 (the code is numerical). Was: raw `numpy.linalg.LinAlgError` "Singular matrix" (only the linear transient loop was unwrapped) | `test_singular_linear_transient_step_raises_nr_convergence_with_the_step_time` (was `test_singular_linear_transient_step_raises_raw_numpy_linalg_error`) | decided: NR_CONVERGENCE with the step time (2.4.1) |

### Parser conflicts, duplicates, ground aliases, NaN/Inf

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Parallel V sources (same polarity, reversed, V and VCVS) | `NetlistError` `NETLIST_FATAL` "Voltage source parallel conflict: ..." | `test_parallel_voltage_sources_are_rejected` | yes |
| V source with both terminals on (aliased) ground / same node | `NETLIST_FATAL` "Voltage source has both terminals on ground." / `COMPONENT_ERROR` | `test_voltage_source_with_both_terminals_on_ground_is_rejected` | yes |
| Duplicate branch names (V/V, L/L, `V1` and `V1:L`) | Parser accepts. Builder raises `NETLIST_FATAL` "Duplicate component name 'X'. ..." | `test_duplicate_branch_component_names_are_rejected_at_build_time` | yes |
| Duplicate non-branch names (R1 twice, C1 twice) | Silently accepted, both stamped (C doubles). Names are case-sensitive (`R1` and `r1` are distinct) | `test_duplicate_non_branch_component_names_are_silently_accepted` | unclear |
| GND aliases: several aliases, alias declared after use, lowercase `gnd`, alias used as a control node, `GND 0`, wrong arity | All aliases map to "0" in any order. `GND`/`GND a b` give `COMPONENT_ERROR` "'GND' GND statement expects '<node>' (got N arg(s))." | `test_gnd_aliases` | yes |
| `nan`, `NaN`, `inf` literals | Rejected by the value regex: `NETLIST_FATAL` "'R2' has invalid value format 'nan'." / "'V1' has invalid value 'nan'." | `test_nan_and_inf_literals_are_rejected_by_the_value_pattern` | yes |
| `1e999` literal | Parses to `inf`. As R: an open circuit. As a V source: `SCE` `NR_CONVERGENCE` "non-finite solution" after every gmin step. As a SIN frequency: `TypeError` from the expression compiler. Was: `status="ok"` with a non-finite solution (`[nan, -inf]` with LAPACK) | `test_overflowing_literals_parse_to_infinity` | decided: NR_CONVERGENCE for non-finite solutions (2.4.1) |
| `FUNC 0 nan` | `SCE` `NR_CONVERGENCE` "non-finite solution" (the initial operating point is NaN; from a given state the first step fails with "non-finite solution at t=1.00e-03 s"). Was: `status="ok"`, all-NaN waveform | `test_func_expression_nan_raises_non_finite_solution` (was `test_func_expression_nan_produces_an_ok_all_nan_waveform`) | decided: NR_CONVERGENCE for non-finite solutions (2.4.1) |

### Conditioning and device models

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Values from 1e-15 to 1e9 in one circuit (1G/1m divider, 1e-15 Ω loads, 1e9 A source, 1f, 1n) | `.op` matches the analytic values to 1e-12 relative (currents up to 1e15 A). `.tran` is finite | `test_magnitude_spread_1e_minus15_to_1e9_in_one_circuit` | yes |
| Diode directly across 1 V / 5 V / 20 V source | 1 V and 5 V converge at gmin 0 with `max_iter=200` to V = V1, I = -Is (exp(V1/0.025) - 1) (5 V: -7.2e72 A) through the P3b junction strategy (highlight 2). 20 V: `SCE` `NR_CONVERGENCE` "Newton step could not reduce residual at iteration N." (exp(800) overflows). Was: `NR_CONVERGENCE` "Solver did not converge after N iterations." for 5 V and 20 V (also at the default N=10 000, ~6 s) | `test_diode_clamped_by_an_ideal_source_converges_with_junction_limiting`, `test_diode_clamped_by_an_ideal_source_at_20v_still_fails` (was `test_diode_clamped_by_an_ideal_source_never_converges`) | decided: junction voltage-step limiting (2.4.1) |
| Diode exp overflow | `f`/`J` evaluate to `inf` at 20 V (RuntimeWarning). Damped Newton treats non-finite trial residuals as +inf and halves the step | `test_diode_expression_overflows_to_inf` | yes |
| Diode with 5 V / 100 V / 1000 V through 1 Ω | Converges (V_D = 0.842 / 0.921 / 0.979 V) and satisfies the device equation | `test_diode_huge_forward_bias_through_a_resistor_converges` | yes |
| Level-1 BJT, `vbe/vt > 80` (`vt = 0.02585`) | Beyond `|v/vt| = 80` the junction exponential continues linearly, `exp(80) * (1 + x - 80)` (and `exp(-80) * (1 + x + 80)` below -80), so the currents keep growing with the clip-point slope and dIc/dVb (2.24e21 at the clip) is that slope: value and derivative are continuous and consistent. Was: `exp` argument clipped to [-80, 80], currents flat beyond 2.068 V while the reported dIc/dVb stayed at its clip value | `test_level1_bjt_exponent_continues_linearly_beyond_80_vt` (was `test_level1_bjt_exponent_is_clipped_at_80_vt`) | decided: linear extrapolation together with junction limiting (2.4.1); a zero slope was rejected because it made the HB Newton Jacobian singular |
| Level-1 BJT, `vbe/vt < -80` | Reverse saturation currents (pinned) | `test_level1_bjt_negative_clip_gives_reverse_saturation_currents` | yes |
| Level-1 MOS, `vgs == vth` exactly | Off (`vgs <= vth`): zero current and zero derivatives. Region "off" | `test_level1_mos_exactly_at_threshold_is_off` | yes |
| Level-1 MOS, `vds == vgs - vth` exactly | Saturation branch (`vds < vov` is triode). Model is C1-continuous at the boundary. Region "saturation" | `test_level1_mos_triode_saturation_boundary_uses_saturation_and_is_c1_continuous` | yes |
| Level-1 MOS, `vd < vs` | Symmetric device: source and drain swap, current negative. Region "reverse". `vd == vs` takes the forward branch | `test_level1_mos_reverse_mode_swaps_source_and_drain` | yes |
| PMOS polarity | `I = -I_nmos(-vd, -vg, -vs)`. Compiled path and direct function agree | `test_level1_pmos_polarity` | yes |
| Negative `vth` in a Level-1 netlist | `abs()` applied: `-0.4` behaves as 0.4 | `test_level1_mos_negative_vth_is_taken_as_absolute_value` | unclear |
| CMOS inverter chain, 3 / 4 / 6 stages, VIN=0, `max_iter=200` | All at gmin 0: 3 directly (rails within ~2e-9 V); 4 and 6 through the P3b gmin continuation of `.op` (rails within the f_tol stop, ~5e-7 V; 8 stages too). 10 Meg keepers give gmin 0 directly. Was: 4 only at gmin=1e-5 (rails 1.179 V / 8.5 mV), 6 failed at every gmin with `NR_CONVERGENCE` "Newton step could not reduce residual at iteration 1." | `test_cmos_inverter_chain_dc_converges_at_gmin_0_at_any_depth` (was `test_cmos_inverter_chain_dc_convergence_depends_on_depth`) | decided: gmin continuation and node-to-ground gmin (2.4.1) |

### Transient time grid

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| `_transient_time_points(0, h)` | `[0.0]` | `test_transient_time_points_with_zero_t_stop` | yes |
| `solve_transient(t_stop=0)` directly | One sample: the DC operating point | `test_solve_transient_with_zero_t_stop_returns_only_the_initial_point` | yes |
| Negative `t_stop`, `t_step <= 0` (direct) | `NetlistError` with code `TRAN_ERROR` ("t_stop must be non-negative." / "t_step must be positive."); since P3 also for a non-finite value ("t_stop must be finite." / "t_step must be finite.", the oracle produced `[nan]` or looped forever). Was: base `CSE`, because `error_handler` had no mapping for `TRAN_ERROR` | `test_transient_time_points_reject_bad_arguments_with_tran_error` | decided: NetlistError (2.4.1) |
| Negative `t_stop` via *basic* / *service* | Same `NetlistError` `TRAN_ERROR` (the service raises it at once since P3: structural, no gmin retries; HTTP 400 with the code). Was: base `CSE`, re-raised after trying every gmin | `test_negative_t_stop_through_core_and_service_raises_tran_error` | decided: NetlistError (2.4.1) |
| `t_step > t_stop` | `[0, t_stop]` (one shortened step) | `test_t_step_larger_than_t_stop_gives_a_single_step` | yes |
| `t_step` not dividing `t_stop` | Grid ends exactly at `t_stop` with a shorter last step: `(1e-3, 3e-4)` gives 5 points | `test_t_step_not_dividing_t_stop_ends_exactly_at_t_stop` | yes |
| Grid construction | Repeated addition, not `i*h`: `(1.0, 0.1)` gives `0.30000000000000004`, `0.7999999999999999`, ... A bit-exact port must accumulate the same way | `test_time_grid_is_built_by_repeated_addition` | unclear |
| Snapping tolerance `max(|t_stop|, |t_step|, eps) * 1e-9` | Overshoot inside it snaps (11 points). Just outside it, an extra ~1e-11 s step is appended (12 points; a tiny BE step makes C/h huge). Undershoot shortens the last step | `test_time_grid_snapping_tolerance_is_1e_minus_9_relative` | unclear |
| `tran_stop=0` / unset / `tran_step=0` via *basic* | Only `None` is unset (0.1 s / 1e-5). `tran_stop=0` returns the single initial sample (the operating point), `tran_step=0` is `NetlistError` `TRAN_ERROR` "t_step must be positive." (raised at once by the service). Was: `or` substitution, `t_stop` became 0.1 s and `t_step` 1e-5 (so `.tran 0` ran 10 001 steps) | `test_core_basic_treats_only_none_as_unset_tran_stop_and_tran_step` (was `test_core_basic_replaces_falsy_tran_stop_and_tran_step_with_defaults`) | decided: only None is unset (2.4.1) |
| `.tran 0`, `.tran 1m`, bare `.tran` | `{t_stop: 0, t_step: 1e-5}`, `{1e-3, 1e-5}`, `NETLIST_FATAL` "Use .tran <time> (<step_time>)." | `test_tran_directive_defaults` | yes |
| Service options `t_stop="1m"` | Parsed by `build_options` with `parse_value`, like `.tran` (`"1m"` is 1e-3; also `t_step`, `f_start`, `f_stop`, `time_window`, `krylov_tol`, `v_tol`, `f_tol`). An unparsable string is a plain `ValueError` "t_stop must be a number (metric suffixes allowed)". Was: not parsed, `TypeError` ("'<' not supported ...") | `test_service_parses_string_time_options` (was `test_service_does_not_parse_string_time_options`) | decided: coerce (2.4.1) |

### Non-convergence

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Diode `.op` with `max_iter` 1 or 2 | `SCE` `NR_CONVERGENCE` "Solver did not converge after N iterations." | `test_nr_convergence_after_max_iter` | yes |
| Nonlinear circuit plus floating island / MOS with a floating gate | *basic* (gmin 0): `SCE` `JACOBIAN_SINGULAR` "Jacobian is singular at iteration 0." (the gmin continuation does not hide it: its final solve at the requested gmin must leave a nonsingular Newton matrix). *service*: converges at gmin=1e-12, node-to-ground gmin holds the floating nodes at 0 V (P3b). Was: `JACOBIAN_SINGULAR` at every gmin in the *service* too | `test_jacobian_singular_is_reachable` | decided: node-to-ground gmin (2.4.1) |
| `.tran` Newton failure (array IC, `max_iter=1`) | `SCE` `NR_CONVERGENCE` "NR failed to converge at t=1.00e-07 s" | `test_transient_newton_failure_reports_the_time_point` | yes |
| HB Newton, `max_iter=1` | `SCE` `HB_ERROR` "Harmonic balance solver did not converge within limit." The HB Newton starts from x=0 with the reference's undamped iteration; since P3b a breakdown of it (non-finite residual or singular Newton matrix) restarts from x=0 with a damped Newton (residual backtracking, 28 halvings, pnjlim on the sampled junction voltages) for the rest of `max_iter`, which climbs the 2-diode DC string of `rand_diode_s2` (was: "Singular HB Jacobian."). Runs the undamped iteration solves are bit for bit unchanged | `test_harmonic_balance_newton_failure` | yes; damped restart decided (2.4.1 / 2.6) |
| `.ac` on a diode circuit with `max_iter=1` | `SCE` `NR_CONVERGENCE` with the same message as `.op` with the same options: the AC operating point receives `max_iter`/`v_tol`/`f_tol` and equals the `.op` solution for them. Was: succeeds, the AC operating point ignored the options (10 000 / 1e-9 / 1e-9) while `.op` failed | `test_ac_operating_point_honours_max_iter` (was `test_ac_operating_point_ignores_max_iter`) | decided: pass the options through (2.4.1) |

### `init_condition`

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| `"DEFAULT"`, `"RANDOM"` (global NumPy RNG), ndarray start | Same operating point. The array is copied, not mutated | `test_init_condition_default_random_and_array_reach_the_same_point` | yes |
| Unknown string on nonlinear `.op` / nonlinear `.tran` | `NetlistError` `NETLIST_FATAL` "Unsupported init condition 'x'." (core level). `default`/`random` in any case (trimmed) are accepted since P3. Through the service, `build_options` upper-cases `default`/`random` and rejects other strings first (row "JSON list" below). Was: lowercase `default`/`random` also rejected; nonlinear `.tran` ignored every string | `test_init_condition_unknown_string_raises_on_nonlinear_op`, `test_init_condition_strings_are_case_insensitive` | decided: case-insensitive (2.4.1) |
| Strings on linear paths, `DEFAULT`/`RANDOM` on nonlinear `.tran` | Linear `.op`/`.tran`: ignored with a diagnostic ("init_condition 'x' ignored: the circuit is linear, ...") in `SimulationResult.diagnostics`. Nonlinear `.tran`: the Newton starting point of the initial operating point (`op_used` unchanged). Was: silently ignored on linear `.op` and on every `.tran` | `test_init_condition_strings_on_linear_and_nonlinear_paths` (was `test_init_condition_strings_are_ignored_on_linear_op_and_on_tran`) | decided: strings start every nonlinear Newton solve (2.4.1) |
| ndarray on `.tran` | Used as the t=0 state. No OP is computed (`op_used=False`) | `test_init_condition_array_sets_the_transient_initial_state` | yes |
| ndarray of the wrong length | `NetlistError` `NETLIST_FATAL` "init_condition has k values, expected n (one per unknown)." on every `.op` (linear too) and `.tran`. Was: `TypeError` from the lambdified function on nonlinear `.op`, ignored on linear `.op` | `test_init_condition_with_wrong_length_raises_netlist_fatal` (was `test_init_condition_with_wrong_length_raises_type_error`) | decided: length validated (2.4.1) |
| JSON list through the service | `build_options` converts a numeric list/tuple to a float column array, so nonlinear `.op` and `.tran` use it like an ndarray; linear `.op` ignores it. Other strings than `DEFAULT`/`RANDOM` (any case) and non-numeric lists are a plain `ValueError` "init_condition must be 'DEFAULT', 'RANDOM', or a numeric array". A wrong-length list is `NetlistError` `NETLIST_FATAL` (P3 length check, raised at once). Was: `AttributeError` "'list' object has no attribute 'flatten'" after every gmin; then (option coercion applied) `TypeError` inside the solve | `test_init_condition_json_list_through_the_service` | decided: lists accepted, length validated (2.4.1) |

### `probe_nodes`, output decimation

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Unknown `probe_nodes` without MOR | Silently ignored; `probe_nodes` only matters for MOR | `test_unknown_probe_nodes_are_ignored_without_mor` | unclear |
| Unknown `probe_nodes` with MOR | `NetlistError` `MOR_CONFIG` "MOR output label 'V(nope)' is not present in simulation labels (...)." | `test_unknown_probe_nodes_are_rejected_with_mor` | yes |
| `output_max_points` = 0, None, negative, >= length | No decimation, no `output_decimation` metadata | `test_output_decimation_boundaries` | yes |
| `output_max_points` = 1 / 2 (11 points) | Returns 2 / 3 points: the last sample is always appended, so `returned_points > max_points` | `test_output_decimation_boundaries` | no |
| `output_max_points` = 10 (11 points) | Stride `ceil(11/10)=2` returns 6 points | `test_output_decimation_boundaries` | unclear |
| `output_max_points` = `"7"` / `7.9` | `int()` coercion gives 7 | `test_output_decimation_boundaries` | yes |
| Decimation scope | Applies to `.tran` and HB time windows. Never to `.op`/`.ac`/spectra | `test_output_decimation_applies_to_hb_windows_but_not_to_op` | yes |

### Krylov options

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| `krylov_max_iter=1`, rank 1 on a general 3x3 | GMRES fails, so it falls back to the direct solve (exact answer). `krylov_converged=False`, `krylov_used_direct_fallback=True` | `test_krylov_non_convergence_falls_back_to_the_direct_solve` | yes |
| `krylov_rank` 0 / -3 / `"abc"` / `""` / 2.7 / `"2.5"` | Plain `ValueError` from `build_options`, raised before the gmin loop. No `.code`. 0 / -3: "krylov_rank must be 'auto' or a positive integer."; the rest: "krylov_rank must be an integer". Was: `int()`'s "invalid literal ..." message for `"abc"`/`""`, and 2.7 truncated | `test_invalid_krylov_rank_raises_plain_value_error` | decided: plain ValueError, HTTP 400 (2.4.1) |
| `krylov_rank` 2.0 / `"1"` / None / `"AUTO"` / `" auto "` | 2 manual / 1 manual / auto / auto / auto (integral floats and strings accepted; same rule for `krylov_max_iter`, `krylov_restart`, `max_iter`, `points`, `harmonics`, `mor_order`). Was: 2.7 gave 2 manual (truncated) | `test_krylov_rank_coercion` | decided: strict integers (2.4.1) |
| Only `krylov_restart` given | Used as a manual rank | `test_krylov_restart_alone_sets_a_manual_rank` | yes |
| `krylov_method` `"foo"` / `"hermitian"` | `ValueError` "Unknown Krylov method 'x'." even with `use_krylov=False`. `hermitian` is accepted by `solvers._krylov_route` but not by the backend alias table | `test_invalid_krylov_method_raises_even_when_krylov_is_off` | decided: preserve the validation (2.4.1) |
| `krylov_method` aliases (`GMRES`, `minres`, None, `""`) | Normalised. `minres` maps to `conjugate_residual` | `test_krylov_method_aliases` | yes |
| `use_krylov="false"`, `krylov="no"` | Krylov OFF. Booleans (`use_krylov`/`krylov`, `use_mor`/`mor`, `mor_validate`) accept bools, 0/1 and `true/false/yes/no/on/off/1/0` (case-insensitive, trimmed); anything else is a plain `ValueError` "use_krylov must be a boolean" (HTTP 400). None means unset (default). Was: Krylov ON (`bool()` of a non-empty string) | `test_use_krylov_string_false_keeps_krylov_off` (was `test_use_krylov_string_false_enables_krylov`) | decided: strict booleans (2.4.1) |

### MOR options

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Invalid output node | `NetlistError` `MOR_CONFIG` "MOR output label 'V(nope)' is not present ..." | `test_mor_with_an_invalid_output_node` | yes |
| No output nodes | `MOR_CONFIG` "MOR requires at least one output node, e.g. V(out) or I(L1)." | `test_mor_without_output_nodes` | yes |
| `probe_nodes` not a subset of the outputs | `MOR_CONFIG` "Display nodes must be a subset ...". Skipped when `mor_validate=False` | `test_mor_with_probe_nodes_not_a_subset_of_outputs` | yes |
| Output labels `"out, V(out), I(V1)"` | Bare names resolve to `V(...)`, duplicates are dropped, giving `["V(out)", "I(V1)"]` | `test_mor_output_labels_are_normalised_and_deduplicated` | yes |
| `tpwl` on `.ac`; `linear` on nonlinear `.tran` | `MOR_CONFIG` with the pinned messages | `test_mor_method_and_mode_mismatches` | yes |
| `mor_method="xyz"`, `mor_order=0` | Plain `ValueError` (not `MOR_CONFIG`). A non-integral `mor_order` (2.5, `"2.5"`) is "mor_order must be an integer" (2.4.1) | `test_invalid_mor_method_and_order_raise_plain_value_error` | decided: preserve plain ValueError, HTTP 400 (2.4.1) |
| MOR on `show_matrix` | `mor_used=False`, reason ".show_matrix requires the full MNA system." | `test_mor_on_show_matrix_reports_not_used` | yes |
| Linear MOR basis on near-dependent seeds (random corpus `rand_rlc_ladder_s2` `.ac`, `rand_rlc_mesh_s1` `.ac`, `rand_rlc_mesh_s2` `.tran`: an output or input direction the Krylov loop reproduces) | The basis is built with classical Gram-Schmidt twice ("twice is enough") and stays orthonormal (`||V^T V - I|| < 1e-12`); the reduced outputs match the full-order solution of the same analysis within 1e-11 of its maximum (measured below 6e-15). `rand_rlc_ladder_s2` keeps 11 basis columns, the other two 12 and 6 (k == n). A rounding residual that survives both passes is still accepted (the 1e-14 threshold stays absolute), as an orthonormal direction. Was: one pass accepted rounding residuals above the absolute 1e-14 threshold (e.g. 3e-13 on a vector of norm 2.6e3) whose direction lay mostly along the basis, the basis became numerically singular (cond 1e18..1e21 with k == n) and the outputs were rounding noise: `rand_rlc_mesh_s2` reached 7e5 V where the full-order solution peaks at 0.33 V; `rand_rlc_ladder_s2` had 13 columns and a 29 % error | `test_linear_mor_basis_stays_orthonormal_on_near_dependent_seeds` | decided: Gram-Schmidt twice, improve not parity (2.4.1) |

### Harmonic balance

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| No SIN/COS source; SIN at 0 Hz | `SCE` `HB_ERROR` "No sinusoidal sources found." | `test_hb_without_a_usable_sinusoidal_source` | yes |
| Tone below the 1e-5 Hz GCD resolution | `HB_ERROR` "Could not determine HB base frequency." | `test_hb_source_frequency_below_the_gcd_resolution` | yes |
| `harmonics` smaller than needed (1 kHz + 3 kHz, H=2) | Expanded to needed+4 = 7. H=3 is kept. `base_frequency_hz` is `1000.0000000000001` (integer GCD times the 1e-5 resolution) | `test_hb_harmonics_smaller_than_needed_are_expanded_to_needed_plus_4` | expansion: yes; float artifact: unclear |
| `harmonics=0` / negative / unset | `harmonics <= 0` (option or `.hb 0`) is `SCE` `HB_ERROR` "harmonics must be a positive integer, got <n>."; unset (`None`) gives 8. A bare `.hb` directive gives 16. Was: 0 became 8 (`or 8`) and a negative count was expanded to needed+4 | `test_hb_harmonics_unset_means_8_and_non_positive_is_an_error` | decided: only `None` means unset, `<= 0` is `HB_ERROR` (2.4.1); 8 vs 16: unclear |
| `.hb`, `.hb 5`, `.hb 1m`, `.hb 5 1m`, `.hb 2.5 1m` | `{H:16}`, `{5}`, `{16, window 1m}`, `{5, 1m}`, `{2, 1m}` (2.5 truncated) | `test_hb_directive_parameters` | yes |

### AC sweep

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| `points=1` | Single frequency `f_start` | `test_ac_single_point` | yes |
| `f_start=0`, `f_stop=0`, `points=0` (options or `.ac 0 1k 1`) | Only `None` means unset (defaults 1 Hz, 1 MHz, 100 points). `f_start=0` / `f_stop=0` are DC points (`.ac 0 1k 1` gives `[0.0]`; an RC low-pass reads 1 at 0 Hz). `points <= 0` is `SCE` `AC_ERROR` "AC sweep needs at least one frequency point (points = 0)." on the full and the MOR sweep. Was: 0 replaced by 1 Hz, 1 MHz and 100 points | `test_ac_zero_start_and_stop_are_dc_points_and_nonpositive_points_raise` (was `test_ac_falsy_start_stop_and_points_are_replaced_by_defaults`) | decided: only `None` means unset (2.4.1) |
| Sweep shape | **Linear** `np.linspace` (SPICE uses dec/oct/lin). Descending and negative frequencies are accepted (\|H(-f)\| = \|H(f)\|) | `test_ac_sweep_is_linear_and_accepts_descending_and_negative_frequencies` | linear: unclear; negative f: no |
| `points=2.5` / `"3"` | `"3"` and `3.0` are coerced to 3; `2.5` and `"2.5"` are a plain `ValueError` "points must be an integer" (HTTP 400). Was: `TypeError` from `np.linspace` for both | `test_ac_points_must_be_an_integer` | decided: strict integers (2.4.1) |
| No AC/SIN/COS source | All-zero magnitudes, `status="ok"` | `test_ac_without_an_ac_source_returns_zero_magnitudes` | yes |

### `show_matrix`, mode selection, source semantics

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| `mode=show_matrix` request | Only `matrices` (`G, C, f_str, b_dc, b_ac, b_time_str`). `f_str`/`b_time_str` carry the NumPy-style expression text. One complex entry turns every `b_ac` entry into `{real, imag}` | `test_show_matrix_mode_request_payload` | yes (exact expression text: unclear for the port) |
| `.show_matrix` directive | `NETLIST_FATAL` "Unsupported directive '.show_matrix'." | `test_show_matrix_is_not_a_netlist_directive` | unclear (it is the default mode) |
| Mode selection | No directive and no mode gives `show_matrix`. The last directive wins. Request `mode` overrides the directive (its params still apply). `dyn` runs as `tran` | `test_mode_selection_rules` | yes |
| `.op` with FUNC/STEP/COS sources that are non-zero at t=0 | `.op` uses `b_dc` only, so these sources are 0 V. The OP inside `.tran` uses `b_dc + b_time(0)` and gives 1.5 / 2 / 2 V | `test_op_ignores_time_dependent_sources_even_when_nonzero_at_t0` | unclear |
| Controlled-source signs (E, G, F, H) | SPICE-consistent (pinned values) | `test_controlled_source_sign_conventions` | yes |

### SPICE dialect front end (phase M1)

Rulings of `HIERARCHY_AND_MODELS_PLAN.md` section 3.4 ("plan 3.4") and the ones the M1 parser had to make ("M1"). Test ids are functions in `tests/test_spice_dialect.py`; the frozen oracle knows none of these constructs.

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Title line | None: the first line is a netlist line (`My amplifier` is a MOS line and fails); comment a title out with `*` | `test_no_title_line` | decided: plan 3.4 |
| Which parser | Netlists with `{`, `;` or `$` go to the front end at once; the single-pass loop hands over at the first `.model`/`.subckt`/`.param`/`.include`/`.lib`/`.global`/`.control` line, `X` line or `+` line, or after any error when such a line exists further down (a model defined after its use) | `test_corpus_netlists_parse_exactly_as_before` | decided: M1 |
| Case | Keywords, model, subcircuit and parameter names and node names are case-insensitive; a node keeps the spelling of its first occurrence (applies to every netlist). Component names stay case-sensitive for the duplicate rules (`R1`/`r1` are two parts), but a CCCS/CCVS controlling name resolves case-insensitively; subcircuit instance names are unique case-insensitively | `test_case_insensitive_keywords_models_keys_and_nodes`, `test_controlling_source_names_are_case_insensitive` | decided: plan 3.4 (component names: M1) |
| Unit letters | Read after a value only from a list of unit words (`V A Ohm F H S Hz W deg`, plural and long forms): `10kOhm` -> `10k`, `1x` stays invalid (`.tran 1x`, duty cycle `2x` keep their errors). `T` and `MIL` become computed values. The dialect's own LEVEL1/small-signal transistor parameter lists stay strict | `test_value_suffixes_and_unit_letters` | decided: M1 |
| Comments | `;` anywhere, `$` only after whitespace (`1k$ohm` is a value), `*` and `#` lines; comment lines may sit between a line and its `+` continuation | `test_inline_semicolon_and_dollar_comments`, `test_continuation_lines_and_comment_lines_between_them` | decided: M1 |
| Node names | `.`, `+`, `-` allowed (flattened `X1.n`, vendor ports `IN+`); `#` stays reserved for the builder's internal nodes | `test_nested_flattening_names_nets_ports_ground_and_globals` | decided: M1 |
| Model token | A token after the terminals that names a card in scope wins; otherwise the dialect's keywords and numbers keep their meaning (`D1 a k 1e-15`, `Q1 ... QNPN LEVEL1`) and another identifier is an undefined model. A 4-token `R/C/L` line whose value names no card keeps "invalid value format" | `test_dialect_device_lines_keep_their_meaning`, `test_model_card_errors` | decided: M1 |
| Instance `M=` | BJT and MOS: `metadata["m"]` (instance multiplier). Diode: dropped with a diagnostic, since its `metadata["m"]` is the card's grading coefficient and `area` is its only multiplier | `test_diode_card_record_follows_the_contract`, `test_bjt_card_record_follows_the_contract` | decided: M2 ruling |
| `inf` card values | `VAF=inf`, `IKF=INF`, ... are passed on as `inf` | `test_infinite_card_parameters_pass_through` | decided: M2 ruling |
| Compatibility fields | Diode `value` = `is` (SPICE default `1e-14` when the card has none), BJT `value/value2/value3` = `is/bf/br` (defaults `1e-16/100/1`); MOS leaves them `None`; card defaults are never written into `metadata` | `test_bjt_card_record_follows_the_contract`, `test_mos_level3_card_three_and_four_terminal_instances` | decided: M1 |
| MOS cards | LEVEL 2/3 map with a diagnostic; any other level is an error when a device references the card; card `W`/`L` share the instance keys (the instance's win); three-terminal instance: `ctrl_node2 = None` | `test_mos_levels_beyond_3_are_errors_naming_the_level` | decided: plan 3.4 (W/L: M1) |
| Passive cards | `R`/`C`/`L` card: its `R=`/`C=`/`L=` is the value an instance omits; `RES`/`CAP`/`IND` card: a multiplier of the instance value; `M=` means parallel parts; `TC1`/`TC2`/... a diagnostic; records stay plain (no metadata) | `test_passive_cards_flatten_to_plain_values` | decided: M1 |
| Card diagnostics | Only for referenced cards, once per card definition: unknown parameters ("not used by the ... equations") and table parameters the current equations ignore ("recorded but not modelled yet"); unreferenced cards of unsupported types: one summary line | `test_unknown_card_parameters_are_copied_and_reported_once_per_card`, `test_unreferenced_cards_of_unsupported_types_are_a_diagnostic` | decided: M1 |
| Duplicates | Two `.model`, `.subckt` or `.param` of one name in one scope: error naming both lines | `test_subcircuit_errors`, `test_model_card_errors`, `test_parameter_errors` | decided: M1 |
| Scopes | Lexical: nested `.subckt` and `.model` are visible in their parent only; `PARAMS:` defaults are evaluated in the instance scope, overrides in the caller's; parameters are evaluated eagerly (an unused bad one is still an error); a parameter named `pi`/`e` shadows the constant; `log` is the natural log | `test_param_scopes_overrides_and_expressions`, `test_parameters_shadow_the_constants` | decided: M1 |
| Tied ports | A subcircuit `R`/`C` whose two terminals land on one net is dropped with a diagnostic; any other element is an error | `test_tied_ports_drop_a_shorted_resistor_with_a_diagnostic` | decided: M1 |
| Includes | Netlist text: error asking to paste the library; files: relative to the including file, then `--lib` directories; CLI `run-text`: current directory; one-argument `.lib x` is a file when quoted or containing `.`, `/`, `\`, else a section start; sections in a file read whole, or pasted into text, are skipped with a diagnostic | `test_include_from_netlist_text_asks_for_the_library_to_be_pasted`, `test_pasted_library_sections_are_skipped_with_a_diagnostic` | decided: plan 3.4 (`.lib` forms: M1) |
| Source forms | The dialect's arities keep their meaning (`SIN amp freq` with two values is this simulator's sine and the `.ac` stimulus); parenthesised `SIN(...)` or three or more values is SPICE's (no `.ac` contribution); `SINE` = `SIN` | `test_source_records_follow_the_contract` | decided: plan 3.2 (`SINE`: M1) |
| Source analyses | `.op`: DC value only (0 without one, no fallback to the waveform at t = 0); `.tran`/`.hb`: the waveform (`b_dc + (waveform - dc)`); `.ac`: `ac_mag * exp(j ac_phase deg)` (builder change in the `V`/`I` branches) | `test_op_uses_dc_ac_uses_the_phasor_and_tran_the_waveform`, `test_ac_phase_reaches_the_ac_analysis` | decided: M1 |
| Transient defaults | Zero or omitted `tr tf` / `tau` take the `.tran` step, `pw per` the stop time, SIN `freq` `1/t_stop` (the netlist's last `.tran`, filled at parse time); without `.tran`: ideal edges, endless pulse, single pulse, constant SIN | `test_transient_defaults_come_from_the_tran_directive`, `test_pulse_waveform_has_linear_edges_and_repeats` | decided: M1 |
| `I` sign | Independent current sources keep this simulator's sign inside subcircuits too; a diagnostic names them per subcircuit (vendor models assume SPICE's) | `test_current_sources_in_subcircuits_get_a_polarity_diagnostic` | unclear (coordinator) |
| `.tran` / `.ac` forms | One or two values: the dialect's `<t_stop> [<t_step>]` (diagnostic when `t_stop < t_step`); three or four: SPICE `tstep tstop tstart [tmax]`; `uic` ignored. `.ac lin` exact, `dec`/`oct` a linear sweep with the same point count (diagnostic) | `test_spice_analysis_forms` | decided: M1 |
| Bare directives | `.noise`, `.tf`, `.four`, `.meas`, `.step`, `.temp`, `.ic`, `.nodeset`, `.print`, `.plot` without arguments keep "Unsupported directive" (pinned for `.noise`); with arguments they are diagnostics | `test_bare_noise_stays_an_unsupported_directive`, `test_harmless_directives_are_diagnostics` | decided: M1 |
| Unsupported letters | `J Z S W T O U K` (and `A N P Y`) lines are errors only from the element's shortest real form (`Zfoo a 0 1k` keeps "Cannot infer type") | `test_short_lines_with_unsupported_letters_keep_the_dialect_error` | decided: M1 |
| Behavioural forms | `TABLE`, `LAPLACE`, `VALUE`, `POLY`, `FREQ`, `CHEBYSHEV` on `E/F/G/H`, `B` with `V=`/`I=`, `.func`, `.if`, `.alter`: errors; `.control` blocks: skipped with a diagnostic | `test_answer_changing_constructs_are_errors` | decided: plan 3.4 (`VALUE`, `POLY`, `B`, `.if`, `.alter`: M1) |

### Vendor model cards (phase M2)

Model-card devices (`HIERARCHY_AND_MODELS_PLAN.md` 3.2 and 3.4: `metadata["model"]` = `spice`,
`gummel_poon`, `spice_level1`) are new behaviour with no oracle counterpart, so the rows of this
section are pinned by `tests/test_spice_devices.py`, `tests/device_reference/` and
`cpp/tests/test_devices.cpp` (doctest case names in quotes) instead of `tests/test_edge_cases.py`.
"decided (M2)" marks a choice made while implementing phase M2 where SPICE offers variants.

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| Thermal voltage of the card equations | `vt = 8.617333262e-5 * 300.15` (0.025865 V, 27 °C) for the SPICE diode, Gummel-Poon and card MOS; the ideal diode keeps 0.025 V and the Level-1 BJT 0.02585 V, so the oracle comparison holds | `test_card_vt_is_kt_over_q_at_27_celsius`, `test_constants_match` | decided (plan 3.4) |
| Diode reverse breakdown | SPICE3f5: below `-bv_eff` the current is `-is exp(-(bv_eff + v) / vt)` (plain `vt`, not `n vt`), where `bv_eff` solves `is (exp((bv - bv_eff) / vt) - 1 + bv_eff / vt) = ibv` by SPICE's fixed-point iteration (RELTOL 1e-3, at most 25 steps); when `ibv < is bv / vt` SPICE raises `ibv` and `bv_eff = bv`. At `v = -bv_eff` the value jumps by `is (3 n vt / (e bv_eff))^3` (1e-10 relative at 100 V) and the slope from about 0 to `is / vt`, as in SPICE3f5. `ibv` scales with `area` (the shift is area-independent). No warning is emitted for the raised `ibv` | `test_zener_breakdown_follows_the_shifted_breakdown_branch`, `test_diode_breakdown_voltage_and_current`, "SPICE diode: breakdown voltage is shifted so the current at -BV is IBV" | decided (M2): SPICE3f5 `diotemp.c`/`dioload.c` |
| Diode reverse region between `-bv_eff` and `-3 n vt` | SPICE3f5's `-is (1 + (3 n vt / (e v))^3)`, value and slope continuous with the forward law at `-3 n vt`; without `bv` it extends to `-inf` | "SPICE diode: reverse region joins the forward branch with value and slope at -3 n vt" | decided (M2): SPICE3f5 |
| Card exponentials beyond `abs(x) = 80` | Every card exponential (diode forward and breakdown, the four Gummel-Poon junction terms) continues along its tangent beyond 80, like the P3b Level-1 BJT: a SPICE diode clamped by a 20 V source converges (the ideal diode at 20 V still overflows and fails) | `test_spice_diode_across_an_ideal_source_converges_through_junction_limiting`, `test_diode_breakdown_voltage_and_current` (clip points), "SPICE diode: exponent clips continue linearly with continuous value and slope" | decided (M2): reuse of the 2.4.1 clip |
| Junction limiting of card devices | The SPICE diode junction (intrinsic anode, cathode, `vt = n vt`, `is`) and both Gummel-Poon junctions (`vt`, `is`, as `bjtload.c`) are appended to `Circuit::junctions()` after the Level-1 ones, so existing circuits are limited in the same order; SPICE3f5's breakdown-side limiting of the diode is not applied | `test_spice_diode_across_an_ideal_source_converges_through_junction_limiting`, `test_gummel_poon_base_driven_by_an_ideal_source_converges` | decided (M2) |
| Gummel-Poon details that differ from `bjtload.c` | No reverse-bias linearization below `-5 nf vt` (SPICE: constant `-is` with the secant conductance `-is / vbe`); `sqrt(max(0, 1 + 4 q2))` without SPICE's jump to 1 at exactly 0 (its derivative is 0 where clamped); no gmin inside the junction currents (the builder stamps gmin across b-e and b-c). All three keep the Jacobian the exact derivative of the currents | `tests/device_reference/test_reference_jacobians.py`, "Gummel-Poon: Jacobian matches central differences, NPN and PNP" | decided (M2) |
| `vaf`, `var`, `ikf`, `ikr` equal to 0 | Treated as infinite (SPICE's "not given"), bit-identical to omitting them | "Gummel-Poon: ISE = ISC = 0, IKF = IKR = inf, VAF = VAR = inf give the simple transport form" | yes (SPICE) |
| MOS threshold sign | The n-channel-frame threshold is `type * VTO` (SPICE): the absolute value of `VTO` for every enhancement card, as plan 3.2 states, but a depletion card (NMOS `VTO < 0`, PMOS `VTO > 0`) keeps its sign instead of being folded to its absolute value | `test_pmos_three_node_instance_puts_the_bulk_on_the_source`, `test_mos_card_with_zero_gamma_is_bit_identical_to_level1` | decided (M2), differs from the literal plan text for depletion cards |
| MOS body effect for `vbs > 0` | `vth` held at its `vbs = 0` value (slope jumps from `-gamma / (2 sqrt(phi))` to 0 at `vbs = 0`); in reverse mode the body voltage is measured from the drain | `test_nmos_inverter_body_effect_threshold_shift_matches_the_formula`, "MOS card: body effect shifts the threshold by gamma (sqrt(phi - vbs) - sqrt(phi))" | decided (plan 3.2) |
| Card MOS gmin and floor | gmin across drain-source (as Level-1) and bulk-source when the bulk is a separate node; no 1e-12 S drain-source floor (the Level-1 builder adds one); `rd`/`rs` of the card are not stamped | `test_cmos_inverter_transient_swings_rail_to_rail` | decided (M2) |
| Internal nodes | `<name>#a` (diode `rs`), `<name>#b`, `#c`, `#e` (BJT `rb`, `rc`, `re`), only for positive values after the area rule; numbered after the sorted external nodes and before the branch currents, so branch-current indices shift by their count (a `B` source reading `x[i]` of a branch current must account for it); a circuit node of the same name or two components with the same name are `NETLIST_FATAL` | `test_internal_nodes_follow_the_sorted_external_nodes_and_precede_branches`, `test_internal_node_collisions_and_bad_parameters_are_netlist_errors` | decided (plan 3.4) |
| Instance multipliers | `area` scales `is ise isc ikf ikr ibv cjo cje cjc` up and `rs rb re rc` down; `m` multiplies the area of a BJT and `beta` and the overlap capacitances of a MOS; on a diode `m` is the card's grading coefficient (unused before M4), not a multiplier | `test_diode_area_scales_is_and_rs`, `test_diode_m_is_the_grading_coefficient_not_a_multiplier` | decided (M2): the plan's common `m` key collides with the diode card's `M` |
| BJT substrate node | Ignored: a node used only by substrate terminals is not an unknown | `test_substrate_node_is_ignored` | decided (plan 3.2) |
| Unsupported card values | Non-numeric values, `is <= 0`, `n <= 0`, `bv <= 0`, negative resistances or capacitances, `phi <= 0`, `w`/`l`/`kp <= 0`, MOS LEVEL other than 1-3: `NETLIST_FATAL` naming the component, the card and the parameter | `test_internal_node_collisions_and_bad_parameters_are_netlist_errors`, "compiled circuit: invalid model-card parameters are NETLIST_FATAL" | yes |
| Operating points of card devices | Region codes (diode off/forward/reverse/breakdown, BJT and MOS as Level-1 with the body-adjusted threshold), currents and conductances from `_core.operating_points`; non-finite values (`bv_eff` without breakdown, `rpi` without base conductance) are `None` so the JSON response never carries `inf` | `test_backend_with_gmin_steps_and_json_safe_metadata`, `test_2n2222_common_emitter_bias_matches_the_gummel_poon_beta` | yes |

## Differential corpus notes

- The random MOS chain generator adds 10 Meg keepers because of highlight 3 (kept although P3b
  makes keeperless chains converge, so the random cases stay oracle-comparable).
- P3b deliberate changes are recorded per analysis as `corpus.Deviation` and checked by
  `differential.compare.check_deviation` (the oracle must still fail or behave as recorded; the
  live result must match the oracle on an equivalent netlist or an analytic answer):
  `floating_resistor_island` op (converges at gmin 1e-12; equivalent netlist with the 1e-12 S
  node shunts as resistors), `capacitor_only_node_gmin` op and tran (gmin-defined node moves
  from V(b)/2 to V(b)/3; equivalent netlist with every gmin conductance as a resistor), the new
  `diode_ideal_source` op (analytic answer) and `cmos_chain6_no_keepers` op (oracle on the keeper
  version, 1e-6 absolute: the keeperless solve stops at f_tol), and `rand_diode_s2` hb (the
  two-diode DC string HB, added now that the damped HB restart converges; success only).
- `.ac` cases on circuits whose OP needs gmin stepping are avoided: with `max_iter` ignored
  (highlight 4), each failing gmin step can run 10 000 Newton iterations in the oracle. The
  live core now passes `max_iter` through (2.4.1); every corpus `ac` case converges at gmin 0
  within its `max_iter` (200 for `ac_nl`), so the oracle and the live core still agree. No
  corpus case uses `f_start=0`, `f_stop=0` or `points <= 0`, so the `.ac` option ruling needs
  no corpus edits either.
- The three MOR analyses of the Gram-Schmidt ruling (`rand_rlc_ladder_s2-ac-mor`,
  `rand_rlc_mesh_s1-ac-mor`, `rand_rlc_mesh_s2-tran-mor`) are compared with the oracle's
  full-order solution of the same analysis (`MOR_FULL_ORDER_REFERENCE` in
  `tests/differential/test_reference_parity.py`, `mor` tolerance); status, labels, grids,
  shapes and contract metadata are still compared with the oracle's MOR run.
- The oracle keeps the old option coercion and raises `TRAN_ERROR` as the base `CSE`. No
  corpus case uses a non-canonical option value (string boolean, non-integral number,
  string time, list `init_condition`) or reaches `TRAN_ERROR`, so the 2.4.1 option and
  `TRAN_ERROR` changes need no corpus edits; both packages' `build_options` agree on every
  corpus analysis. The full set of accepted and rejected forms is in
  `tests/test_option_coercion.py`.
