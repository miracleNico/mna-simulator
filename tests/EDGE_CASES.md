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
`TRAN_ERROR`, `MOR_CONFIG`, `ZeroDivisionError`, ...) are raised at once. The error after the last
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
3. **CMOS inverter chains of 4 or more Level-1 stages do not converge at gmin=0.** Four
   stages succeed only at gmin=1e-5 (rails visibly pulled to 1.179 V / 8.5 mV). Six stages
   fail at every gmin ("Newton step could not reduce residual at iteration 1."). The corpus
   adds 10 Meg keeper resistors to its random chains to stay robust.
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
| Floating island, *service* | Fails at every gmin, because gmin is stamped only across C, D and Level-1 BJT/MOS terminals and never node-to-ground (unlike SPICE GMIN). Same error | `test_floating_island_fails_every_gmin_step_in_the_service` | unclear |
| Capacitor-only internal node, *basic* | Same singular `NR_CONVERGENCE` | `test_capacitor_only_node_is_singular_at_gmin0` | yes |
| Capacitor-only internal node, *service* | Succeeds at gmin=1e-12 (1k + 1T + 1T divider). I(V1) ≈ -5e-13 agrees with the analytic value to only ~7 digits (ill-conditioned) | `test_capacitor_only_node_is_solved_by_gmin_stepping_in_the_service` | yes |
| Voltage-source loop (KVL-consistent or not) | Passes the parser (only *parallel* sources are checked). Singular `NR_CONVERGENCE`. The code says "NR" for a linear singular matrix | `test_voltage_source_loop_passes_the_parser_and_is_singular` | unclear (no topology check; misleading code) |
| Inductor-only loop / parallel inductors | Passes the parser. Singular `NR_CONVERGENCE` for `.op` and for `.tran` (its OP) at every gmin | `test_inductor_only_loop_is_singular_at_dc_for_op_and_tran` | unclear |

### Zero and negative values

| Case | Current behaviour | Test id | Looks intentional? |
| --- | --- | --- | --- |
| R = `0`, `0.0`, `0k`, `1e-999` (underflows to 0) | Raw `ZeroDivisionError` "float division by zero" (`.op` and `show_matrix`), no `.code` | `test_zero_resistor_raises_zero_division_error` | no |
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
| Diode directly across 5 V / 20 V source | `SCE` `NR_CONVERGENCE` "Solver did not converge after N iterations." (also at the default N=10 000, ~6 s) | `test_diode_clamped_by_an_ideal_source_never_converges` | no |
| Diode exp overflow | `f`/`J` evaluate to `inf` at 20 V (RuntimeWarning). Damped Newton treats non-finite trial residuals as +inf and halves the step | `test_diode_expression_overflows_to_inf` | yes |
| Diode with 5 V / 100 V / 1000 V through 1 Ω | Converges (V_D = 0.842 / 0.921 / 0.979 V) and satisfies the device equation | `test_diode_huge_forward_bias_through_a_resistor_converges` | yes |
| Level-1 BJT, `vbe/vt > 80` (`vt = 0.02585`) | `exp` argument clipped to [-80, 80]: currents are flat beyond 2.068 V, but the reported dIc/dVb stays at its clip value (2.24e21), so the Jacobian disagrees with the flat function | `test_level1_bjt_exponent_is_clipped_at_80_vt` | decided: keep until P3; a zero slope beyond the clip made the HB Newton Jacobian singular when an iterate crossed the clip, so P3 replaces the hard clip by junction limiting with linear extrapolation (2.4.1) |
| Level-1 BJT, `vbe/vt < -80` | Reverse saturation currents (pinned) | `test_level1_bjt_negative_clip_gives_reverse_saturation_currents` | yes |
| Level-1 MOS, `vgs == vth` exactly | Off (`vgs <= vth`): zero current and zero derivatives. Region "off" | `test_level1_mos_exactly_at_threshold_is_off` | yes |
| Level-1 MOS, `vds == vgs - vth` exactly | Saturation branch (`vds < vov` is triode). Model is C1-continuous at the boundary. Region "saturation" | `test_level1_mos_triode_saturation_boundary_uses_saturation_and_is_c1_continuous` | yes |
| Level-1 MOS, `vd < vs` | Symmetric device: source and drain swap, current negative. Region "reverse". `vd == vs` takes the forward branch | `test_level1_mos_reverse_mode_swaps_source_and_drain` | yes |
| PMOS polarity | `I = -I_nmos(-vd, -vg, -vs)`. Compiled path and direct function agree | `test_level1_pmos_polarity` | yes |
| Negative `vth` in a Level-1 netlist | `abs()` applied: `-0.4` behaves as 0.4 | `test_level1_mos_negative_vth_is_taken_as_absolute_value` | unclear |
| CMOS inverter chain, 3 / 4 / 6 stages, VIN=0, `max_iter=200` | 3: gmin 0 (rails within ~2e-9 V). 4: only gmin=1e-5, rails 1.179 V / 8.5 mV. 6: every gmin fails with `NR_CONVERGENCE` "Newton step could not reduce residual at iteration 1.". 10 Meg keepers give gmin 0 | `test_cmos_inverter_chain_dc_convergence_depends_on_depth` | no |

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
| Nonlinear circuit plus floating island / MOS with a floating gate | `SCE` `JACOBIAN_SINGULAR` "Jacobian is singular at iteration 0." (*basic* and *service*) | `test_jacobian_singular_is_reachable` | yes |
| `.tran` Newton failure (array IC, `max_iter=1`) | `SCE` `NR_CONVERGENCE` "NR failed to converge at t=1.00e-07 s" | `test_transient_newton_failure_reports_the_time_point` | yes |
| HB Newton, `max_iter=1` | `SCE` `HB_ERROR` "Harmonic balance solver did not converge within limit." The HB Newton is undamped and starts from x=0 (it also cannot climb a 2-diode DC string; see the corpus note in `random_diode_net`) | `test_harmonic_balance_newton_failure` | yes (undamped HB: unclear) |
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

## Differential corpus notes

- The random MOS chain generator adds 10 Meg keepers because of highlight 3. The diode
  generator skips HB on its two-diode DC string because the HB Newton is undamped.
- `.ac` cases on circuits whose OP needs gmin stepping are avoided: with `max_iter` ignored
  (highlight 4), each failing gmin step can run 10 000 Newton iterations in the oracle. The
  live core now passes `max_iter` through (2.4.1); every corpus `ac` case converges at gmin 0
  within its `max_iter` (200 for `ac_nl`), so the oracle and the live core still agree. No
  corpus case uses `f_start=0`, `f_stop=0` or `points <= 0`, so the `.ac` option ruling needs
  no corpus edits either.
- The oracle keeps the old option coercion and raises `TRAN_ERROR` as the base `CSE`. No
  corpus case uses a non-canonical option value (string boolean, non-integral number,
  string time, list `init_condition`) or reaches `TRAN_ERROR`, so the 2.4.1 option and
  `TRAN_ERROR` changes need no corpus edits; both packages' `build_options` agree on every
  corpus analysis. The full set of accepted and rejected forms is in
  `tests/test_option_coercion.py`.
