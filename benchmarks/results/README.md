# Performance comparison results

`benchmarks/compare_backends.py` writes one JSON document per run into this
directory (default name `<UTC timestamp>_<hostname>.json`, e.g.
`20261011T173510Z_build-host.json`). Each document compares the frozen Python
reference (`mna_reference`, `tests/reference/`) with the C++ core
(`mna_simulation._core`) for every unit in `benchmarks/perf_units.py`.

## File format (`schema_version: 1`)

- `timestamp`: UTC time of the run (ISO 8601).
- `host`: hostname, `platform`, CPU count, Python/NumPy/SciPy versions, BLAS
  pools (threadpoolctl), `core_version` and `core_build_info` of the native
  core, `git_commit` (+ `git_dirty`), `loadavg` (where the OS reports it) and
  `load_warning` when the 1-minute load average was above 0.5.
- `methodology`: one warm-up run and `reps` timed repetitions per series
  (`time.perf_counter_ns`, GC disabled while timing), `statistic: median`
  with the MAD recorded, BLAS pinned to one thread, and the core `threads`
  measured.
- `units[]`: `id`, `category`, `size`, `available` and `unavailable_reason`
  (the phase that will provide the native side, e.g. `lands in P3: ...`),
  `reference_s` and `core_s["<threads>"]` as `{median, mad, samples}` in
  seconds, `speedup["<threads>"]` (reference median / core median), and
  `parity{max_rel_error, ok, tolerance}`. `error` is set when a unit raised.

`--markdown PATH` writes the same results as a Markdown table with a host
header (the table is always printed to stdout).

## What is tracked

Run outputs are ignored by git (`.gitignore`: `benchmarks/results/*.json`).
Only the per-platform baselines are committed:

- `baseline-linux.json`
- `baseline-windows.json`

Baselines are refreshed deliberately, at release time, from the artifacts of
the `Perf` workflow (`.github/workflows/perf.yml`, `perf-results.json` per
platform): download the artifact, check its `host` block (commit, versions,
load), rename it to `baseline-<platform>.json`, and commit it together with
any change to the floors in `tests/perf/test_perf_gate.py` that the new numbers
justify. Never commit a baseline from a developer machine that was doing other
work. None is committed yet: baselines start with the first release that ships
a migrated unit.

## Measure on an idle machine

Timings are only meaningful on an otherwise idle machine. Multithreaded BLAS
and the native thread pool both collapse under CPU contention
(`PERFORMANCE_PLAN.md` section 4.2 measured a 21x slowdown), so a busy machine
can turn a real speedup into noise. The script prints a warning and records
`host.load_warning` when the 1-minute load average is above 0.5; rerun such
results before comparing them with a baseline. Speedups (ratios) are more
stable than absolute times across machines, which is why the perf gate and the
baselines compare ratios.
