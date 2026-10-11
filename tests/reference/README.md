# Frozen reference oracle (`mna_reference`)

`tests/reference/mna_reference/` is a byte-for-byte copy of the pure-Python
`mna_simulation/` package, renamed. It is the **test-only oracle** for the
C++ migration: the differential harness (`tests/differential/`) runs every
corpus circuit (`tests/corpus.py`) through both `mna_reference` and the live
`mna_simulation` package and asserts that the results agree within the
tolerance class of each case.

- Frozen at commit `e5172cb508e485eeb451ce5eb77ea6c148c13067`
  (2026-10-11, "Add performance review, multithreading plan, and C++ migration
  research"). See `mna_reference/FROZEN_AT.txt`.
- All imports inside the package are relative, so the rename needs no edits.
- `FROZEN_AT.txt` records a `tree_sha256` of the package;
  `tests/differential/test_harness_selftest.py::test_oracle_tree_matches_frozen_digest`
  fails on any edit, addition or removal under `mna_reference/`.
- `tests/conftest.py` prepends `tests/reference` to `sys.path`; the package is
  importable as `mna_reference` only while running the test suite. It lives
  under `tests/`, and the wheel packs only `mna_simulation`
  (`[tool.scikit-build] wheel.packages = ["mna_simulation"]`; the earlier
  setuptools config used `include = ["mna_simulation*"]`), so it never ships in
  a wheel. Checked on the editable install: `import mna_reference` fails outside
  the test run and the `mna-simulation` distribution records no `mna_reference`
  or `tests/` files.
- Later commits: up to `d8fb23e` ("Add native core build skeleton") the only
  change under `mna_simulation/` is the new `native/` subpackage (the C++
  extension accessor). It is deliberately not part of the oracle, which is the
  pure-Python core.

## Rules

1. **Never edit** anything under `mna_reference/`. Behaviour changes, speedups,
   refactors and the C++ port all go into `mna_simulation/` only. The oracle
   must stay the behaviour the port is measured against.
2. The single exception is a **confirmed oracle bug**: a defect in the frozen
   Python core that the team has agreed is wrong (not merely different from
   the new core). Fix it with the smallest possible change, add an entry to the
   CHANGELOG below (date, file, what was wrong, the fix, who confirmed it, and
   which corpus cases / edge-case tests changed), update `tree_sha256` in
   `FROZEN_AT.txt` (recipe in that file), and update `tests/EDGE_CASES.md` if
   the behaviour was listed there.
3. Do not re-copy `mna_simulation/` over the oracle to "refresh" it. A refresh
   is a deliberate re-baseline: record it in the CHANGELOG with the new commit
   hash and update `FROZEN_AT.txt`.
4. Intentional, agreed behaviour differences between the new core and the
   oracle belong in the test side (a changed tolerance class or expectation in
   `tests/corpus.py`, an `xfail` with a reason in the parity test, or an updated
   row in `tests/EDGE_CASES.md`), never in the oracle.

## CHANGELOG

| Date | File | Change | Confirmed by | Affected tests |
| --- | --- | --- | --- | --- |
| 2026-10-11 | (all) | Initial freeze from `e5172cb`. | - | - |
