"""Corpus-wide parity: live ``mna_simulation`` must reproduce the frozen oracle.

One test per ``(corpus case, analysis)``. Ids read ``<case>-<mode>[-flags]``,
e.g. ``rc_ladder_ac_mor-ac-mor`` or ``workload_rlc_mesh_23x23-tran-krylov``.
Run only this harness with ``pytest -m differential``; deselect the large
circuits with ``-m "differential and not slow"``.
"""

from __future__ import annotations

from typing import Any

import pytest

from corpus import Case, iter_analyses
from differential.compare import assert_outcomes_match, run_request

pytestmark = pytest.mark.differential


def _flags(options: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if options.get("use_mor"):
        flags.append("mor" + (f"_{options['mor_method']}" if options.get("mor_method") not in (None, "auto") else ""))
    if options.get("use_krylov"):
        flags.append("krylov")
    if "time_window" in options:
        flags.append("window")
    if "output_max_points" in options:
        flags.append(f"maxpts{options['output_max_points']}")
    if "max_iter" in options and options["max_iter"] != 200:
        flags.append(f"maxiter{options['max_iter']}")
    return flags


def _params() -> list[Any]:
    params = []
    seen: dict[str, int] = {}
    for case, _index, mode, options in iter_analyses():
        base = "-".join([case.name, mode, *_flags(options)])
        seen[base] = seen.get(base, 0) + 1
        test_id = base if seen[base] == 1 else f"{base}#{seen[base]}"
        marks = [pytest.mark.slow] if "slow" in case.tags else []
        params.append(pytest.param(case, mode, options, id=test_id, marks=marks))
    return params


@pytest.mark.parametrize(("case", "mode", "options"), _params())
def test_reference_parity(case: Case, mode: str, options: dict[str, Any], reference_pkg, current_pkg) -> None:
    tolerance = case.tolerance_for(mode, options)
    ref = run_request(reference_pkg, case.netlist_text, mode, options)
    cur = run_request(current_pkg, case.netlist_text, mode, options)
    assert_outcomes_match(
        ref,
        cur,
        tolerance=tolerance,
        expects_error=case.expects_error,
        context=f"{case.name} {mode} {options} [{tolerance}]",
    )
