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
from differential.compare import (
    assert_mor_matches_full_order,
    assert_outcomes_match,
    full_order_options,
    run_request,
)

pytestmark = pytest.mark.differential

# MOR analyses whose oracle values are rounding noise (CPP_MIGRATION_PLAN.md 2.4.1). The
# reference's one-pass classical Gram-Schmidt accepts a rounding residual above its absolute
# 1e-14 threshold and its basis becomes numerically singular (cond(V) ~ 1e18..1e21 with k == n),
# so its reduced outputs miss the full-order solution of the same netlist (by up to six orders
# of magnitude). The live basis is built with Gram-Schmidt twice (an "improve, not parity"
# ruling) and stays orthonormal: for these analyses the live MOR outputs are compared with the
# oracle's FULL-ORDER solution (same analysis without MOR, restricted to the MOR output labels)
# under the mor tolerance, and everything else (status, labels, grids, shapes, contract
# metadata) with the oracle's MOR run. tests/test_native_mor.py::
# test_former_noise_mor_analyses_oracle_singular_live_orthonormal checks both bases.
MOR_FULL_ORDER_REFERENCE: frozenset[str] = frozenset(
    {
        "rand_rlc_ladder_s2-ac-mor",
        "rand_rlc_mesh_s1-ac-mor",
        "rand_rlc_mesh_s2-tran-mor",
    }
)


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
def test_reference_parity(
    case: Case, mode: str, options: dict[str, Any], reference_pkg, current_pkg, request: pytest.FixtureRequest
) -> None:
    tolerance = case.tolerance_for(mode, options)
    ref = run_request(reference_pkg, case.netlist_text, mode, options)
    cur = run_request(current_pkg, case.netlist_text, mode, options)
    context = f"{case.name} {mode} {options} [{tolerance}]"
    if request.node.callspec.id in MOR_FULL_ORDER_REFERENCE:
        full = run_request(reference_pkg, case.netlist_text, mode, full_order_options(options))
        assert_mor_matches_full_order(ref, cur, full, tolerance=tolerance, context=context)
        return
    assert_outcomes_match(
        ref,
        cur,
        tolerance=tolerance,
        expects_error=case.expects_error,
        context=context,
        current_error=case.current_error,
    )
