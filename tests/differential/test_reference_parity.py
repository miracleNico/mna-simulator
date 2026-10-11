"""Corpus-wide parity: live ``mna_simulation`` must reproduce the frozen oracle.

One test per ``(corpus case, analysis)``. Ids read ``<case>-<mode>[-flags]``,
e.g. ``rc_ladder_ac_mor-ac-mor`` or ``workload_rlc_mesh_23x23-tran-krylov``.
Run only this harness with ``pytest -m differential``; deselect the large
circuits with ``-m "differential and not slow"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from corpus import Case, iter_analyses
from differential.compare import Outcome, assert_outcomes_match, run_request

pytestmark = pytest.mark.differential

# MOR analyses whose oracle values are rounding noise (P5, CPP_MIGRATION_PLAN.md 2.7). The
# reference's single-pass classical Gram-Schmidt accepts a candidate whose residual is pure
# rounding error (relative size ~1e-16, absolute size above the 1e-14 threshold), which leaves
# its basis numerically singular (cond(V) ~ 1e18..1e21) although k == n. The reduced outputs
# then depend on the last bits of the LU solves (in the oracle they miss the full-order
# solution of the same netlist by up to six orders of magnitude), so no port can reproduce
# them. For these analyses the numeric series (waveform values, spectrum magnitudes) are not
# compared; status, labels, time/frequency grids, shapes and the contract metadata (mor_used,
# mor_method, ...) still are. tests/test_native_mor.py::
# test_noise_dominated_mor_analyses_are_singular_in_the_oracle checks that every entry really
# has a numerically singular oracle basis.
NOISE_DOMINATED_MOR: frozenset[str] = frozenset(
    {
        "rand_rlc_ladder_s2-ac-mor",
        "rand_rlc_mesh_s1-ac-mor",
        "rand_rlc_mesh_s2-tran-mor",
    }
)


def _without_values(outcome: Outcome) -> Outcome:
    """The outcome with its waveform values / spectrum magnitudes replaced by zeros of the same shape."""

    if outcome.response is None:
        return outcome
    response = outcome.response.model_copy(deep=True)
    for block_name, key in (("waveform", "values"), ("spectrum", "magnitudes")):
        block = getattr(response, block_name)
        if block is not None and block.get(key) is not None:
            block[key] = np.zeros(np.shape(block[key])).tolist()
    return Outcome(response=response, error=outcome.error)


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
    if request.node.callspec.id in NOISE_DOMINATED_MOR:
        ref, cur = _without_values(ref), _without_values(cur)
    assert_outcomes_match(
        ref,
        cur,
        tolerance=tolerance,
        expects_error=case.expects_error,
        context=f"{case.name} {mode} {options} [{tolerance}]",
    )
