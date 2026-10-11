"""Self-tests for the differential plumbing: oracle integrity, comparator, corpus rules.

The parity tests can only catch regressions if (a) the oracle really is the
frozen copy, (b) the comparator really fails on differences, and (c) the
corpus keeps to its size budget. These tests check all three.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import re
from pathlib import Path

import pytest

import corpus
from differential.compare import (
    TOLERANCES,
    Outcome,
    assert_errors_match,
    assert_outcomes_match,
    assert_responses_match,
    check_deviation,
    run_request,
)

pytestmark = pytest.mark.differential

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference" / "mna_reference"


# ---------------------------------------------------------------------------
# Oracle integrity
# ---------------------------------------------------------------------------


def _tree_digest(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != "FROZEN_AT.txt"
    )
    for path in files:
        # Windows checkouts may convert line endings; the recipe hashes LF-normalized bytes.
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + content + b"\0")
    return len(files), digest.hexdigest()


def test_oracle_tree_matches_frozen_digest() -> None:
    frozen = dict(
        line.split(": ", 1) for line in (REFERENCE_ROOT / "FROZEN_AT.txt").read_text().splitlines() if ": " in line
    )
    count, digest = _tree_digest(REFERENCE_ROOT)
    assert count == int(frozen["files"]), "files were added to / removed from tests/reference/mna_reference"
    assert digest == frozen["tree_sha256"], (
        "tests/reference/mna_reference changed. The oracle is frozen: revert the edit, or (confirmed oracle bug only) "
        "record it in tests/reference/README.md CHANGELOG and update tree_sha256 in FROZEN_AT.txt."
    )


def test_oracle_and_current_are_distinct_packages(reference_pkg, current_pkg) -> None:
    import mna_reference.errors as ref_errors
    import mna_simulation.errors as cur_errors

    assert reference_pkg.__name__ == "mna_reference"
    assert current_pkg.__name__ == "mna_simulation"
    assert Path(reference_pkg.__file__).resolve().parent == REFERENCE_ROOT
    assert Path(reference_pkg.__file__).resolve().parent != Path(current_pkg.__file__).resolve().parent
    assert ref_errors.NetlistError is not cur_errors.NetlistError


def test_oracle_never_imports_current_package(reference_pkg) -> None:
    for path in REFERENCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+mna_simulation\b", text, re.MULTILINE), path


# ---------------------------------------------------------------------------
# Comparator behaviour
# ---------------------------------------------------------------------------

_RC_NETLIST = "V1 a 0 SIN 1 1k\nR1 a b 1k\nC1 b 0 1u\n.end\n"


@pytest.fixture(scope="module")
def op_response(reference_pkg):
    outcome = run_request(reference_pkg, "V1 a 0 DC 5\nR1 a b 1k\nR2 b 0 1k\n.op\n", "op")
    assert outcome.ok
    return outcome.response


@pytest.fixture(scope="module")
def tran_response(reference_pkg):
    outcome = run_request(reference_pkg, _RC_NETLIST, "tran", {"t_stop": 1e-3, "t_step": 1e-4})
    assert outcome.ok
    return outcome.response


@pytest.fixture(scope="module")
def ac_response(reference_pkg):
    outcome = run_request(reference_pkg, _RC_NETLIST, "ac", {"f_start": 10, "f_stop": 1e4, "points": 5})
    assert outcome.ok
    return outcome.response


def _copy(response, **updates):
    return response.model_copy(update=updates, deep=True)


def test_identical_responses_match(op_response, tran_response, ac_response) -> None:
    for response in (op_response, tran_response, ac_response):
        assert_responses_match(response, _copy(response), "linear")


def test_cross_package_identical_run_matches(reference_pkg, current_pkg) -> None:
    ref = run_request(reference_pkg, _RC_NETLIST, "tran", {"t_stop": 1e-3, "t_step": 1e-4})
    cur = run_request(current_pkg, _RC_NETLIST, "tran", {"t_stop": 1e-3, "t_step": 1e-4})
    assert type(ref.response) is not type(cur.response)
    assert_outcomes_match(ref, cur, "linear")


@pytest.mark.parametrize("tolerance", sorted(TOLERANCES))
def test_drift_inside_and_outside_each_tolerance_class(op_response, tolerance) -> None:
    rtol, atol = TOLERANCES[tolerance]
    base = list(op_response.dc_solution)
    inside = [value * (1 + 0.1 * rtol) for value in base]
    outside = [value * (1 + 10 * rtol) + 10 * atol for value in base]
    assert_responses_match(op_response, _copy(op_response, dc_solution=inside), tolerance)
    with pytest.raises(AssertionError, match="dc_solution"):
        assert_responses_match(op_response, _copy(op_response, dc_solution=outside), tolerance)


def test_waveform_value_drift_and_shape_change_are_detected(tran_response) -> None:
    drifted = copy.deepcopy(tran_response.waveform)
    drifted["values"][1][5] += 1e-6
    with pytest.raises(AssertionError, match="waveform.values"):
        assert_responses_match(tran_response, _copy(tran_response, waveform=drifted), "linear")

    shorter = copy.deepcopy(tran_response.waveform)
    shorter["time"] = shorter["time"][:-1]
    with pytest.raises(AssertionError, match="waveform.time: shape"):
        assert_responses_match(tran_response, _copy(tran_response, waveform=shorter), "linear")

    with pytest.raises(AssertionError, match="waveform: present in reference only"):
        assert_responses_match(tran_response, _copy(tran_response, waveform=None), "linear")


def test_spectrum_drift_is_detected(ac_response) -> None:
    drifted = copy.deepcopy(ac_response.spectrum)
    drifted["magnitudes"][0][0] *= 1.001
    with pytest.raises(AssertionError, match="spectrum.magnitudes"):
        assert_responses_match(ac_response, _copy(ac_response, spectrum=drifted), "nonlinear")
    assert_responses_match(ac_response, _copy(ac_response, spectrum=drifted), "mor")


def test_nan_positions_must_agree(tran_response) -> None:
    with_nan = copy.deepcopy(tran_response.waveform)
    with_nan["values"][0][-1] = math.nan
    assert_responses_match(_copy(tran_response, waveform=with_nan), _copy(tran_response, waveform=with_nan), "linear")
    with pytest.raises(AssertionError, match="waveform.values"):
        assert_responses_match(tran_response, _copy(tran_response, waveform=with_nan), "linear")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("gmin", 1e-12),
        ("num_nodes", 99),
        ("labels", ["V(x)"]),
        ("op_used", True),
        ("krylov_method", "arnoldi_gmres"),
        ("output_decimation", {"original_points": 1, "returned_points": 1, "max_points": 1}),
    ],
)
def test_contract_metadata_changes_are_detected(op_response, key, value) -> None:
    metadata = dict(op_response.metadata)
    metadata[key] = value
    with pytest.raises(AssertionError, match=re.escape(f"metadata[{key!r}]")):
        assert_responses_match(op_response, _copy(op_response, metadata=metadata), "mor")


def test_non_contract_metadata_is_ignored(op_response) -> None:
    metadata = dict(op_response.metadata)
    metadata["G_nnz"] = -1
    metadata["some_new_diagnostic"] = "anything"
    assert_responses_match(op_response, _copy(op_response, metadata=metadata), "linear")


def test_status_labels_and_matrices_are_compared(op_response) -> None:
    with pytest.raises(AssertionError, match="status"):
        assert_responses_match(op_response, _copy(op_response, status="error"), "linear")
    with pytest.raises(AssertionError, match="labels"):
        assert_responses_match(op_response, _copy(op_response, labels=["V(z)"]), "linear")
    matrices = copy.deepcopy(op_response.matrices)
    matrices["G"][0][0] += 1e-3
    with pytest.raises(AssertionError, match=re.escape("matrices['G']")):
        assert_responses_match(op_response, _copy(op_response, matrices=matrices), "linear")
    matrices = copy.deepcopy(op_response.matrices)
    matrices["f_str"] = [["x"] for _ in matrices["f_str"]]  # symbolic text is not compared, only shape
    assert_responses_match(op_response, _copy(op_response, matrices=matrices), "linear")


def test_error_comparison_uses_class_name_and_code(reference_pkg, current_pkg) -> None:
    from mna_reference.errors import NetlistError as RefNetlistError
    from mna_reference.errors import SolverConvergenceError as RefSolverError
    from mna_simulation.errors import NetlistError as CurNetlistError

    assert_errors_match(RefNetlistError("x", code="PARSE_ERROR"), CurNetlistError("y", code="PARSE_ERROR"))
    with pytest.raises(AssertionError, match="error mismatch"):
        assert_errors_match(RefNetlistError("x", code="PARSE_ERROR"), CurNetlistError("x", code="NETLIST_FATAL"))
    with pytest.raises(AssertionError, match="error mismatch"):
        assert_errors_match(RefSolverError("x", code="NR_CONVERGENCE"), CurNetlistError("x", code="NR_CONVERGENCE"))
    with pytest.raises(AssertionError, match="current succeeded"):
        assert_errors_match(ZeroDivisionError("x"), None)
    with pytest.raises(AssertionError, match="reference succeeded"):
        assert_errors_match(None, ZeroDivisionError("x"))


def test_expected_error_is_checked_against_the_oracle(op_response) -> None:
    ok = Outcome(response=op_response)
    with pytest.raises(AssertionError, match="corpus expects NetlistError but the reference succeeded"):
        assert_outcomes_match(ok, ok, expects_error="NetlistError")
    wrong = Outcome(error=ValueError("boom"))
    with pytest.raises(AssertionError, match="reference raised ValueError"):
        assert_outcomes_match(wrong, wrong, expects_error="NetlistError")
    with pytest.raises(AssertionError, match="reference raised unexpectedly"):
        assert_outcomes_match(wrong, wrong)


def test_ruling_error_is_checked_against_the_current_package(op_response) -> None:
    from mna_simulation.errors import NetlistError as CurNetlistError

    ruling = ("NetlistError", "NETLIST_FATAL")
    ref = Outcome(error=ZeroDivisionError("float division by zero"))
    new = Outcome(error=CurNetlistError("'R1' has zero resistance"))
    assert_outcomes_match(ref, new, expects_error="ZeroDivisionError", current_error=ruling)
    with pytest.raises(AssertionError, match="ruling expects current NetlistError"):
        assert_outcomes_match(ref, ref, expects_error="ZeroDivisionError", current_error=ruling)
    with pytest.raises(AssertionError, match="current succeeded"):
        assert_outcomes_match(ref, Outcome(response=op_response), expects_error="ZeroDivisionError", current_error=ruling)
    with pytest.raises(AssertionError, match="ruling expects current"):
        assert_outcomes_match(
            ref, Outcome(error=CurNetlistError("x", code="PARSE_ERROR")), expects_error="ZeroDivisionError", current_error=ruling
        )
    # The oracle is still checked against its own expectation.
    with pytest.raises(AssertionError, match="corpus expects ZeroDivisionError"):
        assert_outcomes_match(Outcome(error=ValueError("x")), new, expects_error="ZeroDivisionError", current_error=ruling)


# ---------------------------------------------------------------------------
# Corpus rules
# ---------------------------------------------------------------------------


def test_corpus_names_are_unique_and_tagged() -> None:
    names = [case.name for case in corpus.CASES]
    assert len(names) == len(set(names))
    for case in corpus.CASES:
        assert case.tags, case.name
        assert {"existing", "workload", "random", "extra"} & set(case.tags), case.name


def test_corpus_covers_every_random_family_with_3_to_5_seeds() -> None:
    for family, (generator, seeds) in corpus.RANDOM_GENERATORS.items():
        assert 3 <= len(seeds) <= 5, family
        assert generator(seeds[0]).netlist_text == generator(seeds[0]).netlist_text, f"{family} is not deterministic"
        assert len({generator(seed).netlist_text for seed in seeds}) == len(seeds), f"{family} seeds collide"


def test_corpus_analyses_respect_the_size_budget() -> None:
    for case, _index, mode, options in corpus.iter_analyses():
        label = f"{case.name} {mode} {options}"
        if mode == "tran":
            steps = math.ceil(options["t_stop"] / options["t_step"] - 1e-9)
            assert steps <= 60, label
        elif mode == "ac":
            assert options["points"] <= 40, label
        elif mode == "hb" and "harmonics" in options:
            assert options["harmonics"] <= 8 or "hb_ghz_mixer" in case.name, label
        if case.tolerance == "nonlinear" and case.expects_error is None and mode != "show_matrix":
            assert options.get("max_iter") == corpus.NL_MAX_ITER, label


def test_iter_cases_filters_by_tags() -> None:
    mor_cases = list(corpus.iter_cases("mor"))
    assert mor_cases and all("mor" in case.tags for case in mor_cases)
    no_slow = list(corpus.iter_cases(exclude="slow"))
    assert no_slow and all("slow" not in case.tags for case in no_slow)
    assert len(list(corpus.iter_cases())) == len(corpus.CASES)
    assert corpus.get_case("hb_ghz_mixer").tolerance == "nonlinear"


# ---------------------------------------------------------------------------
# Deliberate changes (corpus.Deviation, CPP_MIGRATION_PLAN.md 2.4.1)
# ---------------------------------------------------------------------------


def test_deviations_name_existing_analyses_and_a_ruling() -> None:
    with pytest.raises(ValueError, match="does not match an analysis"):
        corpus.Case("x", "R1 a 0 1\n", [corpus.op()], tags=("extra",), deviations=(("tran", corpus.Deviation("r")),))
    recorded = [(case.name, mode) for case in corpus.CASES for mode, _ in case.deviations]
    assert recorded, "the P3b changes are recorded as deviations"
    for case in corpus.CASES:
        for mode, deviation in case.deviations:
            assert deviation.ruling and case.deviation_for(mode) is deviation
            assert mode in {analysis_mode for analysis_mode, _ in case.analyses}


def test_deviation_check_requires_the_recorded_oracle_outcome_and_a_live_match(reference_pkg, current_pkg) -> None:
    island = corpus.get_case("floating_resistor_island")
    deviation = island.deviation_for("op")
    assert deviation is not None
    check_deviation(reference_pkg, current_pkg, island.netlist_text, "op", {}, deviation)
    # A stale record: the oracle no longer fails as recorded.
    with pytest.raises(AssertionError, match="expects the reference to raise SolverConvergenceError but it succeeded"):
        check_deviation(reference_pkg, current_pkg, "V1 a 0 DC 1\nR1 a 0 1k\n.op\n", "op", {}, deviation)
    # The live package must succeed (an inconsistent voltage-source loop is singular at every gmin).
    with pytest.raises(AssertionError, match="expects the current package to succeed"):
        check_deviation(
            reference_pkg, current_pkg, "V1 a 0 DC 1\nV2 b 0 DC 2\nV3 a b DC 0.5\nR1 a 0 1k\n.op\n", "op", {}, deviation
        )
    with pytest.raises(AssertionError, match="metadata gmin"):
        check_deviation(
            reference_pkg, current_pkg, island.netlist_text, "op", {}, dataclasses.replace(deviation, gmin=1e-9)
        )
    # The live result must match the oracle on the equivalent netlist (here a 10x larger shunt at a).
    wrong = deviation.equivalent_netlist.replace("RGA a 0 1e12", "RGA a 0 1e11")
    with pytest.raises(AssertionError, match="vs equivalent netlist"):
        check_deviation(
            reference_pkg, current_pkg, island.netlist_text, "op", {}, dataclasses.replace(deviation, equivalent_netlist=wrong)
        )
    diode = corpus.get_case("diode_ideal_source")
    diode_deviation = diode.deviation_for("op")
    options = dict(diode.analyses[0][1])
    check_deviation(reference_pkg, current_pkg, diode.netlist_text, "op", options, diode_deviation, tolerance="nonlinear")
    off = dataclasses.replace(diode_deviation, expected_dc=(1.0, -2000.0))
    with pytest.raises(AssertionError, match="expected operating point"):
        check_deviation(reference_pkg, current_pkg, diode.netlist_text, "op", options, off, tolerance="nonlinear")
