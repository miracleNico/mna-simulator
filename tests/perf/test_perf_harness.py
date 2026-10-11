"""Always-on checks of the Python-vs-C++ comparison harness.

Covers ``benchmarks/perf_units.py`` (the unit registry) and
``benchmarks/compare_backends.py`` (timing, JSON, Markdown, CLI). Timings are
not asserted here; the perf gate (``test_perf_gate.py``, ``--run-perf``) does
that. Every reference callable runs once on the tiny ``smoke`` inputs, and the
units whose native side exists are checked for parity.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import compare_backends  # noqa: E402
import perf_units  # noqa: E402
from perf_units import CATEGORIES, UnitUnavailable  # noqa: E402

# Every unit of CPP_MIGRATION_PLAN.md section 5 (and the sizes of PERFORMANCE_PLAN.md 4.1).
PLAN_UNITS = {
    # device evaluation: 622 MOS, BJT stage, diode net
    "device_eval_sram_mos_622",
    "device_eval_bjt_3stage",
    "device_eval_diode_mixer",
    # circuit build and Jacobian assembly
    "circuit_build_sram_248",
    "circuit_build_rlc_1035",
    "assembly_jacobian_sram_248",
    # factor and solve, n = 248 and 1035, real and complex
    "direct_solve_rlc_1035",
    "direct_solve_sram_248",
    "direct_solve_complex_rlc_1035",
    "direct_solve_complex_sram_248",
    # each Krylov method on the RLC operator
    "krylov_gmres_rlc_1035",
    "krylov_minres_rlc_1035",
    "krylov_cg_rlc_1035",
    # Newton .op
    "op_sram_248",
    # transient: SRAM 0.25n, BJT 2001 steps, RLC 40 steps dense and Krylov
    "tran_sram_248",
    "tran_bjt_3stage_2001",
    "tran_rlc_1035_dense",
    "tran_rlc_1035_krylov",
    # AC: RLC 100 points, BJT 400 points
    "ac_rlc_1035",
    "ac_bjt_3stage",
    # HB mixer H=205 and H=64
    "hb_mixer_h205",
    "hb_mixer_h64",
    # MOR basis and TPWL training
    "mor_basis_rlc_1035",
    "tpwl_train_sram_248",
}

# Units whose native side exists today (P0: _core.klu_solve).
AVAILABLE_NOW = ("direct_solve_rlc_1035", "direct_solve_sram_248")


def test_registry_lists_every_unit_from_the_plan() -> None:
    ids = [unit.id for unit in perf_units.all_units()]
    assert len(ids) == len(set(ids))
    assert set(ids) == PLAN_UNITS


def test_units_are_well_formed() -> None:
    for unit in perf_units.all_units():
        assert unit.category in CATEGORIES, unit.id
        assert unit.size and all(isinstance(v, int) and v > 0 for v in unit.size.values()), unit.id
        assert re.fullmatch(r"P[0-9]", unit.phase), unit.id
        assert unit.tolerance > 0, unit.id
        assert unit.description, unit.id
        for name in ("setup", "reference", "core", "core_setup", "parity"):
            assert callable(getattr(unit, name)), (unit.id, name)
    covered = {unit.category for unit in perf_units.all_units()}
    assert covered == set(CATEGORIES)


def test_select_units_by_id_and_category() -> None:
    picked = perf_units.select_units(["direct_solve_sram_248", "op_sram_248"])
    assert [unit.id for unit in picked] == ["direct_solve_sram_248", "op_sram_248"]
    transient = perf_units.select_units(categories=["transient"])
    assert transient and all(unit.category == "transient" for unit in transient)
    both = perf_units.select_units(["direct_solve_sram_248", "op_sram_248"], ["newton"])
    assert [unit.id for unit in both] == ["op_sram_248"]
    with pytest.raises(KeyError):
        perf_units.select_units(["no_such_unit"])
    with pytest.raises(KeyError):
        perf_units.select_units(categories=["no_such_category"])


@pytest.mark.parametrize("unit_id", sorted(PLAN_UNITS))
def test_reference_runs_on_smoke_inputs(unit_id: str) -> None:
    """Every frozen reference callable runs (1 rep on tiny inputs); available cores match it."""

    unit = perf_units.get_unit(unit_id)
    record = compare_backends.run_unit(unit, threads=[1], reps=1, smoke=True)
    assert record["error"] is None, record["error"]
    assert record["reference_s"] is not None
    assert len(record["reference_s"]["samples"]) == 1
    if record["available"]:
        assert record["parity"]["ok"] is True, record["parity"]
        assert set(record["core_s"]) == {"1"}
    else:
        assert compare_backends.unavailable_tag(record["unavailable_reason"]) == unit.phase
        assert record["core_s"] == {} and record["speedup"] == {}


@pytest.mark.parametrize("unit_id", AVAILABLE_NOW)
def test_available_core_units_match_reference_at_full_size(unit_id: str) -> None:
    unit = perf_units.get_unit(unit_id)
    inputs = unit.setup()
    assert inputs["A"].shape == (unit.size["n"], unit.size["n"])
    assert np.count_nonzero(inputs["A"]) == unit.size["nnz"]
    core_inputs = unit.core_setup(inputs)
    assert core_inputs["Ap"].dtype == np.int32 and core_inputs["Ai"].dtype == np.int32
    reference = unit.reference(inputs)
    for threads in (1, 2):
        error = unit.parity(reference, unit.core(core_inputs, threads))
        assert error is not None and error <= 1e-9, (unit_id, threads, error)


def test_unavailable_core_raises_with_phase() -> None:
    unit = perf_units.get_unit("tpwl_train_sram_248")
    with pytest.raises(UnitUnavailable) as info:
        unit.core(None, 1)
    assert info.value.phase == "P5"
    assert compare_backends.unavailable_tag(str(info.value)) == "P5"


def test_parity_metrics() -> None:
    a = np.array([1.0, -2.0, 4.0])
    assert perf_units.max_rel_error(a, a) == 0.0
    assert perf_units.max_rel_error(a, a + np.array([0.0, 0.0, 4e-6])) == pytest.approx(1e-6)
    assert perf_units.max_rel_error(a, a[:2]) == float("inf")
    assert perf_units.max_rel_error(np.array([np.nan, 1.0]), np.array([np.nan, 1.0])) == 0.0
    assert perf_units.max_rel_error(np.array([np.nan, 1.0]), np.array([0.0, 1.0])) == float("inf")
    assert perf_units.max_rel_error(np.zeros(0), np.zeros(0)) == 0.0
    basis = np.linalg.qr(np.random.default_rng(0).standard_normal((20, 4)))[0]
    rotation = np.linalg.qr(np.random.default_rng(1).standard_normal((4, 4)))[0]
    assert perf_units.subspace_distance(basis, basis @ rotation) < 1e-12
    other = np.linalg.qr(np.random.default_rng(2).standard_normal((20, 4)))[0]
    assert perf_units.subspace_distance(basis, other) > 0.1


def test_measure_and_summarize() -> None:
    calls = []
    stats, output = compare_backends.measure(lambda: calls.append(1) or len(calls), reps=3)
    assert len(calls) == 4  # one warm-up + three timed runs
    assert output == 1  # the warm-up output
    assert len(stats["samples"]) == 3 and stats["median"] >= 0 and stats["mad"] >= 0
    summary = compare_backends.summarize([1.0, 2.0, 10.0])
    assert summary["median"] == 2.0 and summary["mad"] == 1.0


def test_thread_parsing() -> None:
    cpus = os.cpu_count() or 1
    assert compare_backends.parse_threads("4,1,2,2") == [1, 2, 4]
    assert compare_backends.parse_threads("1,N") == sorted({1, cpus})
    with pytest.raises(ValueError):
        compare_backends.parse_threads("0")
    with pytest.raises(ValueError):
        compare_backends.parse_threads("two")
    defaults = compare_backends.default_threads()
    assert defaults[:3] == [1, 2, 4]
    assert defaults == ([1, 2, 4, cpus] if cpus > 4 else [1, 2, 4])


def test_default_json_path_is_timestamped_under_results() -> None:
    path = compare_backends.default_json_path(hostname="my host/1")
    assert path.parent == BENCH_DIR / "results"
    assert re.fullmatch(r"\d{8}T\d{6}Z_my-host-1\.json", path.name)


def test_json_and_markdown_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    json_path = tmp_path / "perf.json"
    md_path = tmp_path / "perf.md"
    # One available unit and one whose native side has not landed yet (dynamic, so the test
    # keeps working as phases land; skipped once every unit is available).
    pending_units = [unit for unit in perf_units.UNITS.values() if _is_unavailable(unit)]
    if not pending_units:
        pytest.skip("every unit has a native implementation")
    pending_unit = pending_units[0]
    status = compare_backends.main(
        [
            "--units",
            f"direct_solve_sram_248,{pending_unit.id}",
            "--smoke",
            "--reps",
            "1",
            "--threads",
            "1,2",
            "--json",
            str(json_path),
            "--markdown",
            str(md_path),
        ]
    )
    assert status == 0
    stdout = capsys.readouterr().out

    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert set(document) >= {"timestamp", "host", "methodology", "units"}
    host = document["host"]
    for key in ("platform", "cpu_count", "python", "numpy", "scipy", "core_version", "core_build_info", "git_commit", "loadavg"):
        assert key in host, key
    method = document["methodology"]
    assert method["warmup"] == 1 and method["reps"] == 1 and method["statistic"] == "median"
    assert method["threads"] == [1, 2] and method["smoke"] is True

    records = {record["id"]: record for record in document["units"]}
    assert list(records) == ["direct_solve_sram_248", pending_unit.id]
    for record in records.values():
        for key in ("id", "category", "size", "available", "unavailable_reason", "reference_s", "core_s", "speedup", "parity"):
            assert key in record, key
        assert set(record["reference_s"]) == {"median", "mad", "samples"}
        assert set(record["parity"]) >= {"max_rel_error", "ok"}

    solve = records["direct_solve_sram_248"]
    assert solve["available"] is True and solve["unavailable_reason"] is None
    assert set(solve["core_s"]) == {"1", "2"} and set(solve["speedup"]) == {"1", "2"}
    assert set(solve["core_s"]["1"]) == {"median", "mad", "samples"}
    assert solve["speedup"]["1"] == pytest.approx(solve["reference_s"]["median"] / solve["core_s"]["1"]["median"])
    assert solve["parity"]["ok"] is True and solve["parity"]["max_rel_error"] <= 1e-9

    pending = records[pending_unit.id]
    assert pending["available"] is False and pending_unit.phase in pending["unavailable_reason"]
    assert pending["core_s"] == {} and pending["speedup"] == {}
    assert pending["parity"]["max_rel_error"] is None and pending["parity"]["ok"] is None

    markdown = md_path.read_text(encoding="utf-8")
    header = "| unit | size | reference ms | core ms (1t) | core ms (2t) | speedup 1t | best speedup | parity |"
    assert header in markdown and header in stdout
    rows = {line.split("|")[1].strip(): line for line in markdown.splitlines() if line.startswith("| `")}
    assert f"n/a ({pending_unit.phase})" in rows[f"`{pending_unit.id}`"]
    assert re.search(r"\| \d+(\.\d+)?x \|", rows["`direct_solve_sram_248`"])
    assert " ok |" in rows["`direct_solve_sram_248`"]
    assert "Native core:" in markdown and "Commit:" in markdown


def test_render_table_marks_unavailable_and_failed_units() -> None:
    document = {
        "methodology": {"threads": [1, 4]},
        "units": [
            {
                "id": "a",
                "size": {"n": 3},
                "available": False,
                "unavailable_reason": "lands in P3: driver",
                "reference_s": {"median": 0.5},
            },
            {
                "id": "b",
                "size": {"n": 3},
                "available": False,
                "unavailable_reason": None,
                "reference_s": None,
                "error": "reference: boom",
            },
            {
                "id": "c",
                "size": {"n": 3},
                "available": True,
                "reference_s": {"median": 0.01},
                "core_s": {"1": {"median": 0.001}, "4": {"median": 0.0005}},
                "speedup": {"1": 10.0, "4": 20.0},
                "parity": {"max_rel_error": 3e-3, "ok": False, "tolerance": 1e-9},
            },
        ],
    }
    lines = compare_backends.render_table(document).splitlines()
    assert lines[0].startswith("| unit | size | reference ms | core ms (1t) | core ms (4t) |")
    assert lines[2].count("n/a (P3)") == 4 and "500.0" in lines[2]
    assert "error" in lines[3]
    assert "10.0x" in lines[4] and "20.0x" in lines[4] and "FAIL" in lines[4]
    problems = compare_backends.failures({"units": [dict(r, error=r.get("error")) for r in document["units"]]}, strict=True)
    assert len(problems) == 3


def _is_unavailable(unit) -> bool:
    try:
        inputs = unit.setup(smoke=True)
        core_inputs = unit.core_setup(inputs) if unit.core_setup is not None else inputs
        unit.core(core_inputs, 1)
    except perf_units.UnitUnavailable:
        return True
    except Exception:  # noqa: BLE001 - any other failure means the unit exists but broke
        return False
    return False


def test_strict_fails_when_a_unit_is_unavailable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Pick a unit whose native side has not landed yet (skips once every unit is available).
    unavailable = [unit.id for unit in perf_units.UNITS.values() if _is_unavailable(unit)]
    if not unavailable:
        pytest.skip("every unit has a native implementation")
    args = ["--units", unavailable[0], "--smoke", "--reps", "1", "--threads", "1", "--json", str(tmp_path / "s.json")]
    assert compare_backends.main(args) == 0
    assert compare_backends.main([*args, "--strict"]) == 1
    capsys.readouterr()


def test_list_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(BENCH_DIR / "compare_backends.py"), "--list"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for unit_id in PLAN_UNITS:
        assert unit_id in completed.stdout
    assert "linear_solve" in completed.stdout and "P0" in completed.stdout
