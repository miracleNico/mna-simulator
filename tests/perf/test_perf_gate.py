"""Performance gate: the C++ core must beat the frozen Python reference by generous floors.

Marked ``perf``: skipped unless ``pytest --run-perf`` (``tests/conftest.py``).
Each floor is roughly a third of the speedup the migration is expected to
deliver (``CPP_MIGRATION_PLAN.md`` section 5), measured at one core thread
against the reference with BLAS pinned to one thread, with the ``--quick``
methodology of ``benchmarks/compare_backends.py`` (one warm-up, three
repetitions, median). Units whose native side has not landed yet
(``UnitUnavailable``) are skipped without timing the reference. Run it on an
idle machine; ratios, not absolute times, are compared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import compare_backends  # noqa: E402
import perf_units  # noqa: E402

# Minimum speedup (reference median / core median at 1 thread), keyed by unit id.
FLOORS: dict[str, float] = {
    # device evaluation >= 20x
    "device_eval_sram_mos_622": 20.0,
    "device_eval_bjt_3stage": 20.0,
    # 3 unknowns, ~16 us in the reference: a native call plus its output arrays is ~1 us,
    # so 20x is out of reach for this net.
    "device_eval_diode_mixer": 5.0,
    # assembly >= 10x
    "assembly_jacobian_sram_248": 10.0,
    # direct solve RLC >= 10x (KLU vs dense LAPACK, ~40x measured for the real solve)
    "direct_solve_rlc_1035": 10.0,
    "direct_solve_complex_rlc_1035": 10.0,
    # n = 248: dense LAPACK is only ~1.3 ms; KLU measured ~5-7x
    "direct_solve_sram_248": 1.5,
    "direct_solve_complex_sram_248": 1.5,
    # Krylov >= 1x (SciPy's Krylov kernels are already compiled)
    "krylov_gmres_rlc_1035": 1.0,
    "krylov_minres_rlc_1035": 1.0,
    "krylov_cg_rlc_1035": 1.0,
    # .op SRAM >= 5x
    "op_sram_248": 5.0,
    # transients: SRAM >= 5x, BJT 2001 steps >= 5x, RLC >= 10x
    "tran_sram_248": 5.0,
    "tran_bjt_3stage_2001": 5.0,
    "tran_rlc_1035_dense": 10.0,
    "tran_rlc_1035_krylov": 10.0,
    # AC RLC >= 5x
    "ac_rlc_1035": 5.0,
    # HB mixer >= 2x
    "hb_mixer_h205": 2.0,
    "hb_mixer_h64": 2.0,
    # MOR basis >= 2x, TPWL training >= 2x
    "mor_basis_rlc_1035": 2.0,
    "tpwl_train_sram_248": 2.0,
}

# Thread scaling: core time at 1 thread / core time at N threads, on machines with >= N CPUs.
SCALING_FLOORS: dict[str, tuple[int, float]] = {
    "ac_rlc_1035": (4, 1.5),
}

REPS = compare_backends.QUICK_REPS


def _measure(unit_id: str, threads: list[int]) -> dict[str, Any]:
    unit = perf_units.get_unit(unit_id)
    record = compare_backends.run_unit(unit, threads, REPS, reference_when_unavailable=False)
    if record["error"]:
        pytest.fail(f"{unit_id}: {record['error']}")
    if not record["available"]:
        pytest.skip(f"{unit_id}: {record['unavailable_reason']}")
    parity = record["parity"]
    assert parity["ok"] is not False, (
        f"{unit_id}: parity {parity['max_rel_error']:.3e} exceeds the tolerance {parity['tolerance']:.0e}"
    )
    return record


def test_floor_tables_name_registered_units() -> None:
    """Runs without --run-perf: the tables must not drift from the registry."""

    known = set(perf_units.UNITS)
    assert set(FLOORS) <= known, sorted(set(FLOORS) - known)
    assert set(SCALING_FLOORS) <= known, sorted(set(SCALING_FLOORS) - known)
    assert all(floor >= 1.0 for floor in FLOORS.values())


@pytest.mark.perf
@pytest.mark.parametrize("unit_id", list(FLOORS))
def test_speedup_floor(unit_id: str) -> None:
    floor = FLOORS[unit_id]
    record = _measure(unit_id, [1])
    reference = record["reference_s"]["median"]
    core = record["core_s"]["1"]["median"]
    speedup = reference / core if core > 0 else float("inf")
    assert speedup >= floor, (
        f"{unit_id}: speedup {speedup:.2f}x is below the floor {floor:g}x "
        f"(reference {reference * 1e3:.3f} ms, core {core * 1e3:.3f} ms at 1 thread; median of {REPS})"
    )


@pytest.mark.perf
@pytest.mark.parametrize("unit_id", list(SCALING_FLOORS))
def test_thread_scaling_floor(unit_id: str) -> None:
    threads, floor = SCALING_FLOORS[unit_id]
    cpus = os.cpu_count() or 1
    if cpus < threads:
        pytest.skip(f"needs >= {threads} CPUs, have {cpus}")
    record = _measure(unit_id, [1, threads])
    one = record["core_s"]["1"]["median"]
    many = record["core_s"][str(threads)]["median"]
    scaling = one / many if many > 0 else float("inf")
    assert scaling >= floor, (
        f"{unit_id}: {threads}-thread scaling {scaling:.2f}x is below the floor {floor:g}x "
        f"(core {one * 1e3:.3f} ms at 1 thread, {many * 1e3:.3f} ms at {threads} threads; median of {REPS})"
    )
