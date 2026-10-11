"""Thread-count determinism: every corpus analysis gives bitwise-identical responses at 1 and 4 threads.

``CPP_MIGRATION_PLAN.md`` section 2.5: chunk boundaries of every parallel loop
depend only on the item count, outputs are reduced in a fixed order, and no
kernel uses atomics or fast-math, so the thread count may change speed but
never a single bit of a result. This module checks that end to end, through
``SimulationService.run_request`` (option coercion, gmin stepping, MOR and HB
facades, response conversion), for every ``(case, analysis)`` of
``tests/corpus.py``:

* the run at ``num_threads=1`` and the run at ``num_threads=4`` (granted by a
  private 4-thread ``ThreadBudget``, so the grant does not depend on the CPU
  count; the native core still clamps to the hardware concurrency) must
  produce the same response: ``status``, ``labels``, ``diagnostics``,
  ``dc_solution``, ``waveform``, ``spectrum``, ``matrices`` and every metadata
  field, with floats compared bit for bit (``struct.pack``), except the fields
  that report the thread grant, timings or memory (``EXCLUDED_KEYS``,
  ``EXCLUDED_SUFFIXES``);
* analyses that raise on both runs are skipped after checking that both runs
  raised the same exception class with the same code and message; an analysis
  that raises on only one of them fails.

``RANDOM`` initial conditions draw from NumPy's global RNG, which is reseeded
before every run, and the TPWL cache is cleared before every run so the second
run trains its own model instead of reusing the first one's.
"""

from __future__ import annotations

import copy
import struct
import warnings
from typing import Any

import numpy as np
import pytest

from corpus import Case, iter_analyses
from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest
from mna_simulation.api.service import SimulationService
from mna_simulation.mor import tpwl
from mna_simulation.native.threads import ThreadBudget

THREAD_COUNTS = (1, 4)

# Metadata that legitimately depends on the thread grant or on the clock: the grant itself
# (top level and inside solver statistics such as ``hb_solver``), wall times, and the
# per-thread workspace accounting of harmonic balance.
EXCLUDED_KEYS: frozenset[str] = frozenset({"threads_requested", "threads_used", "elapsed_ms", "memory_bytes"})
EXCLUDED_SUFFIXES: tuple[str, ...] = ("_ms", "_seconds")
RNG_SEED = 20261011


@pytest.fixture(scope="module")
def service() -> SimulationService:
    return SimulationService(budget=ThreadBudget(max(THREAD_COUNTS)))


def _params() -> list[Any]:
    params = []
    for case, index, mode, options in iter_analyses():
        marks = [pytest.mark.slow] if "slow" in case.tags else []
        params.append(pytest.param(case, mode, options, id=f"{case.name}-{index}-{mode}", marks=marks))
    return params


def _run(service: SimulationService, case: Case, mode: str, options: dict[str, Any], threads: int):
    tpwl._TPWL_CACHE.clear()  # noqa: SLF001 - each run must train its own TPWL model
    np.random.seed(RNG_SEED)
    request = AnalysisRequest(
        netlist_text=case.netlist_text,
        mode=AnalysisMode(mode),
        options={**copy.deepcopy(options), "num_threads": threads},
    )
    with warnings.catch_warnings():
        # exp() overflow inside damped Newton trials is expected on some cases.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return service.run_request(request), None
        except Exception as exc:  # noqa: BLE001 - outcomes are compared, not hidden
            return None, exc


def _excluded(key: object) -> bool:
    return isinstance(key, str) and (key in EXCLUDED_KEYS or key.endswith(EXCLUDED_SUFFIXES))


def _differences(a: Any, b: Any, path: str = "response") -> list[str]:
    """Paths where ``a`` and ``b`` differ; floats are compared by their IEEE-754 bit patterns."""

    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = {key for key in a if not _excluded(key)}
        keys_b = {key for key in b if not _excluded(key)}
        found = [f"{path}.{key} (present in one run only)" for key in sorted(keys_a ^ keys_b, key=str)]
        for key in sorted(keys_a & keys_b, key=str):
            found.extend(_differences(a[key], b[key], f"{path}.{key}"))
        return found
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            found = _differences(item_a, item_b, f"{path}[{index}]")
            if found:
                return found  # the first differing element is enough
        return []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _differences(np.asarray(a).tolist(), np.asarray(b).tolist(), path)
    if isinstance(a, float) and isinstance(b, float):
        if struct.pack("<d", a) != struct.pack("<d", b):
            return [f"{path} ({a!r} vs {b!r})"]
        return []
    if isinstance(a, complex) and isinstance(b, complex):
        return _differences([a.real, a.imag], [b.real, b.imag], path)
    if type(a) is not type(b) or a != b:
        return [f"{path} ({a!r} vs {b!r})"]
    return []


def test_difference_finder_is_bitwise() -> None:
    """The comparator itself: -0.0 vs 0.0 and NaN payloads count, excluded keys do not."""

    assert _differences({"x": [0.0, 1.0]}, {"x": [0.0, 1.0]}) == []
    assert _differences({"x": [0.0]}, {"x": [-0.0]}) != []
    assert _differences({"x": float("nan")}, {"x": float("nan")}) == []
    assert _differences({"x": 1.0}, {"x": 1.0 + 2**-52}) != []
    assert _differences({"x": [1, 2]}, {"x": [1, 2, 3]}) != []
    assert _differences({"elapsed_ms": 1.0, "hb": {"threads_used": 1}}, {"elapsed_ms": 2.0, "hb": {"threads_used": 4}}) == []
    assert _differences({"a": 1}, {"a": 1, "b": 2}) != []


@pytest.mark.parametrize(("case", "mode", "options"), _params())
def test_responses_are_bitwise_identical_for_1_and_4_threads(
    service: SimulationService, case: Case, mode: str, options: dict[str, Any]
) -> None:
    (one, one_error), (many, many_error) = (_run(service, case, mode, options, threads) for threads in THREAD_COUNTS)

    if one_error is not None or many_error is not None:
        assert one_error is not None and many_error is not None, (
            f"{case.name} {mode}: raised only at "
            f"{THREAD_COUNTS[0] if one_error is not None else THREAD_COUNTS[1]} thread(s): {one_error or many_error!r}"
        )
        assert type(one_error) is type(many_error), (one_error, many_error)
        assert getattr(one_error, "code", None) == getattr(many_error, "code", None)
        assert str(one_error) == str(many_error)
        pytest.skip(f"errors on both thread counts: {type(one_error).__name__}: {one_error}")

    assert one.metadata["threads_used"] == THREAD_COUNTS[0]
    assert many.metadata["threads_used"] == THREAD_COUNTS[1]
    found = _differences(one.model_dump(mode="python"), many.model_dump(mode="python"))
    assert not found, f"{case.name} {mode} {options}: results differ between 1 and 4 threads:\n" + "\n".join(found[:20])
