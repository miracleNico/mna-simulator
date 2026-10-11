"""Thread budget: grants, blocking, release, MNA_NUM_THREADS parsing, validation, BLAS pinning."""

from __future__ import annotations

import os
import threading
import time
import warnings

import pytest

from mna_simulation.native import threads
from mna_simulation.native.threads import (
    NUM_THREADS_ERROR,
    ThreadBudget,
    budget_total,
    coerce_num_threads,
    default_budget,
    pin_blas_threads,
    reset_default_budget,
)


def test_auto_takes_fair_share_of_free_threads() -> None:
    budget = ThreadBudget(8)
    with budget.lease("auto") as first:
        # Nobody active: "auto" gets everything that is free.
        assert first == 8
        assert budget.free == 0
        assert budget.active == 1
    assert budget.free == 8 and budget.active == 0

    with budget.lease(2) as explicit:
        assert explicit == 2
        # One active run: fair share is total // 2 = 4, bounded by the 6 free threads.
        with budget.lease(None) as second:
            assert second == 4
            # Two active runs: total // 3 = 2, and 2 are free.
            with budget.lease("auto") as third:
                assert third == 2
                assert budget.free == 0
                assert budget.active == 3
    assert budget.snapshot() == {"total": 8, "free": 8, "active": 0}


def test_auto_fair_share_is_bounded_by_free() -> None:
    budget = ThreadBudget(4)
    with budget.lease(3):
        # Fair share would be 4 // 2 = 2, but only 1 thread is free.
        with budget.lease("auto") as granted:
            assert granted == 1


def test_explicit_request_capped_by_free_and_total() -> None:
    budget = ThreadBudget(4)
    with budget.lease(16) as granted:
        assert granted == 4  # capped by total
    with budget.lease(3) as first:
        assert first == 3
        with budget.lease(3) as second:
            assert second == 1  # capped by free
    with budget.lease("2") as from_string:
        assert from_string == 2


def test_lease_blocks_at_capacity_until_release() -> None:
    budget = ThreadBudget(2)
    holder_entered = threading.Event()
    release_holder = threading.Event()
    waiter_granted: list[int] = []
    waiter_done = threading.Event()

    def holder() -> None:
        with budget.lease(2):
            holder_entered.set()
            release_holder.wait(5)

    def waiter() -> None:
        with budget.lease(1) as granted:
            waiter_granted.append(granted)
        waiter_done.set()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_entered.wait(5)
    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()

    time.sleep(0.1)
    assert not waiter_done.is_set(), "lease must block while no thread is free"
    assert waiter_granted == []

    release_holder.set()
    assert waiter_done.wait(5)
    holder_thread.join(5)
    waiter_thread.join(5)
    assert waiter_granted == [1]
    assert budget.snapshot() == {"total": 2, "free": 2, "active": 0}


def test_wait_timeout_raises_clear_error() -> None:
    budget = ThreadBudget(1, wait_timeout=0.05)
    with budget.lease(1):
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="Timed out .* waiting for a free simulation thread"):
            with budget.lease(1):
                pass
        assert time.perf_counter() - started < 2
    # A per-call timeout overrides the default.
    with budget.lease(1):
        with pytest.raises(RuntimeError):
            with budget.lease(1, timeout=0.01):
                pass
    assert budget.free == 1 and budget.active == 0


def test_release_on_exception() -> None:
    budget = ThreadBudget(3)
    with pytest.raises(ZeroDivisionError):
        with budget.lease(2):
            assert budget.free == 1
            1 / 0  # noqa: B018
    assert budget.snapshot() == {"total": 3, "free": 3, "active": 0}


def test_concurrent_leases_never_exceed_total() -> None:
    budget = ThreadBudget(3)
    lock = threading.Lock()
    in_use = 0
    peak = 0
    errors: list[BaseException] = []

    def worker(request: object) -> None:
        nonlocal in_use, peak
        try:
            for _ in range(20):
                with budget.lease(request) as granted:
                    with lock:
                        in_use += granted
                        peak = max(peak, in_use)
                    time.sleep(0.001)
                    with lock:
                        in_use -= granted
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=worker, args=(req,)) for req in ("auto", 1, 2, None, 5, "auto")]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(10)
    assert not errors
    assert 1 <= peak <= 3
    assert budget.snapshot() == {"total": 3, "free": 3, "active": 0}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, os.cpu_count() or 1),
        ("", os.cpu_count() or 1),
        ("1", 1),
        (" 2 ", min(2, os.cpu_count() or 1)),
        ("100000", os.cpu_count() or 1),
    ],
)
def test_budget_total_reads_env(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int) -> None:
    if raw is None:
        monkeypatch.delenv("MNA_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("MNA_NUM_THREADS", raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert budget_total() == expected


@pytest.mark.parametrize("raw", ["0", "-3", "four", "2.5", "auto"])
def test_budget_total_ignores_invalid_env_with_warning(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("MNA_NUM_THREADS", raw)
    with pytest.warns(RuntimeWarning, match="MNA_NUM_THREADS"):
        assert budget_total() == (os.cpu_count() or 1)


def test_budget_total_caps_at_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 2)
    monkeypatch.setenv("MNA_NUM_THREADS", "16")
    assert budget_total() == 2
    monkeypatch.setattr(threads.os, "cpu_count", lambda: None)
    monkeypatch.delenv("MNA_NUM_THREADS")
    assert budget_total() == 1


def test_default_budget_is_process_wide_and_resettable(monkeypatch: pytest.MonkeyPatch) -> None:
    original = default_budget()
    try:
        assert default_budget() is original
        monkeypatch.setenv("MNA_NUM_THREADS", "1")
        replaced = reset_default_budget()
        assert replaced.total == 1
        assert default_budget() is replaced
    finally:
        monkeypatch.delenv("MNA_NUM_THREADS", raising=False)
        reset_default_budget()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("auto", "auto"), (" AUTO ", "auto"), (1, 1), (4, 4), ("3", 3), (" 7 ", 7)],
)
def test_coerce_num_threads_accepts(value: object, expected: object) -> None:
    assert coerce_num_threads(value) == expected


@pytest.mark.parametrize("value", [0, -1, "0", "-2", "", "many", "2.5", 2.0, 1.5, True, False, [2], {"n": 2}])
def test_coerce_num_threads_rejects(value: object) -> None:
    with pytest.raises(ValueError, match="num_threads must be 'auto' or a positive integer"):
        coerce_num_threads(value)


def test_lease_validates_request() -> None:
    budget = ThreadBudget(2)
    with pytest.raises(ValueError, match=NUM_THREADS_ERROR):
        with budget.lease(0):
            pass
    assert budget.snapshot() == {"total": 2, "free": 2, "active": 0}
    with pytest.raises(ValueError):
        ThreadBudget(0)


def test_pin_blas_threads_limits_and_restores() -> None:
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy  # noqa: F401  (loads the BLAS library)

    def blas_threads() -> list[int]:
        return [info["num_threads"] for info in threadpoolctl.threadpool_info() if info["user_api"] == "blas"]

    before = blas_threads()
    with pin_blas_threads():
        assert all(count == 1 for count in blas_threads())
        with pin_blas_threads():  # nested/concurrent pins are reference-counted
            assert threads.blas_pin_depth() == 2
        assert all(count == 1 for count in blas_threads())
    assert threads.blas_pin_depth() == 0
    assert blas_threads() == before


def test_pin_blas_threads_without_threadpoolctl_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "threadpoolctl":
            raise ImportError("no threadpoolctl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(threads, "_missing_threadpoolctl_warned", False)
    monkeypatch.setattr(threads, "_pin_controller", None)
    with pytest.warns(RuntimeWarning, match="threadpoolctl is not installed"):
        with pin_blas_threads():
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pin_blas_threads():
            pass
    assert threads.blas_pin_depth() == 0
