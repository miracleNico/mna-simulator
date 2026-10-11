"""Thread budget, ``num_threads`` validation, and BLAS pinning.

Every simulation run takes a lease of worker threads from a process-wide
:class:`ThreadBudget` before it starts and returns it when it ends, so
concurrent requests divide the machine instead of oversubscribing it:

* ``budget_total()`` is the size of the budget: ``MNA_NUM_THREADS`` when set to
  a positive integer, otherwise ``os.cpu_count()``; never more than the CPU
  count and never less than 1.
* ``ThreadBudget.lease(requested)`` grants ``min(requested, free, total)``
  threads (at least 1) for an explicit integer, or a fair share
  ``max(1, total // (active + 1))`` bounded by what is free for ``"auto"`` /
  ``None``. It blocks while no thread is free.
* ``pin_blas_threads()`` limits NumPy/SciPy BLAS to one thread for the duration
  of a run (via ``threadpoolctl``). Multithreaded OpenBLAS busy-waits when its
  threads are descheduled by a competing run, which made four concurrent
  simulations 21x slower (``PERFORMANCE_PLAN.md`` section 4.2). Parallelism
  belongs to the native core, which reads the leased thread count.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

ENV_VAR = "MNA_NUM_THREADS"
NUM_THREADS_ERROR = "num_threads must be 'auto' or a positive integer"

ThreadRequest = int | str | None

_UNSET: Any = object()


def cpu_count() -> int:
    """Number of CPUs reported by the OS (at least 1)."""

    return max(1, os.cpu_count() or 1)


def budget_total(environ: Mapping[str, str] | None = None) -> int:
    """Size of the machine-wide thread budget.

    ``MNA_NUM_THREADS`` (a positive integer) caps the budget; values above
    ``os.cpu_count()`` are capped at the CPU count. An unset or empty variable
    means ``os.cpu_count()``. Anything else is ignored with a ``RuntimeWarning``.
    """

    cpus = cpu_count()
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR)
    if raw is None or not str(raw).strip():
        return cpus
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = 0
    if value < 1:
        warnings.warn(
            f"Ignoring {ENV_VAR}={raw!r}: expected a positive integer; using os.cpu_count() = {cpus}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cpus
    return min(value, cpus)


def coerce_num_threads(value: object) -> ThreadRequest:
    """Validate a ``num_threads`` request.

    Returns ``None`` when absent, ``"auto"``, or a positive ``int``. Strings of
    integers (``"4"``) are accepted; anything else raises ``ValueError``.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(NUM_THREADS_ERROR)
    if isinstance(value, int):
        if value >= 1:
            return value
        raise ValueError(NUM_THREADS_ERROR)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "auto":
            return "auto"
        try:
            parsed = int(cleaned)
        except ValueError:
            raise ValueError(NUM_THREADS_ERROR) from None
        if parsed >= 1:
            return parsed
    raise ValueError(NUM_THREADS_ERROR)


class ThreadBudget:
    """Counting semaphore of worker threads shared by concurrent runs.

    ``total`` threads exist; each active lease holds at least one. ``lease()``
    is a context manager that blocks until a thread is free, yields the number
    of threads granted, and returns them on exit (also on exceptions).
    ``wait_timeout`` (seconds, default ``None`` = wait forever) bounds the wait;
    on timeout a ``RuntimeError`` is raised.
    """

    def __init__(self, total: int | None = None, *, wait_timeout: float | None = None) -> None:
        resolved = budget_total() if total is None else int(total)
        if resolved < 1:
            raise ValueError("ThreadBudget total must be a positive integer.")
        self._total = resolved
        self._free = resolved
        self._active = 0
        self._cond = threading.Condition()
        self.wait_timeout = wait_timeout

    @property
    def total(self) -> int:
        return self._total

    @property
    def free(self) -> int:
        with self._cond:
            return self._free

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    def snapshot(self) -> dict[str, int]:
        """Consistent ``{"total", "free", "active"}`` view for health reporting."""

        with self._cond:
            return {"total": self._total, "free": self._free, "active": self._active}

    def acquire(self, requested: object = None, *, cap: int | None = None, timeout: float | None = _UNSET) -> int:
        """Block until a thread is free, then take a grant; prefer :meth:`lease`.

        ``cap`` bounds an ``"auto"`` grant (see :func:`useful_threads`); an explicit
        integer request is the caller's own upper limit and ignores it.
        """

        normalized = coerce_num_threads(requested)
        wait = self.wait_timeout if timeout is _UNSET else timeout
        with self._cond:
            if not self._cond.wait_for(lambda: self._free >= 1, timeout=wait):
                raise RuntimeError(
                    f"Timed out after {wait:g} s waiting for a free simulation thread "
                    f"(budget total={self._total}, active runs={self._active}). "
                    f"Retry later or raise {ENV_VAR}."
                )
            if isinstance(normalized, int):
                granted = min(normalized, self._free, self._total)
            else:
                granted = min(self._free, max(1, self._total // (self._active + 1)))
                if cap is not None:
                    granted = min(granted, max(1, int(cap)))
            granted = max(1, granted)
            self._free -= granted
            self._active += 1
            return granted

    def release(self, granted: int) -> None:
        """Return a grant obtained from :meth:`acquire`."""

        with self._cond:
            self._free = min(self._total, self._free + int(granted))
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    @contextmanager
    def lease(
        self, requested: object = None, *, cap: int | None = None, timeout: float | None = _UNSET
    ) -> Iterator[int]:
        """Hold ``min(requested, free, total)`` threads (or a fair share for ``"auto"``, at most ``cap``)."""

        granted = self.acquire(requested, cap=cap, timeout=timeout)
        try:
            yield granted
        finally:
            self.release(granted)

    def __repr__(self) -> str:
        state = self.snapshot()
        return f"ThreadBudget(total={state['total']}, free={state['free']}, active={state['active']})"


# ``"auto"`` leases only what an analysis can use (the core's own thresholds):
# device evaluation and the Newton/transient drivers run on one thread below this
# many devices (``kParallelDeviceMinCount`` in cpp/include/mna/analyses/newton.hpp),
# the AC sweep spreads frequencies over at most ``points // 2`` threads, and
# harmonic balance of a small circuit is dominated by one serial dense LU.
PARALLEL_DEVICE_MIN_COUNT = 2048
HB_PARALLEL_MIN_NODES = 64


def useful_threads(mode: object, *, devices: int, nodes: int, ac_points: int | None = None) -> int | None:
    """Upper bound on the threads an ``"auto"`` request of ``mode`` can put to use.

    ``None`` means "no bound" (the fair share of the budget). ``mode`` is an
    ``AnalysisMode`` or its string value (``"op"``, ``"tran"``, ``"ac"``, ``"hb"``).
    Concurrent small requests then run side by side instead of waiting for one
    another's lease; an explicit ``num_threads`` is never capped.
    """

    kind = str(getattr(mode, "value", mode)).lower()
    if kind in ("op", "tran"):
        return 1 if devices < PARALLEL_DEVICE_MIN_COUNT else None
    if kind == "ac":
        points = 100 if ac_points is None else int(ac_points)
        return max(1, points // 2)
    if kind == "hb":
        return 1 if nodes < HB_PARALLEL_MIN_NODES else None
    return None


_default_budget: ThreadBudget | None = None
_default_budget_lock = threading.Lock()


def default_budget() -> ThreadBudget:
    """The process-wide budget used by the service (created on first use)."""

    global _default_budget
    with _default_budget_lock:
        if _default_budget is None:
            _default_budget = ThreadBudget(budget_total())
        return _default_budget


def reset_default_budget(total: int | None = None) -> ThreadBudget:
    """Replace the process-wide budget (re-reading ``MNA_NUM_THREADS`` when ``total`` is None).

    Intended for tests and start-up configuration; leases held on the old
    budget are released to the old budget.
    """

    global _default_budget
    with _default_budget_lock:
        _default_budget = ThreadBudget(total)
        return _default_budget


# --- BLAS pinning -----------------------------------------------------------------------------

_pin_lock = threading.Lock()
_pin_depth = 0
_pin_limiter: Any = None
_pin_controller: Any = None
_missing_threadpoolctl_warned = False


def _warn_missing_threadpoolctl() -> None:
    global _missing_threadpoolctl_warned
    if _missing_threadpoolctl_warned:
        return
    _missing_threadpoolctl_warned = True
    warnings.warn(
        "threadpoolctl is not installed, so BLAS threads are not pinned during simulations; "
        "concurrent runs may oversubscribe the CPU. Install threadpoolctl>=3 or set OPENBLAS_NUM_THREADS=1.",
        RuntimeWarning,
        stacklevel=3,
    )


def _limit_blas() -> Any:
    """Apply a one-thread BLAS limit and return the limiter (None if unavailable)."""

    global _pin_controller
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        _warn_missing_threadpoolctl()
        return None
    # Same effect as ``threadpool_limits(limits=1)``, which rediscovers every loaded library
    # on each call (~3 ms); the cached controller applies the limit in ~0.05 ms. It is built
    # on the first run (after NumPy/SciPy are imported) and rebuilt while it has found none.
    if _pin_controller is None or not _pin_controller.lib_controllers:
        _pin_controller = ThreadpoolController()
    return _pin_controller.limit(limits=1)


@contextmanager
def pin_blas_threads() -> Iterator[None]:
    """Limit NumPy/SciPy BLAS to one thread while the block runs.

    BLAS thread limits are process-wide, so nested and concurrent pins are
    reference-counted: the first entrant applies the limit and the last one to
    leave restores the original setting. Without ``threadpoolctl`` this is a
    no-op that warns once.
    """

    global _pin_depth, _pin_limiter
    with _pin_lock:
        if _pin_depth == 0:
            _pin_limiter = _limit_blas()
        _pin_depth += 1
    try:
        yield
    finally:
        with _pin_lock:
            _pin_depth -= 1
            if _pin_depth == 0 and _pin_limiter is not None:
                limiter, _pin_limiter = _pin_limiter, None
                limiter.restore_original_limits()


def blas_pin_depth() -> int:
    """Number of runs currently inside :func:`pin_blas_threads` (diagnostics)."""

    with _pin_lock:
        return _pin_depth
