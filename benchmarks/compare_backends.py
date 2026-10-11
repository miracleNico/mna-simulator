"""Compare the frozen Python reference with the C++ core, unit by unit.

Usage::

    python benchmarks/compare_backends.py --list
    python benchmarks/compare_backends.py                       # every unit, 7 reps
    python benchmarks/compare_backends.py --quick               # 3 reps
    python benchmarks/compare_backends.py --units direct_solve_rlc_1035,direct_solve_sram_248
    python benchmarks/compare_backends.py --category transient --threads 1,4
    python benchmarks/compare_backends.py --json out.json --markdown out.md

The units live in ``benchmarks/perf_units.py`` (``CPP_MIGRATION_PLAN.md``
section 5). For each unit the reference (``mna_reference``) is timed once with
BLAS pinned to one thread, and the native core at each ``--threads`` count.
Method: ``gc.disable()`` during the timed repetitions (with a ``gc.collect()``
between them, untimed), one warm-up run, then ``--reps`` repetitions timed with
``time.perf_counter_ns``; median and median absolute deviation (MAD). Parity
(``perf_units.max_rel_error`` or the unit's own metric) is computed from the
warm-up outputs at every thread count and the worst value is reported.

The Markdown table always goes to stdout; ``--markdown`` also writes it with a
host header. The JSON document (``schema_version`` 1) goes to ``--json``,
default ``benchmarks/results/<UTC timestamp>_<hostname>.json``. Units whose
native side has not landed yet report ``n/a (P<phase>)``; with ``--strict``
they make the run fail. Run on an idle machine: a warning is printed when the
1-minute load average is above 0.5.

Exit status: 0 on success; 1 when a unit raised, a parity check failed, or
(with ``--strict``) a unit was unavailable; 2 for usage errors.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gc
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import perf_units  # noqa: E402
from perf_units import CATEGORIES, Unit, UnitUnavailable  # noqa: E402

SCHEMA_VERSION = 1
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_REPS = 7
QUICK_REPS = 3
WARMUP = 1
LOAD_WARNING_THRESHOLD = 0.5

Log = Callable[[str], None]


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# Threads and BLAS pinning
# ---------------------------------------------------------------------------------------------


def default_threads() -> list[int]:
    """``1, 2, 4`` plus ``os.cpu_count()`` when it is larger."""

    threads = [1, 2, 4]
    cpus = os.cpu_count() or 1
    if cpus > 4:
        threads.append(cpus)
    return threads


def parse_threads(text: str) -> list[int]:
    """``"1,2,4,N"`` -> ``[1, 2, 4, cpu_count]`` (sorted, unique; ``N`` = ``os.cpu_count()``)."""

    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.upper() == "N":
            values.add(os.cpu_count() or 1)
            continue
        try:
            value = int(token)
        except ValueError:
            raise ValueError(f"invalid thread count {token!r} (expected positive integers or N)") from None
        if value < 1:
            raise ValueError(f"invalid thread count {token!r} (expected positive integers or N)")
        values.add(value)
    if not values:
        raise ValueError("no thread counts given")
    return sorted(values)


def blas_single_thread() -> contextlib.AbstractContextManager[Any]:
    """``threadpoolctl.threadpool_limits(1)``, or a no-op (with a warning) without threadpoolctl."""

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a runtime dependency
        _stderr("warning: threadpoolctl is not installed; BLAS is not pinned to one thread")
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


# ---------------------------------------------------------------------------------------------
# Host information
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _load_average() -> list[float] | None:
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return [round(value, 2) for value in getloadavg()]
    except OSError:
        return None


def host_info() -> dict[str, Any]:
    """Machine, interpreter, library, native-core and repository facts for the report header."""

    import numpy
    import scipy

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import threadpoolctl

        info["threadpoolctl"] = threadpoolctl.__version__
        info["blas"] = [
            {key: pool.get(key) for key in ("internal_api", "version", "num_threads", "threading_layer")}
            for pool in threadpoolctl.threadpool_info()
            if pool.get("user_api") == "blas"
        ]
    except ImportError:  # pragma: no cover
        info["threadpoolctl"] = None
        info["blas"] = []
    try:
        from mna_simulation.native import build_info, core_version

        info["core_version"] = core_version()
        info["core_build_info"] = build_info()
    except Exception as exc:  # noqa: BLE001 - recorded, the reference side still runs
        info["core_version"] = None
        info["core_build_info"] = None
        info["core_error"] = f"{type(exc).__name__}: {exc}"
    info["git_commit"] = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    info["git_dirty"] = None if status is None else bool(status)
    load = _load_average()
    info["loadavg"] = load
    info["load_warning"] = None
    if load is not None and load[0] > LOAD_WARNING_THRESHOLD:
        info["load_warning"] = (
            f"1-minute load average is {load[0]:.2f} (> {LOAD_WARNING_THRESHOLD}); timings are not "
            "representative. Rerun on an idle machine before comparing or committing a baseline."
        )
    return info


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """``{"median", "mad", "samples"}`` in seconds; MAD = median of ``|x - median|``."""

    median = statistics.median(samples)
    mad = statistics.median(abs(sample - median) for sample in samples)
    return {"median": median, "mad": mad, "samples": list(samples)}


def measure(fn: Callable[[], Any], reps: int, warmup: int = WARMUP) -> tuple[dict[str, Any], Any]:
    """Time ``fn``: ``warmup`` untimed runs, then ``reps`` timed runs with the GC disabled.

    Returns the summary and the output of the last warm-up run (the output of
    the first timed run when ``warmup`` is 0). Each output is released outside
    the timed region; a full collection runs (untimed) before every repetition.
    """

    if reps < 1:
        raise ValueError("reps must be >= 1")
    output: Any = None
    have_output = False
    for _ in range(warmup):
        output = fn()
        have_output = True
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            gc.collect()
            start = time.perf_counter_ns()
            result = fn()
            stop = time.perf_counter_ns()
            samples.append((stop - start) * 1e-9)
            if not have_output:
                output, have_output = result, True
            del result
    finally:
        if gc_was_enabled:
            gc.enable()
    return summarize(samples), output


def _short_error(stage: str, exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    where = f" ({Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
    return f"{stage}: {type(exc).__name__}: {exc}{where}"


def unavailable_tag(reason: str | None, unit: Unit | None = None) -> str:
    """``"P1"`` from ``"lands in P1: ..."`` (falls back to the unit's phase)."""

    if reason:
        match = re.search(r"\bP\d+\b", reason)
        if match:
            return match.group(0)
    return unit.phase if unit is not None else "unavailable"


def run_unit(
    unit: Unit,
    threads: Sequence[int],
    reps: int,
    *,
    smoke: bool = False,
    warmup: int = WARMUP,
    inputs: Any = None,
    reference_when_unavailable: bool = True,
    log: Log | None = None,
) -> dict[str, Any]:
    """Measure one unit and return its JSON record.

    The native side is probed first (``core_setup`` plus one call at
    ``threads[0]``); with ``reference_when_unavailable=False`` an unavailable
    unit returns without timing the reference (used by the perf gate).
    Exceptions other than ``UnitUnavailable`` are recorded in ``error``.
    """

    record: dict[str, Any] = {
        "id": unit.id,
        "category": unit.category,
        "size": dict(unit.size),
        "description": unit.description,
        "phase": unit.phase,
        "available": False,
        "unavailable_reason": None,
        "reference_s": None,
        "core_s": {},
        "speedup": {},
        "parity": {"max_rel_error": None, "ok": None, "tolerance": unit.tolerance},
        "error": None,
    }
    say = log or (lambda _message: None)
    threads = list(threads)

    try:
        if inputs is None:
            inputs = unit.setup(smoke=smoke)
    except Exception as exc:  # noqa: BLE001
        record["error"] = _short_error("setup", exc)
        return record

    core_inputs: Any = None
    try:
        core_inputs = unit.core_setup(inputs)
        with blas_single_thread():
            unit.core(core_inputs, threads[0])
        record["available"] = True
    except UnitUnavailable as exc:
        record["unavailable_reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        record["error"] = _short_error("core", exc)
        return record

    if not record["available"] and not reference_when_unavailable:
        return record

    try:
        with blas_single_thread():
            record["reference_s"], ref_out = measure(lambda: unit.reference(inputs), reps, warmup)
    except Exception as exc:  # noqa: BLE001
        record["error"] = _short_error("reference", exc)
        return record
    say(f"  reference {record['reference_s']['median'] * 1e3:10.3f} ms")

    if not record["available"]:
        say(f"  core      n/a ({unavailable_tag(record['unavailable_reason'], unit)})")
        return record

    worst: float | None = None
    for count in threads:
        try:
            with blas_single_thread():
                stats, core_out = measure(lambda count=count: unit.core(core_inputs, count), reps, warmup)
        except Exception as exc:  # noqa: BLE001
            record["error"] = _short_error(f"core[{count} threads]", exc)
            break
        key = str(count)
        record["core_s"][key] = stats
        record["speedup"][key] = _ratio(record["reference_s"]["median"], stats["median"])
        say(f"  core      {stats['median'] * 1e3:10.3f} ms at {count} thread(s)  ({record['speedup'][key]:.2f}x)")
        try:
            error = unit.parity(ref_out, core_out)
        except Exception as exc:  # noqa: BLE001
            record["error"] = _short_error("parity", exc)
            break
        if error is not None:
            error = float(error)
            worst = error if worst is None else max(worst, error)
        del core_out
    record["parity"]["max_rel_error"] = worst
    record["parity"]["ok"] = unit.parity_ok(worst)
    return record


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("inf")


def run_units(
    units: Sequence[Unit],
    threads: Sequence[int],
    reps: int,
    *,
    smoke: bool = False,
    host: dict[str, Any] | None = None,
    log: Log | None = None,
) -> dict[str, Any]:
    """Measure every unit and return the full JSON document."""

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host if host is not None else host_info(),
        "methodology": {
            "warmup": WARMUP,
            "reps": reps,
            "statistic": "median",
            "dispersion": "mad",
            "timer": "time.perf_counter_ns",
            "gc": "disabled during timed repetitions; gc.collect() between them (untimed)",
            "reference_blas_threads": 1,
            "core_blas_threads": 1,
            "threads": list(threads),
            "speedup": "reference median / core median at each thread count",
            "parity": "worst over thread counts of the unit's parity metric on warm-up outputs",
            "smoke": smoke,
        },
        "units": [],
    }
    for index, unit in enumerate(units, start=1):
        if log is not None:
            log(f"[{index}/{len(units)}] {unit.id} ({unit.size_text()})")
        document["units"].append(run_unit(unit, threads, reps, smoke=smoke, log=log))
    return document


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Non-finite floats become ``None`` so the JSON stays strict."""

    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):  # NumPy scalars
        return _json_safe(value.item())
    return value


def default_json_path(timestamp: _dt.datetime | None = None, hostname: str | None = None) -> Path:
    moment = timestamp or _dt.datetime.now(_dt.timezone.utc)
    host = re.sub(r"[^A-Za-z0-9._-]+", "-", hostname or socket.gethostname()).strip("-") or "host"
    return RESULTS_DIR / f"{moment.strftime('%Y%m%dT%H%M%SZ')}_{host}.json"


def write_json(document: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _ms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    value = seconds * 1e3
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _speed(value: float | None) -> str:
    if value is None or value != value:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.1f}x" if value >= 10 else f"{value:.2f}x"


def _parity_cell(record: dict[str, Any]) -> str:
    parity = record.get("parity") or {}
    error = parity.get("max_rel_error")
    if error is None:
        return "n/a" if parity.get("ok") is None else "?"
    verdict = "ok" if parity.get("ok") else f"FAIL (tol {parity.get('tolerance'):.0e})"
    return f"{error:.1e} {verdict}"


def render_table(document: dict[str, Any]) -> str:
    """The Markdown comparison table (one row per unit)."""

    threads = [str(count) for count in document["methodology"]["threads"]]
    header = (
        ["unit", "size", "reference ms"]
        + [f"core ms ({count}t)" for count in threads]
        + ["speedup 1t", "best speedup", "parity"]
    )
    align = ["---", "---", "---:"] + ["---:"] * len(threads) + ["---:", "---:", "---"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for record in document["units"]:
        size = " ".join(f"{key}={value}" for key, value in record["size"].items())
        reference = record.get("reference_s") or {}
        cells = [f"`{record['id']}`", size, _ms(reference.get("median"))]
        if record.get("error"):
            marker = "error"
            cells += [marker] * len(threads) + [marker, marker, marker]
        elif not record.get("available"):
            marker = f"n/a ({unavailable_tag(record.get('unavailable_reason'))})"
            cells += [marker] * len(threads) + [marker, marker, "n/a"]
        else:
            core = record.get("core_s") or {}
            speedups = record.get("speedup") or {}
            cells += [_ms((core.get(count) or {}).get("median")) for count in threads]
            finite = [value for value in speedups.values() if value is not None]
            best = max(finite) if finite else None
            cells += [_speed(speedups.get("1")), _speed(best), _parity_cell(record)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_header(document: dict[str, Any]) -> str:
    host = document["host"]
    method = document["methodology"]
    build = host.get("core_build_info") or {}
    core = host.get("core_version") or f"unavailable ({host.get('core_error', 'unknown error')})"
    if build:
        core += f" ({build.get('compiler', '?')} {build.get('compiler_version', '')}, {build.get('build_type', '?')})"
    commit = host.get("git_commit") or "unknown"
    if host.get("git_dirty"):
        commit += " (dirty)"
    load = host.get("loadavg")
    lines = [
        f"### Python reference vs C++ core ({document['timestamp']})",
        "",
        f"- Host: `{host.get('hostname')}`, {host.get('platform')}, {host.get('cpu_count')} CPUs",
        f"- Python {host.get('python')}, NumPy {host.get('numpy')}, SciPy {host.get('scipy')}",
        f"- Native core: {core}",
        f"- Commit: `{commit}`",
        f"- Load average: {', '.join(f'{value:.2f}' for value in load) if load else 'n/a'}",
        (
            f"- Method: {method['warmup']} warm-up + {method['reps']} reps, median (MAD recorded), "
            f"GC off while timing, BLAS pinned to 1 thread, core threads {', '.join(map(str, method['threads']))}"
            + (" -- SMOKE inputs, timings meaningless" if method.get("smoke") else "")
        ),
    ]
    if host.get("load_warning"):
        lines.append(f"- **Warning:** {host['load_warning']}")
    errors = [record for record in document["units"] if record.get("error")]
    if errors:
        lines.append("")
        lines.extend(f"- **Error** `{record['id']}`: {record['error']}" for record in errors)
    return "\n".join(lines)


def render_markdown(document: dict[str, Any]) -> str:
    return render_header(document) + "\n\n" + render_table(document) + "\n"


def render_list(units: Iterable[Unit]) -> str:
    rows = [("unit", "category", "phase", "size", "description")]
    rows += [(unit.id, unit.category, unit.phase, unit.size_text(), unit.description) for unit in units]
    widths = [max(len(row[col]) for row in rows) for col in range(4)]
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row[:4], widths)) + "  " + row[4])
    return "\n".join(lines)


def failures(document: dict[str, Any], strict: bool) -> list[str]:
    problems: list[str] = []
    for record in document["units"]:
        if record.get("error"):
            problems.append(f"{record['id']}: {record['error']}")
        elif record.get("available") and record["parity"].get("ok") is False:
            problems.append(
                f"{record['id']}: parity {record['parity']['max_rel_error']:.3e} exceeds {record['parity']['tolerance']:.0e}"
            )
        elif strict and not record.get("available"):
            problems.append(f"{record['id']}: unavailable ({record.get('unavailable_reason')})")
    return problems


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the frozen Python reference (mna_reference) with the C++ core per unit.",
    )
    parser.add_argument("--list", action="store_true", help="list the units and exit")
    parser.add_argument("--units", help="comma-separated unit ids (default: all)")
    parser.add_argument(
        "--category",
        help=f"comma-separated categories to run ({', '.join(CATEGORIES)})",
    )
    parser.add_argument(
        "--threads",
        help="comma-separated core thread counts; N = os.cpu_count() (default: 1,2,4 plus cpu_count when larger)",
    )
    parser.add_argument("--reps", type=int, help=f"timed repetitions per series (default {DEFAULT_REPS})")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_REPS} repetitions instead of {DEFAULT_REPS}")
    parser.add_argument(
        "--json",
        type=Path,
        help="JSON output path (default: benchmarks/results/<UTC timestamp>_<hostname>.json)",
    )
    parser.add_argument("--markdown", type=Path, help="also write the Markdown report (header + table) here")
    unavailable = parser.add_mutually_exclusive_group()
    unavailable.add_argument(
        "--skip-unavailable",
        dest="strict",
        action="store_false",
        help="report units without a native implementation as n/a (P<phase>) (default)",
    )
    unavailable.add_argument(
        "--strict", dest="strict", action="store_true", help="fail when a selected unit has no native implementation"
    )
    parser.set_defaults(strict=False)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny inputs for exercising the harness (tests); timings are meaningless",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    def split(text: str | None) -> list[str]:
        return [item.strip() for item in (text or "").split(",") if item.strip()]

    try:
        units = perf_units.select_units(split(args.units), split(args.category))
    except KeyError as exc:
        parser.error(str(exc.args[0] if exc.args else exc))
    if args.list:
        print(render_list(units))
        return 0
    if not units:
        parser.error("no units selected")
    try:
        threads = parse_threads(args.threads) if args.threads else default_threads()
    except ValueError as exc:
        parser.error(str(exc))
    reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else DEFAULT_REPS)
    if reps < 1:
        parser.error("--reps must be >= 1")

    host = host_info()
    if host.get("load_warning"):
        _stderr(f"warning: {host['load_warning']}")
    _stderr(
        f"{len(units)} unit(s), {reps} rep(s) + {WARMUP} warm-up, core threads {threads}"
        + (" [smoke inputs]" if args.smoke else "")
    )
    document = run_units(units, threads, reps, smoke=args.smoke, host=host, log=_stderr)

    json_path = args.json or default_json_path(hostname=host.get("hostname"))
    write_json(document, json_path)
    report = render_markdown(document)
    print(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
    _stderr(f"wrote {json_path}" + (f" and {args.markdown}" if args.markdown else ""))

    problems = failures(document, args.strict)
    for problem in problems:
        _stderr(f"FAILED {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
