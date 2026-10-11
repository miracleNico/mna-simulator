"""Plain-table interface to the native compiled circuit (``mna_simulation._core.Circuit``).

``build_circuit`` takes the tables of ``CPP_MIGRATION_PLAN.md`` section 2.1 as
NumPy arrays (node indices are 0-based, ``-1`` is ground) and returns the core
handle, which owns the union CSC pattern of ``G``, ``C``, the unit gmin stamps
and every device Jacobian footprint. Table layout of the spec dict passed to
``_core.build_circuit``::

    {"n": int, "gmin": float,
     "G" / "C" / "gmin_stamps": {"rows": int64, "cols": int64, "values": float64},
     "diodes": {"anode", "cathode": int64, "is": float64},
     "mos": {"drain", "gate", "source": int64, "beta", "vth", "lambda": float64,
             "sign": int32 (+1 NMOS, -1 PMOS)},
     "bjt": {"collector", "base", "emitter": int64, "is", "bf", "br", "vaf", "var": float64,
             "sign": int32 (+1 NPN, -1 PNP)},
     "behavioral": {"node_p", "node_n": int64, "programs": [_core.Program, ...]}}

``compile_device_functions`` is the engine behind
``solvers.compile_nl_functions``: it compiles the nonlinear rows of
``solver_f_str`` (one behavioral program per non-zero row, injected at that row,
which mirrors the reference ``sympify`` of each row) together with the Level-1
device dictionaries into a temporary circuit and returns ``f(*x)`` / ``J(*x)``
callables evaluated by the core.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from operator import itemgetter
from typing import Any

import numpy as np

from . import require_core
from .expressions import Program, compile_expression

__all__ = [
    "behavioral_table",
    "bjt_table",
    "build_circuit",
    "compile_device_functions",
    "diode_table",
    "mos_table",
    "triplets",
]


def _int64(values: Iterable[Any]) -> np.ndarray:
    return np.ascontiguousarray(np.fromiter((int(v) for v in values), dtype=np.int64))


def _float64(values: Iterable[Any]) -> np.ndarray:
    return np.ascontiguousarray(np.fromiter((float(v) for v in values), dtype=np.float64))


def triplets(rows: Sequence[int], cols: Sequence[int], values: Sequence[float]) -> dict[str, np.ndarray]:
    return {
        "rows": np.ascontiguousarray(rows, dtype=np.int64).reshape(-1),
        "cols": np.ascontiguousarray(cols, dtype=np.int64).reshape(-1),
        "values": np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
    }


def diode_table(anode: Sequence[int], cathode: Sequence[int], is_: Sequence[float]) -> dict[str, np.ndarray]:
    return {"anode": _int64(anode), "cathode": _int64(cathode), "is": _float64(is_)}


def _columns(devices: Sequence[dict[str, Any]], keys: tuple[str, ...]) -> list[tuple[Any, ...]]:
    """Per-key value tuples of a list of device dicts (column-major)."""

    if not devices:
        return [() for _ in keys]
    getter = itemgetter(*keys)
    return list(zip(*map(getter, devices)))


def _signs(types: Sequence[Any], negative: str) -> np.ndarray:
    """-1 where ``str(type).upper() == negative`` (PMOS / QPNP), +1 elsewhere."""

    sign_of = {kind: (-1 if str(kind).upper() == negative else 1) for kind in set(types)}
    return np.asarray([sign_of[kind] for kind in types], dtype=np.int32).reshape(-1)


def mos_table(devices: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Table of Level-1 MOS dictionaries as built by ``mna_builder`` (keys drain/gate/source/beta/vth/lambda/type)."""

    drain, gate, source, beta, vth, lambda_, types = _columns(
        devices, ("drain", "gate", "source", "beta", "vth", "lambda", "type")
    )
    return {
        "drain": np.asarray(drain, dtype=np.int64).reshape(-1),
        "gate": np.asarray(gate, dtype=np.int64).reshape(-1),
        "source": np.asarray(source, dtype=np.int64).reshape(-1),
        "beta": np.asarray(beta, dtype=np.float64).reshape(-1),
        "vth": np.asarray(vth, dtype=np.float64).reshape(-1),
        "lambda": np.asarray(lambda_, dtype=np.float64).reshape(-1),
        "sign": _signs(types, "PMOS"),
    }


def bjt_table(devices: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Table of Level-1 BJT dictionaries (keys collector/base/emitter/is/bf/br/vaf/var/type)."""

    collector, base, emitter, is_, bf, br, vaf, var, types = _columns(
        devices, ("collector", "base", "emitter", "is", "bf", "br", "vaf", "var", "type")
    )
    return {
        "collector": np.asarray(collector, dtype=np.int64).reshape(-1),
        "base": np.asarray(base, dtype=np.int64).reshape(-1),
        "emitter": np.asarray(emitter, dtype=np.int64).reshape(-1),
        "is": np.asarray(is_, dtype=np.float64).reshape(-1),
        "bf": np.asarray(bf, dtype=np.float64).reshape(-1),
        "br": np.asarray(br, dtype=np.float64).reshape(-1),
        "vaf": np.asarray(vaf, dtype=np.float64).reshape(-1),
        "var": np.asarray(var, dtype=np.float64).reshape(-1),
        "sign": _signs(types, "QPNP"),
    }


def behavioral_table(node_p: Sequence[int], node_n: Sequence[int], programs: Sequence[Program]) -> dict[str, Any]:
    return {
        "node_p": _int64(node_p),
        "node_n": _int64(node_n),
        "programs": [program.handle for program in programs],
    }


def build_circuit(
    n: int,
    *,
    gmin: float = 0.0,
    G: dict[str, np.ndarray] | None = None,
    C: dict[str, np.ndarray] | None = None,
    gmin_stamps: dict[str, np.ndarray] | None = None,
    diodes: dict[str, np.ndarray] | None = None,
    mos: dict[str, np.ndarray] | None = None,
    bjt: dict[str, np.ndarray] | None = None,
    behavioral: dict[str, Any] | None = None,
) -> Any:
    """Compile the tables into a ``_core.Circuit`` handle (see the module docstring for the layout)."""

    spec: dict[str, Any] = {"n": int(n), "gmin": float(gmin)}
    for key, table in (
        ("G", G),
        ("C", C),
        ("gmin_stamps", gmin_stamps),
        ("diodes", diodes),
        ("mos", mos),
        ("bjt", bjt),
        ("behavioral", behavioral),
    ):
        if table is not None:
            spec[key] = table
    return require_core().build_circuit(spec)


def _state_vector(args: tuple[Any, ...], size: int) -> np.ndarray:
    if len(args) != size:
        raise TypeError(f"expected {size} state values, got {len(args)}")
    return np.fromiter(args, dtype=np.float64, count=size)


def compile_device_functions(
    f_str_vector: np.ndarray,
    size: int,
    level1_mos_devices: Sequence[dict[str, Any]] | None = None,
    level1_bjt_devices: Sequence[dict[str, Any]] | None = None,
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """``(f_func, j_func)`` for ``solvers.compile_nl_functions`` backed by a native circuit."""

    rows: list[int] = []
    programs: list[Program] = []
    for row, expr in enumerate(np.asarray(f_str_vector, dtype=object).reshape(-1)):
        text = str(expr)
        if text == "0":
            continue
        rows.append(row)
        programs.append(compile_expression(text, time_dependent=False, n_unknowns=size))

    mos_devices = list(level1_mos_devices or [])
    bjt_devices = list(level1_bjt_devices or [])
    circuit = build_circuit(
        size,
        mos=mos_table(mos_devices) if mos_devices else None,
        bjt=bjt_table(bjt_devices) if bjt_devices else None,
        behavioral=behavioral_table(rows, [-1] * len(rows), programs) if programs else None,
    )
    core = require_core()
    device_f = core.device_f
    device_jacobian = core.device_jacobian_dense

    def f_func(*args: Any) -> np.ndarray:
        return device_f(circuit, _state_vector(args, size), 0.0)

    def j_func(*args: Any) -> np.ndarray:
        return device_jacobian(circuit, _state_vector(args, size), 0.0)

    f_func.circuit = circuit  # type: ignore[attr-defined]
    j_func.circuit = circuit  # type: ignore[attr-defined]
    return f_func, j_func
