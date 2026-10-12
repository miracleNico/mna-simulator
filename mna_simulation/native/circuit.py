"""Plain-table interface to the native compiled circuit (``mna_simulation._core.Circuit``).

``build_circuit`` takes the tables of ``CPP_MIGRATION_PLAN.md`` section 2.1 as
NumPy arrays (node indices are 0-based, ``-1`` is ground) and returns the core
handle, which owns the union CSC pattern of ``G``, ``C``, the unit gmin stamps
and every device Jacobian footprint. Table layout of the spec dict passed to
``_core.build_circuit``::

    {"n": int, "gmin": float, "node_count": int,
     "G" / "C" / "gmin_stamps": {"rows": int64, "cols": int64, "values": float64},
     "diodes": {"anode", "cathode": int64, "is": float64},
     "mos": {"drain", "gate", "source": int64, "beta", "vth", "lambda": float64,
             "sign": int32 (+1 NMOS, -1 PMOS)},
     "bjt": {"collector", "base", "emitter": int64, "is", "bf", "br", "vaf", "var": float64,
             "sign": int32 (+1 NPN, -1 PNP)},
     "behavioral": {"node_p", "node_n": int64, "programs": [_core.Program, ...]},
     "spice_diodes": {"anode", "cathode": int64, "is", "n", "bv", "ibv": float64},
     "gummel_poon": {"collector", "base", "emitter": int64, "is", "bf", "nf", "br", "nr", "vaf", "var",
                     "ikf", "ikr", "ise", "ne", "isc", "nc": float64, "sign": int32 (+1 NPN, -1 PNP)},
     "mos4": {"drain", "gate", "source", "bulk": int64, "beta", "vto", "lambda", "gamma", "phi": float64,
              "sign": int32 (+1 NMOS, -1 PMOS)}}

The last three are the model-card devices of ``HIERARCHY_AND_MODELS_PLAN.md`` 3.2 (phase M2;
``cpp/include/mna/devices/{spice_diode,gummel_poon,mos_card}.hpp``): every parameter is
already scaled by the instance area / multiplier, ``bv``/``vaf``/``var``/``ikf``/``ikr`` may be
``inf`` (not given), ``vto`` is the card's VTO (the kernel applies the PMOS sign) and the
terminals are the intrinsic ones (the builder's internal nodes when series resistances exist).

``node_count`` (P3b, ``CPP_MIGRATION_PLAN.md`` 2.4.1): the unknowns ``[0, node_count)``
are node voltages; the core appends one unit node-to-ground stamp per node to the gmin
pattern, so ``G + gmin * gmin_stamps`` shunts every node (never a branch-current row) by
gmin during gmin stepping. ``build_circuit`` infers it from the tables when the caller
does not pass it (:func:`infer_node_count`); circuits that only evaluate devices pass 0.

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
    "gummel_poon_table",
    "infer_node_count",
    "mos4_table",
    "mos_table",
    "spice_diode_table",
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


SPICE_DIODE_COLUMNS = ("is", "n", "bv", "ibv")
GUMMEL_POON_COLUMNS = ("is", "bf", "nf", "br", "nr", "vaf", "var", "ikf", "ikr", "ise", "ne", "isc", "nc")
MOS4_COLUMNS = ("beta", "vto", "lambda", "gamma", "phi")


def _device_columns(
    devices: Sequence[dict[str, Any]], nodes: tuple[str, ...], values: tuple[str, ...]
) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for key in nodes:
        table[key] = np.asarray([int(d[key]) for d in devices], dtype=np.int64).reshape(-1)
    for key in values:
        table[key] = np.asarray([float(d[key]) for d in devices], dtype=np.float64).reshape(-1)
    return table


def spice_diode_table(devices: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Table of SPICE diode dictionaries (keys anode/cathode/is/n/bv/ibv, area already applied)."""

    return _device_columns(devices, ("anode", "cathode"), SPICE_DIODE_COLUMNS)


def gummel_poon_table(devices: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Table of Gummel-Poon BJT dictionaries (intrinsic collector/base/emitter, card values, type)."""

    table = _device_columns(devices, ("collector", "base", "emitter"), GUMMEL_POON_COLUMNS)
    table["sign"] = _signs([d["type"] for d in devices], "QPNP")
    return table


def mos4_table(devices: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Table of four-terminal card MOS dictionaries (drain/gate/source/bulk, beta/vto/lambda/gamma/phi, type)."""

    table = _device_columns(devices, ("drain", "gate", "source", "bulk"), MOS4_COLUMNS)
    table["sign"] = _signs([d["type"] for d in devices], "PMOS")
    return table


def behavioral_table(node_p: Sequence[int], node_n: Sequence[int], programs: Sequence[Program]) -> dict[str, Any]:
    return {
        "node_p": _int64(node_p),
        "node_n": _int64(node_n),
        "programs": [program.handle for program in programs],
    }


def _table_column(table: dict[str, Any] | None, key: str) -> np.ndarray:
    if table is None or key not in table:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(table[key], dtype=np.int64).reshape(-1)


def infer_node_count(
    n: int,
    *,
    G: dict[str, np.ndarray] | None = None,
    C: dict[str, np.ndarray] | None = None,
    gmin_stamps: dict[str, np.ndarray] | None = None,
    diodes: dict[str, np.ndarray] | None = None,
    mos: dict[str, np.ndarray] | None = None,
    bjt: dict[str, np.ndarray] | None = None,
    behavioral: dict[str, Any] | None = None,
    spice_diodes: dict[str, np.ndarray] | None = None,
    gummel_poon: dict[str, np.ndarray] | None = None,
    mos4: dict[str, np.ndarray] | None = None,
) -> int:
    """Number of node-voltage unknowns of a builder-ordered circuit (nodes first, then branch currents).

    ``mna_builder`` numbers the nodes ``0 .. num_nodes - 1`` and the branch currents of V, L,
    E and H after them. A branch-current unknown ``k`` of a source between nodes ``p`` and
    ``q`` has the unit incidence pairs ``(k, p) = (p, k) = +1`` / ``(k, q) = (q, k) = -1`` with
    ``p, q < k``, no diagonal ``G`` entry, no off-diagonal ``C`` entry (an inductor adds only
    ``C[k, k]``), and is never a device terminal or a gmin stamp. The count is one past the
    last unknown that does not look like that, so every branch current is excluded. A node
    can only be missed in the degenerate case where it is numbered last and touches nothing
    but voltage-defined branches through unit pairs to lower indices (none in the builder's
    output). Callers that know the count (``problem.metadata["num_nodes"]``) should pass it.
    """

    size = int(n)
    if size <= 0 or size > np.iinfo(np.int32).max:
        return 0  # empty, or rejected by the core
    node = np.zeros(size, dtype=bool)
    for table, keys in (
        (diodes, ("anode", "cathode")),
        (mos, ("drain", "gate", "source")),
        (bjt, ("collector", "base", "emitter")),
        (behavioral, ("node_p", "node_n")),
        (spice_diodes, ("anode", "cathode")),
        (gummel_poon, ("collector", "base", "emitter")),
        (mos4, ("drain", "gate", "source", "bulk")),
        (gmin_stamps, ("rows", "cols")),
    ):
        for key in keys:
            index = _table_column(table, key)
            node[index[(index >= 0) & (index < size)]] = True  # out-of-range indices: the core reports them
    g_rows, g_cols = _table_column(G, "rows"), _table_column(G, "cols")
    g_values = np.asarray(G["values"], dtype=np.float64).reshape(-1) if G is not None and "values" in G else np.zeros(0)
    c_rows, c_cols = _table_column(C, "rows"), _table_column(C, "cols")
    if g_values.shape != g_rows.shape or g_cols.shape != g_rows.shape or c_cols.shape != c_rows.shape:
        return 0  # malformed: the core reports it
    keep = (g_rows >= 0) & (g_cols >= 0) & (g_rows < size) & (g_cols < size)
    g_rows, g_cols, g_values = g_rows[keep], g_cols[keep], g_values[keep]
    keep = (c_rows >= 0) & (c_cols >= 0) & (c_rows < size) & (c_cols < size)
    c_rows, c_cols = c_rows[keep], c_cols[keep]
    node[g_rows[g_rows == g_cols]] = True
    off = c_rows != c_cols
    node[c_rows[off]] = True
    node[c_cols[off]] = True
    # Branch-like rows: a unit incidence pair (k, j) / (j, k) with equal sign and j < k, or an
    # inductor whose terminals are both grounded (no G entry, only C[k, k]).
    unit = np.abs(g_values) == 1.0
    sign = (g_values > 0.0).astype(np.int64)
    keys = (g_rows * size + g_cols) * 2 + sign
    lower = unit & (g_cols < g_rows)
    transposed = (g_cols * size + g_rows) * 2 + sign
    paired = np.zeros(size, dtype=bool)
    hits = lower & np.isin(transposed, keys[unit])
    paired[g_rows[hits]] = True
    has_g_row = np.zeros(size, dtype=bool)
    has_g_row[g_rows] = True
    c_diagonal = np.zeros(size, dtype=bool)
    c_diagonal[c_rows[~off]] = True
    branch_like = ~node & (paired | (~has_g_row & c_diagonal))
    not_branch = np.flatnonzero(~branch_like)
    return int(not_branch[-1]) + 1 if not_branch.size else 0


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
    node_count: int | None = None,
    spice_diodes: dict[str, np.ndarray] | None = None,
    gummel_poon: dict[str, np.ndarray] | None = None,
    mos4: dict[str, np.ndarray] | None = None,
) -> Any:
    """Compile the tables into a ``_core.Circuit`` handle (see the module docstring for the layout).

    ``node_count``: node-voltage unknowns ``[0, node_count)`` that get a node-to-ground gmin
    stamp; ``None`` infers it (:func:`infer_node_count`), 0 adds none.
    """

    if node_count is None:
        node_count = infer_node_count(
            n,
            G=G,
            C=C,
            gmin_stamps=gmin_stamps,
            diodes=diodes,
            mos=mos,
            bjt=bjt,
            behavioral=behavioral,
            spice_diodes=spice_diodes,
            gummel_poon=gummel_poon,
            mos4=mos4,
        )
    spec: dict[str, Any] = {"n": int(n), "gmin": float(gmin), "node_count": int(node_count)}
    for key, table in (
        ("G", G),
        ("C", C),
        ("gmin_stamps", gmin_stamps),
        ("diodes", diodes),
        ("mos", mos),
        ("bjt", bjt),
        ("behavioral", behavioral),
        ("spice_diodes", spice_diodes),
        ("gummel_poon", gummel_poon),
        ("mos4", mos4),
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
        node_count=0,
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
