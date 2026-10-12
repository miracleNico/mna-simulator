"""MNA matrix construction and indexing metadata."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .api.contracts import CircuitIR, ComponentRecord, IndexMap, MnaProblem
from .device_models import (
    diode_current_expression,
    mos_level1_current_expression,
    mos_level1_current_template,
    render_mos_level1_current,
    source_expression,
    spice_source_terms,
)
from .errors import CircuitSimulatorError, error_handler
from .utils import format_voltage_labels, parse_value


def _ground_aliases(components: list[ComponentRecord]) -> set[str]:
    """Return the set of node names that should be treated as ground.

    Always includes the canonical "0". Any node referenced by a ``GND <node>``
    component is added to the alias set, so the rest of the MNA pipeline can
    treat that node as the reference rail.
    """

    aliases: set[str] = {"0"}
    for component in components:
        if component.type == "GND" and component.node1:
            aliases.add(component.node1)
    return aliases


_RECORD_FIELDS = tuple(f.name for f in dataclasses.fields(ComponentRecord))
# Record classes whose ``dataclasses.replace`` copy is a positional constructor call
# (ComponentRecord, and look-alikes such as the frozen oracle's record class).
_POSITIONAL_COPY: dict[type, bool] = {ComponentRecord: True}


def _copies_positionally(cls: type) -> bool:
    known = _POSITIONAL_COPY.get(cls)
    if known is None:
        fields = dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else ()
        known = tuple(f.name for f in fields) == _RECORD_FIELDS and all(
            f.init and f.kw_only is not True for f in fields
        )
        _POSITIONAL_COPY[cls] = known
    return known


def _normalize_components(components: list[ComponentRecord]) -> list[ComponentRecord]:
    """Strip GND symbols and rewrite any ground-aliased nodes to "0".

    The MNA pipeline expects ground references to be the literal string "0".
    This helper normalizes a netlist that may use other identifiers (declared
    via ``GND <node>`` lines) so downstream code stays unchanged. Every kept
    record is a fresh shallow copy (``dataclasses.replace`` semantics); records
    of ``ComponentRecord``'s shape are copied with one positional constructor
    call, which builds the same object without ``replace``'s per-field overhead.
    """

    aliases = _ground_aliases(components)

    def alias(node: str | None) -> str | None:
        if node is None:
            return None
        return "0" if node in aliases else node

    out: list[ComponentRecord] = []
    append = out.append
    for component in components:
        if component.type == "GND":
            continue
        record = type(component)
        if record is not ComponentRecord and not _copies_positionally(record):
            append(
                replace(
                    component,
                    node1=alias(component.node1),
                    node2=alias(component.node2),
                    ctrl_node1=alias(component.ctrl_node1),
                    ctrl_node2=alias(component.ctrl_node2),
                )
            )
            continue
        # ``alias`` inlined: None is never in ``aliases``, so it stays None.
        node1 = component.node1
        if node1 in aliases:
            node1 = "0"
        node2 = component.node2
        if node2 in aliases:
            node2 = "0"
        ctrl_node1 = component.ctrl_node1
        if ctrl_node1 is not None and ctrl_node1 in aliases:
            ctrl_node1 = "0"
        ctrl_node2 = component.ctrl_node2
        if ctrl_node2 is not None and ctrl_node2 in aliases:
            ctrl_node2 = "0"
        append(
            record(
                component.name,
                component.type,
                node1,
                node2,
                component.value,
                component.subtype,
                ctrl_node1,
                ctrl_node2,
                component.ctrl_source,
                component.value2,
                component.value3,
                component.metadata,
            )
        )
    return out


_BRANCH_TYPES = frozenset({"V", "VCVS", "CCVS", "L"})


def _build_index_map(components: list[ComponentRecord]) -> IndexMap:
    node_names: set[str] = set()
    add_node = node_names.add
    branch_names: list[str] = []
    seen_branch_names: set[str] = set()

    for component in components:
        node = component.node1
        if node:
            add_node(node)
        node = component.node2
        if node:
            add_node(node)
        node = component.ctrl_node1
        if node:
            add_node(node)
        node = component.ctrl_node2
        if node:
            add_node(node)

        if component.type in _BRANCH_TYPES:
            name = component.name
            if name in seen_branch_names:
                error_handler(
                    f"NETLIST_FATAL: Duplicate component name '{name}'. "
                    f"Use unique names or rename via '<name>:<TYPE>'."
                )
            seen_branch_names.add(name)
            branch_names.append(name)

    node_names.discard("0")
    sorted_nodes = sorted(node_names)
    num_nodes = len(sorted_nodes)
    node_to_index = {name: index for index, name in enumerate(sorted_nodes)}
    branch_to_index = {name: index + num_nodes for index, name in enumerate(branch_names)}

    return IndexMap(
        node_names=sorted_nodes,
        branch_names=branch_names,
        node_to_index=node_to_index,
        branch_to_index=branch_to_index,
    )


class _Stamps:
    """Matrix contributions in stamping order, expanded to COO triplets with NumPy.

    Each record is either a two-terminal stamp ``(j, k, v)`` - the entries
    ``(j, j, v)``, ``(k, k, v)``, ``(j, k, -v)``, ``(k, j, -v)`` minus those on a
    ground (``-1``) row or column - or a single entry ``(r, c, v)``, stored with
    ``a = -2 - r`` so that ``a <= -2`` marks it. Zero values are never recorded.
    ``arrays()`` expands the records once, keeping their order and each stamp's
    entry order, which yields exactly the triplet list entry-by-entry stamping
    produced, with three list appends per record instead of twelve.
    """

    __slots__ = ("a", "b", "v", "_expanded")

    def __init__(self) -> None:
        self.a: list[int] = []
        self.b: list[int] = []
        self.v: list[float] = []
        self._expanded: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def stamp(self, node_j: int, node_k: int, value: float) -> None:
        if value == 0.0:
            return
        self.a.append(node_j)
        self.b.append(node_k)
        self.v.append(value)

    def entry(self, row: int, col: int, value: float) -> None:
        if row == -1 or col == -1 or value == 0.0:
            return
        self.a.append(-2 - row)
        self.b.append(col)
        self.v.append(value)

    def _expand(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._expanded is None:
            count = len(self.a)
            a = np.fromiter(self.a, dtype=np.int64, count=count)
            b = np.fromiter(self.b, dtype=np.int64, count=count)
            v = np.fromiter(self.v, dtype=np.float64, count=count)
            single = a <= -2
            r = np.where(single, -2 - a, a)
            rows = np.stack((r, b, r, b), axis=1)
            cols = np.stack((np.where(single, b, r), b, b, r), axis=1)
            negative = -v
            values = np.stack((v, v, negative, negative), axis=1)
            valid = (rows != -1) & (cols != -1)
            valid[single, 1:] = False
            self._expanded = (rows[valid], cols[valid], values[valid], valid)
        return self._expanded

    def arrays(self, exclude_records: list[int] | None = None) -> dict[str, np.ndarray]:
        """Triplet arrays (``rows``/``cols`` int64, ``values`` float64), optionally without some records."""

        rows, cols, values, valid = self._expand()
        if exclude_records:
            keep = np.ones(valid.shape, dtype=bool)
            keep[np.asarray(exclude_records, dtype=np.int64)] = False
            keep = keep[valid]
            rows, cols, values = rows[keep], cols[keep], values[keep]
        return {"rows": rows, "cols": cols, "values": values}


def _summed_csr(stamps: _Stamps, size: int) -> csr_matrix:
    """CSR matrix of the triplets with duplicates summed in stamping order.

    ``np.bincount`` accumulates its weights sequentially from 0.0, exactly as the
    former dense ``G[r, c] += v`` (and the dok ``s[r, c] = s[r, c] + v``) did, so
    the values are bit-identical to the reference builder. Entries whose sum is
    exactly 0 are dropped, as the dok matrix dropped them. The unique keys come
    out of ``np.unique`` sorted in row-major order, so the CSR arrays are formed
    directly (the same canonical arrays a COO-to-CSR conversion produces).
    """

    arrays = stamps.arrays()
    if size == 0 or arrays["rows"].size == 0:
        return csr_matrix((size, size), dtype=float)
    keys, inverse = np.unique(arrays["rows"] * size + arrays["cols"], return_inverse=True)
    sums = np.bincount(inverse.reshape(-1), weights=arrays["values"], minlength=keys.size)
    keep = sums != 0.0
    keys = keys[keep]
    data = sums[keep]
    # The index dtype SciPy settles on (int32 whenever the sizes fit), passed in that
    # dtype so the constructor does not convert.
    index_dtype = np.int32 if max(size, data.size) <= np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(size + 1, dtype=index_dtype)
    np.cumsum(np.bincount(keys // size, minlength=size), out=indptr[1:])
    return csr_matrix((data, (keys % size).astype(index_dtype), indptr), shape=(size, size), dtype=float)


def _compile_native(
    size: int,
    gmin: float,
    g_stamps: _Stamps,
    g_gmin_records: list[int],
    c_stamps: _Stamps,
    gmin_stamps: _Stamps,
    diode_devices: list[dict[str, Any]],
    behavioral_devices: list[dict[str, Any]],
    level1_mos_devices: list[dict[str, Any]],
    level1_bjt_devices: list[dict[str, Any]],
) -> Any | None:
    """Compile the native circuit; ``None`` when a behavioral expression cannot be compiled.

    The native ``G`` excludes the gmin stamps (they go to the unit gmin table)
    so one compiled circuit serves every gmin step; ``g_gmin_records`` lists the
    records of those stamps in ``g_stamps`` (empty when ``gmin <= 0``).
    """

    from .native.circuit import behavioral_table, bjt_table, build_circuit, diode_table, mos_table
    from .native.expressions import compile_expression

    try:
        programs = [
            compile_expression(str(device["expression"]), time_dependent=False, n_unknowns=size)
            for device in behavioral_devices
        ]
    except (CircuitSimulatorError, TypeError):
        # Same timing as the reference core: the error surfaces when an analysis
        # evaluates the expression (solvers.compile_nl_functions), not at build time.
        return None
    return build_circuit(
        size,
        gmin=gmin,
        G=g_stamps.arrays(exclude_records=g_gmin_records),
        C=c_stamps.arrays(),
        gmin_stamps=gmin_stamps.arrays(),
        diodes=diode_table(
            [d["anode"] for d in diode_devices], [d["cathode"] for d in diode_devices], [d["is"] for d in diode_devices]
        )
        if diode_devices
        else None,
        mos=mos_table(level1_mos_devices) if level1_mos_devices else None,
        bjt=bjt_table(level1_bjt_devices) if level1_bjt_devices else None,
        behavioral=behavioral_table(
            [d["node_p"] for d in behavioral_devices], [d["node_n"] for d in behavioral_devices], programs
        )
        if behavioral_devices
        else None,
    )


# Symbolic vectors (f_str, solver_f_str, b_time_str) are accumulated per row as a
# list of pieces and joined once at the end: a node shared by hundreds of devices
# (the SRAM's vdd row holds ~420 MOS currents) would otherwise be rebuilt by
# quadratic repeated concatenation, and a long expression is referenced rather
# than copied into " + (expr)". ``None`` is an untouched row (value "0"). A row
# with one piece has that piece as its value (which may be "0" or, for a numeric
# behavioral value, not a string); a row with more pieces holds only strings and
# its value is ``"".join(pieces)``. Every update reproduces the reference
# ``vector[i] = expr`` / ``vector[i] = f"-({expr})"`` / ``vector[i] += f" {sign} ({expr})"``.


def _add_symbolic(rows: list[list[Any] | None], index: int, expression: Any, positive: bool = True) -> None:
    """``stamp_symbolic_rhs`` of the reference builder on piece-list rows."""

    if index == -1:
        return
    pieces = rows[index]
    if pieces is None or (len(pieces) == 1 and pieces[0] == "0"):
        if positive:
            rows[index] = [expression]
        elif type(expression) is str:
            rows[index] = ["-(", expression, ")"]
        else:
            rows[index] = [f"-({expression})"]
        return
    if len(pieces) == 1 and type(pieces[0]) is not str:
        # Not a plain string (e.g. a numeric behavioral value): apply the
        # reference's in-place ``+=`` to it (raising what that raises).
        sign = "+" if positive else "-"
        pieces[0] += f" {sign} ({expression})"
        return
    if type(expression) is str:
        pieces.extend((" + (" if positive else " - (", expression, ")"))
    else:
        pieces.append(f" + ({expression})" if positive else f" - ({expression})")


def _add_nonlinear_current(rows: list[list[Any] | None], node_p: int, node_n: int, expression: Any) -> None:
    _add_symbolic(rows, node_p, expression, positive=True)
    _add_symbolic(rows, node_n, expression, positive=False)


def _symbolic_column(rows: list[list[Any] | None]) -> np.ndarray:
    """``(n, 1)`` object array of the row values (as ``np.full((n, 1), "0", dtype=object)`` stamped in place)."""

    column = np.empty((len(rows), 1), dtype=object)
    column[:, 0] = ["0" if pieces is None else pieces[0] if len(pieces) == 1 else "".join(pieces) for pieces in rows]
    return column


def build_mna_problem(circuit: CircuitIR, gmin: float = 0.0) -> MnaProblem:
    """Construct the MNA matrices and symbolic source/nonlinear vectors.

    Linear elements are recorded as stamps (``_Stamps``: G, C and a unit-valued
    list of every gmin stamp) that expand to COO triplets in stamping order;
    ``G_sparse``/``C_sparse`` sum them in that order, the dense ``G``/``C`` are
    materialized lazily from them, and the native compiled circuit is stored in
    ``problem.native``.

    The loop appends stamp records directly (no per-entry calls), builds the
    symbolic vectors from piece lists, and formats each distinct Level-1 MOS
    parameter set once (``mos_level1_current_template``); triplet order, values
    and every string are the same as entry-by-entry stamping produces.
    """

    # Pre-process components: drop GND symbols (used only to flag ground nodes)
    # and rewrite their referenced node names to the canonical ground "0".
    components = _normalize_components(circuit.components)

    index_map = _build_index_map(components)
    num_nodes = len(index_map.node_names)
    num_branches = len(index_map.branch_names)
    size = num_nodes + num_branches

    G_stamps = _Stamps()
    C_stamps = _Stamps()
    gmin_stamps = _Stamps()
    stamp_g = G_stamps.stamp
    stamp_c = C_stamps.stamp
    entry_g = G_stamps.entry
    entry_c = C_stamps.entry
    g_a, g_b, g_v = G_stamps.a, G_stamps.b, G_stamps.v
    c_a, c_b, c_v = C_stamps.a, C_stamps.b, C_stamps.v
    m_a, m_b, m_v = gmin_stamps.a, gmin_stamps.b, gmin_stamps.v
    # Records of the gmin stamps inside G_stamps (they stay out of the native G).
    g_gmin_records: list[int] = []
    f_str: list[list[Any] | None] = [None] * size
    solver_f_str: list[list[Any] | None] = [None] * size
    b_time_str: list[list[Any] | None] = [None] * size
    b_dc = np.zeros((size, 1), dtype=float)
    b_ac = np.zeros((size, 1), dtype=complex)
    controlling_branch_map: dict[str, int] = {}
    level1_mos_devices: list[dict[str, float | int | str]] = []
    level1_bjt_devices: list[dict[str, float | int | str]] = []
    diode_devices: list[dict[str, float | int | str]] = []
    behavioral_devices: list[dict[str, float | int | str]] = []
    branch_to_index = index_map.branch_to_index
    # Node index lookup: ground ("0") and absent nodes (None) are -1; any other
    # name missing from the map raises KeyError(name), as a plain dict lookup does.
    node_index = dict(index_map.node_to_index)
    node_index[None] = -1  # type: ignore[index]
    node_index["0"] = -1
    get_index = node_index.__getitem__
    # "x[i]" for i in range(size), then "0" at position -1 (ground).
    x_exprs: list[str] | None = None
    # Level-1 MOS parameters per distinct parameter strings (built once per set).
    mos_parameters: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    def get_branch_index(branch_name: str) -> int:
        if branch_name not in branch_to_index:
            error_handler(f"NETLIST_FATAL: Unknown controlling branch '{branch_name}'.")
        return branch_to_index[branch_name]

    def stamp_gmin(node_j: int, node_k: int) -> None:
        # The unit pattern is recorded at every gmin (0 included) so the native
        # circuit can form G + gmin * pattern; G itself gets gmin only when > 0.
        m_a.append(node_j)
        m_b.append(node_k)
        m_v.append(1.0)
        if gmin > 0.0:
            g_gmin_records.append(len(g_a))
            g_a.append(node_j)
            g_b.append(node_k)
            g_v.append(gmin)

    def stamp_numeric_rhs(vector: np.ndarray, index: int, value: complex, positive: bool = True) -> None:
        if index == -1:
            return
        vector[index] += value if positive else -value

    def stamp_voltage_source(kv: int, j: int, k: int) -> None:
        # Entries (j, kv, 1), (kv, j, 1), (k, kv, -1), (kv, k, -1); kv is a branch row (never -1).
        if j != -1:
            g_a.extend((-2 - j, -2 - kv))
            g_b.extend((kv, j))
            g_v.extend((1.0, 1.0))
        if k != -1:
            g_a.extend((-2 - k, -2 - kv))
            g_b.extend((kv, k))
            g_v.extend((-1.0, -1.0))

    def stamp_vccs_g(out_p: int, out_n: int, ctrl_p: int, ctrl_n: int, gain: float) -> None:
        if gain == 0.0:
            return
        entry_g(out_p, ctrl_p, gain)
        entry_g(out_p, ctrl_n, -gain)
        entry_g(out_n, ctrl_p, -gain)
        entry_g(out_n, ctrl_n, gain)

    for component in components:
        comp_type = component.type
        j = get_index(component.node1)
        k = get_index(component.node2)

        if comp_type == "NMOS" or comp_type == "PMOS":
            gate = get_index(component.ctrl_node1)
            source = k
            drain = j
            metadata = component.metadata
            if metadata.get("model") == "level1":
                if x_exprs is None:
                    x_exprs = [f"x[{index}]" for index in range(size)]
                    x_exprs.append("0")
                key = (
                    component.value,
                    component.value2,
                    component.value3,
                    comp_type,
                    metadata.get("cgs", "0"),
                    metadata.get("cgd", "0"),
                    metadata.get("gds_floor", "1e-12"),
                )
                try:
                    parameters = mos_parameters.get(key)
                except TypeError:  # an unhashable parameter object: no memo
                    parameters = None
                    key = None
                if parameters is not None:
                    template, c_gs, c_gd, g_ds_floor, beta, vth, lambda_ = parameters
                    mos_expr = render_mos_level1_current(template, comp_type, x_exprs[drain], x_exprs[gate], x_exprs[source])
                else:
                    # First device with these parameters: the calls (and any error) in the reference order.
                    mos_expr = mos_level1_current_expression(component, x_exprs[drain], x_exprs[gate], x_exprs[source])
                    c_gs = parse_value(metadata.get("cgs", "0"))
                    c_gd = parse_value(metadata.get("cgd", "0"))
                    g_ds_floor = parse_value(metadata.get("gds_floor", "1e-12"))
                    beta = parse_value(component.value or "1m")
                    vth = abs(parse_value(component.value2 or "0.4"))
                    lambda_ = parse_value(component.value3 or "0")
                    if key is not None:
                        template = mos_level1_current_template(
                            component.value, component.value2, component.value3, comp_type
                        )
                        mos_parameters[key] = (template, c_gs, c_gd, g_ds_floor, beta, vth, lambda_)
                # _add_nonlinear_current(f_str, drain, source, mos_expr) with the
                # common case (a row already holding plain-string pieces) inlined.
                if drain != -1:
                    pieces = f_str[drain]
                    if pieces is not None and (len(pieces) > 1 or (type(pieces[0]) is str and pieces[0] != "0")):
                        pieces.extend((" + (", mos_expr, ")"))
                    else:
                        _add_symbolic(f_str, drain, mos_expr, True)
                if source != -1:
                    pieces = f_str[source]
                    if pieces is not None and (len(pieces) > 1 or (type(pieces[0]) is str and pieces[0] != "0")):
                        pieces.extend((" - (", mos_expr, ")"))
                    else:
                        _add_symbolic(f_str, source, mos_expr, False)
                level1_mos_devices.append(
                    {
                        "type": comp_type,
                        "drain": drain,
                        "gate": gate,
                        "source": source,
                        "beta": beta,
                        "vth": vth,
                        "lambda": lambda_,
                    }
                )
                # Stamps (the values are > 0, so never skipped as zero), then gmin.
                if g_ds_floor > 0.0:
                    g_a.append(drain)
                    g_b.append(source)
                    g_v.append(g_ds_floor)
                if c_gs > 0.0:
                    c_a.append(gate)
                    c_b.append(source)
                    c_v.append(c_gs)
                if c_gd > 0.0:
                    c_a.append(gate)
                    c_b.append(drain)
                    c_v.append(c_gd)
                stamp_gmin(drain, source)
                continue
            gm = parse_value(component.value or "5m")
            r_o = parse_value(component.value2 or "50k")
            c_gs = parse_value(component.value3 or "5p")
            c_gd = parse_value(metadata.get("cgd", "1p"))
            g_mb = parse_value(metadata.get("gmb", "0"))
            c_bs = parse_value(metadata.get("cbs", "0"))
            c_bd = parse_value(metadata.get("cbd", "0"))
            if r_o > 0.0:
                stamp_g(drain, source, 1.0 / r_o)
            if c_gs > 0.0:
                stamp_c(gate, source, c_gs)
            if c_gd > 0.0:
                stamp_c(gate, drain, c_gd)
            if c_bs > 0.0:
                stamp_c(source, -1, c_bs)
            if c_bd > 0.0:
                stamp_c(drain, -1, c_bd)
            stamp_vccs_g(drain, source, gate, source, gm)
            # Body is implicit ground for the current three-pin symbol. gmb is
            # present for richer models but defaults to zero.
            stamp_vccs_g(drain, source, -1, source, g_mb)
        elif comp_type == "R":
            resistance = parse_value(component.value or "0")
            if resistance == 0.0:
                # CPP_MIGRATION_PLAN.md 2.4.1 (was: ZeroDivisionError "float division by zero").
                error_handler(f"NETLIST_FATAL: '{component.name}' has zero resistance")
            stamp_g(j, k, 1.0 / resistance)
        elif comp_type == "C":
            stamp_c(j, k, parse_value(component.value or "0"))
            stamp_gmin(j, k)
        elif comp_type == "L":
            branch_idx = get_branch_index(component.name)
            stamp_voltage_source(branch_idx, j, k)
            # MNA form is C dx/dt + G x = b. For branch current flowing from
            # node1 to node2, an inductor satisfies v(node1)-v(node2)-L di/dt=0.
            entry_c(branch_idx, branch_idx, -parse_value(component.value or "0"))
            controlling_branch_map[component.name] = branch_idx
        elif comp_type == "I":
            if component.metadata:  # SPICE source form: DC, AC (with phase) and waveform parts
                dc, phasor, expression = spice_source_terms(component)
                if j != -1:
                    b_dc[j] += dc
                if k != -1:
                    b_dc[k] -= dc
                if phasor:
                    stamp_numeric_rhs(b_ac, j, phasor, positive=True)
                    stamp_numeric_rhs(b_ac, k, phasor, positive=False)
                if expression is not None:
                    _add_symbolic(b_time_str, j, expression, positive=True)
                    _add_symbolic(b_time_str, k, expression, positive=False)
            elif (component.subtype or "").upper() == "DC":
                current = parse_value(component.value or "0")
                if j != -1:
                    b_dc[j] += current
                if k != -1:
                    b_dc[k] -= current
            elif (component.subtype or "").upper() == "AC":
                phasor = parse_value(component.value or "0")
                stamp_numeric_rhs(b_ac, j, phasor, positive=True)
                stamp_numeric_rhs(b_ac, k, phasor, positive=False)
            elif (component.subtype or "").upper() in {"SIN", "COS"}:
                phase = 0.0 if (component.subtype or "").upper() == "COS" else -np.pi / 2.0
                phasor = parse_value(component.value or "0") * np.exp(1j * phase)
                stamp_numeric_rhs(b_ac, j, phasor, positive=True)
                stamp_numeric_rhs(b_ac, k, phasor, positive=False)
                expression = source_expression(component)
                _add_symbolic(b_time_str, j, expression, positive=True)
                _add_symbolic(b_time_str, k, expression, positive=False)
            else:
                expression = source_expression(component)
                _add_symbolic(b_time_str, j, expression, positive=True)
                _add_symbolic(b_time_str, k, expression, positive=False)
        elif comp_type == "V":
            branch_idx = get_branch_index(component.name)
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            if component.metadata:  # SPICE source form: DC, AC (with phase) and waveform parts
                dc, phasor, expression = spice_source_terms(component)
                b_dc[branch_idx] += dc
                if phasor:
                    stamp_numeric_rhs(b_ac, branch_idx, phasor, positive=True)
                if expression is not None:
                    _add_symbolic(b_time_str, branch_idx, expression, positive=True)
            elif (component.subtype or "").upper() == "DC":
                b_dc[branch_idx] += parse_value(component.value or "0")
            elif (component.subtype or "").upper() == "AC":
                stamp_numeric_rhs(b_ac, branch_idx, parse_value(component.value or "0"), positive=True)
            elif (component.subtype or "").upper() in {"SIN", "COS"}:
                phase = 0.0 if (component.subtype or "").upper() == "COS" else -np.pi / 2.0
                phasor = parse_value(component.value or "0") * np.exp(1j * phase)
                stamp_numeric_rhs(b_ac, branch_idx, phasor, positive=True)
                _add_symbolic(b_time_str, branch_idx, source_expression(component), positive=True)
            else:
                _add_symbolic(b_time_str, branch_idx, source_expression(component), positive=True)
        elif comp_type == "VCCS":
            c1 = get_index(component.ctrl_node1)
            c2 = get_index(component.ctrl_node2)
            gain = parse_value(component.value or "0")
            stamp_vccs_g(j, k, c1, c2, gain)
        elif comp_type == "VCVS":
            branch_idx = get_branch_index(component.name)
            c1 = get_index(component.ctrl_node1)
            c2 = get_index(component.ctrl_node2)
            gain = parse_value(component.value or "0")
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            if c1 != -1:
                entry_g(branch_idx, c1, -gain)
            if c2 != -1:
                entry_g(branch_idx, c2, gain)
        elif comp_type == "CCCS":
            control_branch = get_branch_index(component.ctrl_source or "")
            gain = parse_value(component.value or "0")
            entry_g(j, control_branch, gain)
            entry_g(k, control_branch, -gain)
        elif comp_type == "CCVS":
            branch_idx = get_branch_index(component.name)
            control_branch = get_branch_index(component.ctrl_source or "")
            gain = parse_value(component.value or "0")
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            entry_g(branch_idx, control_branch, -gain)
        elif comp_type == "D":
            j_expr = f"x[{j}]" if j != -1 else "0"
            k_expr = f"x[{k}]" if k != -1 else "0"
            diode_expr = diode_current_expression(component, j_expr, k_expr)
            _add_nonlinear_current(f_str, j, k, diode_expr)
            _add_nonlinear_current(solver_f_str, j, k, diode_expr)
            diode_devices.append(
                {"name": component.name, "anode": j, "cathode": k, "is": parse_value(component.value or "1e-15")}
            )
            stamp_gmin(j, k)
        elif comp_type == "QNPN" or comp_type == "QPNP":
            base = get_index(component.ctrl_node1)
            emitter = k
            collector = j
            metadata = component.metadata
            if metadata.get("model") == "level1":
                cje = parse_value(metadata.get("cje", "4p"))
                cjc = parse_value(metadata.get("cjc", "2p"))
                level1_bjt_devices.append(
                    {
                        "type": comp_type,
                        "collector": collector,
                        "base": base,
                        "emitter": emitter,
                        "is": parse_value(component.value or "1e-15"),
                        "bf": parse_value(component.value2 or "150"),
                        "br": parse_value(component.value3 or "3"),
                        "vaf": parse_value(metadata.get("vaf", "100")),
                        "var": parse_value(metadata.get("var", "25")),
                        "rb": parse_value(metadata.get("rb", "50")),
                        "re": parse_value(metadata.get("re", "0.5")),
                        "rc": parse_value(metadata.get("rc", "5")),
                    }
                )
                if cje > 0.0:
                    stamp_c(base, emitter, cje)
                if cjc > 0.0:
                    stamp_c(base, collector, cjc)
                stamp_gmin(base, emitter)
                stamp_gmin(base, collector)
                continue
            gm = parse_value(component.value or "40m")
            r_pi = parse_value(component.value2 or "2.5k")
            r_o = parse_value(component.value3 or "100k")
            c_pi = parse_value(metadata.get("cpi", "8p"))
            c_mu = parse_value(metadata.get("cmu", "3p"))
            c_cs = parse_value(metadata.get("ccs", "0"))
            if r_pi == 0.0:
                # The 2.4.1 zero-resistance ruling applies to rpi as well (was: ZeroDivisionError).
                error_handler(f"NETLIST_FATAL: '{component.name}' has zero resistance (rpi)")
            stamp_g(base, emitter, 1.0 / r_pi)
            if r_o > 0.0:
                stamp_g(collector, emitter, 1.0 / r_o)
            if c_pi > 0.0:
                stamp_c(base, emitter, c_pi)
            if c_mu > 0.0:
                stamp_c(base, collector, c_mu)
            if c_cs > 0.0:
                stamp_c(collector, emitter, c_cs)
            # Hybrid-pi transconductance: gm * Vbe flowing collector -> emitter.
            stamp_vccs_g(collector, emitter, base, emitter, gm)
        elif comp_type == "B":
            expression = component.value or "0"
            _add_nonlinear_current(f_str, j, k, expression)
            _add_nonlinear_current(solver_f_str, j, k, expression)
            behavioral_devices.append({"name": component.name, "node_p": j, "node_n": k, "expression": expression})
        else:
            error_handler(f"NETLIST_FATAL: Unsupported device type '{comp_type}'.")

    labels = format_voltage_labels(index_map.node_names, index_map.branch_names)
    G_csr = _summed_csr(G_stamps, size)
    C_csr = _summed_csr(C_stamps, size)
    metadata = {
        "num_nodes": num_nodes,
        "num_branches": num_branches,
        "labels": labels,
        "controlling_branch_map": controlling_branch_map,
        "matrix_storage": "dense+csr",
        "G_nnz": int(G_csr.nnz),
        "C_nnz": int(C_csr.nnz),
        "nonlinear_mos_devices": len(level1_mos_devices),
        "nonlinear_bjt_devices": len(level1_bjt_devices),
    }
    native = _compile_native(
        size,
        gmin,
        G_stamps,
        g_gmin_records,
        C_stamps,
        gmin_stamps,
        diode_devices,
        behavioral_devices,
        level1_mos_devices,
        level1_bjt_devices,
    )

    return MnaProblem(
        G=None,
        C=None,
        G_sparse=G_csr,
        C_sparse=C_csr,
        f_str=_symbolic_column(f_str),
        solver_f_str=_symbolic_column(solver_f_str),
        b_dc=b_dc,
        b_ac=b_ac,
        b_time_str=_symbolic_column(b_time_str),
        index_map=index_map,
        components=components,
        level1_mos_devices=level1_mos_devices,
        level1_bjt_devices=level1_bjt_devices,
        gmin=gmin,
        metadata=metadata,
        diode_devices=diode_devices,
        behavioral_devices=behavioral_devices,
        native=native,
    )
