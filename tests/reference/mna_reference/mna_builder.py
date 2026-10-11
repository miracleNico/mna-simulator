"""MNA matrix construction and indexing metadata."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import dok_matrix

from .api.contracts import CircuitIR, ComponentRecord, IndexMap, MnaProblem
from .device_models import diode_current_expression, mos_level1_current_expression, source_expression
from .errors import error_handler
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


def _normalize_components(components: list[ComponentRecord]) -> list[ComponentRecord]:
    """Strip GND symbols and rewrite any ground-aliased nodes to "0".

    The MNA pipeline expects ground references to be the literal string "0".
    This helper normalizes a netlist that may use other identifiers (declared
    via ``GND <node>`` lines) so downstream code stays unchanged.
    """

    aliases = _ground_aliases(components)

    def alias(node: str | None) -> str | None:
        if node is None:
            return None
        return "0" if node in aliases else node

    out: list[ComponentRecord] = []
    for component in components:
        if component.type == "GND":
            continue
        out.append(
            replace(
                component,
                node1=alias(component.node1),
                node2=alias(component.node2),
                ctrl_node1=alias(component.ctrl_node1),
                ctrl_node2=alias(component.ctrl_node2),
            )
        )
    return out


def _build_index_map(components: list[ComponentRecord]) -> IndexMap:
    node_names: set[str] = set()
    extra_current_components: list[ComponentRecord] = []
    seen_branch_names: set[str] = set()

    for component in components:
        if component.node1:
            node_names.add(component.node1)
        if component.node2:
            node_names.add(component.node2)
        if component.ctrl_node1:
            node_names.add(component.ctrl_node1)
        if component.ctrl_node2:
            node_names.add(component.ctrl_node2)

        if component.type in {"V", "VCVS", "CCVS", "L"}:
            if component.name in seen_branch_names:
                error_handler(
                    f"NETLIST_FATAL: Duplicate component name '{component.name}'. "
                    f"Use unique names or rename via '<name>:<TYPE>'."
                )
            seen_branch_names.add(component.name)
            extra_current_components.append(component)

    node_names.discard("0")
    sorted_nodes = sorted(node_names)
    branch_names = [component.name for component in extra_current_components]
    node_to_index = {name: index for index, name in enumerate(sorted_nodes)}
    branch_to_index = {name: index + len(sorted_nodes) for index, name in enumerate(branch_names)}

    return IndexMap(
        node_names=sorted_nodes,
        branch_names=branch_names,
        node_to_index=node_to_index,
        branch_to_index=branch_to_index,
    )


def build_mna_problem(circuit: CircuitIR, gmin: float = 0.0) -> MnaProblem:
    """Construct the MNA matrices and symbolic source/nonlinear vectors."""

    # Pre-process components: drop GND symbols (used only to flag ground nodes)
    # and rewrite their referenced node names to the canonical ground "0".
    components = _normalize_components(circuit.components)

    index_map = _build_index_map(components)
    num_nodes = len(index_map.node_names)
    num_branches = len(index_map.branch_names)
    size = num_nodes + num_branches

    G = np.zeros((size, size), dtype=float)
    C = np.zeros((size, size), dtype=float)
    G_sparse = dok_matrix((size, size), dtype=float)
    C_sparse = dok_matrix((size, size), dtype=float)
    f_str = np.full((size, 1), "0", dtype=object)
    solver_f_str = np.full((size, 1), "0", dtype=object)
    b_dc = np.zeros((size, 1), dtype=float)
    b_ac = np.zeros((size, 1), dtype=complex)
    b_time_str = np.full((size, 1), "0", dtype=object)
    controlling_branch_map: dict[str, int] = {}
    level1_mos_devices: list[dict[str, float | int | str]] = []
    level1_bjt_devices: list[dict[str, float | int | str]] = []

    def get_index(node_name: str | None) -> int:
        if node_name in {None, "0"}:
            return -1
        return index_map.node_to_index[node_name]

    def get_branch_index(branch_name: str) -> int:
        if branch_name not in index_map.branch_to_index:
            error_handler(f"NETLIST_FATAL: Unknown controlling branch '{branch_name}'.")
        return index_map.branch_to_index[branch_name]

    def add_entry(matrix: np.ndarray, row: int, col: int, value: float) -> None:
        if row == -1 or col == -1 or value == 0.0:
            return
        matrix[row, col] += value
        sparse_matrix = G_sparse if matrix is G else C_sparse if matrix is C else None
        if sparse_matrix is not None:
            sparse_matrix[row, col] = sparse_matrix[row, col] + value

    def stamp(matrix: np.ndarray, node_j: int, node_k: int, value: float) -> None:
        add_entry(matrix, node_j, node_j, value)
        add_entry(matrix, node_k, node_k, value)
        add_entry(matrix, node_j, node_k, -value)
        add_entry(matrix, node_k, node_j, -value)

    def stamp_symbolic_rhs(vector: np.ndarray, index: int, expression: str, positive: bool = True) -> None:
        if index == -1:
            return
        sign = "+" if positive else "-"
        if vector[index, 0] == "0":
            vector[index, 0] = expression if positive else f"-({expression})"
        else:
            vector[index, 0] += f" {sign} ({expression})"

    def stamp_nonlinear_current(vector: np.ndarray, node_p: int, node_n: int, expression: str) -> None:
        stamp_symbolic_rhs(vector, node_p, expression, positive=True)
        stamp_symbolic_rhs(vector, node_n, expression, positive=False)

    def stamp_numeric_rhs(vector: np.ndarray, index: int, value: complex, positive: bool = True) -> None:
        if index == -1:
            return
        vector[index] += value if positive else -value

    def stamp_voltage_source(kv: int, j: int, k: int) -> None:
        add_entry(G, j, kv, 1.0)
        add_entry(G, kv, j, 1.0)
        add_entry(G, k, kv, -1.0)
        add_entry(G, kv, k, -1.0)

    def stamp_vccs_matrix(matrix: np.ndarray, out_p: int, out_n: int, ctrl_p: int, ctrl_n: int, gain: float) -> None:
        if gain == 0.0:
            return
        add_entry(matrix, out_p, ctrl_p, gain)
        add_entry(matrix, out_p, ctrl_n, -gain)
        add_entry(matrix, out_n, ctrl_p, -gain)
        add_entry(matrix, out_n, ctrl_n, gain)

    for component in components:
        comp_type = component.type
        j = get_index(component.node1)
        k = get_index(component.node2)

        if comp_type == "R":
            stamp(G, j, k, 1.0 / parse_value(component.value or "0"))
        elif comp_type == "C":
            stamp(C, j, k, parse_value(component.value or "0"))
            if gmin > 0.0:
                stamp(G, j, k, gmin)
        elif comp_type == "L":
            branch_idx = get_branch_index(component.name)
            stamp_voltage_source(branch_idx, j, k)
            # MNA form is C dx/dt + G x = b. For branch current flowing from
            # node1 to node2, an inductor satisfies v(node1)-v(node2)-L di/dt=0.
            add_entry(C, branch_idx, branch_idx, -parse_value(component.value or "0"))
            controlling_branch_map[component.name] = branch_idx
        elif comp_type == "I":
            if (component.subtype or "").upper() == "DC":
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
                stamp_symbolic_rhs(b_time_str, j, expression, positive=True)
                stamp_symbolic_rhs(b_time_str, k, expression, positive=False)
            else:
                expression = source_expression(component)
                stamp_symbolic_rhs(b_time_str, j, expression, positive=True)
                stamp_symbolic_rhs(b_time_str, k, expression, positive=False)
        elif comp_type == "V":
            branch_idx = get_branch_index(component.name)
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            if (component.subtype or "").upper() == "DC":
                b_dc[branch_idx] += parse_value(component.value or "0")
            elif (component.subtype or "").upper() == "AC":
                stamp_numeric_rhs(b_ac, branch_idx, parse_value(component.value or "0"), positive=True)
            elif (component.subtype or "").upper() in {"SIN", "COS"}:
                phase = 0.0 if (component.subtype or "").upper() == "COS" else -np.pi / 2.0
                phasor = parse_value(component.value or "0") * np.exp(1j * phase)
                stamp_numeric_rhs(b_ac, branch_idx, phasor, positive=True)
                stamp_symbolic_rhs(b_time_str, branch_idx, source_expression(component), positive=True)
            else:
                stamp_symbolic_rhs(b_time_str, branch_idx, source_expression(component), positive=True)
        elif comp_type == "VCCS":
            c1 = get_index(component.ctrl_node1)
            c2 = get_index(component.ctrl_node2)
            gain = parse_value(component.value or "0")
            stamp_vccs_matrix(G, j, k, c1, c2, gain)
        elif comp_type == "VCVS":
            branch_idx = get_branch_index(component.name)
            c1 = get_index(component.ctrl_node1)
            c2 = get_index(component.ctrl_node2)
            gain = parse_value(component.value or "0")
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            if c1 != -1:
                add_entry(G, branch_idx, c1, -gain)
            if c2 != -1:
                add_entry(G, branch_idx, c2, gain)
        elif comp_type == "CCCS":
            control_branch = get_branch_index(component.ctrl_source or "")
            gain = parse_value(component.value or "0")
            add_entry(G, j, control_branch, gain)
            add_entry(G, k, control_branch, -gain)
        elif comp_type == "CCVS":
            branch_idx = get_branch_index(component.name)
            control_branch = get_branch_index(component.ctrl_source or "")
            gain = parse_value(component.value or "0")
            stamp_voltage_source(branch_idx, j, k)
            controlling_branch_map[component.name] = branch_idx
            add_entry(G, branch_idx, control_branch, -gain)
        elif comp_type == "D":
            j_expr = f"x[{j}]" if j != -1 else "0"
            k_expr = f"x[{k}]" if k != -1 else "0"
            diode_expr = diode_current_expression(component, j_expr, k_expr)
            stamp_nonlinear_current(f_str, j, k, diode_expr)
            stamp_nonlinear_current(solver_f_str, j, k, diode_expr)
            if gmin > 0.0:
                stamp(G, j, k, gmin)
        elif comp_type in {"QNPN", "QPNP"}:
            base = get_index(component.ctrl_node1)
            emitter = k
            collector = j
            if component.metadata.get("model") == "level1":
                cje = parse_value(component.metadata.get("cje", "4p"))
                cjc = parse_value(component.metadata.get("cjc", "2p"))
                level1_bjt_devices.append(
                    {
                        "type": comp_type,
                        "collector": collector,
                        "base": base,
                        "emitter": emitter,
                        "is": parse_value(component.value or "1e-15"),
                        "bf": parse_value(component.value2 or "150"),
                        "br": parse_value(component.value3 or "3"),
                        "vaf": parse_value(component.metadata.get("vaf", "100")),
                        "var": parse_value(component.metadata.get("var", "25")),
                        "rb": parse_value(component.metadata.get("rb", "50")),
                        "re": parse_value(component.metadata.get("re", "0.5")),
                        "rc": parse_value(component.metadata.get("rc", "5")),
                    }
                )
                if cje > 0.0:
                    stamp(C, base, emitter, cje)
                if cjc > 0.0:
                    stamp(C, base, collector, cjc)
                if gmin > 0.0:
                    stamp(G, base, emitter, gmin)
                    stamp(G, base, collector, gmin)
                continue
            gm = parse_value(component.value or "40m")
            r_pi = parse_value(component.value2 or "2.5k")
            r_o = parse_value(component.value3 or "100k")
            c_pi = parse_value(component.metadata.get("cpi", "8p"))
            c_mu = parse_value(component.metadata.get("cmu", "3p"))
            c_cs = parse_value(component.metadata.get("ccs", "0"))
            stamp(G, base, emitter, 1.0 / r_pi)
            if r_o > 0.0:
                stamp(G, collector, emitter, 1.0 / r_o)
            if c_pi > 0.0:
                stamp(C, base, emitter, c_pi)
            if c_mu > 0.0:
                stamp(C, base, collector, c_mu)
            if c_cs > 0.0:
                stamp(C, collector, emitter, c_cs)
            # Hybrid-pi transconductance: gm * Vbe flowing collector -> emitter.
            stamp_vccs_matrix(G, collector, emitter, base, emitter, gm)
        elif comp_type in {"NMOS", "PMOS"}:
            gate = get_index(component.ctrl_node1)
            source = k
            drain = j
            if component.metadata.get("model") == "level1":
                drain_expr = f"x[{drain}]" if drain != -1 else "0"
                gate_expr = f"x[{gate}]" if gate != -1 else "0"
                source_expr = f"x[{source}]" if source != -1 else "0"
                mos_expr = mos_level1_current_expression(component, drain_expr, gate_expr, source_expr)
                c_gs = parse_value(component.metadata.get("cgs", "0"))
                c_gd = parse_value(component.metadata.get("cgd", "0"))
                g_ds_floor = parse_value(component.metadata.get("gds_floor", "1e-12"))
                stamp_nonlinear_current(f_str, drain, source, mos_expr)
                level1_mos_devices.append(
                    {
                        "type": comp_type,
                        "drain": drain,
                        "gate": gate,
                        "source": source,
                        "beta": parse_value(component.value or "1m"),
                        "vth": abs(parse_value(component.value2 or "0.4")),
                        "lambda": parse_value(component.value3 or "0"),
                    }
                )
                if g_ds_floor > 0.0:
                    stamp(G, drain, source, g_ds_floor)
                if c_gs > 0.0:
                    stamp(C, gate, source, c_gs)
                if c_gd > 0.0:
                    stamp(C, gate, drain, c_gd)
                if gmin > 0.0:
                    stamp(G, drain, source, gmin)
                continue
            gm = parse_value(component.value or "5m")
            r_o = parse_value(component.value2 or "50k")
            c_gs = parse_value(component.value3 or "5p")
            c_gd = parse_value(component.metadata.get("cgd", "1p"))
            g_mb = parse_value(component.metadata.get("gmb", "0"))
            c_bs = parse_value(component.metadata.get("cbs", "0"))
            c_bd = parse_value(component.metadata.get("cbd", "0"))
            if r_o > 0.0:
                stamp(G, drain, source, 1.0 / r_o)
            if c_gs > 0.0:
                stamp(C, gate, source, c_gs)
            if c_gd > 0.0:
                stamp(C, gate, drain, c_gd)
            if c_bs > 0.0:
                stamp(C, source, -1, c_bs)
            if c_bd > 0.0:
                stamp(C, drain, -1, c_bd)
            stamp_vccs_matrix(G, drain, source, gate, source, gm)
            # Body is implicit ground for the current three-pin symbol. gmb is
            # present for richer models but defaults to zero.
            stamp_vccs_matrix(G, drain, source, -1, source, g_mb)
        elif comp_type == "B":
            expression = component.value or "0"
            stamp_nonlinear_current(f_str, j, k, expression)
            stamp_nonlinear_current(solver_f_str, j, k, expression)
        else:
            error_handler(f"NETLIST_FATAL: Unsupported device type '{comp_type}'.")

    labels = format_voltage_labels(index_map.node_names, index_map.branch_names)
    G_csr = G_sparse.tocsr()
    C_csr = C_sparse.tocsr()
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

    return MnaProblem(
        G=G,
        C=C,
        G_sparse=G_csr,
        C_sparse=C_csr,
        f_str=f_str,
        solver_f_str=solver_f_str,
        b_dc=b_dc,
        b_ac=b_ac,
        b_time_str=b_time_str,
        index_map=index_map,
        components=components,
        level1_mos_devices=level1_mos_devices,
        level1_bjt_devices=level1_bjt_devices,
        gmin=gmin,
        metadata=metadata,
    )
