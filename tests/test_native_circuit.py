"""Circuit compilation: the COO-triplet builder and the native compiled circuit.

``mna_simulation.mna_builder.build_mna_problem`` must reproduce every observable
field of the frozen reference builder on the same ``CircuitIR`` (parsed by
``mna_reference.netlist.parse_netlist_text``): dense ``G``/``C`` and the CSR
views bit for bit (duplicates are summed in stamping order), ``b_dc``/``b_ac``,
the expression strings, labels, index map and metadata. The native circuit
(``problem.native``) holds the gmin-free ``G``, ``C`` and the unit gmin stamps on
the union pattern of all device Jacobian footprints.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
from corpus import iter_cases, rlc_mesh_netlist
from mna_reference.mna_builder import build_mna_problem as reference_build
from mna_reference.netlist import parse_netlist_text as reference_parse

from mna_simulation import native, solvers
from mna_simulation.errors import NetlistError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.netlist import parse_netlist_text

core = native.require_core()

CASES = list(iter_cases())


def _reference_ir(case):
    try:
        return reference_parse(case.netlist_text)
    except Exception:  # noqa: BLE001 - parser errors are covered by the differential suite
        pytest.skip("netlist does not parse")


def _assert_same_array(current: np.ndarray, reference: np.ndarray) -> None:
    current = np.asarray(current)
    reference = np.asarray(reference)
    assert current.shape == reference.shape
    assert current.dtype == reference.dtype
    np.testing.assert_array_equal(current, reference)  # NaN == NaN, otherwise exact


def _pattern_entries(circuit) -> dict[int, tuple[int, int]]:
    col_ptr, row_idx = circuit.pattern()
    entries = {}
    for col in range(circuit.n):
        for slot in range(col_ptr[col], col_ptr[col + 1]):
            entries[slot] = (int(row_idx[slot]), col)
    return entries


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize("gmin", [0.0, 1e-9])
def test_builder_matches_reference_on_corpus(case, gmin: float) -> None:
    ir = _reference_ir(case)
    try:
        reference = reference_build(ir, gmin=gmin)
    except Exception as exc:  # noqa: BLE001
        with pytest.raises(Exception) as current_exc:
            build_mna_problem(ir, gmin=gmin)
        if case.current_error is not None:
            # An applied 2.4.1 ruling changed the live error (corpus.Case.current_error).
            raised = current_exc.value
            assert (type(raised).__name__, getattr(raised, "code", None)) == case.current_error
            return
        assert type(current_exc.value).__name__ == type(exc).__name__
        assert str(current_exc.value) == str(exc)
        return
    problem = build_mna_problem(ir, gmin=gmin)

    for name in ("G", "C", "b_dc", "b_ac"):
        _assert_same_array(getattr(problem, name), getattr(reference, name))
    for name in ("G_sparse", "C_sparse"):
        current, ref = getattr(problem, name), getattr(reference, name)
        assert current.format == "csr" and current.shape == ref.shape
        assert current.nnz == ref.nnz
        _assert_same_array(current.toarray(), ref.toarray())
    for name in ("f_str", "solver_f_str", "b_time_str"):
        _assert_same_array(getattr(problem, name), getattr(reference, name))
    assert problem.metadata == reference.metadata
    assert problem.metadata["labels"] == reference.metadata["labels"]
    assert dataclasses.asdict(problem.index_map) == dataclasses.asdict(reference.index_map)
    assert problem.level1_mos_devices == reference.level1_mos_devices
    assert problem.level1_bjt_devices == reference.level1_bjt_devices
    assert problem.gmin == reference.gmin
    assert [c.name for c in problem.components] == [c.name for c in reference.components]

    circuit = problem.native
    assert circuit is not None
    size = problem.G.shape[0]
    assert circuit.n == size
    # Native G excludes gmin: G == native G + gmin * unit stamps (exact at gmin = 0; at gmin > 0
    # the reference adds gmin once per stamp, so a node with k stamps differs by ~k ulps).
    effective = circuit.to_dense("G") + gmin * circuit.to_dense("gmin")
    if gmin == 0.0:
        _assert_same_array(effective, reference.G)
    else:
        np.testing.assert_allclose(effective, reference.G, rtol=1e-13, atol=0.0)
        g_eff = np.zeros((size, size))
        for slot, (row, col) in _pattern_entries(circuit).items():
            g_eff[row, col] = circuit.g_effective(gmin)[slot]
        np.testing.assert_array_equal(g_eff, effective)
    _assert_same_array(circuit.to_dense("C"), reference.C)
    # Every static nonzero lies on the pattern.
    for matrix in (reference.G, reference.C):
        for row, col in zip(*np.nonzero(matrix)):
            assert circuit.slot(int(row), int(col)) >= 0
    counts = circuit.device_counts
    assert counts["mos"] == len(problem.level1_mos_devices)
    assert counts["bjt"] == len(problem.level1_bjt_devices)
    assert counts["diode"] == len(problem.diode_devices) == sum(c.type == "D" for c in problem.components)
    assert counts["behavioral"] == len(problem.behavioral_devices)
    assert circuit.has_nonlinear == (sum(counts.values()) > 0)
    assert solvers.is_nonlinear(problem.solver_f_str) == (counts["diode"] + counts["behavioral"] > 0)


def test_native_pattern_and_device_slots() -> None:
    netlist = (
        "V1 vdd 0 DC 1.2\n"
        "R1 vdd a 1k\n"
        "M1 a g 0 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"
        "D1 a b 1e-14\n"
        "C1 b 0 1p\n"
        "Q1 b g 0 QNPN LEVEL1 1e-15 150 3\n"
        "VG g 0 DC 0.8\n"
    )
    problem = build_mna_problem(parse_netlist_text(netlist))
    circuit = problem.native
    index = problem.index_map.node_to_index
    a, b, g, vdd = index["a"], index["b"], index["g"], index["vdd"]
    col_ptr, row_idx = circuit.pattern()
    assert col_ptr.dtype == np.int32 and row_idx.dtype == np.int32
    assert col_ptr.shape == (circuit.n + 1,) and row_idx.shape == (circuit.nnz,)
    assert all(np.all(np.diff(row_idx[col_ptr[c] : col_ptr[c + 1]]) > 0) for c in range(circuit.n))
    entries = _pattern_entries(circuit)
    # MOS rows (drain a, source gnd) x cols (a, g, gnd).
    mos_slots = circuit.device_slots("mos")
    assert [entries.get(int(s)) for s in mos_slots] == [(a, a), (a, g), None, None, None, None]
    assert list(mos_slots[2:]) == [-1, -1, -1, -1]
    diode_slots = circuit.device_slots("diode")
    assert [entries[int(s)] for s in diode_slots] == [(a, a), (a, b), (b, a), (b, b)]
    bjt_slots = circuit.device_slots("bjt")  # (c=b, b=g, e=gnd)
    assert entries[int(bjt_slots[0])] == (b, b) and entries[int(bjt_slots[1])] == (b, g)
    assert bjt_slots[2] == -1 and bjt_slots[6] == -1 and bjt_slots[8] == -1
    # gmin stamps exist in the native pattern even at gmin = 0: C1 (b, gnd), D1 (a, b),
    # Q1 base-emitter (g, gnd) and base-collector (g, b), M1 drain-source (a, gnd).
    stamps = circuit.to_dense("gmin")
    assert stamps[b, b] == 3.0
    assert stamps[a, b] == -1.0 and stamps[a, a] == 2.0 and stamps[g, g] == 2.0 and stamps[vdd, vdd] == 0.0
    assert not np.any(problem.G_sparse.toarray() - circuit.to_dense("G"))
    np.testing.assert_array_equal(circuit.values("G"), [circuit.to_dense("G")[r, c] for r, c in entries.values()])
    assert circuit.device_counts == {"diode": 1, "mos": 1, "bjt": 1, "behavioral": 0}


def test_behavioral_sources_compile_to_native_programs() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nB1 b 0 1e-3*x[1]**2+x[0]*1e-6\n"))
    assert problem.behavioral_devices == [
        {"name": "B1", "node_p": 1, "node_n": -1, "expression": "1e-3*x[1]**2+x[0]*1e-6"}
    ]
    circuit = problem.native
    assert circuit.device_counts["behavioral"] == 1
    assert [_pattern_entries(circuit)[int(s)] for s in circuit.device_slots("behavioral")[:2]] == [(1, 0), (1, 1)]
    x = np.array([1.0, 0.5, 0.0])
    f = core.device_f(circuit, x, 0.0)
    assert f[1, 0] == 1e-3 * 0.5**2 + 1.0 * 1e-6 and f[0, 0] == 0.0
    jac = core.device_jacobian_dense(circuit, x, 0.0)
    assert jac[1, 0] == pytest.approx(1e-6, rel=1e-15) and jac[1, 1] == pytest.approx(1e-3, rel=1e-15)


def test_unsupported_behavioral_expression_defers_the_error_to_evaluation() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nB1 b 0 sinh(x[1])\n"))
    assert problem.native is None  # show_matrix keeps working, as in the reference
    assert problem.G[0, 0] == 1e-3
    with pytest.raises(NetlistError) as exc:
        solvers.compile_nl_functions(problem.solver_f_str, problem.G.shape[0])
    assert exc.value.code == "NETLIST_FATAL" and "Unsupported function 'sinh'" in exc.value.message


def test_dense_matrices_are_lazy_and_cached() -> None:
    problem = build_mna_problem(parse_netlist_text("V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\n"))
    assert problem._G_dense is None and problem._C_dense is None  # nothing materialized yet
    G = problem.G
    assert problem.G is G and problem._G_dense is G
    assert G.shape == (3, 3) and len(problem.G) == 3 and np.asarray(problem.G) is G
    assert problem.G[0, 0] == 1e-3 and problem.C[1, 1] == 1e-6
    copy = dataclasses.replace(problem, gmin=0.0)
    assert copy.G is G and copy.native is problem.native
    problem.G = np.eye(3)
    assert problem.G[0, 1] == 0.0
    assert problem.metadata["matrix_storage"] == "dense+csr"


def test_native_build_rejects_malformed_specs() -> None:
    from mna_simulation.native.circuit import build_circuit, triplets

    with pytest.raises(core.CoreError) as exc:
        build_circuit(2, G=triplets([0], [2], [1.0]))
    assert exc.value.code == "NETLIST_FATAL"
    with pytest.raises(core.CoreError):
        build_circuit(-1)
    with pytest.raises(core.CoreError):
        build_circuit(2**31)
    with pytest.raises(core.CoreError):
        core.build_circuit({"n": 2, "mos": {"drain": np.zeros(1, dtype=np.int64)}})
    with pytest.raises(core.CoreError):
        core.build_circuit({"n": 2, "G": {"rows": "0", "cols": np.zeros(1, dtype=np.int64), "values": np.zeros(1)}})
    with pytest.raises(core.CoreError):
        core.build_circuit({"n": 2, "behavioral": {"node_p": np.zeros(1, dtype=np.int64), "node_n": np.zeros(1, dtype=np.int64), "programs": [1]}})
    with pytest.raises(core.CoreError):
        core.build_circuit({"gmin": 0.0})
    empty = build_circuit(0)
    assert empty.n == 0 and empty.nnz == 0 and not empty.has_nonlinear
    assert empty.to_dense("G").shape == (0, 0)
    with pytest.raises(core.CoreError):
        empty.to_dense("X")


def test_builder_time_for_the_23x23_rlc_mesh() -> None:
    ir = reference_parse(rlc_mesh_netlist(23, 23))

    def best(build) -> float:
        times = []
        for _ in range(5):
            start = time.perf_counter()
            build(ir)
            times.append(time.perf_counter() - start)
        return min(times)

    current = best(build_mna_problem)
    reference = best(reference_build)
    print(f"\nbuild_mna_problem 23x23 RLC mesh (n=1035): {current * 1e3:.1f} ms (reference {reference * 1e3:.1f} ms)")
    assert current < 0.030
