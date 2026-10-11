"""The fast parser/builder front end (phase P6a) against the frozen oracle.

``parse_netlist_text`` (tail memo, memoized pattern checks, first-letter type
lookup), ``build_mna_problem`` (inline COO stamping, piece-list symbolic rows,
memoized Level-1 MOS templates) and the memoized ``parse_value`` must give the
same results as the pure-Python reference ``mna_reference``:

* every corpus case and benchmark workload: an equal ``CircuitIR`` (same records
  in the same order, same field values and types) and, at gmin 0 and 1e-9,
  bitwise-equal dense ``G``/``C``/``b_dc``/``b_ac``, equal symbolic vectors,
  metadata, components and Level-1 device lists; the native circuit's pattern
  and ``G``/``C``/``gmin`` values against dense reconstructions, and the tables
  handed to ``native.circuit.build_circuit`` against the oracle's devices;
* ``parse_value`` on ~200 literals: accepted values bitwise equal, rejects with
  the same exception class, code and message;
* malformed netlists: the same exception class, code and message.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from corpus import iter_cases
from mna_reference.mna_builder import build_mna_problem as reference_build
from mna_reference.netlist import parse_netlist_text as reference_parse
from mna_reference.utils import parse_value as reference_parse_value

import mna_simulation.native.circuit as native_circuit
from mna_simulation.errors import CircuitSimulatorError
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.netlist import parse_netlist_text
from mna_simulation.utils import parse_value

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import workload_netlists as wl  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _netlists() -> list[tuple[str, str]]:
    items = [(f"corpus:{case.name}", case.netlist_text) for case in iter_cases()]
    items += [
        ("workload:sram_10x10", wl.sram_netlist()),
        ("workload:rlc_mesh_23x23", wl.rlc_mesh_netlist()),
        ("workload:bjt_three_stage", wl.BJT_THREE_STAGE),
        ("workload:hb_close_tone_mixer", wl.HB_CLOSE_TONE_MIXER),
    ]
    return items


NETLISTS = _netlists()


# ---------------------------------------------------------------------------
# Canonical, type-aware views (the oracle's classes are distinct types)
# ---------------------------------------------------------------------------


def _canon(value):
    """Nested structure that compares equal only for equal values of equal types (floats bitwise)."""

    if value is None or isinstance(value, (bool, str)):
        return (type(value).__name__, value)
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, value.value)
    if isinstance(value, (int, np.integer)):
        return (type(value).__name__, int(value))
    if isinstance(value, (float, np.floating)):
        return (type(value).__name__, float(value).hex())
    if isinstance(value, (complex, np.complexfloating)):
        return (type(value).__name__, value.real.hex(), value.imag.hex())
    if isinstance(value, dict):
        return ("dict", [(_canon(k), _canon(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_canon(v) for v in value])
    if dataclasses.is_dataclass(value):
        return (type(value).__name__, [(f.name, _canon(getattr(value, f.name))) for f in dataclasses.fields(value)])
    raise TypeError(f"no canonical form for {type(value)!r}")


def _error_view(exc: BaseException) -> tuple[str, str | None, str]:
    return type(exc).__name__, getattr(exc, "code", None), str(exc)


def _assert_bitwise(current: np.ndarray, reference: np.ndarray, what: str) -> None:
    current = np.asarray(current)
    reference = np.asarray(reference)
    assert current.dtype == reference.dtype, what
    assert current.shape == reference.shape, what
    assert np.ascontiguousarray(current).tobytes() == np.ascontiguousarray(reference).tobytes(), what


def _assert_same_objects(current: np.ndarray, reference: np.ndarray, what: str) -> None:
    assert current.dtype == reference.dtype == object, what
    assert current.shape == reference.shape, what
    assert [_canon(v) for v in current.reshape(-1).tolist()] == [_canon(v) for v in reference.reshape(-1).tolist()], what


def _outcome(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - errors are compared
        return None, exc


# ---------------------------------------------------------------------------
# (a) Parse and build parity on the corpus and the workloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "text"), NETLISTS, ids=[name for name, _ in NETLISTS])
def test_parse_matches_reference(name: str, text: str) -> None:
    reference, reference_error = _outcome(reference_parse, text)
    current, current_error = _outcome(parse_netlist_text, text)
    if reference_error is not None or current_error is not None:
        assert reference_error is not None and current_error is not None, (reference_error, current_error)
        assert _error_view(current_error) == _error_view(reference_error)
        return
    assert _canon(current.components) == _canon(reference.components)
    assert [type(record).__name__ for record in current.components] == ["ComponentRecord"] * len(current.components)
    assert [(d.mode.value, _canon(d.params), d.raw_line) for d in current.directives] == [
        (d.mode.value, _canon(d.params), d.raw_line) for d in reference.directives
    ]
    assert current.source_text == reference.source_text
    # Records reusing a parsed parameter tail still own their metadata dict.
    metadata_ids = [id(record.metadata) for record in current.components]
    assert len(set(metadata_ids)) == len(metadata_ids)


def _unit_gmin_dense(problem) -> np.ndarray:
    """Unit gmin stamps of the oracle problem: across C, D, Level-1 BJT b-e / b-c, Level-1 MOS d-s."""

    size = len(problem.index_map.node_names) + len(problem.index_map.branch_names)
    index = problem.index_map.node_to_index
    dense = np.zeros((size, size))

    def node(name):
        return -1 if name in (None, "0") else index[name]

    def stamp(j: int, k: int) -> None:
        for row, col, value in ((j, j, 1.0), (k, k, 1.0), (j, k, -1.0), (k, j, -1.0)):
            if row != -1 and col != -1:
                dense[row, col] += value

    for component in problem.components:
        j, k = node(component.node1), node(component.node2)
        level1 = component.metadata.get("model") == "level1"
        if component.type in ("C", "D"):
            stamp(j, k)
        elif component.type in ("QNPN", "QPNP") and level1:
            base = node(component.ctrl_node1)
            stamp(base, k)
            stamp(base, j)
        elif component.type in ("NMOS", "PMOS") and level1:
            stamp(j, k)
    return dense


def _dense_from_pattern(col_ptr: np.ndarray, row_idx: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    dense = np.zeros((size, size))
    for col in range(size):
        for slot in range(int(col_ptr[col]), int(col_ptr[col + 1])):
            dense[int(row_idx[slot]), col] = values[slot]
    return dense


def _dense_from_triplets(table: dict[str, np.ndarray], size: int) -> np.ndarray:
    dense = np.zeros((size, size))
    np.add.at(dense, (table["rows"], table["cols"]), table["values"])  # sequential, in stamping order
    return dense


def _expected_device_tables(reference_problem):
    """The tables ``build_circuit`` must receive, derived from the oracle problem."""

    index = reference_problem.index_map.node_to_index

    def node(name):
        return -1 if name in (None, "0") else index[name]

    tables = {}
    mos = reference_problem.level1_mos_devices
    if mos:
        tables["mos"] = {
            "drain": np.asarray([d["drain"] for d in mos], dtype=np.int64),
            "gate": np.asarray([d["gate"] for d in mos], dtype=np.int64),
            "source": np.asarray([d["source"] for d in mos], dtype=np.int64),
            "beta": np.asarray([d["beta"] for d in mos], dtype=np.float64),
            "vth": np.asarray([d["vth"] for d in mos], dtype=np.float64),
            "lambda": np.asarray([d["lambda"] for d in mos], dtype=np.float64),
            "sign": np.asarray([-1 if d["type"] == "PMOS" else 1 for d in mos], dtype=np.int32),
        }
    bjt = reference_problem.level1_bjt_devices
    if bjt:
        tables["bjt"] = {
            key: np.asarray([d[key] for d in bjt], dtype=np.int64) for key in ("collector", "base", "emitter")
        } | {key: np.asarray([d[key] for d in bjt], dtype=np.float64) for key in ("is", "bf", "br", "vaf", "var")}
        tables["bjt"]["sign"] = np.asarray([-1 if d["type"] == "QPNP" else 1 for d in bjt], dtype=np.int32)
    diodes = [c for c in reference_problem.components if c.type == "D"]
    if diodes:
        tables["diodes"] = {
            "anode": np.asarray([node(c.node1) for c in diodes], dtype=np.int64),
            "cathode": np.asarray([node(c.node2) for c in diodes], dtype=np.int64),
            "is": np.asarray([reference_parse_value(c.value or "1e-15") for c in diodes], dtype=np.float64),
        }
    behavioral = [c for c in reference_problem.components if c.type == "B"]
    if behavioral:
        tables["behavioral"] = {
            "node_p": np.asarray([node(c.node1) for c in behavioral], dtype=np.int64),
            "node_n": np.asarray([node(c.node2) for c in behavioral], dtype=np.int64),
        }
    return tables, diodes, behavioral


@pytest.mark.parametrize(("name", "text"), NETLISTS, ids=[name for name, _ in NETLISTS])
def test_build_matches_reference(name: str, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        reference_ir = reference_parse(text)
    except Exception:  # noqa: BLE001 - parse errors are compared in test_parse_matches_reference
        pytest.skip("netlist does not parse")
    current_ir = parse_netlist_text(text)

    specs: list[dict] = []
    real_build_circuit = native_circuit.build_circuit

    def capture(n, **tables):
        specs.append({"n": n, **tables})
        return real_build_circuit(n, **tables)

    monkeypatch.setattr(native_circuit, "build_circuit", capture)
    references = {gmin: _outcome(reference_build, reference_ir, gmin=gmin) for gmin in (0.0, 1e-9)}
    reference_at_0 = references[0.0][0]
    for gmin, (reference, reference_error) in references.items():
        specs.clear()
        problem, current_error = _outcome(build_mna_problem, current_ir, gmin=gmin)
        if reference_error is not None or current_error is not None:
            assert reference_error is not None and current_error is not None, (reference_error, current_error)
            if name == "corpus:zero_resistor":
                # CPP_MIGRATION_PLAN.md 2.4.1 ruling, applied in the builder (the oracle keeps ZeroDivisionError).
                assert _error_view(current_error) == ("NetlistError", "NETLIST_FATAL", "'R1' has zero resistance")
                assert type(reference_error) is ZeroDivisionError
            else:
                assert _error_view(current_error) == _error_view(reference_error)
            continue

        for field_name in ("G", "C", "b_dc", "b_ac"):
            _assert_bitwise(getattr(problem, field_name), getattr(reference, field_name), f"{name} {field_name} gmin={gmin}")
        for field_name in ("f_str", "solver_f_str", "b_time_str"):
            _assert_same_objects(getattr(problem, field_name), getattr(reference, field_name), f"{name} {field_name}")
        assert _canon(problem.metadata) == _canon(reference.metadata)
        assert _canon(problem.components) == _canon(reference.components)
        assert _canon(problem.level1_mos_devices) == _canon(reference.level1_mos_devices)
        assert _canon(problem.level1_bjt_devices) == _canon(reference.level1_bjt_devices)
        assert _canon(problem.index_map) == _canon(reference.index_map)
        assert problem.gmin == reference.gmin
        for csr_name, dense_name in (("G_sparse", "G"), ("C_sparse", "C")):
            matrix = getattr(problem, csr_name)
            assert matrix.format == "csr" and matrix.has_canonical_format
            _assert_bitwise(matrix.toarray(), getattr(reference, dense_name), f"{name} {csr_name}")

        expected_tables, diodes, behavioral = _expected_device_tables(reference)
        assert _canon(problem.diode_devices) == _canon(
            [
                {"name": c.name, "anode": int(a), "cathode": int(k), "is": float(i)}
                for c, a, k, i in zip(
                    diodes,
                    expected_tables.get("diodes", {}).get("anode", []),
                    expected_tables.get("diodes", {}).get("cathode", []),
                    expected_tables.get("diodes", {}).get("is", []),
                )
            ]
        )
        assert [d["name"] for d in problem.behavioral_devices] == [c.name for c in behavioral]
        assert [d["expression"] for d in problem.behavioral_devices] == [c.value or "0" for c in behavioral]

        circuit = problem.native
        if circuit is None:
            # A behavioral expression the native VM rejects: no circuit (the error surfaces at analysis time).
            assert behavioral
            continue
        size = reference.G.shape[0]
        assert circuit.n == size and len(specs) == 1
        spec = specs[0]
        g_static = reference_at_0.G  # the native G excludes the gmin stamps
        unit_gmin = _unit_gmin_dense(reference)
        _assert_bitwise(_dense_from_triplets(spec["G"], size), g_static, f"{name} spec G")
        _assert_bitwise(_dense_from_triplets(spec["C"], size), reference.C, f"{name} spec C")
        _assert_bitwise(_dense_from_triplets(spec["gmin_stamps"], size), unit_gmin, f"{name} spec gmin")
        for key in ("mos", "bjt", "diodes", "behavioral"):
            table = spec.get(key)
            if key not in expected_tables:
                assert table is None, key
                continue
            for column, expected in expected_tables[key].items():
                _assert_bitwise(table[column], expected, f"{name} {key}.{column}")
        if "behavioral" in expected_tables:
            assert len(spec["behavioral"]["programs"]) == len(behavioral)

        col_ptr, row_idx = circuit.pattern()
        assert col_ptr.shape == (size + 1,) and col_ptr[0] == 0 and col_ptr[-1] == row_idx.size == circuit.nnz
        assert np.all(np.diff(col_ptr) >= 0)
        for col in range(size):
            rows = row_idx[col_ptr[col] : col_ptr[col + 1]]
            assert np.all(np.diff(rows) > 0)  # sorted, unique
        for kind, expected in (("G", g_static), ("C", reference.C), ("gmin", unit_gmin)):
            values = circuit.values(kind)
            _assert_bitwise(_dense_from_pattern(col_ptr, row_idx, values, size), expected, f"{name} native {kind}")
            _assert_bitwise(circuit.to_dense(kind), expected, f"{name} native to_dense {kind}")
        counts = circuit.device_counts
        assert counts == {
            "diode": len(diodes),
            "mos": len(reference.level1_mos_devices),
            "bjt": len(reference.level1_bjt_devices),
            "behavioral": len(behavioral),
        }


def test_problem_components_are_fresh_copies() -> None:
    """``build_mna_problem`` keeps ``dataclasses.replace`` semantics: new records, shared metadata."""

    ir = parse_netlist_text("GND vss\nV1 a vss DC 1\nR1 a b 1k\nM1 b a vss NMOS LEVEL1 1m 0.4 0.02 2f 1f\n")
    problem = build_mna_problem(ir)
    kept = [c for c in ir.components if c.type != "GND"]
    assert all(new is not old for new, old in zip(problem.components, kept))
    assert all(new.metadata is old.metadata for new, old in zip(problem.components, kept))
    assert [c.node2 for c in problem.components] == ["0", "b", "0"]
    assert ir.components[1].node2 == "vss"


def test_mos_template_memo_shares_parameter_sets() -> None:
    from mna_simulation.device_models import mos_level1_current_template

    mos_level1_current_template.cache_clear()
    build_mna_problem(parse_netlist_text(wl.sram_netlist()))
    info = mos_level1_current_template.cache_info()
    # The 622 MOS of the 10x10 SRAM share six parameter sets: PMOS 3m / 8m / 0.7m, NMOS 8m / 1.8m / 2.5m.
    assert info.currsize == 6
    assert info.misses == 6


# ---------------------------------------------------------------------------
# (b) parse_value table
# ---------------------------------------------------------------------------

_SUFFIXES = ["", "g", "G", "meg", "MEG", "Meg", "mEg", "k", "K", "m", "M", "u", "U", "n", "N", "p", "P", "f", "F", "%"]
_MANTISSAS = ["1", "2.5", "-3", "+.5", "7.", "1e3", "1.5E-2", "-0", "0", "6.02e+23"]
_ACCEPTED = [m + s for m in _MANTISSAS for s in _SUFFIXES if (m, s) != ("6.02e+23", "%")]
_ACCEPTED += [
    " 1k ",
    "\t2.5m\n",
    "1_k",
    "1_%",
    "1_meg",
    "1e3k",
    "1E-3M",
    "1e999",
    "-1e999",
    "1e-999",
    "١٢",
    "١٢k",
    "000123",
    "0.000",
    ".5e-3",
    "5.e2",
    "123456789012345678901234567890",
    "  -0.0  ",
    "1e0",
    "4f",
    "200Meg",
]
_REJECTED = [
    "",
    " ",
    "k",
    "1kohm",
    "1 k",
    "1e",
    "e3",
    "1.2.3",
    "--1",
    "1k5",
    "nan",
    "inf",
    "-inf",
    "NaN",
    "1T",
    "1a",
    "1MA",
    "1mil",
    "1mA",
    "5V",
    "1Hz",
    "1µ",
    "1__k",
    "_1",
    "1,5",
    "0x10",
    "1/2",
    "1e3.5",
    "+-1",
    "1%%",
    "1e+",
    ".",
    "-",
    "1_000",
]
_ACCEPTED.append("1 ")


def test_parse_value_table_is_large_enough() -> None:
    assert len(_ACCEPTED) + len(_REJECTED) >= 200


@pytest.mark.parametrize("literal", _ACCEPTED)
def test_parse_value_accepts_like_the_reference(literal: str) -> None:
    expected = reference_parse_value(literal)
    for _ in range(2):  # the second call is served by the memo
        value = parse_value(literal)
        assert type(value) is float
        assert value.hex() == expected.hex() or (math.isnan(value) and math.isnan(expected))


@pytest.mark.parametrize("literal", _REJECTED)
def test_parse_value_rejects_like_the_reference(literal: str) -> None:
    with pytest.raises(Exception) as reference_exc:  # the oracle's own NetlistError class
        reference_parse_value(literal)
    for _ in range(2):  # errors are never memoized
        with pytest.raises(CircuitSimulatorError) as current_exc:
            parse_value(literal)
        assert _error_view(current_exc.value) == _error_view(reference_exc.value)


@pytest.mark.parametrize("value", [1, 0, -7, 2.5, -0.0, True, np.float64(3.25), float("inf")])
def test_parse_value_numbers_like_the_reference(value) -> None:
    result = parse_value(value)
    expected = reference_parse_value(value)
    assert type(result) is float and result.hex() == expected.hex()


def test_parse_value_non_strings_raise_like_the_reference() -> None:
    for value in (None, np.int64(3), ["1k"]):
        with pytest.raises(Exception) as reference_exc:
            reference_parse_value(value)
        with pytest.raises(Exception) as current_exc:
            parse_value(value)
        assert type(current_exc.value) is type(reference_exc.value)
        assert str(current_exc.value) == str(reference_exc.value)


def test_parse_value_memo_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from mna_simulation import utils

    monkeypatch.setattr(utils, "_VALUE_CACHE_LIMIT", 8)
    utils._VALUE_CACHE.clear()
    for index in range(20):
        assert parse_value(f"{index}k") == reference_parse_value(f"{index}k")
        assert len(utils._VALUE_CACHE) <= 8


# ---------------------------------------------------------------------------
# (c) Error parity on malformed netlists
# ---------------------------------------------------------------------------

_MALFORMED = {
    "bad_value": "V1 a 0 DC 1\nR1 a b 1k\nR2 b 0 bad\n",
    "too_few_parts": "V1 a 0 DC 1\nR1 a 0\n",
    "bad_subtype": "V1 a 0 XX 1\nR1 a 0 1k\n",
    "bad_duty_cycle": "V1 a 0 STEP 1 1k 2x\nR1 a 0 1k\n",
    "digit_name": "V1 a 0 DC 1\n1R a 0 1k\n",
    "unknown_prefix": "V1 a 0 DC 1\nZfoo a 0 1k\n",
    "invalid_explicit_type": "V1 a 0 DC 1\nfoo:ZZ a 0 1k\n",
    "unicode_prefix": "V1 a 0 DC 1\nR1 a b 1k\nſ1 b 0 1k\n",
    "bad_node": "V1 a 0 DC 1\nR1 a b@ 1k\n",
    "same_nodes_after_memo": "V1 a 0 DC 1\nR1 a b 1k\nR2 b c 1k\nR3 c c 1k\n",
    "bad_node_after_memo": "V1 a 0 DC 1\nR1 a b 1k\nR2 b c@ 1k\n",
    "mos_drain_is_source_after_memo": (
        "VDD vdd 0 DC 1\nM1 d g s NMOS LEVEL1 1m 0.4 0.02 2f 1f\nM2 d2 g2 d2 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"
    ),
    "mos_bad_gate_after_memo": (
        "VDD vdd 0 DC 1\nM1 d g s NMOS LEVEL1 1m 0.4 0.02 2f 1f\nM2 d2 g@ s2 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"
    ),
    "mos_node_order_after_memo": (
        "VDD vdd 0 DC 1\nM1 d g s NMOS LEVEL1 1m 0.4 0.02 2f 1f\nM2 d@ g@ s2 NMOS LEVEL1 1m 0.4 0.02 2f 1f\n"
    ),
    "bjt_collector_is_emitter_after_memo": "VCC vcc 0 DC 5\nQ1 c b e QNPN LEVEL1\nQ2 c2 b e QNPN LEVEL1\nQ3 c3 b c3 QNPN LEVEL1\n",
    "diode_too_few_after_memo": "V1 a 0 DC 1\nD1 a b\nD2 b c\nD3 c\n",
    "mos_bad_param": "V1 a 0 DC 1\nM1 d g s NMOS LEVEL1 1m 0.4 0.02 2f bad\n",
    "bjt_too_many_params": "V1 a 0 DC 1\nQ1 c b e QNPN LEVEL1 1 2 3 4 5 6 7 8 9 10 11\n",
    "gnd_arity": "V1 a 0 DC 1\nGND a b\n",
    "parallel_sources": "V1 a 0 DC 1\nV2 a 0 DC 2\n",
    "series_current_sources": "V1 a 0 DC 1\nI1 a b DC 1m\nI2 b 0 DC 1m\n",
    "bad_tran": "V1 a 0 DC 1\nR1 a 0 1k\n.tran 1x\n",
    "unsupported_directive": "V1 a 0 DC 1\nR1 a 0 1k\n.noise\n",
    "duplicate_branch": "V1 a 0 DC 1\nV1 b 0 DC 2\nR1 a b 1k\n",
    "unknown_controlling_branch": "V1 a 0 DC 1\nR1 a 0 1k\nF1 a 0 VX 2\n",
    "empty": "* nothing\n.op\n",
}


@pytest.mark.parametrize("netlist", list(_MALFORMED.values()), ids=list(_MALFORMED))
def test_malformed_netlists_raise_like_the_reference(netlist: str) -> None:
    def run(parse, build):
        return build(parse(netlist))

    _, reference_error = _outcome(run, reference_parse, reference_build)
    _, current_error = _outcome(run, parse_netlist_text, build_mna_problem)
    assert reference_error is not None and current_error is not None
    assert _error_view(current_error) == _error_view(reference_error)
