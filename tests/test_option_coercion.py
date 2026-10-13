"""Strict request-option coercion in ``PythonBackend.build_options``.

Rulings from CPP_MIGRATION_PLAN.md section 2.4.1 ("runtime work"): booleans and
integers are coerced strictly, time/frequency values accept metric suffixes like
the netlist directives, and ``init_condition`` accepts DEFAULT/RANDOM in any case
or a numeric array/list. Every rejection is a plain ``ValueError`` that the
service tags so the HTTP layer answers 400. ``TRAN_ERROR`` is a ``NetlistError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.server.main as server
from mna_simulation.api.contracts import AnalysisMode, AnalysisRequest
from mna_simulation.api.service import SimulationService, is_invalid_option_error
from mna_simulation.backends.python_backend import PythonBackend
from mna_simulation.errors import NetlistError

RC_TRAN = "V1 a 0 DC 1\nR1 a b 1k\nC1 b 0 1u\n.tran 1m\n"
OP_NETLIST = "V1 n1 0 DC 5\nR1 n1 n2 1k\nR2 n2 0 1k\n.op\n"


@pytest.fixture(scope="module")
def backend() -> PythonBackend:
    return PythonBackend()


@pytest.fixture(scope="module")
def circuit(backend: PythonBackend):
    return backend.parse_text(RC_TRAN)


def _options(backend: PythonBackend, circuit, **overrides):
    return backend.build_options(circuit, overrides=overrides)


def _rejects(backend: PythonBackend, circuit, message: str, **overrides) -> None:
    with pytest.raises(ValueError) as exc:
        backend.build_options(circuit, overrides=overrides)
    assert type(exc.value) is ValueError
    assert str(exc.value) == message


# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------

BOOL_OPTIONS = {
    "use_krylov": "use_krylov",
    "krylov": "use_krylov",
    "use_mor": "use_mor",
    "mor": "use_mor",
    "mor_validate": "mor_validate",
    "breakpoints": "breakpoints",
}
TRUE_FORMS = [True, 1, np.True_, "true", "TRUE", " True ", "yes", "YES", "on", "On", "1", " 1 "]
FALSE_FORMS = [False, 0, np.False_, "false", "False", " FALSE ", "no", "No", "off", "OFF", "0", " 0 "]
BAD_BOOLS = ["maybe", "", "  ", "tru", "y", "n", "enabled", 2, -1, 1.0, 0.0, [], [True], {"a": 1}]


@pytest.mark.parametrize("name", BOOL_OPTIONS)
@pytest.mark.parametrize("value", TRUE_FORMS, ids=repr)
def test_boolean_true_forms(backend, circuit, name: str, value: object) -> None:
    assert getattr(_options(backend, circuit, **{name: value}), BOOL_OPTIONS[name]) is True


@pytest.mark.parametrize("name", BOOL_OPTIONS)
@pytest.mark.parametrize("value", FALSE_FORMS, ids=repr)
def test_boolean_false_forms(backend, circuit, name: str, value: object) -> None:
    assert getattr(_options(backend, circuit, **{name: value}), BOOL_OPTIONS[name]) is False


@pytest.mark.parametrize("name", BOOL_OPTIONS)
@pytest.mark.parametrize("value", BAD_BOOLS, ids=repr)
def test_boolean_rejects_everything_else(backend, circuit, name: str, value: object) -> None:
    _rejects(backend, circuit, f"{name} must be a boolean", **{name: value})


def test_boolean_defaults_and_none(backend, circuit) -> None:
    defaults = _options(backend, circuit)
    assert (defaults.use_krylov, defaults.use_mor, defaults.mor_validate) == (False, False, True)
    assert defaults.breakpoints is True and _options(backend, circuit, breakpoints=None).breakpoints is True
    explicit_none = _options(backend, circuit, use_krylov=None, use_mor=None, mor_validate=None)
    assert (explicit_none.use_krylov, explicit_none.use_mor, explicit_none.mor_validate) == (False, False, True)


def test_boolean_alias_precedence(backend, circuit) -> None:
    # The canonical key wins when both are given, as before; errors name the key that was used.
    assert _options(backend, circuit, use_krylov="off", krylov="on").use_krylov is False
    assert _options(backend, circuit, use_mor="yes", mor="no").use_mor is True
    _rejects(backend, circuit, "use_krylov must be a boolean", use_krylov="nope", krylov=True)


# ---------------------------------------------------------------------------
# Integers
# ---------------------------------------------------------------------------

INT_OPTIONS = {
    "krylov_rank": "krylov_rank",
    "krylov_max_iter": "krylov_max_iter",
    "krylov_restart": "krylov_restart",
    "max_iter": "max_iter",
    "points": "ac_points",
    "harmonics": "hb_harmonics",
    "mor_order": "mor_order",
}
GOOD_INTS = [(3, 3), ("3", 3), (" 3 ", 3), ("+3", 3), (3.0, 3), ("3.0", 3), ("1e1", 10), (np.int64(4), 4), (np.float64(5.0), 5)]
BAD_INTS = [2.7, "2.5", "abc", "1k", "3 x", True, False, float("nan"), float("inf"), "inf", "nan", [3], (3,), {"n": 3}]


@pytest.mark.parametrize("name", INT_OPTIONS)
@pytest.mark.parametrize(("value", "expected"), GOOD_INTS, ids=repr)
def test_integer_accepts_ints_and_integral_floats_and_strings(backend, circuit, name: str, value, expected: int) -> None:
    result = getattr(_options(backend, circuit, **{name: value}), INT_OPTIONS[name])
    assert result == expected and type(result) is int


@pytest.mark.parametrize("name", INT_OPTIONS)
@pytest.mark.parametrize("value", BAD_INTS, ids=repr)
def test_integer_rejects_non_integral_values(backend, circuit, name: str, value: object) -> None:
    _rejects(backend, circuit, f"{name} must be an integer", **{name: value})


def test_integer_none_and_defaults(backend, circuit) -> None:
    defaults = _options(backend, circuit)
    assert (defaults.max_iter, defaults.krylov_max_iter, defaults.krylov_restart) == (10000, 2000, 80)
    assert (defaults.ac_points, defaults.hb_harmonics, defaults.krylov_rank, defaults.mor_order) == (None, None, "auto", "auto")
    explicit_none = _options(
        backend, circuit, max_iter=None, krylov_max_iter=None, points=None, harmonics=None, krylov_rank=None, mor_order=None
    )
    assert (explicit_none.max_iter, explicit_none.krylov_max_iter) == (10000, 2000)
    assert (explicit_none.ac_points, explicit_none.hb_harmonics) == (None, None)
    assert (explicit_none.krylov_rank, explicit_none.mor_order) == ("auto", "auto")


def test_krylov_rank_and_mor_order_keep_auto_and_positive_rules(backend, circuit) -> None:
    for value in ("auto", "AUTO", "  Auto "):
        assert _options(backend, circuit, krylov_rank=value).krylov_rank == "auto"
        assert _options(backend, circuit, mor_order=value).mor_order == "auto"
    assert _options(backend, circuit, mor_order="").mor_order == "auto"  # unchanged: empty means auto
    _rejects(backend, circuit, "krylov_rank must be an integer", krylov_rank="")
    for value in (0, -3, "0", "-2", 0.0):
        _rejects(backend, circuit, "krylov_rank must be 'auto' or a positive integer.", krylov_rank=value)
        _rejects(backend, circuit, "mor_order must be 'auto' or a positive integer.", mor_order=value)


def test_krylov_restart_alone_is_a_manual_rank(backend, circuit) -> None:
    options = _options(backend, circuit, krylov_restart="5.0")
    assert (options.krylov_restart, options.krylov_rank) == (5, 5)
    assert _options(backend, circuit, krylov_restart=5, krylov_rank="auto").krylov_rank == "auto"


def test_integer_zero_and_negative_values_pass_through_unchanged(backend, circuit) -> None:
    # Zero handling is a P3/P4 ruling for the solvers; build_options does not add positivity checks.
    options = _options(backend, circuit, points=0, harmonics="0", max_iter=-1)
    assert (options.ac_points, options.hb_harmonics, options.max_iter) == (0, 0, -1)


# ---------------------------------------------------------------------------
# Floats with metric suffixes
# ---------------------------------------------------------------------------

FLOAT_OPTIONS = {
    "t_stop": "tran_stop",
    "t_step": "tran_step",
    "f_start": "ac_start",
    "f_stop": "ac_stop",
    "time_window": "hb_time_window",
    "krylov_tol": "krylov_tol",
    "v_tol": "v_tol",
    "f_tol": "f_tol",
}
GOOD_FLOATS = [
    (1e-3, 1e-3),
    (2, 2.0),
    (np.float32(0.5), 0.5),
    (np.int64(7), 7.0),
    ("1m", 1e-3),
    (" 1m ", 1e-3),
    ("1M", 1e-3),  # SPICE: M is milli
    ("2meg", 2e6),
    ("2MEG", 2e6),
    ("10k", 1e4),
    ("1.5u", 1.5e-6),
    ("3n", 3e-9),
    ("4p", 4e-12),
    ("5f", 5e-15),
    ("1G", 1e9),
    ("1e-3", 1e-3),
    ("-1m", -1e-3),
    (".5", 0.5),
    ("50%", 0.5),
    ("1_m", 1e-3),
]
BAD_FLOATS = ["abc", "", "   ", "1 m", "1x", "1ms", "m", "1e", "nan", "inf", True, False, [1.0], (1.0,), {"v": 1}, 1j]


@pytest.mark.parametrize("name", FLOAT_OPTIONS)
@pytest.mark.parametrize(("value", "expected"), GOOD_FLOATS, ids=repr)
def test_float_accepts_numbers_and_metric_strings(backend, circuit, name: str, value, expected: float) -> None:
    result = getattr(_options(backend, circuit, **{name: value}), FLOAT_OPTIONS[name])
    assert result == pytest.approx(expected, rel=1e-15) and type(result) is float


@pytest.mark.parametrize("name", FLOAT_OPTIONS)
@pytest.mark.parametrize("value", BAD_FLOATS, ids=repr)
def test_float_rejects_invalid_values(backend, circuit, name: str, value: object) -> None:
    _rejects(backend, circuit, f"{name} must be a number (metric suffixes allowed)", **{name: value})


def test_float_defaults_none_and_zero(backend, circuit) -> None:
    defaults = _options(backend, circuit)  # .tran 1m from the directive
    assert (defaults.tran_stop, defaults.tran_step) == (1e-3, 1e-5)
    assert (defaults.ac_start, defaults.ac_stop, defaults.hb_time_window) == (None, None, None)
    assert (defaults.krylov_tol, defaults.v_tol, defaults.f_tol) == (1e-9, 1e-9, 1e-9)
    explicit_none = _options(backend, circuit, f_start=None, krylov_tol=None, v_tol=None, f_tol=None)
    assert (explicit_none.ac_start, explicit_none.krylov_tol, explicit_none.v_tol, explicit_none.f_tol) == (None, 1e-9, 1e-9, 1e-9)
    # A zero is passed through; the solvers still treat it as unset (P3/P4 ruling, not applied here).
    zero = _options(backend, circuit, t_stop=0, t_step="0", f_start=0)
    assert (zero.tran_stop, zero.tran_step, zero.ac_start) == (0.0, 0.0, 0.0)


def test_string_time_options_run_like_the_tran_directive() -> None:
    service = SimulationService()
    by_options = service.run_request(
        AnalysisRequest(netlist_text=RC_TRAN, mode=AnalysisMode.TRAN, options={"t_stop": "1m", "t_step": "0.1m"})
    )
    by_directive = service.run_request(AnalysisRequest(netlist_text=RC_TRAN.replace(".tran 1m", ".tran 1m 0.1m")))
    assert by_options.waveform == by_directive.waveform


# ---------------------------------------------------------------------------
# init_condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("DEFAULT", "DEFAULT"), ("default", "DEFAULT"), (" Random ", "RANDOM"), ("RANDOM", "RANDOM"), (None, None)],
)
def test_init_condition_strings_are_normalised(backend, circuit, value, expected) -> None:
    assert _options(backend, circuit, init_condition=value).init_condition == expected


def test_init_condition_ndarray_is_passed_through(backend, circuit) -> None:
    start = np.array([[1.0], [0.0], [0.0]])
    assert _options(backend, circuit, init_condition=start).init_condition is start


@pytest.mark.parametrize(
    "value",
    [[1, 0.5, -2], (1, 0.5, -2), [[1], [0.5], [-2]], [np.int64(1), np.float64(0.5), -2]],
    ids=["list", "tuple", "nested_column", "numpy_scalars"],
)
def test_init_condition_numeric_sequence_becomes_a_float_column(backend, circuit, value) -> None:
    result = _options(backend, circuit, init_condition=value).init_condition
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.shape == (3, 1)
    np.testing.assert_array_equal(result, [[1.0], [0.5], [-2.0]])


@pytest.mark.parametrize(
    "value",
    ["garbage", "", "DEFAULTS", 5, 1.5, True, {"a": 1}, ["1", "2"], [True, False], [None], [[1, 2], [3]], [1, "x"]],
    ids=repr,
)
def test_init_condition_rejects_everything_else(backend, circuit, value) -> None:
    _rejects(backend, circuit, "init_condition must be 'DEFAULT', 'RANDOM', or a numeric array", init_condition=value)


def test_init_condition_length_is_not_validated_by_build_options(backend, circuit) -> None:
    # Wrong length is a P3 ruling for the solvers; build_options only converts.
    assert _options(backend, circuit, init_condition=[1.0]).init_condition.shape == (1, 1)


# ---------------------------------------------------------------------------
# Unchanged options
# ---------------------------------------------------------------------------


def test_label_lists_are_unchanged(backend, circuit) -> None:
    options = _options(backend, circuit, probe_nodes="V(a), V(b)", mor_output_nodes=["V(b)", " "])
    assert options.probe_nodes == ["V(a)", "V(b)"]
    assert options.mor_output_nodes == ["V(b)"]


# ---------------------------------------------------------------------------
# Service, HTTP and CLI paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{"use_krylov": "maybe"}, {"krylov_rank": 2.7}, {"t_stop": "soon"}, {"init_condition": "warm"}, {"points": "2.5"}],
    ids=repr,
)
def test_service_tags_coercion_errors_as_invalid_options(options: dict) -> None:
    with pytest.raises(ValueError) as exc:
        SimulationService().run_request(AnalysisRequest(netlist_text=OP_NETLIST, mode=AnalysisMode.OP, options=options))
    assert type(exc.value) is ValueError
    assert is_invalid_option_error(exc.value)


@pytest.fixture()
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def test_http_string_boolean_is_rejected_with_400(client: TestClient) -> None:
    response = client.post(
        "/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"use_krylov": "maybe"}}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == {"code": "INVALID_OPTIONS", "message": "use_krylov must be a boolean"}


def test_http_string_false_keeps_krylov_off(client: TestClient) -> None:
    off = client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"use_krylov": "false"}})
    assert off.status_code == 200
    assert "linear_solver" not in off.json()["metadata"]
    on = client.post("/api/simulate", json={"netlist_text": OP_NETLIST, "mode": "op", "options": {"use_krylov": "on"}})
    assert on.status_code == 200
    assert on.json()["metadata"]["linear_solver"] == "krylov"


def test_http_string_time_options_and_json_list_init_condition(client: TestClient) -> None:
    response = client.post(
        "/api/simulate",
        json={
            "netlist_text": RC_TRAN,
            "mode": "tran",
            "options": {"t_stop": "2m", "t_step": "1m", "init_condition": [1.0, 0.0, 0.0]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["waveform"]["time"] == [0.0, 1e-3, 2e-3]
    assert body["metadata"]["op_used"] is False


def test_http_tran_error_is_a_400_netlist_error(client: TestClient) -> None:
    response = client.post(
        "/api/simulate", json={"netlist_text": RC_TRAN, "mode": "tran", "options": {"t_stop": "-1m", "t_step": "0.1m"}}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == {"code": "TRAN_ERROR", "message": "t_stop must be non-negative."}


def test_tran_error_is_a_netlist_error_through_the_service() -> None:
    with pytest.raises(NetlistError) as exc:
        SimulationService().run_request(
            AnalysisRequest(netlist_text=RC_TRAN, mode=AnalysisMode.TRAN, options={"t_stop": 1e-3, "t_step": -1e-4})
        )
    assert type(exc.value) is NetlistError and exc.value.code == "TRAN_ERROR"
    assert str(exc.value) == "t_step must be positive."


def test_cli_still_runs(capsys: pytest.CaptureFixture[str]) -> None:
    from mna_simulation.cli import main

    assert main(["run-text", RC_TRAN.replace(".tran 1m", ".tran 1m 0.25m")]) == 0
    out = capsys.readouterr().out
    assert "Status: ok" in out and "Time samples: 5" in out
    # TRAN_ERROR is now a NetlistError; the CLI still reports it by code.
    assert main(["run-text", RC_TRAN.replace(".tran 1m", ".tran -1m")]) == 1
    assert capsys.readouterr().out.strip() == "TRAN_ERROR: t_stop must be non-negative."
