"""Run identical requests through two simulator packages and compare the responses.

The two packages are normally the frozen oracle ``mna_reference`` and the live
``mna_simulation``. Each package builds its own ``AnalysisRequest`` from its
own ``api.contracts`` module (the two pydantic classes are distinct types) and
runs it through its own ``api.service.SimulationService``.

What is compared (``assert_responses_match``):

* ``mode`` (by enum value), ``status``, top-level ``labels``;
* ``dc_solution``, ``waveform.time/values/labels``,
  ``spectrum.frequencies/magnitudes/labels``: numerically, with the tolerance
  class of the case (``TOLERANCES``), ``|cur - ref| <= atol + rtol * |ref|``,
  NaN == NaN;
* ``matrices``: same key set; numeric matrices (``G``, ``C``, ``b_dc``,
  ``b_ac``) numerically; symbolic string matrices (``f_str``, ``b_time_str``)
  by shape only, because their text is an implementation detail;
* contractual metadata (``CONTRACT_METADATA_KEYS``): present in both or in
  neither, and equal (floats to 1e-12 relative);
* ``metadata.operating_point`` and ``metadata.device_operating_points`` when
  present: discrete fields equal, numeric fields within tolerance.

``diagnostics`` and the remaining metadata (timings, cache counters, Krylov
iteration counts, ...) are deliberately not compared.

Error outcomes (``assert_errors_match``) must have the same exception class
name and the same ``.code`` attribute (``None`` when absent).
"""

from __future__ import annotations

import copy
import importlib
import math
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

TOLERANCES: dict[str, tuple[float, float]] = {
    # class: (rtol, atol)
    "linear": (1e-9, 1e-12),
    "nonlinear": (1e-6, 1e-9),
    "hb": (1e-6, 1e-9),
    "mor": (5e-2, 1e-6),
}

CONTRACT_METADATA_KEYS: tuple[str, ...] = (
    "labels",
    "num_nodes",
    "num_branches",
    "gmin",
    "op_used",
    "base_frequency_hz",
    "harmonics",
    "krylov_method",
    "krylov_matrix_kind",
    "krylov_converged",
    "mor_used",
    "mor_method",
    "output_decimation",
)

NUMERIC_MATRICES: tuple[str, ...] = ("G", "C", "b_dc", "b_ac")
SYMBOLIC_MATRICES: tuple[str, ...] = ("f_str", "b_time_str")
DEVICE_POINT_DISCRETE_FIELDS: tuple[str, ...] = ("name", "type", "model", "region", "note")
METADATA_FLOAT_RTOL = 1e-12


@dataclass(slots=True)
class Outcome:
    """Result of one request: either a response or the exception it raised."""

    response: Any | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_SERVICES: dict[str, Any] = {}


def _module(pkg: ModuleType | str, suffix: str) -> ModuleType:
    name = pkg if isinstance(pkg, str) else pkg.__name__
    return importlib.import_module(f"{name}.{suffix}")


def build_request(pkg: ModuleType | str, netlist_text: str, mode: str | None, options: dict[str, Any] | None = None):
    """Build an ``AnalysisRequest`` with ``pkg``'s own contracts module."""

    contracts = _module(pkg, "api.contracts")
    return contracts.AnalysisRequest(
        netlist_text=netlist_text,
        mode=contracts.AnalysisMode(mode) if mode is not None else None,
        options=copy.deepcopy(options or {}),
    )


def run_request(
    pkg: ModuleType | str,
    netlist_text: str,
    mode: str | None,
    options: dict[str, Any] | None = None,
) -> Outcome:
    """Run one request through ``pkg.api.service.SimulationService``."""

    name = pkg if isinstance(pkg, str) else pkg.__name__
    service = _SERVICES.get(name)
    if service is None:
        service = _module(pkg, "api.service").SimulationService()
        _SERVICES[name] = service
    request = build_request(pkg, netlist_text, mode, options)
    with warnings.catch_warnings():
        # exp() overflow inside damped Newton trials is expected on some cases.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            response = service.run_request(request)
        except Exception as exc:  # noqa: BLE001 - errors are compared, not hidden
            return Outcome(error=exc)
    return Outcome(response=response)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class _Mismatches:
    def __init__(self, context: str) -> None:
        self.context = context
        self.items: list[str] = []

    def add(self, where: str, message: str) -> None:
        self.items.append(f"{where}: {message}")

    def raise_if_any(self) -> None:
        if self.items:
            header = f"reference/current mismatch{f' ({self.context})' if self.context else ''}:"
            raise AssertionError("\n  - ".join([header, *self.items]))


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _to_array(value: Any) -> np.ndarray:
    def convert(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"real", "imag"}:
            return complex(item["real"], item["imag"])
        if isinstance(item, (list, tuple)):
            return [convert(entry) for entry in item]
        return item

    array = np.asarray(convert(value))
    if array.dtype == object:
        try:
            array = array.astype(complex)
        except (TypeError, ValueError):
            pass
    return array


def _compare_arrays(out: _Mismatches, where: str, ref: Any, cur: Any, rtol: float, atol: float) -> None:
    if ref is None or cur is None:
        if (ref is None) != (cur is None):
            out.add(where, f"present in {'current' if ref is None else 'reference'} only")
        return
    ref_arr = _to_array(ref)
    cur_arr = _to_array(cur)
    if ref_arr.shape != cur_arr.shape:
        out.add(where, f"shape {cur_arr.shape} != reference {ref_arr.shape}")
        return
    if ref_arr.dtype == object or cur_arr.dtype == object:
        if ref_arr.tolist() != cur_arr.tolist():
            out.add(where, "non-numeric contents differ")
        return
    try:
        np.testing.assert_allclose(cur_arr, ref_arr, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as exc:
        detail = " ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
        out.add(where, f"not close (rtol={rtol:g}, atol={atol:g}): {detail[:600]}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _values_equal(ref: Any, cur: Any, rel: float = METADATA_FLOAT_RTOL) -> bool:
    if _is_number(ref) and _is_number(cur):
        ref_f, cur_f = float(ref), float(cur)
        if math.isnan(ref_f) and math.isnan(cur_f):
            return True
        return ref_f == cur_f or math.isclose(ref_f, cur_f, rel_tol=rel, abs_tol=0.0)
    if isinstance(ref, dict) and isinstance(cur, dict):
        return ref.keys() == cur.keys() and all(_values_equal(ref[key], cur[key], rel) for key in ref)
    if isinstance(ref, (list, tuple)) and isinstance(cur, (list, tuple)):
        return len(ref) == len(cur) and all(_values_equal(a, b, rel) for a, b in zip(ref, cur))
    return _enum_value(ref) == _enum_value(cur)


def _compare_contract_metadata(out: _Mismatches, ref_md: dict[str, Any], cur_md: dict[str, Any]) -> None:
    for key in CONTRACT_METADATA_KEYS:
        in_ref, in_cur = key in ref_md, key in cur_md
        if in_ref != in_cur:
            out.add(f"metadata[{key!r}]", f"present in {'reference' if in_ref else 'current'} only")
        elif in_ref and not _values_equal(ref_md[key], cur_md[key]):
            out.add(f"metadata[{key!r}]", f"{cur_md[key]!r} != reference {ref_md[key]!r}")


def _compare_operating_points(out: _Mismatches, ref_md: dict[str, Any], cur_md: dict[str, Any], rtol: float, atol: float) -> None:
    ref_op, cur_op = ref_md.get("operating_point"), cur_md.get("operating_point")
    if (ref_op is None) != (cur_op is None):
        out.add("metadata['operating_point']", f"present in {'current' if ref_op is None else 'reference'} only")
    elif ref_op is not None:
        if _field(ref_op, "labels") != _field(cur_op, "labels"):
            out.add("metadata['operating_point'].labels", "labels differ")
        _compare_arrays(out, "metadata['operating_point'].values", _field(ref_op, "values"), _field(cur_op, "values"), rtol, atol)

    ref_dev, cur_dev = ref_md.get("device_operating_points"), cur_md.get("device_operating_points")
    if (ref_dev is None) != (cur_dev is None):
        out.add("metadata['device_operating_points']", f"present in {'current' if ref_dev is None else 'reference'} only")
        return
    if ref_dev is None:
        return
    if len(ref_dev) != len(cur_dev):
        out.add("metadata['device_operating_points']", f"{len(cur_dev)} devices != reference {len(ref_dev)}")
        return
    for index, (ref_point, cur_point) in enumerate(zip(ref_dev, cur_dev)):
        where = f"metadata['device_operating_points'][{index}]"
        if set(ref_point) != set(cur_point):
            out.add(where, f"keys {sorted(cur_point)} != reference {sorted(ref_point)}")
            continue
        for key, ref_value in ref_point.items():
            cur_value = cur_point[key]
            if _is_number(ref_value) and _is_number(cur_value):
                _compare_arrays(out, f"{where}[{key!r}]", [float(ref_value)], [float(cur_value)], rtol, atol)
            elif not _values_equal(ref_value, cur_value):
                out.add(f"{where}[{key!r}]", f"{cur_value!r} != reference {ref_value!r}")


def assert_responses_match(ref: Any, cur: Any, tolerance: str = "linear", context: str = "") -> None:
    """Assert two ``SimulationResponse`` objects agree under ``tolerance``."""

    rtol, atol = TOLERANCES[tolerance]
    out = _Mismatches(context)

    if _enum_value(ref.mode) != _enum_value(cur.mode):
        out.add("mode", f"{_enum_value(cur.mode)!r} != reference {_enum_value(ref.mode)!r}")
    if ref.status != cur.status:
        out.add("status", f"{cur.status!r} != reference {ref.status!r}")
    if list(ref.labels) != list(cur.labels):
        out.add("labels", f"{cur.labels!r} != reference {ref.labels!r}")

    _compare_arrays(out, "dc_solution", ref.dc_solution, cur.dc_solution, rtol, atol)

    for name, series in (("waveform", ("time", "values")), ("spectrum", ("frequencies", "magnitudes"))):
        ref_block, cur_block = getattr(ref, name), getattr(cur, name)
        if (ref_block is None) != (cur_block is None):
            out.add(name, f"present in {'current' if ref_block is None else 'reference'} only")
            continue
        if ref_block is None:
            continue
        if list(ref_block.get("labels") or []) != list(cur_block.get("labels") or []):
            out.add(f"{name}.labels", f"{cur_block.get('labels')!r} != reference {ref_block.get('labels')!r}")
        for key in series:
            _compare_arrays(out, f"{name}.{key}", ref_block.get(key), cur_block.get(key), rtol, atol)

    ref_mat, cur_mat = ref.matrices, cur.matrices
    if (ref_mat is None) != (cur_mat is None):
        out.add("matrices", f"present in {'current' if ref_mat is None else 'reference'} only")
    elif ref_mat is not None:
        if set(ref_mat) != set(cur_mat):
            out.add("matrices", f"keys {sorted(cur_mat)} != reference {sorted(ref_mat)}")
        for key in sorted(set(ref_mat) & set(cur_mat)):
            if key in SYMBOLIC_MATRICES:
                ref_shape, cur_shape = np.shape(ref_mat[key]), np.shape(cur_mat[key])
                if ref_shape != cur_shape:
                    out.add(f"matrices[{key!r}]", f"shape {cur_shape} != reference {ref_shape}")
            else:
                _compare_arrays(out, f"matrices[{key!r}]", ref_mat[key], cur_mat[key], rtol, atol)

    ref_md, cur_md = dict(ref.metadata or {}), dict(cur.metadata or {})
    _compare_contract_metadata(out, ref_md, cur_md)
    _compare_operating_points(out, ref_md, cur_md, rtol, atol)

    out.raise_if_any()


def assert_errors_match(ref_error: BaseException | None, cur_error: BaseException | None, context: str = "") -> None:
    """Assert both runs raised the same exception class name and ``.code``."""

    suffix = f" ({context})" if context else ""
    if ref_error is None and cur_error is None:
        return
    if ref_error is None:
        raise AssertionError(
            f"reference succeeded but current raised {type(cur_error).__name__}"
            f"(code={getattr(cur_error, 'code', None)!r}): {cur_error}{suffix}"
        ) from cur_error
    if cur_error is None:
        raise AssertionError(
            f"reference raised {type(ref_error).__name__}(code={getattr(ref_error, 'code', None)!r}): {ref_error} "
            f"but current succeeded{suffix}"
        )
    ref_desc = (type(ref_error).__name__, getattr(ref_error, "code", None))
    cur_desc = (type(cur_error).__name__, getattr(cur_error, "code", None))
    if ref_desc != cur_desc:
        raise AssertionError(
            f"error mismatch{suffix}: current {cur_desc[0]}(code={cur_desc[1]!r}): {cur_error} != "
            f"reference {ref_desc[0]}(code={ref_desc[1]!r}): {ref_error}"
        ) from cur_error


def assert_outcomes_match(
    ref: Outcome,
    cur: Outcome,
    tolerance: str = "linear",
    expects_error: str | None = None,
    context: str = "",
) -> None:
    """Check the oracle against the case expectation, then current against the oracle."""

    suffix = f" ({context})" if context else ""
    if expects_error is not None:
        if ref.error is None:
            raise AssertionError(f"corpus expects {expects_error} but the reference succeeded{suffix}")
        if type(ref.error).__name__ != expects_error:
            raise AssertionError(
                f"corpus expects {expects_error} but the reference raised {type(ref.error).__name__}: {ref.error}{suffix}"
            ) from ref.error
    elif ref.error is not None:
        raise AssertionError(
            f"reference raised unexpectedly {type(ref.error).__name__}"
            f"(code={getattr(ref.error, 'code', None)!r}): {ref.error}{suffix}"
        ) from ref.error

    if ref.error is not None or cur.error is not None:
        assert_errors_match(ref.error, cur.error, context)
        return
    assert_responses_match(ref.response, cur.response, tolerance=tolerance, context=context)


def run_and_compare(
    reference_pkg: ModuleType | str,
    current_pkg: ModuleType | str,
    netlist_text: str,
    mode: str | None,
    options: dict[str, Any] | None = None,
    tolerance: str = "linear",
    expects_error: str | None = None,
    context: str = "",
) -> tuple[Outcome, Outcome]:
    """Convenience wrapper: run both packages and assert parity."""

    ref = run_request(reference_pkg, netlist_text, mode, options)
    cur = run_request(current_pkg, netlist_text, mode, options)
    assert_outcomes_match(ref, cur, tolerance=tolerance, expects_error=expects_error, context=context)
    return ref, cur
