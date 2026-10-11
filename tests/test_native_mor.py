"""Native model-order reduction (``mna_simulation/native/mor.py`` over ``_core.mor_*``) vs the frozen reference.

* Every corpus MOR analysis: the native rational Krylov basis accepts as many columns as the
  reference's (same acceptance rules on the same seeds, shifts and order), and the reduced
  outputs of ``solve_ac_mor`` / ``solve_transient_mor`` / ``solve_transient_tpwl`` match the
  oracle's within the ``mor`` tolerance. The analyses whose oracle basis is numerically
  singular (``MOR_FULL_ORDER_REFERENCE``; the native basis uses Gram-Schmidt twice, ruling
  2.4.1) are compared with the oracle's full-order solution instead, and their live bases are
  checked to be orthonormal.
* ``_safe_solve`` stages: LU, least squares on a singular matrix (minimum-norm, like
  ``np.linalg.lstsq``), and a singular shifted system inside the basis loop.
* POD rank rule on constructed snapshot matrices (empty, zero, rank-deficient, tiny scale).
* Training-index rule: identical index lists to the reference's ``_training_indexes`` on 20
  random snapshot matrices (plus plateaus and repeated columns).
* TPWL: the reduced step loop against the reference on a diode ladder (k < n), and the LRU
  cache (miss, hit, size, eviction).
* Threads: the reduced AC sweep is bitwise identical for 1 and 4 threads.
* Bindings: shapes, dtypes and ``CoreError(NETLIST_FATAL)`` for malformed input.
"""

from __future__ import annotations

import numpy as np
import pytest
from corpus import iter_analyses
from differential.compare import TOLERANCES, full_order_options, run_request
from differential.test_reference_parity import MOR_FULL_ORDER_REFERENCE
from mna_reference.mna_builder import build_mna_problem as reference_build
from mna_reference.mor import linear as ref_linear
from mna_reference.mor import metadata as ref_metadata
from mna_reference.mor import selectors as ref_selectors
from mna_reference.mor import tpwl as ref_tpwl
from mna_reference.netlist import parse_netlist_text as reference_parse
from scipy import sparse

from mna_simulation import native
from mna_simulation.mna_builder import build_mna_problem
from mna_simulation.mor import linear, tpwl
from mna_simulation.native import mor as native_mor
from mna_simulation.netlist import parse_netlist_text

core = native.require_core()

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

MOR_RTOL, MOR_ATOL = TOLERANCES["mor"]

DIODE_LADDER = """I1 in 0 STEP 1m 1u
R1 in n1 100
C1 n1 0 1n
D1 n1 0 1e-12
R2 n1 n2 200
C2 n2 0 2n
D2 n2 0 1e-12
R3 n2 out 300
C3 out 0 1n
R4 out 0 1k
.end
"""


def _flags_id(case, mode: str, options: dict) -> str:
    flag = "mor" + (f"_{options['mor_method']}" if options.get("mor_method") not in (None, "auto") else "")
    return f"{case.name}-{mode}-{flag}"


def _mor_analyses() -> list:
    params = []
    for case, index, mode, options in iter_analyses():
        if not options.get("use_mor") or case.expects_error or mode not in ("ac", "tran"):
            continue
        params.append(pytest.param(case, index, mode, options, id=f"{case.name}-{index}-{mode}"))
    return params


MOR_ANALYSES = _mor_analyses()


def _problems(netlist: str):
    return reference_build(reference_parse(netlist)), build_mna_problem(parse_netlist_text(netlist))


def _basis_arguments(ref_problem, mode: str, options: dict):
    """(G_lin, C, inputs, selector, shifts, order) exactly as the reference solve_*_mor builds them."""

    labels = list(ref_problem.metadata["labels"])
    selector, outputs = ref_selectors.build_output_selector(labels, options["mor_output_nodes"])
    inputs = ref_linear._source_directions(ref_problem)
    order, _mode = ref_metadata.resolve_mor_order(
        ref_problem.G.shape[0],
        ref_linear.count_input_directions(ref_problem),
        len(outputs),
        options.get("mor_order", "auto"),
    )
    G = ref_problem.G
    if mode == "ac":
        f_start, f_stop = options["f_start"], options["f_stop"]
        f_mid = float(np.sqrt(max(abs(f_start), 1e-30) * max(abs(f_stop), 1e-30)))
        shifts = [0.0, 1j * 2.0 * np.pi * f_mid, 1j * 2.0 * np.pi * float(f_stop)]
        G, _metadata = ref_linear._linearized_system(ref_problem, options.get("max_iter", 10000), 1e-9, 1e-9, {})
    else:
        shifts = [0.0, 1.0 / max(float(options["t_step"]), np.finfo(float).tiny)]
    return G, ref_problem.C, inputs, selector, shifts, order


def _run_mor(module, problem, mode: str, options: dict):
    common = {
        "mor_output_nodes": options["mor_output_nodes"],
        "mor_order": options.get("mor_order", "auto"),
        "requested_method": options.get("mor_method", "auto"),
        "validate": True,
        "probe_nodes": options.get("probe_nodes", []),
        "max_iter": options.get("max_iter", 10000),
        "v_tol": 1e-9,
        "f_tol": 1e-9,
    }
    if mode == "ac":
        spectrum = module.solve_ac_mor(
            problem, f_start=options["f_start"], f_stop=options["f_stop"], points=options["points"], **common
        )
        return np.asarray(spectrum.magnitudes), spectrum.metadata
    waveform, metadata = module.solve_transient_mor(
        problem, t_stop=options["t_stop"], t_step=options["t_step"], **common
    )
    return np.asarray(waveform.values), metadata


def _nonlinear(problem) -> bool:
    f = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    return any(str(e) != "0" for e in np.asarray(f).flatten()) or bool(
        problem.level1_mos_devices or problem.level1_bjt_devices
    )


# ---------------------------------------------------------------------------------------------
# Corpus parity
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("case", "index", "mode", "options"), MOR_ANALYSES)
def test_corpus_basis_columns_and_reduced_outputs_match_the_reference(case, index, mode, options) -> None:
    ref_problem, live_problem = _problems(case.netlist_text)
    if mode == "tran" and _nonlinear(ref_problem):
        # TPWL (the linear Krylov path never runs on nonlinear transients).
        common = {
            "t_stop": options["t_stop"],
            "t_step": options["t_step"],
            "mor_output_nodes": options["mor_output_nodes"],
            "mor_order": options.get("mor_order", "auto"),
            "requested_method": "tpwl",
            "validate": True,
            "probe_nodes": options.get("probe_nodes", []),
            "max_iter": options.get("max_iter", 10000),
            "v_tol": 1e-9,
            "f_tol": 1e-9,
        }
        ref_waveform, ref_md = ref_tpwl.solve_transient_tpwl(ref_problem, **common)
        live_waveform, live_md = tpwl.solve_transient_tpwl(live_problem, **common)
        for key in ("mor_reduced_dimension", "mor_training_snapshots", "mor_tpwl_local_models"):
            assert live_md[key] == ref_md[key], key
        np.testing.assert_allclose(live_waveform.values, ref_waveform.values, rtol=MOR_RTOL, atol=MOR_ATOL)
        return

    ruled = _flags_id(case, mode, options) in MOR_FULL_ORDER_REFERENCE
    G, C, inputs, selector, shifts, order = _basis_arguments(ref_problem, mode, options)
    ref_basis = ref_linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
    live_basis = linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
    if not ruled:  # the ruled cases' one-pass oracle accepted rounding residuals as columns
        assert live_basis.shape == ref_basis.shape

    ref_values, ref_md = _run_mor(ref_linear, ref_problem, mode, options)
    live_values, live_md = _run_mor(linear, live_problem, mode, options)
    keys = ["mor_used", "mor_method", "mor_resolved_order", "mor_num_inputs", "mor_basis_expansion_points"]
    keys += ["op_used", "linearized_device_count"] + ([] if ruled else ["mor_reduced_dimension"])
    for key in keys:
        assert live_md.get(key) == ref_md.get(key), key
    if ruled:
        full = _full_order_outputs(case.netlist_text, mode, options, list(live_md["mor_output_nodes"]))
        np.testing.assert_allclose(live_values, full, rtol=MOR_RTOL, atol=MOR_ATOL)
        return
    np.testing.assert_allclose(live_values, ref_values, rtol=MOR_RTOL, atol=MOR_ATOL)


def _full_order_outputs(netlist: str, mode: str, options: dict, labels: list[str]) -> np.ndarray:
    """The oracle's full-order solution of the same analysis (without MOR), restricted to ``labels``."""

    outcome = run_request("mna_reference", netlist, mode, full_order_options(options))
    assert outcome.ok, outcome.error
    block = outcome.response.waveform if mode == "tran" else outcome.response.spectrum
    values = np.asarray(block["values" if mode == "tran" else "magnitudes"], dtype=float)
    return values[[list(block["labels"]).index(label) for label in labels]]


def test_former_noise_mor_analyses_oracle_singular_live_orthonormal() -> None:
    """Every MOR_FULL_ORDER_REFERENCE entry: the one-pass oracle basis is numerically singular
    (cond > 1e15 with k == n), the live two-pass basis is orthonormal (||V^T V - I|| < 1e-12)
    and the live MOR outputs match the oracle's full-order solution to 1e-11 (max-normalized)."""

    found = set()
    for case, _index, mode, options in iter_analyses():
        if not options.get("use_mor") or case.expects_error:
            continue
        test_id = _flags_id(case, mode, options)
        if test_id not in MOR_FULL_ORDER_REFERENCE:
            continue
        found.add(test_id)
        ref_problem, live_problem = _problems(case.netlist_text)
        G, C, inputs, selector, shifts, order = _basis_arguments(ref_problem, mode, options)
        basis = ref_linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
        assert basis.shape[1] == basis.shape[0]  # k == n: an exact reduced model if V were regular
        assert np.linalg.cond(basis) > 1e15, test_id
        live = linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
        assert np.linalg.norm(live.T @ live - np.eye(live.shape[1])) < 1e-12, test_id
        values, metadata = _run_mor(linear, live_problem, mode, options)
        full = _full_order_outputs(case.netlist_text, mode, options, list(metadata["mor_output_nodes"]))
        assert np.max(np.abs(values - full)) <= 1e-11 * np.max(np.abs(full)), test_id
    assert found == set(MOR_FULL_ORDER_REFERENCE)


def test_well_conditioned_basis_equals_the_reference_up_to_rotation() -> None:
    """Outputs away from the driven node: both bases orthonormal, same column space."""

    netlist = "\n".join(
        ["IIN n_0_0 0 STEP 1m 20n"]
        + [
            line
            for r in range(5)
            for c in range(5)
            for line in (
                f"C_{r}_{c} n_{r}_{c} 0 200p",
                f"RG_{r}_{c} n_{r}_{c} 0 50Meg",
                *([f"R_H_{r}_{c} n_{r}_{c} n_{r}_{c + 1} 25"] if c < 4 else []),
                *([f"L_V_{r}_{c} n_{r}_{c} n_{r + 1}_{c} 20n"] if r < 4 else []),
            )
        ]
        + ["RLOAD n_4_4 0 50", ".end"]
    )
    ref_problem, live_problem = _problems(netlist)
    G, C, inputs, selector, shifts, order = _basis_arguments(
        ref_problem, "tran", {"mor_output_nodes": ["V(n_2_2)", "V(n_4_4)"], "t_step": 1e-9}
    )
    ref_basis = ref_linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
    live_basis = linear.build_linear_krylov_basis(G, C, inputs, selector, shifts, order)
    sparse_basis = linear.build_linear_krylov_basis(
        sparse.csr_matrix(G), sparse.csc_matrix(C), inputs, selector, shifts, order
    )
    assert live_basis.shape == ref_basis.shape == (G.shape[0], 12)
    np.testing.assert_allclose(live_basis.T @ live_basis, np.eye(12), atol=1e-10)
    residual = live_basis - ref_basis @ (ref_basis.T @ live_basis)
    assert np.linalg.norm(residual, 2) < 1e-9
    np.testing.assert_array_equal(sparse_basis, live_basis)
    # The problem path (native circuit pattern) spans the same space.
    _circuit, _gmin, operands = linear._problem_operands(live_problem)
    problem_basis = native_mor.krylov_basis(operands, inputs, selector, shifts, order)
    residual = problem_basis - ref_basis @ (ref_basis.T @ problem_basis)
    assert np.linalg.norm(residual, 2) < 1e-9


# ---------------------------------------------------------------------------------------------
# _safe_solve and the least-squares fallback
# ---------------------------------------------------------------------------------------------


def test_safe_solve_matches_the_reference_on_regular_and_singular_systems() -> None:
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 6))
    b = rng.standard_normal(6)
    np.testing.assert_allclose(linear._safe_solve(A, b), ref_linear._safe_solve(A, b), rtol=1e-12)
    B = rng.standard_normal((6, 3))
    np.testing.assert_allclose(linear._safe_solve(A, B), ref_linear._safe_solve(A, B), rtol=1e-12)
    Az = A + 1j * rng.standard_normal((6, 6))
    np.testing.assert_allclose(linear._safe_solve(Az, b), ref_linear._safe_solve(Az, b), rtol=1e-12)
    # Exactly singular (duplicate rows stay identical through the elimination, so every LU meets an
    # exactly zero pivot): np.linalg.solve raises, lstsq gives the minimum-norm solution. (Merely
    # numerically singular matrices hit an exact zero pivot or not depending on rounding, which differs
    # between LAPACK's blocked FMA updates and Eigen's; such systems are noise in both implementations.)
    S = rng.standard_normal((6, 6))
    S[4] = S[1]
    S[:, 5] = 0.0
    rhs = rng.standard_normal((6, 2))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(S, rhs)
    np.testing.assert_allclose(linear._safe_solve(S, rhs), ref_linear._safe_solve(S, rhs), rtol=1e-8, atol=1e-10)
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(linear._safe_solve(singular, np.array([2.0, 2.0])), [1.0, 1.0])
    # A singular matrix with a NaN: lstsq fails, the regularized retry too -> LinAlgError.
    with pytest.raises(np.linalg.LinAlgError):
        linear._safe_solve(np.array([[0.0, 0.0], [np.nan, 0.0]]), np.ones(2))


def test_singular_shift_uses_least_squares_like_the_reference() -> None:
    # Node c hangs on capacitors only: G is singular at shift 0.
    netlist = "I1 a 0 STEP 1m 1u\nR1 a b 1k\nR2 b 0 2k\nC1 b c 1n\nC2 c 0 2n\nC3 a 0 1n\n.end\n"
    ref_problem, _live = _problems(netlist)
    labels = list(ref_problem.metadata["labels"])
    selector, _outputs = ref_selectors.build_output_selector(labels, ["V(c)"])
    inputs = ref_linear._source_directions(ref_problem)
    shifts = [0.0, 1e6]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(ref_problem.G, inputs)
    ref_basis = ref_linear.build_linear_krylov_basis(ref_problem.G, ref_problem.C, inputs, selector, shifts, 3)
    operands = native_mor.matrix_operands(ref_problem.G, ref_problem.C)
    basis, info = core.mor_krylov_basis(
        operands.col_ptr,
        operands.row_idx,
        operands.g,
        operands.c,
        inputs,
        selector,
        np.asarray(shifts, dtype=complex),
        [False, False],
        3,
        1,
    )
    assert info["lstsq_shifts"] == 1
    assert basis.shape == ref_basis.shape
    residual = basis - ref_basis @ (ref_basis.T @ basis)
    assert np.linalg.norm(residual, 2) < 1e-9


# ---------------------------------------------------------------------------------------------
# POD and training indexes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snapshots, order",
    [
        (np.zeros((5, 0)), 3),
        (np.zeros((4, 6)), 3),
        (np.outer(np.arange(1.0, 7.0), np.linspace(0.0, 1.0, 9)), 4),
        (np.random.default_rng(3).standard_normal((8, 3)) @ np.random.default_rng(4).standard_normal((3, 12)), 10),
        (np.random.default_rng(5).standard_normal((10, 7)), 4),
        (1e-14 * np.random.default_rng(6).standard_normal((6, 5)), 5),
        (np.random.default_rng(8).standard_normal((3, 20)), 0),
    ],
    ids=["empty", "zero", "rank1", "rank3", "full_clipped", "tiny_scale", "order0"],
)
def test_pod_basis_rank_rule(snapshots: np.ndarray, order: int) -> None:
    ref = ref_tpwl._pod_basis(snapshots, order)
    live = tpwl._pod_basis(snapshots, order)
    assert live.shape == ref.shape
    if snapshots.size == 0 or not np.any(snapshots):
        np.testing.assert_array_equal(live, ref)
        return
    np.testing.assert_allclose(live.T @ live, np.eye(live.shape[1]), atol=1e-12)
    residual = live - ref @ (ref.T @ live)
    assert np.linalg.norm(residual, 2) < 1e-8


def _random_snapshots(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(1, 30))
    steps = int(rng.integers(1, 45))
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return rng.standard_normal((n, steps))
    if kind == 1:  # plateaus: repeated columns give zero deltas
        values = np.repeat(rng.standard_normal((n, max(1, steps // 3))), 3, axis=1)[:, :steps]
        return (
            values
            if values.shape[1] == steps
            else np.hstack([values, values[:, -1:].repeat(steps - values.shape[1], 1)])
        )
    if kind == 2:  # a step response
        t = np.linspace(0.0, 1.0, steps)
        return np.outer(rng.standard_normal(n), 1.0 - np.exp(-5.0 * t))
    return np.round(rng.standard_normal((n, steps)), 1)  # coarse values: many equal deltas


def test_training_indexes_equal_the_reference_on_random_snapshots() -> None:
    rng = np.random.default_rng(2024)
    for trial in range(20):
        snapshots = _random_snapshots(rng)
        basis = np.zeros((snapshots.shape[0], int(rng.integers(1, 20))))
        output_count = int(rng.integers(0, 5))
        expected = ref_tpwl._training_indexes(snapshots, basis, output_count)
        assert tpwl._training_indexes(snapshots, basis, output_count) == expected, trial
    # Shapes at the rule boundaries.
    for steps in (1, 2, 3, 4, 17, 64):
        snapshots = np.cumsum(rng.standard_normal((3, steps)), axis=1)
        for columns in (0, 1, 3, 15, 30):
            basis = np.zeros((3, columns))
            assert tpwl._training_indexes(snapshots, basis, 1) == ref_tpwl._training_indexes(snapshots, basis, 1)


# ---------------------------------------------------------------------------------------------
# TPWL
# ---------------------------------------------------------------------------------------------


def _tpwl_kwargs(**overrides):
    kwargs = {
        "t_stop": 5e-6,
        "t_step": 0.25e-6,
        "mor_output_nodes": ["V(out)", "V(n1)"],
        "mor_order": 2,
        "requested_method": "tpwl",
        "validate": True,
        "probe_nodes": ["V(out)", "V(n1)"],
        "max_iter": 200,
        "v_tol": 1e-9,
        "f_tol": 1e-9,
    }
    kwargs.update(overrides)
    return kwargs


def test_tpwl_reduced_step_loop_matches_the_reference() -> None:
    ref_problem, live_problem = _problems(DIODE_LADDER)
    for order in (1, 2, 3):
        ref_waveform, ref_md = ref_tpwl.solve_transient_tpwl(ref_problem, **_tpwl_kwargs(mor_order=order))
        live_waveform, live_md = tpwl.solve_transient_tpwl(live_problem, **_tpwl_kwargs(mor_order=order))
        assert live_md["mor_reduced_dimension"] == ref_md["mor_reduced_dimension"] == order
        assert live_md["mor_tpwl_local_models"] == ref_md["mor_tpwl_local_models"]
        np.testing.assert_allclose(live_waveform.time, ref_waveform.time, rtol=0, atol=0)
        np.testing.assert_allclose(live_waveform.values, ref_waveform.values, rtol=1e-9, atol=1e-12)


def test_tpwl_training_matches_the_reference() -> None:
    ref_problem, live_problem = _problems(DIODE_LADDER)
    params = {
        "t_stop": 5e-6,
        "t_step": 0.25e-6,
        "order": 3,
        "max_iter": 200,
        "v_tol": 1e-9,
        "f_tol": 1e-9,
        "output_count": 2,
    }
    ref_entry = ref_tpwl._train_tpwl(ref_problem, **params)
    live_entry = tpwl._train_tpwl(live_problem, **params)
    np.testing.assert_allclose(live_entry["snapshots"], ref_entry["snapshots"], rtol=1e-9, atol=1e-12)
    ref_basis, live_basis = ref_entry["basis"], live_entry["basis"]
    assert live_basis.shape == ref_basis.shape
    assert np.linalg.norm(live_basis - ref_basis @ (ref_basis.T @ live_basis), 2) < 1e-8
    assert len(live_entry["local_models"]) == len(ref_entry["local_models"])
    col_ptr, row_idx = live_problem.native.pattern()
    n = live_problem.native.n
    for ref_model, live_model in zip(ref_entry["local_models"], live_entry["local_models"]):
        np.testing.assert_allclose(live_model["x"], ref_model["x"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(live_model["f"], ref_model["f"], rtol=1e-6, atol=1e-12)
        J = sparse.csc_matrix((live_model["J_values"], row_idx, col_ptr), shape=(n, n)).toarray()
        np.testing.assert_allclose(J, ref_model["J"], rtol=1e-6, atol=1e-12)


def test_tpwl_cache_hit_miss_and_eviction() -> None:
    tpwl._TPWL_CACHE.clear()
    _ref, problem = _problems(DIODE_LADDER)
    first, md1 = tpwl.solve_transient_tpwl(problem, **_tpwl_kwargs())
    second, md2 = tpwl.solve_transient_tpwl(problem, **_tpwl_kwargs())
    assert md1["mor_cache_hit"] is False and md2["mor_cache_hit"] is True
    assert md1["mor_cache_size"] == md2["mor_cache_size"] == 1
    np.testing.assert_array_equal(first.values, second.values)
    # A rebuilt problem of the same netlist hits; another t_stop misses.
    _ref, rebuilt = _problems(DIODE_LADDER)
    assert tpwl.solve_transient_tpwl(rebuilt, **_tpwl_kwargs())[1]["mor_cache_hit"] is True
    third = tpwl.solve_transient_tpwl(problem, **_tpwl_kwargs(t_stop=4e-6))[1]
    assert third["mor_cache_hit"] is False and third["mor_cache_size"] == 2
    # LRU eviction at 8 entries.
    for k in range(10):
        tpwl.solve_transient_tpwl(problem, **_tpwl_kwargs(t_stop=1e-6 + k * 0.25e-6))
    assert len(tpwl._TPWL_CACHE) == tpwl._TPWL_CACHE_LIMIT == 8
    assert tpwl.solve_transient_tpwl(problem, **_tpwl_kwargs())[1]["mor_cache_hit"] is False
    tpwl._TPWL_CACHE.clear()


# ---------------------------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------------------------


def test_reduced_ac_sweep_is_bitwise_identical_for_1_and_4_threads() -> None:
    netlist = (
        "IIN n_0_0 0 AC 1m\n"
        + "\n".join(
            line
            for r in range(6)
            for c in range(6)
            for line in (
                f"C_{r}_{c} n_{r}_{c} 0 200p",
                f"RG_{r}_{c} n_{r}_{c} 0 50Meg",
                *([f"R_H_{r}_{c} n_{r}_{c} n_{r}_{c + 1} 25"] if c < 5 else []),
                *([f"L_V_{r}_{c} n_{r}_{c} n_{r + 1}_{c} 20n"] if r < 5 else []),
            )
        )
        + "\nRLOAD n_5_5 0 50\n.end\n"
    )
    _ref, problem = _problems(netlist)
    kwargs = {
        "f_start": 1e6,
        "f_stop": 1e9,
        "points": 64,
        "mor_output_nodes": ["V(n_3_3)", "V(n_5_5)"],
        "mor_order": "auto",
        "requested_method": "auto",
        "validate": False,
        "probe_nodes": [],
        "max_iter": 100,
        "v_tol": 1e-9,
        "f_tol": 1e-9,
    }
    one = linear.solve_ac_mor(problem, num_threads=1, **kwargs).magnitudes
    four = linear.solve_ac_mor(problem, num_threads=4, **kwargs).magnitudes
    assert one.shape == (2, 64)
    np.testing.assert_array_equal(one.view(np.uint64), four.view(np.uint64))
    # Directly on the kernel, big enough for the frequency-parallel path (points * k^3 >= 2e6).
    rng = np.random.default_rng(11)
    k = 30
    Gr = rng.standard_normal((k, k)) + k * np.eye(k)
    Cr = 1e-9 * rng.standard_normal((k, k))
    Br = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    Sr = rng.standard_normal((3, k))
    f = np.linspace(0.0, 1e9, 101)
    m1, info1 = core.mor_reduced_ac(Gr, Cr, Br, Sr, f, 1)
    m4, info4 = core.mor_reduced_ac(Gr, Cr, Br, Sr, f, 4)
    assert info1["threads_used"] == 1 and info4["threads_used"] == min(4, core.effective_threads(4))
    np.testing.assert_array_equal(m1.view(np.uint64), m4.view(np.uint64))
    expected = np.abs(np.stack([Sr @ np.linalg.solve(Gr + 1j * 2.0 * np.pi * fi * Cr, Br) for fi in f], axis=1))
    np.testing.assert_allclose(m1, expected, rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# Binding validation
# ---------------------------------------------------------------------------------------------


def _expect_fatal(fn, *args) -> None:
    with pytest.raises(core.CoreError) as info:
        fn(*args)
    assert info.value.code == "NETLIST_FATAL"


def test_bindings_validate_shapes() -> None:
    G = np.array([[2.0, -1.0], [-1.0, 2.0]])
    C = np.eye(2) * 1e-9
    ops = native_mor.matrix_operands(G, C)
    sel = np.array([[0.0, 1.0]])
    shifts = np.array([0.0], dtype=complex)
    good = (ops.col_ptr, ops.row_idx, ops.g, ops.c, np.zeros((2, 0)), sel, shifts, [False], 2, 1)
    basis, info = core.mor_krylov_basis(*good)
    assert basis.shape == (2, 2) and basis.flags.f_contiguous and info["columns"] == 2
    _expect_fatal(
        core.mor_krylov_basis, ops.col_ptr, ops.row_idx, ops.g[:-1], ops.c, np.zeros((2, 0)), sel, shifts, [False], 2, 1
    )
    _expect_fatal(
        core.mor_krylov_basis, ops.col_ptr, ops.row_idx, ops.g, ops.c, np.zeros((3, 1)), sel, shifts, [False], 2, 1
    )
    _expect_fatal(
        core.mor_krylov_basis,
        ops.col_ptr,
        ops.row_idx,
        ops.g,
        ops.c,
        np.zeros((2, 0)),
        np.zeros((1, 3)),
        shifts,
        [False],
        2,
        1,
    )
    _expect_fatal(
        core.mor_krylov_basis,
        ops.col_ptr,
        ops.row_idx,
        ops.g,
        ops.c,
        np.zeros((2, 0)),
        sel,
        shifts,
        [False, True],
        2,
        1,
    )
    _expect_fatal(
        core.mor_krylov_basis,
        np.zeros(1, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 0)),
        np.zeros((1, 0)),
        shifts,
        [False],
        2,
        1,
    )
    _expect_fatal(core.mor_project, ops.col_ptr, ops.row_idx, ops.g, np.zeros((3, 1)))
    _expect_fatal(core.mor_project, ops.col_ptr, ops.row_idx, ops.g[:1], np.zeros((2, 1)))
    Gr = np.eye(2)
    _expect_fatal(core.mor_reduced_ac, Gr, np.eye(3), np.ones(2, dtype=complex), np.ones((1, 2)), np.ones(3), 1)
    _expect_fatal(core.mor_reduced_ac, Gr, Gr, np.ones(3, dtype=complex), np.ones((1, 2)), np.ones(3), 1)
    _expect_fatal(core.mor_reduced_ac, Gr, Gr, np.ones(2, dtype=complex), np.ones((1, 3)), np.ones(3), 1)
    V = np.eye(2)
    rows = np.zeros(0, dtype=np.int64)
    _expect_fatal(
        core.mor_reduced_transient, Gr, Gr, V, np.ones((1, 2)), np.ones(3), np.array([0.0, 1.0]), np.ones(2), rows, []
    )
    _expect_fatal(
        core.mor_reduced_transient, Gr, Gr, V, np.ones((1, 2)), np.ones(2), np.array([0.0, 0.0]), np.ones(2), rows, []
    )
    _expect_fatal(
        core.mor_reduced_transient,
        Gr,
        Gr,
        V,
        np.ones((1, 2)),
        np.ones(2),
        np.array([0.0, 1.0]),
        np.ones(2),
        np.array([5], dtype=np.int64),
        [],
    )
    _expect_fatal(
        core.mor_tpwl_simulate,
        Gr,
        Gr,
        V,
        np.ones((1, 2)),
        np.zeros((0, 2)),
        np.zeros((0, 2, 2)),
        np.zeros((0, 2)),
        np.ones(2),
        np.array([0.0, 1.0]),
        np.ones(2),
        rows,
        [],
    )
    _expect_fatal(core.mor_pod_basis, np.zeros((0, 3)), 2)
    _expect_fatal(core.mor_safe_solve, np.ones((2, 3)), np.ones((2, 1)))
    _expect_fatal(core.mor_safe_solve, np.eye(2), np.ones((3, 1)))
    _expect_fatal(core.mor_dense_operands, np.eye(2), np.eye(3))
    # Results: shapes and dtypes.
    values, info = core.mor_reduced_transient(
        Gr, Gr, V, np.ones((1, 2)), np.ones(2), np.array([0.0, 1.0]), np.ones(2), rows, []
    )
    assert values.shape == (1, 2) and values.dtype == np.float64 and info["failed"] is False
    indexes = core.mor_training_indexes(np.ones((2, 5)), 1, 1)
    assert indexes.dtype == np.int64 and indexes.tolist() == [0, 1, 2, 4]
    x, path = core.mor_safe_solve(np.eye(2), np.ones((2, 1)))
    assert path == "lu" and x.shape == (2, 1)
    _x, path = core.mor_safe_solve(np.ones((2, 2), dtype=complex), np.ones((2, 1), dtype=complex))
    assert path == "least_squares"


def test_linearized_system_matches_the_reference() -> None:
    netlist = next(case.netlist_text for case, _i, _m, _o in iter_analyses() if case.name == "ce_level1_ac_mor")
    ref_problem, live_problem = _problems(netlist)
    ref_G, ref_md = ref_linear._linearized_system(ref_problem, 200, 1e-9, 1e-9, {})
    live_G, live_md = linear._linearized_system(live_problem, 200, 1e-9, 1e-9, {})
    np.testing.assert_allclose(live_G, ref_G, rtol=1e-6, atol=1e-12)
    assert live_md["op_used"] is True and live_md["linearized_device_count"] == ref_md["linearized_device_count"]
    np.testing.assert_allclose(
        live_md["operating_point"]["values"], ref_md["operating_point"]["values"], rtol=1e-6, atol=1e-9
    )
    assert live_md["operating_point"]["labels"] == ref_md["operating_point"]["labels"]
