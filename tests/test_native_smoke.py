"""Smoke tests for the mandatory C++ extension module mna_simulation._core."""

from __future__ import annotations

import numpy as np
import pytest

from mna_simulation import native

core = native.require_core()

# A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]] in CSC form.
AP = np.array([0, 2, 5, 7], dtype=np.int32)
AI = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
AX = np.array([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0])
DENSE = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])


def test_version_and_build_info() -> None:
    assert core.version() == "0.70.0-dev"
    assert native.core_version() == core.version()
    info = core.build_info()
    expected_keys = {
        "version",
        "compiler",
        "compiler_version",
        "cxx_standard",
        "build_type",
        "threads_impl",
        "hardware_concurrency",
        "eigen_version",
        "suitesparse_version",
        "klu_version",
        "platform",
        "sanitizers",
        "coverage",
        "native_arch",
    }
    assert expected_keys <= set(info)
    assert info["version"] == core.version()
    assert info["threads_impl"] == "std::thread"
    assert info["cxx_standard"] >= 201703
    assert info["eigen_version"] == "3.4.0"
    assert info["suitesparse_version"] == "7.8.3"
    assert info["klu_version"] == "2.3.5"
    assert native.build_info() == info


def test_klu_solve_matches_numpy() -> None:
    x_true = np.array([1.0, -2.0, 0.5])
    b = DENSE @ x_true
    x = core.klu_solve(AP, AI, AX, b)
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    np.testing.assert_allclose(x, np.linalg.solve(DENSE, b), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(x, x_true, rtol=1e-14, atol=1e-14)


def test_singular_matrix_raises_core_error() -> None:
    ap = np.array([0, 2, 4], dtype=np.int32)
    ai = np.array([0, 1, 0, 1], dtype=np.int32)
    ax = np.array([1.0, 2.0, 2.0, 4.0])  # [[1, 2], [2, 4]], rank 1
    with pytest.raises(core.CoreError) as excinfo:
        core.klu_solve(ap, ai, ax, np.ones(2))
    assert excinfo.value.code == "JACOBIAN_SINGULAR"
    assert isinstance(excinfo.value, RuntimeError)


def test_malformed_input_raises_netlist_fatal() -> None:
    with pytest.raises(core.CoreError) as excinfo:
        core.klu_solve(AP, AI, AX[:-1], np.ones(3))
    assert excinfo.value.code == "NETLIST_FATAL"


def test_parallel_row_sums_is_deterministic() -> None:
    rng = np.random.default_rng(1234)
    a = rng.standard_normal((997, 513))
    results = {k: core.parallel_row_sums(a, k) for k in (1, 2, 4)}
    reference, used = results[1]
    assert used == 1
    for k, (sums, threads_used) in results.items():
        assert 1 <= threads_used <= k
        assert sums.shape == (a.shape[0],)
        assert np.array_equal(sums, reference), f"row sums differ with {k} threads"
    np.testing.assert_allclose(reference, a.sum(axis=1), rtol=1e-12, atol=1e-12)
