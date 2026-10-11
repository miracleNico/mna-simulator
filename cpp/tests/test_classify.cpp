// Matrix classification for Krylov method selection (sparse rule).
#include <doctest/doctest.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "mna/linear/classify.hpp"

using mna::linear::Complex;
using mna::linear::MatrixKind;

namespace {

using IVec = std::vector<std::int32_t>;

// Dense column-major n x n -> CSC (drops exact zeros) -> classification.
template <class T>
MatrixKind classify_dense(std::int32_t n, const std::vector<T>& colmajor, double tol = 1e-10) {
    IVec Ap = {0};
    IVec Ai;
    std::vector<T> Ax;
    for (std::int32_t j = 0; j < n; ++j) {
        for (std::int32_t i = 0; i < n; ++i) {
            const T v = colmajor[static_cast<std::size_t>(i + j * n)];
            if (v != T{}) {
                Ai.push_back(i);
                Ax.push_back(v);
            }
        }
        Ap.push_back(static_cast<std::int32_t>(Ai.size()));
    }
    return mna::linear::classify_matrix<T>(n, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size(),
                                           tol);
}

std::string name(MatrixKind kind) { return mna::linear::matrix_kind_name(kind); }

}  // namespace

TEST_CASE("the three matrices of test_krylov_solvers classify like the reference") {
    // Column-major; general, symmetric indefinite, symmetric positive definite.
    CHECK(name(classify_dense<double>(3, {4, 2, 0, 1, 3, 1, 0, 1, 2})) == "general");
    CHECK(name(classify_dense<double>(3, {4, 1, 0, 1, -3, 1, 0, 1, 2})) == "symmetric");
    CHECK(name(classify_dense<double>(3, {4, 1, 0, 1, 3, 1, 0, 1, 2})) == "positive_definite");
}

TEST_CASE("1x1, empty and diagonal matrices") {
    CHECK(classify_dense<double>(1, {1e-3}) == MatrixKind::PositiveDefinite);
    CHECK(classify_dense<double>(1, {-1e-3}) == MatrixKind::Symmetric);
    CHECK(classify_dense<double>(0, {}) == MatrixKind::PositiveDefinite);
    // A structurally empty 2x2 (all zeros) is symmetric but not positive definite.
    CHECK(classify_dense<double>(2, {0, 0, 0, 0}) == MatrixKind::Symmetric);
}

TEST_CASE("symmetry tolerance is absolute on the largest difference over the pattern") {
    CHECK(classify_dense<double>(2, {2, 1, 1 + 1e-11, 2}) == MatrixKind::PositiveDefinite);
    CHECK(classify_dense<double>(2, {2, 1, 1 + 1e-9, 2}) == MatrixKind::General);
    // An entry stored on one side only counts with its full magnitude.
    CHECK(classify_dense<double>(2, {2, 0, 5e-11, 2}) == MatrixKind::PositiveDefinite);
    CHECK(classify_dense<double>(2, {2, 0, 5e-10, 2}) == MatrixKind::General);
    // Large entries do not get a relative allowance (unlike np.allclose).
    CHECK(classify_dense<double>(2, {1e8, 1, 1 + 1e-6, 1e8}) == MatrixKind::General);
    // Custom tolerance.
    CHECK(classify_dense<double>(2, {2, 1, 1.001, 2}, 0.01) == MatrixKind::PositiveDefinite);
}

TEST_CASE("complex matrices: Hermitian vs complex symmetric") {
    // Hermitian positive definite [[2, 1-i], [1+i, 3]].
    CHECK(classify_dense<Complex>(2, {{2, 0}, {1, 1}, {1, -1}, {3, 0}}) == MatrixKind::PositiveDefinite);
    // Hermitian indefinite [[1, 2i], [-2i, 1]] (eigenvalues -1, 3).
    CHECK(classify_dense<Complex>(2, {{1, 0}, {0, -2}, {0, 2}, {1, 0}}) == MatrixKind::Symmetric);
    // Complex symmetric (A^T = A) but not Hermitian: the G + j w C of AC analysis.
    CHECK(classify_dense<Complex>(2, {{1, 1}, {0, 1}, {0, 1}, {1, 1}}) == MatrixKind::General);
    // A non-real diagonal makes A != A^H.
    CHECK(classify_dense<Complex>(1, {{1, 1}}) == MatrixKind::General);
}

TEST_CASE("the Cholesky test only runs up to n = 256") {
    for (std::int32_t n : {256, 257}) {
        std::vector<double> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        for (std::int32_t i = 0; i < n; ++i) {
            A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)] = 2.0;
            if (i + 1 < n) {
                A[static_cast<std::size_t>(i + (i + 1) * n)] = -1.0;
                A[static_cast<std::size_t>((i + 1) + i * n)] = -1.0;
            }
        }
        const MatrixKind expected = n <= mna::linear::kCholeskyMaxN ? MatrixKind::PositiveDefinite : MatrixKind::Symmetric;
        CHECK(classify_dense<double>(n, A) == expected);
    }
}

TEST_CASE("NaN follows NumPy: a NaN difference never makes the matrix general") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN in the upper triangle plus a clearly asymmetric pair: np.max(...) is
    // NaN, so the reference does not classify it as general; the Cholesky of
    // the lower triangle [[2], [1, 2], [0, 5, 2]] then fails (pivot < 0).
    CHECK(classify_dense<double>(3, {2, 1, 0, nan, 2, 5, 0, 0, 2}) == MatrixKind::Symmetric);
    // NaN pivots pass numpy.linalg.cholesky (OpenBLAS potrf) and propagate.
    CHECK(classify_dense<double>(2, {nan, 0, 0, 1}) == MatrixKind::PositiveDefinite);
    CHECK(classify_dense<double>(2, {nan, 0, 0, -1}) == MatrixKind::PositiveDefinite);
    CHECK(classify_dense<double>(2, {-1, 0, 0, nan}) == MatrixKind::Symmetric);
    // Infinite asymmetry is general.
    CHECK(classify_dense<double>(2, {1, std::numeric_limits<double>::infinity(), 0, 1}) == MatrixKind::General);
}

TEST_CASE("kind names") {
    CHECK(name(MatrixKind::General) == "general");
    CHECK(name(MatrixKind::Symmetric) == "symmetric");
    CHECK(name(MatrixKind::PositiveDefinite) == "positive_definite");
}
