// Dense Eigen LU solver (DenseSolver) behind the LinearSolver interface.
#include <doctest/doctest.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/dense_solver.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/linear/linear_solver.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::linear::Complex;
using mna::linear::CscPattern;
using mna::linear::DenseSolver;
using mna::linear::FactorPath;

namespace {

using IVec = std::vector<std::int32_t>;

// Column-major dense multiply y = A x (A n x n).
template <class T>
std::vector<T> dense_multiply(const std::vector<T>& A, const std::vector<T>& x, bool transpose = false) {
    const std::size_t n = x.size();
    std::vector<T> y(n, T{});
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            if (transpose) {
                y[j] += A[i + j * n] * x[i];
            } else {
                y[i] += A[i + j * n] * x[j];
            }
        }
    }
    return y;
}

template <class T>
void check_close(const std::vector<T>& got, const std::vector<T>& expected, double tol) {
    REQUIRE(got.size() == expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        CHECK(std::abs(got[i] - expected[i]) <= tol * (1.0 + std::abs(expected[i])));
    }
}

template <class T>
ErrorCode factor_code(std::int32_t n, const std::vector<T>& colmajor) {
    try {
        DenseSolver<T> lu(n);
        lu.factor(colmajor.data(), colmajor.size());
    } catch (const Error& e) {
        return e.code();
    }
    return ErrorCode::Unknown;
}

// A = [[4, 1, 0], [2, 3, 1], [0, 1, 2]] column-major, and its CSC form.
const std::vector<double> kDense = {4.0, 2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
const IVec kAp = {0, 2, 5, 7};
const IVec kAi = {0, 1, 0, 1, 2, 1, 2};
const std::vector<double> kAx = {4.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0};

}  // namespace

TEST_CASE("dense solver: dense input, known solution, transpose solve, several right-hand sides") {
    DenseSolver<double> lu(3);
    CHECK_FALSE(lu.has_pattern());
    CHECK(std::string(lu.kind()) == "dense");
    CHECK_FALSE(lu.factored());
    lu.factor(kDense.data(), kDense.size());
    CHECK(lu.factored());
    CHECK(lu.size() == 3);
    CHECK(lu.nnz_lu() == 12);

    const std::vector<double> x = {1.0, -2.0, 0.5};
    std::vector<double> b = dense_multiply(kDense, x);
    lu.solve(b.data(), b.size(), 1);
    check_close(b, x, 1e-14);

    std::vector<double> bt = dense_multiply(kDense, x, /*transpose=*/true);
    lu.tsolve(bt.data(), bt.size(), 1, false);
    check_close(bt, x, 1e-14);

    std::vector<double> two = dense_multiply(kDense, x);
    const std::vector<double> e2 = dense_multiply(kDense, std::vector<double>{0.0, 1.0, 0.0});
    two.insert(two.end(), e2.begin(), e2.end());
    lu.solve(two.data(), two.size(), 2);
    check_close(two, std::vector<double>{1.0, -2.0, 0.5, 0.0, 1.0, 0.0}, 1e-14);

    // refactor has no pivot order to reuse.
    CHECK(lu.refactor(kDense.data(), kDense.size()) == FactorPath::FullFactor);
    const double rc = lu.rcond();
    CHECK(rc > 0.0);
    CHECK(rc <= 1.0);
}

TEST_CASE("dense solver: CSC input is scattered and matches KLU") {
    const CscPattern pattern = CscPattern::from_arrays(3, kAp.data(), kAp.size(), kAi.data(), kAi.size());
    DenseSolver<double> lu(pattern);
    CHECK(lu.has_pattern());
    lu.factor(kAx.data(), kAx.size());
    mna::linear::KluSolver klu;
    klu.analyze(pattern);
    klu.factor(kAx);
    const std::vector<double> x = {0.25, 3.0, -1.0};
    std::vector<double> b_dense = dense_multiply(kDense, x);
    std::vector<double> b_klu = b_dense;
    lu.solve(b_dense.data(), b_dense.size(), 1);
    klu.solve(b_klu);
    check_close(b_dense, x, 1e-14);
    check_close(b_dense, b_klu, 1e-14);
}

TEST_CASE("dense solver: complex system, transpose and adjoint solves") {
    // A = [[2+i, 1], [-i, 3-2i]] column-major.
    const std::vector<Complex> A = {{2, 1}, {0, -1}, {1, 0}, {3, -2}};
    DenseSolver<Complex> lu(2);
    lu.factor(A.data(), A.size());
    const std::vector<Complex> x = {{1, 1}, {-0.5, 2}};
    std::vector<Complex> b = dense_multiply(A, x);
    lu.solve(b.data(), b.size(), 1);
    check_close(b, x, 1e-14);

    std::vector<Complex> bt = dense_multiply(A, x, /*transpose=*/true);
    lu.tsolve(bt.data(), bt.size(), 1, false);
    check_close(bt, x, 1e-14);

    std::vector<Complex> Ah(A.size());
    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < 2; ++i) {
            Ah[i + j * 2] = std::conj(A[j + i * 2]);
        }
    }
    std::vector<Complex> bh = dense_multiply(Ah, x);
    lu.tsolve(bh.data(), bh.size(), 1, /*conjugate=*/true);
    check_close(bh, x, 1e-14);
}

TEST_CASE("dense solver: singular and nearly singular matrices raise JACOBIAN_SINGULAR") {
    // Exactly singular: rank 1.
    CHECK(factor_code<double>(2, {1.0, 2.0, 2.0, 4.0}) == ErrorCode::JacobianSingular);
    // All zeros.
    CHECK(factor_code<double>(2, {0.0, 0.0, 0.0, 0.0}) == ErrorCode::JacobianSingular);
    // Zero column.
    CHECK(factor_code<double>(3, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}) == ErrorCode::JacobianSingular);
    // Nearly singular: |u_22| = 1e-17 < eps * max|u| = 2.2e-16.
    CHECK(factor_code<double>(2, {1.0, 0.0, 0.0, 1e-17}) == ErrorCode::JacobianSingular);
    // Nearly singular after elimination: [[1, 1], [1, 1 + 1e-17]] rounds to rank 1.
    CHECK(factor_code<double>(2, {1.0, 1.0, 1.0, 1.0 + 1e-17}) == ErrorCode::JacobianSingular);
    // Just above the threshold: |u_22| = 1e-15 > 2.2e-16 is accepted.
    CHECK(factor_code<double>(2, {1.0, 0.0, 0.0, 1e-15}) == ErrorCode::Unknown);
    // Non-finite input.
    CHECK(factor_code<double>(2, {1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0}) ==
          ErrorCode::JacobianSingular);
    CHECK(factor_code<double>(2, {std::numeric_limits<double>::infinity(), 0.0, 0.0, 1.0}) ==
          ErrorCode::JacobianSingular);
    // Complex singular.
    CHECK(factor_code<Complex>(2, {{1, 1}, {1, 0}, {2, 2}, {2, 0}}) == ErrorCode::JacobianSingular);

    // The solver is usable again after a failed factorization.
    DenseSolver<double> lu(2);
    const std::vector<double> bad = {1.0, 2.0, 2.0, 4.0};
    CHECK_THROWS_AS(lu.factor(bad.data(), bad.size()), Error);
    CHECK_FALSE(lu.factored());
    const std::vector<double> good = {2.0, 0.0, 0.0, 4.0};
    lu.factor(good.data(), good.size());
    std::vector<double> b = {2.0, 4.0};
    lu.solve(b.data(), b.size(), 1);
    check_close(b, std::vector<double>{1.0, 1.0}, 1e-15);
    // rcond is the pivot ratio.
    CHECK(lu.rcond() == doctest::Approx(0.5));
}

TEST_CASE("dense solver: n = 0, n = 1 and malformed input") {
    DenseSolver<double> empty(0);
    empty.factor(nullptr, 0);
    CHECK(empty.factored());
    empty.solve(nullptr, 0, 1);
    CHECK(empty.rcond() == 1.0);
    CHECK(empty.nnz_lu() == 0);

    DenseSolver<double> one(1);
    const std::vector<double> a = {-4.0};
    one.factor(a.data(), a.size());
    std::vector<double> b = {2.0};
    one.solve(b.data(), b.size(), 1);
    CHECK(b[0] == doctest::Approx(-0.5));
    CHECK(factor_code<double>(1, {0.0}) == ErrorCode::JacobianSingular);

    auto code = [](auto fn) {
        try {
            fn();
        } catch (const Error& e) {
            return e.code();
        }
        return ErrorCode::Unknown;
    };
    CHECK(code([] {
              DenseSolver<double> lu(2);
              const std::vector<double> v = {1.0, 2.0, 3.0};
              lu.factor(v.data(), v.size());
          }) == ErrorCode::NetlistFatal);
    CHECK(code([] {
              DenseSolver<double> lu(2);
              std::vector<double> rhs = {1.0, 2.0};
              lu.solve(rhs.data(), rhs.size(), 1);
          }) == ErrorCode::NetlistFatal);
    CHECK(code([] {
              DenseSolver<double> lu(2);
              const std::vector<double> v = {1.0, 0.0, 0.0, 1.0};
              lu.factor(v.data(), v.size());
              std::vector<double> rhs = {1.0, 2.0, 3.0};
              lu.solve(rhs.data(), rhs.size(), 1);
          }) == ErrorCode::NetlistFatal);
    CHECK(code([] { DenseSolver<double> lu(-1); }) == ErrorCode::NetlistFatal);
}
