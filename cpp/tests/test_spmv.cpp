// CSR row view and sparse matrix-vector product (serial and parallel).
#include <doctest/doctest.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/sparse.hpp"

using mna::linear::Complex;
using mna::linear::CscPattern;
using mna::linear::CsrMatrix;

namespace {

using IVec = std::vector<std::int32_t>;

// Deterministic pseudo-random numbers in [-0.5, 0.5).
struct Lcg {
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    double next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(state >> 11) / 9007199254740992.0 - 0.5;
    }
};

// Perturbed 5-point Laplacian on a side x side grid, CSC with sorted rows.
struct Csc {
    std::int32_t n = 0;
    IVec Ap;
    IVec Ai;
    std::vector<double> Ax;
};

Csc laplacian(std::int32_t side, bool symmetric) {
    Csc m;
    m.n = side * side;
    m.Ap.push_back(0);
    Lcg rng;
    std::vector<double> sym(static_cast<std::size_t>(m.n) * 2, 0.0);  // horizontal / vertical edge weights
    for (double& w : sym) {
        w = -1.0 + 0.1 * rng.next();
    }
    for (std::int32_t col = 0; col < m.n; ++col) {
        const std::int32_t r = col / side;
        const std::int32_t c = col % side;
        auto add = [&](std::int32_t row, double v) {
            m.Ai.push_back(row);
            m.Ax.push_back(v);
        };
        auto edge = [&](std::int32_t a, std::int32_t b, int dir) {
            const std::int32_t lo = a < b ? a : b;
            const double w = sym[static_cast<std::size_t>(lo) * 2 + static_cast<std::size_t>(dir)];
            return symmetric ? w : w + 0.05 * rng.next();
        };
        if (r > 0) add(col - side, edge(col, col - side, 1));
        if (c > 0) add(col - 1, edge(col, col - 1, 0));
        add(col, 4.0 + 0.5 * rng.next());
        if (c + 1 < side) add(col + 1, edge(col, col + 1, 0));
        if (r + 1 < side) add(col + side, edge(col, col + side, 1));
        m.Ap.push_back(static_cast<std::int32_t>(m.Ai.size()));
    }
    return m;
}

// Reference y = A x straight from CSC.
std::vector<double> csc_multiply(const Csc& m, const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(m.n), 0.0);
    for (std::int32_t j = 0; j < m.n; ++j) {
        for (std::int32_t p = m.Ap[static_cast<std::size_t>(j)]; p < m.Ap[static_cast<std::size_t>(j) + 1]; ++p) {
            y[static_cast<std::size_t>(m.Ai[static_cast<std::size_t>(p)])] +=
                m.Ax[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(j)];
        }
    }
    return y;
}

}  // namespace

TEST_CASE("CSR view of a CSC matrix lists each row in column order") {
    // A = [[1, 0, 2], [0, 3, 0], [4, 5, 6]] with unsorted row indices in column 2.
    const IVec Ap = {0, 2, 4, 6};
    const IVec Ai = {2, 0, 1, 2, 2, 0};
    const std::vector<double> Ax = {4.0, 1.0, 3.0, 5.0, 6.0, 2.0};
    const auto A = CsrMatrix<double>::from_csc(3, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size());
    CHECK(A.n == 3);
    CHECK(A.nnz() == 6);
    CHECK(A.row_ptr == IVec{0, 2, 3, 6});
    CHECK(A.col_idx == IVec{0, 2, 1, 0, 1, 2});
    CHECK(A.values == std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    const std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y(3);
    CHECK(mna::linear::spmv(A, x.data(), y.data(), 8) == 1);  // small: always serial
    CHECK(y == std::vector<double>{7.0, 6.0, 32.0});
}

TEST_CASE("complex SpMV and empty matrices") {
    // A = [[i, 1], [0, 2 - i]].
    const IVec Ap = {0, 1, 3};
    const IVec Ai = {0, 0, 1};
    const std::vector<Complex> Ax = {{0, 1}, {1, 0}, {2, -1}};
    const auto A =
        CsrMatrix<Complex>::from_csc(2, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size());
    const std::vector<Complex> x = {{1, 1}, {0, 2}};
    std::vector<Complex> y(2);
    mna::linear::spmv(A, x.data(), y.data(), 1);
    CHECK(y[0] == Complex(-1, 3));  // i (1 + i) + 2i
    CHECK(y[1] == Complex(2, 4));   // (2 - i) 2i

    const IVec Ep = {0};
    const auto E = CsrMatrix<double>::from_csc(0, Ep.data(), Ep.size(), nullptr, 0, nullptr, 0);
    CHECK(E.n == 0);
    CHECK(mna::linear::spmv(E, static_cast<const double*>(nullptr), static_cast<double*>(nullptr), 4) == 1);
}

TEST_CASE("CSR view rejects malformed CSC input") {
    auto code = [](auto fn) {
        try {
            fn();
        } catch (const mna::Error& e) {
            return e.code();
        }
        return mna::ErrorCode::Unknown;
    };
    CHECK(code([] {
              const IVec Ap = {0, 1, 2};
              const IVec Ai = {0, 2};
              const std::vector<double> Ax = {1.0, 1.0};
              CsrMatrix<double>::from_csc(2, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size());
          }) == mna::ErrorCode::NetlistFatal);
    CHECK(code([] {
              const IVec Ap = {0, 1, 2};
              const IVec Ai = {0, 1};
              const std::vector<double> Ax = {1.0};
              CsrMatrix<double>::from_csc(2, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size());
          }) == mna::ErrorCode::NetlistFatal);
}

TEST_CASE("parallel SpMV above the threshold is bitwise identical for every thread count") {
    const Csc m = laplacian(70, /*symmetric=*/false);  // n = 4900 >= kParallelSpmvMinN
    REQUIRE(m.n >= mna::linear::kParallelSpmvMinN);
    const auto A = CsrMatrix<double>::from_csc(m.n, m.Ap.data(), m.Ap.size(), m.Ai.data(), m.Ai.size(), m.Ax.data(),
                                               m.Ax.size());
    Lcg rng;
    std::vector<double> x(static_cast<std::size_t>(m.n));
    for (double& v : x) {
        v = rng.next() * 1e3;
    }
    const std::vector<double> expected = csc_multiply(m, x);

    std::vector<double> serial(static_cast<std::size_t>(m.n));
    CHECK(mna::linear::spmv(A, x.data(), serial.data(), 1) == 1);
    for (std::size_t i = 0; i < serial.size(); ++i) {
        CHECK(serial[i] == doctest::Approx(expected[i]).epsilon(1e-12));
    }
    for (int threads : {2, 3, 4, 7}) {
        std::vector<double> parallel(static_cast<std::size_t>(m.n));
        const int used = mna::linear::spmv(A, x.data(), parallel.data(), threads);
        CHECK(used == threads);
        CHECK(std::memcmp(parallel.data(), serial.data(), serial.size() * sizeof(double)) == 0);
    }

    // Below the threshold the product stays on the calling thread.
    const Csc small = laplacian(60, false);  // n = 3600
    const auto S = CsrMatrix<double>::from_csc(small.n, small.Ap.data(), small.Ap.size(), small.Ai.data(),
                                               small.Ai.size(), small.Ax.data(), small.Ax.size());
    std::vector<double> xs(static_cast<std::size_t>(small.n), 1.0);
    std::vector<double> ys(static_cast<std::size_t>(small.n));
    CHECK(mna::linear::spmv(S, xs.data(), ys.data(), 4) == 1);
}
