// Complex KLU (KluSolverZ) and the shared LinearSolver factory.
#include <doctest/doctest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/linear/linear_solver.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::linear::Complex;
using mna::linear::CscPattern;
using mna::linear::FactorPath;
using mna::linear::KluSolverZ;

namespace {

using CVec = std::vector<Complex>;
using IVec = std::vector<std::int32_t>;

// y = A x (or A^T x / A^H x) for a CSC matrix.
CVec multiply(const IVec& Ap, const IVec& Ai, const CVec& Ax, const CVec& x, bool transpose = false,
              bool conjugate = false) {
    CVec y(x.size(), Complex{});
    for (std::size_t j = 0; j + 1 < Ap.size(); ++j) {
        for (auto p = static_cast<std::size_t>(Ap[j]); p < static_cast<std::size_t>(Ap[j + 1]); ++p) {
            const auto i = static_cast<std::size_t>(Ai[p]);
            const Complex a = conjugate ? std::conj(Ax[p]) : Ax[p];
            if (transpose) {
                y[j] += a * x[i];
            } else {
                y[i] += a * x[j];
            }
        }
    }
    return y;
}

void check_close(const CVec& got, const CVec& expected, double tol) {
    REQUIRE(got.size() == expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        CHECK(std::abs(got[i] - expected[i]) <= tol * (1.0 + std::abs(expected[i])));
    }
}

ErrorCode code_of(void (*fn)()) {
    try {
        fn();
    } catch (const Error& e) {
        return e.code();
    }
    return ErrorCode::Unknown;
}

// Non-Hermitian complex 3x3:
//   [[4+1i, 1,    0   ],
//    [2i,   3-1i, 1+1i],
//    [0,    1,    2+2i]]
const IVec kAp = {0, 2, 5, 7};
const IVec kAi = {0, 1, 0, 1, 2, 1, 2};
const CVec kAx = {{4, 1}, {0, 2}, {1, 0}, {3, -1}, {1, 0}, {1, 1}, {2, 2}};

// Hermitian positive definite 3x3:
//   [[4,    1-1i, 0   ],
//    [1+1i, 3,    2i  ],
//    [0,   -2i,   5   ]]
const CVec kHx = {{4, 0}, {1, 1}, {1, -1}, {3, 0}, {0, -2}, {0, 2}, {5, 0}};

}  // namespace

TEST_CASE("complex KLU solves a known non-Hermitian 3x3 system") {
    KluSolverZ lu;
    lu.analyze(3, kAp, kAi);
    CHECK(lu.analyzed());
    CHECK(std::string(lu.kind()) == "klu");
    lu.factor(kAx);
    CHECK(lu.factored());
    CHECK(lu.nnz_lu() >= 3);

    const CVec x = {{1, -1}, {0.5, 2}, {-3, 0.25}};
    CVec b = multiply(kAp, kAi, kAx, x);
    lu.solve(b);
    check_close(b, x, 1e-14);

    const double rc = lu.rcond();
    CHECK(rc > 0.0);
    CHECK(rc <= 1.0);
}

TEST_CASE("complex KLU solves a Hermitian system and several right-hand sides") {
    KluSolverZ lu;
    lu.analyze(3, kAp, kAi);
    lu.factor(kHx);
    const CVec x1 = {{1, 0}, {0, 1}, {-1, -1}};
    const CVec x2 = {{2, 3}, {-1, 0.5}, {0, 4}};
    CVec b1 = multiply(kAp, kAi, kHx, x1);
    CVec b2 = multiply(kAp, kAi, kHx, x2);
    CVec b = b1;
    b.insert(b.end(), b2.begin(), b2.end());
    lu.solve(b, 2);
    check_close(CVec(b.begin(), b.begin() + 3), x1, 1e-14);
    check_close(CVec(b.begin() + 3, b.end()), x2, 1e-14);

    // For a Hermitian matrix, A^H x = b and A x = b have the same solution.
    CVec c = multiply(kAp, kAi, kHx, x1);
    lu.tsolve(c, 1, /*conjugate=*/true);
    check_close(c, x1, 1e-14);
}

TEST_CASE("complex transpose and conjugate-transpose solves match explicit transposes") {
    KluSolverZ lu;
    lu.analyze(3, kAp, kAi);
    lu.factor(kAx);
    const CVec x = {{0.5, 1}, {-2, 0}, {1, -1}};

    CVec bt = multiply(kAp, kAi, kAx, x, /*transpose=*/true);
    lu.tsolve(bt);  // A^T x = b
    check_close(bt, x, 1e-14);

    CVec bh = multiply(kAp, kAi, kAx, x, /*transpose=*/true, /*conjugate=*/true);
    lu.tsolve(bh, 1, /*conjugate=*/true);  // A^H x = b
    check_close(bh, x, 1e-14);

    // Explicit transpose: factor A^T (CSC of A^T = CSR of A) and solve normally.
    // A^T for kAx in CSC: column j of A^T = row j of A.
    const IVec Tp = {0, 2, 5, 7};
    const IVec Ti = {0, 1, 0, 1, 2, 1, 2};
    const CVec Tx = {{4, 1}, {1, 0}, {0, 2}, {3, -1}, {1, 1}, {1, 0}, {2, 2}};
    KluSolverZ lt;
    lt.analyze(3, Tp, Ti);
    lt.factor(Tx);
    CVec b_explicit = multiply(kAp, kAi, kAx, x, /*transpose=*/true);
    lt.solve(b_explicit);
    CVec b_tsolve = multiply(kAp, kAi, kAx, x, /*transpose=*/true);
    lu.tsolve(b_tsolve);
    check_close(b_tsolve, b_explicit, 1e-14);
}

TEST_CASE("singular complex matrices raise JACOBIAN_SINGULAR") {
    // [[1+i, 2+2i], [1, 2]]: second column is twice the first.
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(2, IVec{0, 2, 4}, IVec{0, 1, 0, 1});
              lu.factor(CVec{{1, 1}, {1, 0}, {2, 2}, {2, 0}});
          }) == ErrorCode::JacobianSingular);
    // A pattern without any entries (null value array).
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(2, IVec{0, 0, 0}, IVec{});
              lu.factor(nullptr, 0);
          }) == ErrorCode::JacobianSingular);
    // Structurally singular: empty column.
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(2, IVec{0, 2, 2}, IVec{0, 1});
              lu.factor(CVec{{1, 0}, {0, 1}});
          }) == ErrorCode::JacobianSingular);
    // The solver stays usable afterwards.
    KluSolverZ lu;
    lu.analyze(2, IVec{0, 2, 4}, IVec{0, 1, 0, 1});
    CHECK_THROWS_AS(lu.factor(CVec{{1, 1}, {1, 0}, {2, 2}, {2, 0}}), Error);
    CHECK_FALSE(lu.factored());
    lu.factor(CVec{{0, 2}, {0, 0}, {0, 0}, {4, 0}});
    CVec b = {{0, 2}, {4, 0}};
    lu.solve(b);
    check_close(b, CVec{{1, 0}, {1, 0}}, 1e-15);
}

TEST_CASE("complex refactor reuses the pivot order and falls back when needed") {
    KluSolverZ lu;
    lu.analyze(3, kAp, kAi);
    CHECK(lu.refactor(kAx) == FactorPath::FullFactor);  // no factorization yet
    CVec ax2 = kAx;
    for (Complex& v : ax2) {
        v *= Complex(1.5, -0.5);
    }
    CHECK(lu.refactor(ax2) == FactorPath::Refactor);
    const CVec x = {{1, 2}, {3, 4}, {5, 6}};
    CVec b = multiply(kAp, kAi, ax2, x);
    lu.solve(b);
    check_close(b, x, 1e-13);

    // Zero diagonal pivots in the old order: [[0, i], [i, 0]].
    KluSolverZ two;
    two.analyze(2, IVec{0, 2, 4}, IVec{0, 1, 0, 1});
    two.factor(CVec{{1, 0}, {1, 0}, {1, 0}, {2, 0}});
    CHECK(two.refactor(CVec{{0, 0}, {0, 1}, {0, 1}, {0, 0}}) == FactorPath::FullFactor);
    CVec c = {{0, -2}, {0, 3}};  // A (3, -2) = (i*(-2), i*3)
    two.solve(c);
    check_close(c, CVec{{3, 0}, {-2, 0}}, 1e-15);
}

TEST_CASE("complex KLU handles n = 0 and n = 1") {
    KluSolverZ empty;
    empty.analyze(0, IVec{0}, IVec{});
    empty.factor(CVec{});
    CVec none;
    empty.solve(none);
    empty.tsolve(none, 1, true);
    CHECK(empty.rcond() == 1.0);
    CHECK(empty.nnz_lu() == 0);
    CHECK(empty.refactor(CVec{}) == FactorPath::Refactor);

    KluSolverZ one;
    one.analyze(1, IVec{0, 1}, IVec{0});
    one.factor(CVec{{0, 2}});
    CVec b = {{2, 0}};
    one.solve(b);
    check_close(b, CVec{{0, -1}}, 1e-15);
    b[0] = Complex(2, 0);
    one.tsolve(b, 1, /*conjugate=*/true);  // conj(2i) = -2i -> x = 2 / (-2i) = i
    check_close(b, CVec{{0, 1}}, 1e-15);
    CHECK(one.rcond() == doctest::Approx(1.0));
    CHECK(code_of([] {
              KluSolverZ z;
              z.analyze(1, IVec{0, 1}, IVec{0});
              z.factor(CVec{{0, 0}});
          }) == ErrorCode::JacobianSingular);
}

TEST_CASE("complex KLU rejects malformed input and out-of-order calls") {
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(3, kAp, kAi);
              lu.factor(CVec{{1, 0}});
          }) == ErrorCode::NetlistFatal);
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(3, kAp, kAi);
              CVec b(3);
              lu.tsolve(b);
          }) == ErrorCode::NetlistFatal);
    CHECK(code_of([] {
              KluSolverZ lu;
              lu.analyze(3, kAp, kAi);
              lu.factor(kAx);
              CVec b(2);
              lu.solve(b);
          }) == ErrorCode::NetlistFatal);
}

TEST_CASE("make_linear_solver builds analyzed KLU solvers (dense policy threshold 0)") {
    CHECK(mna::linear::kDenseMaxN == 0);
    for (std::int32_t n : {0, 1, 2, 64, 100000}) {
        CHECK_FALSE(mna::linear::use_dense_solver(n));
    }
    const CscPattern pattern = CscPattern::from_arrays(3, kAp.data(), kAp.size(), kAi.data(), kAi.size());
    auto solver = mna::linear::make_linear_solver<Complex>(pattern);
    CHECK(std::string(solver->kind()) == "klu");
    CHECK(solver->size() == 3);
    solver->factor(kAx.data(), kAx.size());
    const CVec x = {{1, 0}, {2, 0}, {3, 0}};
    CVec b = multiply(kAp, kAi, kAx, x);
    solver->solve(b.data(), b.size(), 1);
    check_close(b, x, 1e-14);

    auto dense = mna::linear::make_linear_solver<Complex>(pattern, mna::linear::SolverKind::Dense);
    CHECK(std::string(dense->kind()) == "dense");
    dense->factor(kAx.data(), kAx.size());
    CVec bd = multiply(kAp, kAi, kAx, x);
    dense->solve(bd.data(), bd.size(), 1);
    check_close(bd, x, 1e-14);
}
