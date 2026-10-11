#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::linear::FactorPath;
using mna::linear::KluSolver;

namespace {

// A = [[4, 1, 0],
//      [1, 3, 1],
//      [0, 1, 2]] in CSC form.
const std::vector<std::int32_t> kAp = {0, 2, 5, 7};
const std::vector<std::int32_t> kAi = {0, 1, 0, 1, 2, 1, 2};
const std::vector<double> kAx = {4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};

// y = A x for the CSC matrix (Ap, Ai, Ax).
std::vector<double> multiply(const std::vector<std::int32_t>& Ap, const std::vector<std::int32_t>& Ai,
                             const std::vector<double>& Ax, const std::vector<double>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t j = 0; j + 1 < Ap.size(); ++j) {
        for (auto p = static_cast<std::size_t>(Ap[j]); p < static_cast<std::size_t>(Ap[j + 1]); ++p) {
            y[static_cast<std::size_t>(Ai[p])] += Ax[p] * x[j];
        }
    }
    return y;
}

ErrorCode code_of(void (*fn)()) {
    try {
        fn();
    } catch (const Error& e) {
        return e.code();
    }
    return ErrorCode::Unknown;
}

}  // namespace

TEST_CASE("error codes use the Python core's names") {
    CHECK(std::string(mna::error_code_name(ErrorCode::JacobianSingular)) == "JACOBIAN_SINGULAR");
    CHECK(std::string(mna::error_code_name(ErrorCode::NetlistFatal)) == "NETLIST_FATAL");
    CHECK(std::string(mna::error_code_name(ErrorCode::Unknown)) == "UNKNOWN");
    const Error e(ErrorCode::NetlistFatal, "bad netlist");
    CHECK(std::string(e.code_name()) == "NETLIST_FATAL");
    CHECK(std::string(e.what()) == "bad netlist");
}

TEST_CASE("KLU solves a known 3x3 system") {
    KluSolver lu;
    CHECK_FALSE(lu.analyzed());
    lu.analyze(3, kAp, kAi);
    CHECK(lu.analyzed());
    CHECK_FALSE(lu.factored());
    CHECK(lu.size() == 3);
    CHECK(lu.nnz() == 7);

    lu.factor(kAx);
    CHECK(lu.factored());
    CHECK(lu.nnz_lu() >= 3);

    // Exact solution x = (1, 2, 3) -> b = A x = (6, 10, 8).
    std::vector<double> b = {6.0, 10.0, 8.0};
    lu.solve(b);
    CHECK(b[0] == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(b[1] == doctest::Approx(2.0).epsilon(1e-14));
    CHECK(b[2] == doctest::Approx(3.0).epsilon(1e-14));

    const double rc = lu.rcond();
    CHECK(rc > 0.0);
    CHECK(rc <= 1.0);
}

TEST_CASE("KLU solves several right-hand sides in place") {
    KluSolver lu;
    lu.analyze(3, kAp, kAi);
    lu.factor(kAx);
    // Column-major 3 x 2: solutions (1, 2, 3) and (-1, 0, 1).
    std::vector<double> b = {6.0, 10.0, 8.0, -4.0, 0.0, 2.0};
    lu.solve(b, 2);
    const std::vector<double> expected = {1.0, 2.0, 3.0, -1.0, 0.0, 1.0};
    for (std::size_t i = 0; i < b.size(); ++i) {
        CHECK(b[i] == doctest::Approx(expected[i]).epsilon(1e-14));
    }
}

TEST_CASE("refactor with changed values reuses the pivot order") {
    KluSolver lu;
    lu.analyze(3, kAp, kAi);

    // refactor before any factor() takes the full path.
    CHECK(lu.refactor(kAx) == FactorPath::FullFactor);

    const std::vector<double> ax2 = {5.0, 0.5, 0.5, 4.0, 2.0, 2.0, 6.0};
    CHECK(lu.refactor(ax2) == FactorPath::Refactor);

    const std::vector<double> x = {0.25, -1.5, 2.0};
    std::vector<double> b = multiply(kAp, kAi, ax2, x);
    lu.solve(b);
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK(b[i] == doctest::Approx(x[i]).epsilon(1e-13));
    }
}

TEST_CASE("refactor falls back to a full factorization when the old pivots fail") {
    // A1 = [[1, 1], [1, 2]] pivots on the diagonal. A2 = [[0, 1], [1, 0]] has
    // zeros in both diagonal pivot positions, so the old order fails whatever
    // the column ordering, while A2 itself is nonsingular.
    const std::vector<std::int32_t> Ap = {0, 2, 4};
    const std::vector<std::int32_t> Ai = {0, 1, 0, 1};
    KluSolver lu;
    lu.analyze(2, Ap, Ai);
    lu.factor(std::vector<double>{1.0, 1.0, 1.0, 2.0});

    const std::vector<double> ax2 = {0.0, 1.0, 1.0, 0.0};
    CHECK(lu.refactor(ax2) == FactorPath::FullFactor);

    const std::vector<double> x = {3.0, -2.0};
    std::vector<double> b = multiply(Ap, Ai, ax2, x);
    lu.solve(b);
    CHECK(b[0] == doctest::Approx(3.0).epsilon(1e-14));
    CHECK(b[1] == doctest::Approx(-2.0).epsilon(1e-14));
}

TEST_CASE("singular matrices raise JACOBIAN_SINGULAR") {
    SUBCASE("numerically singular") {
        // [[1, 2], [2, 4]]: rank 1.
        CHECK(code_of([] {
                  KluSolver lu;
                  lu.analyze(2, std::vector<std::int32_t>{0, 2, 4}, std::vector<std::int32_t>{0, 1, 0, 1});
                  lu.factor(std::vector<double>{1.0, 2.0, 2.0, 4.0});
              }) == ErrorCode::JacobianSingular);
    }
    SUBCASE("structurally singular (empty column)") {
        CHECK(code_of([] {
                  KluSolver lu;
                  lu.analyze(2, std::vector<std::int32_t>{0, 2, 2}, std::vector<std::int32_t>{0, 1});
                  lu.factor(std::vector<double>{1.0, 1.0});
              }) == ErrorCode::JacobianSingular);
    }
    SUBCASE("solver stays usable after a singular factorization") {
        KluSolver lu;
        lu.analyze(2, std::vector<std::int32_t>{0, 2, 4}, std::vector<std::int32_t>{0, 1, 0, 1});
        CHECK_THROWS_AS(lu.factor(std::vector<double>{1.0, 2.0, 2.0, 4.0}), Error);
        CHECK_FALSE(lu.factored());
        lu.factor(std::vector<double>{2.0, 0.0, 0.0, 4.0});
        std::vector<double> b = {2.0, 4.0};
        lu.solve(b);
        CHECK(b[0] == doctest::Approx(1.0));
        CHECK(b[1] == doctest::Approx(1.0));
    }
}

TEST_CASE("1x1 system") {
    KluSolver lu;
    lu.analyze(1, std::vector<std::int32_t>{0, 1}, std::vector<std::int32_t>{0});
    lu.factor(std::vector<double>{4.0});
    std::vector<double> b = {2.0};
    lu.solve(b);
    CHECK(b[0] == doctest::Approx(0.5).epsilon(1e-15));
    CHECK(lu.rcond() == doctest::Approx(1.0));
    CHECK(lu.refactor(std::vector<double>{-8.0}) == FactorPath::Refactor);
    b[0] = 2.0;  // (not b = {2.0}: GCC 13 -Warray-bounds false positive under ASan)
    lu.solve(b);
    CHECK(b[0] == doctest::Approx(-0.25).epsilon(1e-15));
    CHECK(code_of([] {
              KluSolver z;
              z.analyze(1, std::vector<std::int32_t>{0, 1}, std::vector<std::int32_t>{0});
              z.factor(std::vector<double>{0.0});
          }) == ErrorCode::JacobianSingular);
}

TEST_CASE("n = 0 is a valid empty system") {
    KluSolver lu;
    lu.analyze(0, std::vector<std::int32_t>{0}, std::vector<std::int32_t>{});
    CHECK(lu.analyzed());
    CHECK(lu.size() == 0);
    lu.factor(std::vector<double>{});
    CHECK(lu.factored());
    std::vector<double> b;
    lu.solve(b);
    CHECK(b.empty());
    CHECK(lu.rcond() == 1.0);
    CHECK(lu.nnz_lu() == 0);
    CHECK(lu.refactor(std::vector<double>{}) == FactorPath::Refactor);
}

TEST_CASE("malformed input raises NETLIST_FATAL") {
    // Wrong Ap length.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(3, std::vector<std::int32_t>{0, 1, 2}, std::vector<std::int32_t>{0, 1});
          }) == ErrorCode::NetlistFatal);
    // Row index out of range.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(2, std::vector<std::int32_t>{0, 1, 2}, std::vector<std::int32_t>{0, 2});
          }) == ErrorCode::NetlistFatal);
    // Duplicate entry in a column.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(2, std::vector<std::int32_t>{0, 2, 3}, std::vector<std::int32_t>{0, 0, 1});
          }) == ErrorCode::NetlistFatal);
    // Decreasing column pointers.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(2, std::vector<std::int32_t>{0, 2, 1}, std::vector<std::int32_t>{0, 1});
          }) == ErrorCode::NetlistFatal);
    // Negative dimension.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(-1, std::vector<std::int32_t>{0}, std::vector<std::int32_t>{});
          }) == ErrorCode::NetlistFatal);
    // Value count does not match the pattern.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(3, kAp, kAi);
              lu.factor(std::vector<double>{1.0, 2.0});
          }) == ErrorCode::NetlistFatal);
    // Right-hand side of the wrong size.
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(3, kAp, kAi);
              lu.factor(kAx);
              std::vector<double> b = {1.0, 2.0};
              lu.solve(b);
          }) == ErrorCode::NetlistFatal);
    // Calls out of order.
    CHECK(code_of([] {
              KluSolver lu;
              lu.factor(kAx);
          }) == ErrorCode::NetlistFatal);
    CHECK(code_of([] {
              KluSolver lu;
              lu.analyze(3, kAp, kAi);
              std::vector<double> b = {1.0, 2.0, 3.0};
              lu.solve(b);
          }) == ErrorCode::NetlistFatal);
}

TEST_CASE("KluSolver is movable") {
    KluSolver a;
    a.analyze(3, kAp, kAi);
    a.factor(kAx);
    KluSolver b = std::move(a);
    CHECK(b.factored());
    std::vector<double> rhs = {6.0, 10.0, 8.0};
    b.solve(rhs);
    CHECK(rhs[2] == doctest::Approx(3.0));
    // A moved-from solver reports an empty state and can be analyzed again.
    CHECK_FALSE(a.analyzed());  // NOLINT(bugprone-use-after-move)
    a.analyze(1, std::vector<std::int32_t>{0, 1}, std::vector<std::int32_t>{0});
    a.factor(std::vector<double>{2.0});
    std::vector<double> one = {1.0};
    a.solve(one);
    CHECK(one[0] == doctest::Approx(0.5));
}
