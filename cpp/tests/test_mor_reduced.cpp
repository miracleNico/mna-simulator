// Dense _safe_solve (mna/mor/safe_solve.hpp) and the reduced AC / transient loops
// (mna/mor/reduced_ac.hpp, mna/mor/reduced_transient.hpp).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "mna/analyses/transient.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "mna/mor/reduced_ac.hpp"
#include "mna/mor/reduced_transient.hpp"
#include "mna/mor/safe_solve.hpp"
#include "mna/threads.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::mor::Complex;
using mna::mor::DenseSafeSolver;
using mna::mor::SolvePath;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

}  // namespace

TEST_CASE("safe_solve: regular systems take the LU path") {
    const std::vector<double> a = {4.0, 1.0, 2.0, 3.0};  // column-major [[4, 2], [1, 3]]
    const std::vector<double> b = {8.0, 7.0};
    SolvePath path = SolvePath::RegularizedLeastSquares;
    const std::vector<double> x = mna::mor::safe_solve<double>(2, a.data(), b.data(), 1, &path);
    CHECK(path == SolvePath::LU);
    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[1] == doctest::Approx(2.0));
    const std::vector<Complex> az = {Complex(1, 1), Complex(0, 0), Complex(0, 0), Complex(2, 0)};
    const std::vector<Complex> bz = {Complex(2, 0), Complex(4, 2)};
    const std::vector<Complex> xz = mna::mor::safe_solve<Complex>(2, az.data(), bz.data(), 1, &path);
    CHECK(path == SolvePath::LU);
    CHECK(std::abs(xz[0] - Complex(1, -1)) < 1e-14);
    CHECK(std::abs(xz[1] - Complex(2, 1)) < 1e-14);
}

TEST_CASE("safe_solve: an exactly singular system gets the minimum-norm least-squares solution") {
    // [[1, 1], [1, 1]] x = [2, 2]: np.linalg.solve raises, lstsq returns [1, 1].
    const std::vector<double> a = {1.0, 1.0, 1.0, 1.0};
    const std::vector<double> b = {2.0, 2.0, 4.0, 4.0};  // two right-hand sides
    SolvePath path = SolvePath::LU;
    const std::vector<double> x = mna::mor::safe_solve<double>(2, a.data(), b.data(), 2, &path);
    CHECK(path == SolvePath::LeastSquares);
    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[1] == doctest::Approx(1.0));
    CHECK(x[2] == doctest::Approx(2.0));
    CHECK(x[3] == doctest::Approx(2.0));
    // An inconsistent right-hand side: the least-squares solution of [[1, 0], [0, 0]] x = [1, 1].
    const std::vector<double> a2 = {1.0, 0.0, 0.0, 0.0};
    const std::vector<double> b2 = {1.0, 1.0};
    const std::vector<double> x2 = mna::mor::safe_solve<double>(2, a2.data(), b2.data(), 1, &path);
    CHECK(path == SolvePath::LeastSquares);
    CHECK(x2[0] == doctest::Approx(1.0));
    CHECK(std::abs(x2[1]) < 1e-14);
    // The all-zero matrix: x = 0.
    const std::vector<double> zero(4, 0.0);
    const std::vector<double> x3 = mna::mor::safe_solve<double>(2, zero.data(), b2.data(), 1, &path);
    CHECK(path == SolvePath::LeastSquares);
    CHECK(x3 == std::vector<double>{0.0, 0.0});
}

TEST_CASE("safe_solve: known_singular skips the LU; a singular non-finite matrix raises") {
    DenseSafeSolver<double> solver;
    const std::vector<double> a = {2.0, 0.0, 0.0, 4.0};
    solver.factor(2, a.data(), /*known_singular=*/true);
    CHECK(solver.path() == SolvePath::LeastSquares);
    std::vector<double> b = {2.0, 4.0};
    solver.solve(b.data(), 1);
    CHECK(b[0] == doctest::Approx(1.0));
    CHECK(b[1] == doctest::Approx(1.0));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> bad = {0.0, nan, 0.0, 0.0};  // zero first pivot column... and a NaN
    try {
        solver.factor(2, bad.data(), /*known_singular=*/true);
        FAIL("expected JACOBIAN_SINGULAR");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::JacobianSingular);
    }
    DenseSafeSolver<double> unused;
    try {
        unused.solve(b.data(), 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
    // n == 0 is a no-op.
    DenseSafeSolver<Complex> empty;
    empty.factor(0, nullptr);
    CHECK(empty.factored());
}

TEST_CASE("reduced_ac: |Sr (Gr + j w Cr)^-1 Br| per frequency, deterministic for any thread count") {
    // k = 2, two outputs.
    const std::vector<double> gr = {2.0, -1.0, -1.0, 3.0};
    const std::vector<double> cr = {1e-6, 0.0, 0.0, 2e-6};
    const std::vector<Complex> br = {Complex(1.0, 0.0), Complex(0.5, 0.25)};
    const std::vector<double> sr = {1.0, 0.0, 0.5, 1.0};  // [[1, 0.5], [0, 1]] column-major
    std::vector<double> frequencies;
    for (int i = 0; i < 37; ++i) {
        frequencies.push_back(10.0 * i * i);
    }
    const auto one = mna::mor::reduced_ac(2, gr.data(), cr.data(), br.data(), 2, sr.data(), frequencies.data(),
                                          static_cast<std::int64_t>(frequencies.size()), 1);
    CHECK(one.threads_used == 1);
    CHECK(one.lstsq_points == 0);
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        const double w = 2.0 * kPi * frequencies[i];
        const Complex a00(2.0, w * 1e-6), a01(-1.0, 0.0), a10(-1.0, 0.0), a11(3.0, w * 2e-6);
        const Complex det = a00 * a11 - a01 * a10;
        const Complex z0 = (a11 * br[0] - a01 * br[1]) / det;
        const Complex z1 = (a00 * br[1] - a10 * br[0]) / det;
        CHECK(one.magnitudes[i] == doctest::Approx(std::abs(z0 + 0.5 * z1)).epsilon(1e-12));
        CHECK(one.magnitudes[frequencies.size() + i] == doctest::Approx(std::abs(z1)).epsilon(1e-12));
    }
    // Too little work for threads: the sweep stays on the caller.
    const auto small = mna::mor::reduced_ac(2, gr.data(), cr.data(), br.data(), 2, sr.data(), frequencies.data(),
                                            static_cast<std::int64_t>(frequencies.size()), 4);
    CHECK(small.threads_used == 1);
    CHECK(bitwise_equal(one.magnitudes, small.magnitudes));
}

TEST_CASE("reduced_ac: frequency-parallel sweeps are bitwise identical to the serial one") {
    const int k = 40;
    const int points = 64;  // points * k^3 = 4.1e6 >= kReducedAcParallelMinWork
    CHECK(static_cast<double>(points) * k * k * k >= mna::mor::kReducedAcParallelMinWork);
    std::vector<double> gr(static_cast<std::size_t>(k * k)), cr(gr.size());
    std::vector<Complex> br(static_cast<std::size_t>(k));
    std::vector<double> sr(static_cast<std::size_t>(3 * k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            const auto e = static_cast<std::size_t>(i + j * k);
            gr[e] = (i == j ? 50.0 : 0.0) + std::sin(1.0 + i * 0.7 + j * 1.3);
            cr[e] = 1e-9 * std::cos(0.3 * i - 0.9 * j);
        }
        br[static_cast<std::size_t>(j)] = Complex(std::cos(j), std::sin(2.0 * j));
        for (int r = 0; r < 3; ++r) {
            sr[static_cast<std::size_t>(r + 3 * j)] = (r + 1) * std::sin(0.1 * j + r);
        }
    }
    std::vector<double> f(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        f[static_cast<std::size_t>(i)] = 1e6 * i;
    }
    const auto one = mna::mor::reduced_ac(k, gr.data(), cr.data(), br.data(), 3, sr.data(), f.data(), points, 1);
    CHECK(one.threads_used == 1);
    for (int threads : {2, 4, 16}) {
        const auto many =
            mna::mor::reduced_ac(k, gr.data(), cr.data(), br.data(), 3, sr.data(), f.data(), points, threads);
        CHECK(many.threads_used == std::min(mna::effective_threads(threads), points / 2));
        CHECK(bitwise_equal(one.magnitudes, many.magnitudes));
    }
}

TEST_CASE("reduced_ac: a singular reduced system takes the least-squares path") {
    const std::vector<double> gr = {1.0, 1.0, 1.0, 1.0};
    const std::vector<double> cr = {0.0, 0.0, 0.0, 0.0};
    const std::vector<Complex> br = {Complex(2.0, 0.0), Complex(2.0, 0.0)};
    const std::vector<double> sr = {1.0, 0.0};
    const std::vector<double> f = {0.0, 1e3};
    const auto r = mna::mor::reduced_ac(2, gr.data(), cr.data(), br.data(), 1, sr.data(), f.data(), 2, 1);
    CHECK(r.lstsq_points == 2);
    CHECK(r.magnitudes[0] == doctest::Approx(1.0));
}

TEST_CASE("reduced_transient: Backward Euler on the reduced system, one factorization per step size") {
    // One RC node (n = k = 1, V = [1]): (g + c/h) z_k = b + (c/h) z_{k-1}.
    const double g = 1e-3, c = 1e-6, b = 1e-3;
    const std::vector<double> gr = {g}, cr = {c}, V = {1.0}, sr = {1.0}, x0 = {0.0}, b_dc = {b};
    const std::vector<double> grid = {0.0, 0.25e-3, 0.5e-3, 0.75e-3, 1.25e-3};  // exact binary steps
    const auto r = mna::mor::reduced_transient(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), x0.data(), grid,
                                               b_dc.data(), {});
    CHECK_FALSE(r.failed);
    CHECK(r.factorizations == 2);  // h = 0.25e-3 three times (bitwise equal), then 0.5e-3
    double z = 0.0;
    for (std::size_t k = 1; k < grid.size(); ++k) {
        const double h = grid[k] - grid[k - 1];
        const double crh = c / h;
        z = (b + crh * z) / (g + crh);
        CHECK(r.values[k] == doctest::Approx(z).epsilon(1e-13));
    }
}

TEST_CASE("reduced_transient: time sources enter through V^T b(t)") {
    const std::int32_t ops[2] = {2, 12};  // load_t, sin
    const std::int32_t args[2] = {0, 0};
    auto program = std::make_shared<mna::expr::Program>(ops, args, 2, std::vector<double>{}, 0, true);
    std::vector<mna::analyses::TimeSource> sources = {{1, program}};
    // n = 2, k = 1 basis along e1: z = x1.
    const std::vector<double> gr = {1.0}, cr = {1.0}, V = {0.0, 1.0}, sr = {2.0}, x0 = {5.0, 0.5}, b_dc = {7.0, 0.0};
    const std::vector<double> grid = {0.0, 0.5, 1.0};
    const auto r = mna::mor::reduced_transient(2, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), x0.data(), grid,
                                               b_dc.data(), sources);
    double z = 0.5;
    CHECK(r.values[0] == doctest::Approx(1.0));
    for (std::size_t k = 1; k < grid.size(); ++k) {
        z = (std::sin(grid[k]) + 2.0 * z) / 3.0;
        CHECK(r.values[k] == doctest::Approx(2.0 * z).epsilon(1e-14));
    }
}

TEST_CASE("reduced_transient: a blow-up (||z|| > 1e12) NaN-fills that step and the rest") {
    // Gr + Cr/h = 1 - 0.5 / h... chosen so z grows by 1e5 per step.
    const std::vector<double> gr = {-0.99999}, cr = {1.0}, V = {1.0}, sr = {1.0}, x0 = {1.0}, b_dc = {0.0};
    std::vector<double> grid;
    for (int k = 0; k < 8; ++k) {
        grid.push_back(static_cast<double>(k));
    }
    const auto r = mna::mor::reduced_transient(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), x0.data(), grid,
                                               b_dc.data(), {});
    REQUIRE(r.failed);
    CHECK(r.failed_step == 3);  // 1e5, 1e10, 1e15 > 1e12
    CHECK(r.values[1] == doctest::Approx(1e5).epsilon(1e-6));
    CHECK(r.values[2] == doctest::Approx(1e10).epsilon(1e-6));
    for (std::size_t k = 3; k < grid.size(); ++k) {
        CHECK(std::isnan(r.values[k]));
    }
}

TEST_CASE("reduced_transient rejects malformed input") {
    const std::vector<double> one = {1.0};
    try {
        mna::mor::reduced_transient(1, 1, one.data(), one.data(), one.data(), 1, one.data(), one.data(), {}, one.data(),
                                    {});
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
    try {
        mna::mor::reduced_transient(1, 1, one.data(), one.data(), one.data(), 1, one.data(), one.data(), {1.0, 1.0},
                                    one.data(), {});
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
}
