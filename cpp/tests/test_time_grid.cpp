// Transient time grid: bitwise port of solvers._transient_time_points.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "mna/analyses/time_grid.hpp"
#include "mna/error.hpp"

using mna::analyses::transient_time_points;

namespace {

void check_tran_error(double t_stop, double t_step, const std::string& message) {
    try {
        (void)transient_time_points(t_stop, t_step);
        FAIL("expected TRAN_ERROR for (" << t_stop << ", " << t_step << ")");
    } catch (const mna::Error& e) {
        CHECK(e.code() == mna::ErrorCode::TranError);
        CHECK(std::string(e.code_name()) == "TRAN_ERROR");
        CHECK(std::string(e.what()) == message);
    }
}

}  // namespace

TEST_CASE("time grid accumulates current + t_step like the reference") {
    const std::vector<double> grid = transient_time_points(1.0, 0.1);
    const std::vector<double> expected = {0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
                                          0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 1.0};
    REQUIRE(grid.size() == expected.size());
    for (std::size_t k = 0; k < grid.size(); ++k) {
        CHECK(grid[k] == expected[k]);  // exact
    }
}

TEST_CASE("time grid edge cases") {
    CHECK(transient_time_points(0.0, 1e-3) == std::vector<double>{0.0});
    CHECK(transient_time_points(1e-3, 5e-3) == std::vector<double>{0.0, 1e-3});
    const std::vector<double> non_dividing = transient_time_points(1e-3, 3e-4);
    CHECK(non_dividing.size() == 5);
    CHECK(non_dividing.back() == 1e-3);
    CHECK(non_dividing[3] == 3e-4 + 3e-4 + 3e-4);
}

TEST_CASE("time grid snapping tolerance is max(|t_stop|, |t_step|, eps) * 1e-9") {
    const std::vector<double> inside = transient_time_points(1e-3 + 1e-13, 1e-4);
    CHECK(inside.size() == 11);
    CHECK(inside.back() == 1e-3 + 1e-13);
    const std::vector<double> outside = transient_time_points(1e-3 + 1e-11, 1e-4);
    CHECK(outside.size() == 12);
    CHECK(outside.back() == 1e-3 + 1e-11);
    CHECK(outside[11] - outside[10] == doctest::Approx(1e-11).epsilon(1e-3));
    CHECK(transient_time_points(1e-3 - 1e-13, 1e-4).size() == 11);
    CHECK(transient_time_points(1e-3 - 1e-11, 1e-4).size() == 11);
}

TEST_CASE("time grid rejects bad arguments with TRAN_ERROR") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    check_tran_error(-1.0, 1e-3, "t_stop must be non-negative.");
    check_tran_error(-inf, 1e-3, "t_stop must be non-negative.");
    check_tran_error(1e-3, 0.0, "t_step must be positive.");
    check_tran_error(1e-3, -1e-4, "t_step must be positive.");
    check_tran_error(0.0, 0.0, "t_step must be positive.");
    check_tran_error(nan, 1e-3, "t_stop must be finite.");
    check_tran_error(inf, 1e-3, "t_stop must be finite.");
    check_tran_error(1e-3, nan, "t_step must be finite.");
    check_tran_error(1e-3, inf, "t_step must be finite.");
}

TEST_CASE("time grids are strictly increasing and end exactly at t_stop") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> exponent(-12.0, 0.0);
    std::uniform_real_distribution<double> ratio(1.0, 500.0);
    for (int i = 0; i < 200; ++i) {
        const double t_step = std::pow(10.0, exponent(rng));
        const double t_stop = t_step * ratio(rng);
        const std::vector<double> grid = transient_time_points(t_stop, t_step);
        REQUIRE(grid.size() >= 2);
        CHECK(grid.front() == 0.0);
        CHECK(grid.back() == t_stop);
        for (std::size_t k = 1; k < grid.size(); ++k) {
            CHECK(grid[k] > grid[k - 1]);
        }
        // Repeated addition, not k * t_step.
        double current = 0.0;
        for (std::size_t k = 1; k + 1 < grid.size(); ++k) {
            current = current + t_step;
            CHECK(grid[k] == current);
        }
    }
}
