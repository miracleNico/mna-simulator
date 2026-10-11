// Backward-Euler transient driver.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mna/analyses/newton.hpp"
#include "mna/analyses/time_grid.hpp"
#include "mna/analyses/transient.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"

using mna::analyses::TimeSource;
using mna::analyses::TranOptions;
using mna::analyses::TranResult;
using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;

namespace {

void add(mna::circuit::TripletList& t, std::int64_t r, std::int64_t c, double v) {
    if (r < 0 || c < 0) {
        return;
    }
    t.rows.push_back(r);
    t.cols.push_back(c);
    t.values.push_back(v);
}

void stamp(mna::circuit::TripletList& t, std::int64_t a, std::int64_t b, double v) {
    add(t, a, a, v);
    add(t, b, b, v);
    add(t, a, b, -v);
    add(t, b, a, -v);
}

void voltage_source(CircuitSpec& s, std::int64_t branch, std::int64_t p, std::int64_t m) {
    add(s.g, p, branch, 1.0);
    add(s.g, branch, p, 1.0);
    add(s.g, m, branch, -1.0);
    add(s.g, branch, m, -1.0);
}

/// V1 a 0, R1 a b (r), C1 b 0 (c); unknowns a, b, I(V1).
Circuit rc(double r, double c) {
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    stamp(s.g, 0, 1, 1.0 / r);
    stamp(s.c, 1, -1, c);
    stamp(s.gmin_stamps, 1, -1, 1.0);
    return Circuit(s);
}

/// V1 a 0 DC 5, R1 a b 1k, D1 b 0 1e-14, C1 b 0 (c).
Circuit diode_rc(double c) {
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    stamp(s.g, 0, 1, 1e-3);
    stamp(s.c, 1, -1, c);
    s.diodes.anode = {1};
    s.diodes.cathode = {-1};
    s.diodes.is = {1e-14};
    return Circuit(s);
}

std::vector<double> column(const TranResult& r, std::size_t n, std::size_t points, std::size_t k) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = r.values[i * points + k];
    }
    return x;
}

template <class Fn>
void check_error(Fn&& fn, mna::ErrorCode code, const std::string& message) {
    try {
        fn();
        FAIL("expected an error: " << message);
    } catch (const mna::Error& e) {
        CHECK(e.code() == code);
        CHECK(std::string(e.what()) == message);
    }
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

std::shared_ptr<mna::expr::Program> sin_t() {
    const std::int32_t ops[2] = {2, 12};  // load_t, sin
    const std::int32_t args[2] = {0, 0};
    return std::make_shared<mna::expr::Program>(ops, args, 2, std::vector<double>{}, 0, true);
}

}  // namespace

TEST_CASE("format_time is Python's {:.2e}") {
    CHECK(mna::analyses::format_time(1e-7) == "1.00e-07");
    CHECK(mna::analyses::format_time(0.0) == "0.00e+00");
    CHECK(mna::analyses::format_time(1.2345e-3) == "1.23e-03");
    CHECK(mna::analyses::format_time(2.5e-100) == "2.50e-100");
}

TEST_CASE("linear RC: Backward Euler from a given state, one factorization per step size") {
    const Circuit c = rc(1e3, 1e-6);
    const std::vector<double> grid = mna::analyses::transient_time_points(2e-3, 1e-3);
    const double x0[3] = {1.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 1.0};
    const TranResult r = mna::analyses::transient(c, x0, grid, b_dc, {}, TranOptions{});
    REQUIRE(r.values.size() == 9);
    CHECK(column(r, 3, 3, 0) == std::vector<double>{1.0, 0.0, 0.0});
    CHECK(r.values[1 * 3 + 1] == doctest::Approx(0.5).epsilon(1e-14));
    CHECK(r.values[1 * 3 + 2] == doctest::Approx(0.75).epsilon(1e-14));
    CHECK(r.values[2 * 3 + 1] == doctest::Approx(-5e-4).epsilon(1e-12));
    CHECK(r.values[2 * 3 + 2] == doctest::Approx(-2.5e-4).epsilon(1e-12));
    CHECK(r.factorizations == 1);
    CHECK(r.newton_iterations == 0);

    // The accumulated grid's steps change in their last bits a few times: one
    // factorization per run of bitwise-equal steps (the reference's exact h per step).
    const std::vector<double> rounded = mna::analyses::transient_time_points(40e-9, 1e-9);
    std::int64_t runs = 1;
    for (std::size_t k = 2; k < rounded.size(); ++k) {
        runs += (rounded[k] - rounded[k - 1]) != (rounded[k - 1] - rounded[k - 2]) ? 1 : 0;
    }
    CHECK(runs > 1);
    CHECK(runs < 10);
    const TranResult same = mna::analyses::transient(c, x0, rounded, b_dc, {}, TranOptions{});
    CHECK(same.factorizations == runs);

    // A non-dividing grid shortens the last step: a second factorization.
    const std::vector<double> uneven = mna::analyses::transient_time_points(1.1, 0.25);
    REQUIRE(uneven.size() == 6);
    const TranResult u = mna::analyses::transient(c, x0, uneven, b_dc, {}, TranOptions{});
    CHECK(u.factorizations == 2);
}

TEST_CASE("time sources are evaluated at every grid point") {
    CircuitSpec s;
    s.n = 2;  // V1 a 0 (sin t), R1 a 0 1k
    voltage_source(s, 1, 0, -1);
    stamp(s.g, 0, -1, 1e-3);
    const Circuit c(s);
    const std::vector<double> grid = mna::analyses::transient_time_points(1.0, 0.1);
    const double x0[2] = {0.0, 0.0};
    const double b_dc[2] = {0.0, 0.0};
    const std::vector<TimeSource> sources = {{1, sin_t()}};
    const TranResult r = mna::analyses::transient(c, x0, grid, b_dc, sources, TranOptions{});
    for (std::size_t k = 1; k < grid.size(); ++k) {
        CHECK(r.values[k] == doctest::Approx(std::sin(grid[k])).epsilon(1e-14));
        CHECK(r.values[grid.size() + k] == doctest::Approx(-std::sin(grid[k]) * 1e-3).epsilon(1e-12));
    }
    const std::vector<double> b = mna::analyses::source_vector(2, b_dc, sources, 0.5);
    CHECK(b[0] == 0.0);
    CHECK(b[1] == std::sin(0.5));
}

TEST_CASE("nonlinear transient settles at the operating point") {
    const Circuit c = diode_rc(1e-9);
    const std::vector<double> grid = mna::analyses::transient_time_points(20e-6, 1e-6);
    const double x0[3] = {0.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 5.0};
    TranOptions o;
    o.newton.max_iter = 200;
    const TranResult r = mna::analyses::transient(c, x0, grid, b_dc, {}, o);
    const std::size_t points = grid.size();
    CHECK(r.newton_iterations > 0);
    CHECK(r.values[1 * points + points - 1] == doctest::Approx(0.6698509496871571).epsilon(1e-6));
    CHECK(r.factorizations == r.newton_iterations);
}

TEST_CASE("transient failures carry the reference messages") {
    // Newton failure at the first step (max_iter = 1 from an all-zero state).
    const Circuit c = diode_rc(1e-9);
    const std::vector<double> grid = mna::analyses::transient_time_points(1e-6, 1e-7);
    const double zeros[3] = {0.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 5.0};
    TranOptions o;
    o.newton.max_iter = 1;
    check_error([&] { (void)mna::analyses::transient(c, zeros, grid, b_dc, {}, o); },
                mna::ErrorCode::NrConvergence, "NR failed to converge at t=1.00e-07 s");

    // G + C/h exactly singular: R = 1k, C = -1u, h = 1m.
    const Circuit negative = rc(1e3, -1e-6);
    const std::vector<double> coarse = mna::analyses::transient_time_points(5e-3, 1e-3);
    const double b1[3] = {0.0, 0.0, 1.0};
    check_error([&] { (void)mna::analyses::transient(negative, zeros, coarse, b1, {}, TranOptions{}); },
                mna::ErrorCode::NrConvergence,
                "Linear system is singular at t=1.00e-03 s (G + C/h matrix): Singular matrix");

    // A non-finite state is reported with its time.
    const double bad[3] = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    check_error([&] { (void)mna::analyses::transient(rc(1e3, 1e-6), bad, coarse, b1, {}, TranOptions{}); },
                mna::ErrorCode::NrConvergence, "non-finite solution at t=0.00e+00 s");
}

TEST_CASE("progress callback: every progress_every steps and after the last one") {
    const Circuit c = rc(1e3, 1e-6);
    const std::vector<double> grid = mna::analyses::transient_time_points(1.0, 0.1);  // 10 steps
    const double x0[3] = {0.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 1.0};
    for (const auto& [every, expected] : std::vector<std::pair<std::int64_t, std::vector<std::int64_t>>>{
             {0, {}}, {1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}, {3, {3, 6, 9, 10}}, {20, {10}}}) {
        TranOptions o;
        o.progress_every = every;
        std::vector<std::int64_t> steps;
        bool times_match = true;
        const TranResult r = mna::analyses::transient(
            c, x0, grid, b_dc, {}, o, [&](std::int64_t step, double t, const double* x) {
                steps.push_back(step);
                times_match = times_match && t == grid[static_cast<std::size_t>(step)] && std::isfinite(x[1]);
            });
        CHECK(steps == expected);
        CHECK(times_match);
        CHECK(r.progress_calls == static_cast<std::int64_t>(expected.size()));
    }
}

TEST_CASE("Krylov linear transient: one stats entry per step, CSR operator label") {
    const Circuit c = rc(1e3, 1e-6);
    const std::vector<double> grid = mna::analyses::transient_time_points(5e-3, 1e-3);
    const double x0[3] = {1.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 1.0};
    TranOptions o;
    o.newton.krylov.enabled = true;
    o.newton.krylov.rank = 3;
    const TranResult krylov = mna::analyses::transient(c, x0, grid, b_dc, {}, o);
    const TranResult direct = mna::analyses::transient(c, x0, grid, b_dc, {}, TranOptions{});
    REQUIRE(krylov.krylov_stats.size() == grid.size() - 1);
    for (const auto& stat : krylov.krylov_stats) {
        CHECK(stat.csr_operator);
        CHECK(stat.dimension == 3);
    }
    for (std::size_t i = 0; i < direct.values.size(); ++i) {
        CHECK(krylov.values[i] == doctest::Approx(direct.values[i]).epsilon(1e-7).scale(1e-9));
    }
}

TEST_CASE("transient is bitwise identical for every thread count") {
    const int count = static_cast<int>(mna::analyses::kParallelDeviceMinCount) + 10;
    CircuitSpec s;
    s.n = count + 2;
    const std::int64_t branch = count + 1;
    voltage_source(s, branch, 0, -1);
    for (int i = 1; i <= count; ++i) {
        stamp(s.g, 0, i, 1e-3 * (1.0 + 0.001 * i));
        stamp(s.c, i, -1, 1e-9);
        s.diodes.anode.push_back(i);
        s.diodes.cathode.push_back(-1);
        s.diodes.is.push_back(1e-14);
    }
    const Circuit c(s);
    const std::vector<double> grid = mna::analyses::transient_time_points(5e-6, 1e-6);
    std::vector<double> x0(static_cast<std::size_t>(c.n()), 0.0);
    std::vector<double> b_dc(static_cast<std::size_t>(c.n()), 0.0);
    b_dc.back() = 2.0;
    TranOptions o;
    o.newton.max_iter = 200;
    o.newton.num_threads = 1;
    const TranResult serial = mna::analyses::transient(c, x0.data(), grid, b_dc.data(), {}, o);
    o.newton.num_threads = 4;
    const TranResult parallel = mna::analyses::transient(c, x0.data(), grid, b_dc.data(), {}, o);
    CHECK(serial.newton_iterations == parallel.newton_iterations);
    CHECK(bitwise_equal(serial.values, parallel.values));
}

TEST_CASE("malformed transient input is NETLIST_FATAL") {
    const Circuit c = rc(1e3, 1e-6);
    const double x0[3] = {0.0, 0.0, 0.0};
    const double b_dc[3] = {0.0, 0.0, 1.0};
    CHECK_THROWS_AS((void)mna::analyses::transient(c, x0, {}, b_dc, {}, TranOptions{}), mna::Error);
    CHECK_THROWS_AS((void)mna::analyses::transient(c, x0, {0.0, 0.0}, b_dc, {}, TranOptions{}), mna::Error);
    CHECK_THROWS_AS((void)mna::analyses::transient(c, x0, {0.0, 1.0}, b_dc, {{5, sin_t()}}, TranOptions{}),
                    mna::Error);
    CHECK_THROWS_AS((void)mna::analyses::transient(c, x0, {0.0, 1.0}, b_dc, {{0, nullptr}}, TranOptions{}),
                    mna::Error);
}
