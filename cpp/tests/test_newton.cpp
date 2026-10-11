// Damped Newton driver, linear-system front end and .op.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mna/analyses/newton.hpp"
#include "mna/analyses/op.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "mna/threads.hpp"

using mna::analyses::NewtonOptions;
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

void conductance(CircuitSpec& s, std::int64_t a, std::int64_t b, double g) {
    add(s.g, a, a, g);
    add(s.g, b, b, g);
    add(s.g, a, b, -g);
    add(s.g, b, a, -g);
}

void voltage_source(CircuitSpec& s, std::int64_t branch, std::int64_t p, std::int64_t m) {
    add(s.g, p, branch, 1.0);
    add(s.g, branch, p, 1.0);
    add(s.g, m, branch, -1.0);
    add(s.g, branch, m, -1.0);
}

void diode(CircuitSpec& s, std::int64_t a, std::int64_t k, double is) {
    s.diodes.anode.push_back(a);
    s.diodes.cathode.push_back(k);
    s.diodes.is.push_back(is);
    // The builder records a unit gmin stamp across every diode.
    add(s.gmin_stamps, a, a, 1.0);
    add(s.gmin_stamps, k, k, 1.0);
    add(s.gmin_stamps, a, k, -1.0);
    add(s.gmin_stamps, k, a, -1.0);
}

/// V1 a 0 DC 5, R1 a b 1k, D1 b 0 1e-14 (unknowns a, b, I(V1)).
Circuit diode_circuit() {
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    diode(s, 1, -1, 1e-14);
    return Circuit(s);
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

/// x^2 + 1 on unknown 0 (never zero).
std::shared_ptr<mna::expr::Program> square_plus_one(std::int32_t n) {
    const std::int32_t ops[5] = {1, 1, 5, 0, 3};  // load_x, load_x, mul, const, add
    const std::int32_t args[5] = {0, 0, 0, 0, 0};
    return std::make_shared<mna::expr::Program>(ops, args, 5, std::vector<double>{1.0}, n, false);
}

/// `count` parallel branches: node i -- R(1k) -- vdd (unknown 0, set by a source) and a
/// diode from node i to ground; unknowns: vdd, node 1..count, I(V).
Circuit diode_fan(int count) {
    CircuitSpec s;
    s.n = count + 2;
    const std::int64_t branch = count + 1;
    voltage_source(s, branch, 0, -1);
    for (int i = 1; i <= count; ++i) {
        conductance(s, 0, i, 1e-3 * (1.0 + 0.001 * i));
        diode(s, i, -1, 1e-14 * (1.0 + 0.01 * (i % 7)));
    }
    return Circuit(s);
}

}  // namespace

TEST_CASE("finite_norm is NumPy's unscaled 2-norm with +inf for non-finite entries") {
    const double v[3] = {3.0, 4.0, 0.0};
    CHECK(mna::analyses::finite_norm(v, 3) == 5.0);
    const double w[2] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    CHECK(std::isinf(mna::analyses::finite_norm(w, 2)));
    const double big[2] = {1e200, 1e200};  // finite entries, overflowing square: +inf like numpy.linalg.norm
    CHECK(std::isinf(mna::analyses::finite_norm(big, 2)));
    CHECK(mna::analyses::finite_norm(v, 0) == 0.0);
}

TEST_CASE("device evaluation threads: serial below the parallel threshold") {
    const Circuit small = diode_circuit();
    CHECK(mna::analyses::device_eval_threads(small, 8) == 1);
    const Circuit large = diode_fan(static_cast<int>(mna::analyses::kParallelDeviceMinCount));
    CHECK(mna::analyses::device_eval_threads(large, 1) == 1);
    CHECK(mna::analyses::device_eval_threads(large, 4) == mna::effective_threads(4));
}

TEST_CASE("linear .op is one direct solve") {
    CircuitSpec s;
    s.n = 3;  // V1 a 0 DC 1, R1 a b 1k, R2 b 0 2k
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    conductance(s, 1, -1, 0.5e-3);
    const Circuit c(s);
    const double b[3] = {0.0, 0.0, 1.0};
    const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 0.0, NewtonOptions{});
    CHECK(r.iterations == 1);
    CHECK(r.factorizations == 1);
    CHECK(r.x[0] == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(r.x[1] == doctest::Approx(2.0 / 3.0).epsilon(1e-14));
    CHECK(r.x[2] == doctest::Approx(-1.0 / 3000.0).epsilon(1e-12));
    CHECK(r.krylov_stats.empty());
}

TEST_CASE("singular linear .op raises NR_CONVERGENCE with the reference text") {
    CircuitSpec s;
    s.n = 4;  // V1 a 0, R1 a 0, R2 x y (floating island); unknowns a, x, y, I(V1)
    voltage_source(s, 3, 0, -1);
    conductance(s, 0, -1, 1e-3);
    conductance(s, 1, 2, 1e-3);
    const Circuit c(s);
    const double b[4] = {0.0, 0.0, 0.0, 1.0};
    check_error([&] { (void)mna::analyses::op(c, b, nullptr, 0.0, NewtonOptions{}); },
                mna::ErrorCode::NrConvergence, "Linear system is singular (G matrix): Singular matrix");
}

TEST_CASE("Newton .op of a diode circuit reaches the reference operating point") {
    const Circuit c = diode_circuit();
    const double b[3] = {0.0, 0.0, 5.0};
    NewtonOptions o;
    o.max_iter = 200;
    const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 0.0, o);
    CHECK(r.x[0] == doctest::Approx(5.0).epsilon(1e-12));
    CHECK(r.x[1] == doctest::Approx(0.6698509496871571).epsilon(1e-6));
    CHECK(r.x[2] == doctest::Approx(-0.004330149050312843).epsilon(1e-6));
    CHECK(r.iterations > 1);
    CHECK(r.factorizations == r.iterations);
    // Device equation at the returned point.
    CHECK((r.x[0] - r.x[1]) / 1000.0 == doctest::Approx(1e-14 * (std::exp(r.x[1] / 0.025) - 1.0)).epsilon(1e-6));
}

TEST_CASE("Newton failure statuses carry the reference messages") {
    const Circuit c = diode_circuit();
    const double b[3] = {0.0, 0.0, 5.0};
    for (const std::int64_t max_iter : {0, 1, 2}) {
        NewtonOptions o;
        o.max_iter = max_iter;
        check_error([&] { (void)mna::analyses::op(c, b, nullptr, 0.0, o); }, mna::ErrorCode::NrConvergence,
                    "Solver did not converge after " + std::to_string(max_iter) + " iterations.");
    }

    // Diode + floating resistor island: singular Jacobian at iteration 0.
    CircuitSpec s;
    s.n = 5;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    diode(s, 1, -1, 1e-14);
    conductance(s, 3, 4, 1e-3);
    const Circuit island(s);
    const double b5[5] = {0.0, 0.0, 5.0, 0.0, 0.0};
    check_error([&] { (void)mna::analyses::op(island, b5, nullptr, 0.0, NewtonOptions{}); },
                mna::ErrorCode::JacobianSingular, "Jacobian is singular at iteration 0.");

    // F(x) = 1e-12 x + x^2 + 1 has no root: from 0 the full step overshoots by 1e12 and no
    // halving (down to 2^-27) reduces the residual.
    CircuitSpec q;
    q.n = 1;
    add(q.g, 0, 0, 1e-12);
    q.behavioral.node_p = {0};
    q.behavioral.node_n = {-1};
    q.behavioral.programs = {square_plus_one(1)};
    const Circuit no_root(q);
    const double b1[1] = {0.0};
    check_error([&] { (void)mna::analyses::op(no_root, b1, nullptr, 0.0, NewtonOptions{}); },
                mna::ErrorCode::NrConvergence, "Newton step could not reduce residual at iteration 0.");
}

TEST_CASE("the residual is checked before the first solve, x0 is the starting point") {
    const Circuit c = diode_circuit();
    const double b[3] = {0.0, 0.0, 5.0};
    NewtonOptions o;
    o.max_iter = 200;
    const mna::analyses::OpResult first = mna::analyses::op(c, b, nullptr, 0.0, o);
    // Restarting from the converged point needs no solve (||F|| < f_tol) unless the
    // residual is still above f_tol there; either way it stays on the solution.
    const mna::analyses::OpResult again = mna::analyses::op(c, b, first.x.data(), 0.0, o);
    CHECK(again.iterations <= 1);
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(again.x[i] == doctest::Approx(first.x[i]).epsilon(1e-9));
    }
    // max_iter = 0 fails even at the solution (range(0) is empty in the reference).
    o.max_iter = 0;
    CHECK_THROWS_AS((void)mna::analyses::op(c, b, first.x.data(), 0.0, o), mna::Error);
}

TEST_CASE("v_tol exit: a tiny accepted step ends the iteration") {
    const Circuit c = diode_circuit();
    const double b[3] = {0.0, 0.0, 5.0};
    NewtonOptions o;
    o.max_iter = 500;
    o.f_tol = 0.0;  // never satisfied: only ||dx|| < v_tol can stop the loop
    o.v_tol = 1e-9;
    const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 0.0, o);
    CHECK(r.x[1] == doctest::Approx(0.6698509496871571).epsilon(1e-9));
}

TEST_CASE("gmin is applied as G + gmin * gmin_stamps") {
    // V1 a 0 DC 1, R1 a 0 1k; node c is tied to a only by a gmin stamp (unknowns a, c, I(V1)).
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, -1, 1e-3);
    add(s.gmin_stamps, 0, 0, 1.0);
    add(s.gmin_stamps, 1, 1, 1.0);
    add(s.gmin_stamps, 0, 1, -1.0);
    add(s.gmin_stamps, 1, 0, -1.0);
    const Circuit c(s);
    const double b[3] = {0.0, 0.0, 1.0};
    check_error([&] { (void)mna::analyses::op(c, b, nullptr, 0.0, NewtonOptions{}); },
                mna::ErrorCode::NrConvergence, "Linear system is singular (G matrix): Singular matrix");
    const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 1e-12, NewtonOptions{});
    CHECK(r.x[1] == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(r.x[2] == doctest::Approx(-1e-3).epsilon(1e-12));
}

TEST_CASE("a non-finite solution is NR_CONVERGENCE") {
    CircuitSpec s;
    s.n = 2;
    voltage_source(s, 1, 0, -1);
    add(s.g, 0, 0, 1e-3);
    const Circuit c(s);
    const double b[2] = {0.0, std::numeric_limits<double>::infinity()};
    check_error([&] { (void)mna::analyses::op(c, b, nullptr, 0.0, NewtonOptions{}); },
                mna::ErrorCode::NrConvergence, "non-finite solution");
}

TEST_CASE("Krylov path: solve_linear acceptance and stats") {
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    conductance(s, 1, -1, 0.5e-3);
    const Circuit c(s);
    const double b[3] = {0.0, 0.0, 1.0};
    NewtonOptions o;
    o.krylov.enabled = true;
    o.krylov.rank = 3;
    const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 0.0, o);
    REQUIRE(r.krylov_stats.size() == 1);
    const mna::analyses::KrylovStat& stat = r.krylov_stats[0];
    CHECK(stat.dimension == 3);
    CHECK(stat.nnz == 6);
    CHECK_FALSE(stat.csr_operator);
    CHECK(stat.matrix_kind == mna::linear::MatrixKind::Symmetric);
    CHECK(r.x[1] == doctest::Approx(2.0 / 3.0).epsilon(1e-8));

    // A one-iteration budget on a general matrix falls back to the direct solve.
    CircuitSpec g;
    g.n = 3;
    add(g.g, 0, 0, 4.0);
    add(g.g, 0, 1, 1.0);
    add(g.g, 1, 0, 2.0);
    add(g.g, 1, 1, 3.0);
    add(g.g, 1, 2, 1.0);
    add(g.g, 2, 1, -1.0);
    add(g.g, 2, 2, 5.0);
    const Circuit general(g);
    const double rhs[3] = {1.0, 2.0, 3.0};
    NewtonOptions k;
    k.krylov.enabled = true;
    k.krylov.max_iter = 1;
    k.krylov.rank = 1;
    const mna::analyses::OpResult fb = mna::analyses::op(general, rhs, nullptr, 0.0, k);
    REQUIRE(fb.krylov_stats.size() == 1);
    CHECK(fb.krylov_stats[0].used_direct_fallback);
    CHECK(fb.krylov_stats[0].matrix_kind == mna::linear::MatrixKind::General);
    CHECK(fb.factorizations == 1);
    // Exact direct solution: 4a + b = 1, 2a + 3b + c = 2, -b + 5c = 3.
    CHECK(4.0 * fb.x[0] + fb.x[1] == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(-fb.x[1] + 5.0 * fb.x[2] == doctest::Approx(3.0).epsilon(1e-14));

    // Unknown methods are rejected like solvers._krylov_route.
    k.krylov.method = "foo";
    CHECK_THROWS_AS((void)mna::analyses::op(general, rhs, nullptr, 0.0, k), std::invalid_argument);
}

TEST_CASE("Newton .op is bitwise identical for every thread count (parallel device evaluation)") {
    const int count = static_cast<int>(mna::analyses::kParallelDeviceMinCount) + 52;
    const Circuit c = diode_fan(count);
    std::vector<double> b(static_cast<std::size_t>(c.n()), 0.0);
    b.back() = 2.0;
    NewtonOptions o;
    o.max_iter = 200;
    o.num_threads = 1;
    const mna::analyses::OpResult serial = mna::analyses::op(c, b.data(), nullptr, 0.0, o);
    o.num_threads = 4;
    const mna::analyses::OpResult parallel = mna::analyses::op(c, b.data(), nullptr, 0.0, o);
    CHECK(serial.iterations == parallel.iterations);
    CHECK(bitwise_equal(serial.x, parallel.x));
    CHECK(serial.x[1] > 0.4);
    CHECK(serial.x[1] < 0.8);
}
