// Junction voltage-step limiting, node-to-ground gmin and the P3b Newton strategies
// (CPP_MIGRATION_PLAN.md 2.4.1): pnjlim, limit_junctions, Circuit::junctions(), the
// node_count gmin stamps, the junction strategy of the Newton driver and the gmin
// continuation of .op.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "mna/analyses/limiting.hpp"
#include "mna/analyses/newton.hpp"
#include "mna/analyses/op.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/error.hpp"

using mna::analyses::limit_junctions;
using mna::analyses::NewtonOptions;
using mna::analyses::pnjlim;
using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;

namespace {

constexpr double kVt = 0.025;

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

void gmin_stamp(CircuitSpec& s, std::int64_t a, std::int64_t b) {
    add(s.gmin_stamps, a, a, 1.0);
    add(s.gmin_stamps, b, b, 1.0);
    add(s.gmin_stamps, a, b, -1.0);
    add(s.gmin_stamps, b, a, -1.0);
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
    gmin_stamp(s, a, k);
}

void mos(CircuitSpec& s, std::int64_t d, std::int64_t g, std::int64_t src, double beta, int sign) {
    s.mos.drain.push_back(d);
    s.mos.gate.push_back(g);
    s.mos.source.push_back(src);
    s.mos.beta.push_back(beta);
    s.mos.vth.push_back(0.4);
    s.mos.lambda.push_back(0.02);
    s.mos.sign.push_back(sign);
    conductance(s, d, src, 1e-12);  // gds_floor
    gmin_stamp(s, d, src);
}

double vcrit(double vt, double is) { return vt * std::log(vt / (std::sqrt(2.0) * is)); }

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

/// V1 a 0 DC <volts>, D1 a 0 1e-14 (unknowns a, I(V1)): a diode across an ideal source.
Circuit clamped_diode() {
    CircuitSpec s;
    s.n = 2;
    s.node_count = 1;
    voltage_source(s, 1, 0, -1);
    diode(s, 0, -1, 1e-14);
    return Circuit(s);
}

/// Inverter chain: VDD vdd 0 1.2, VIN in 0 0, then `stages` CMOS inverters (no keepers).
/// Unknowns: in = 0, o1 .. o_stages, vdd, I(VDD), I(VIN).
Circuit inverter_chain(int stages) {
    CircuitSpec s;
    const std::int64_t vdd = stages + 1;
    s.n = stages + 4;
    s.node_count = stages + 2;
    voltage_source(s, stages + 2, vdd, -1);
    voltage_source(s, stages + 3, 0, -1);
    for (int k = 1; k <= stages; ++k) {
        mos(s, k, k - 1, vdd, 0.7e-3, -1);
        mos(s, k, k - 1, -1, 1.8e-3, 1);
    }
    return Circuit(s);
}

std::vector<double> chain_rhs(int stages) {
    std::vector<double> b(static_cast<std::size_t>(stages + 4), 0.0);
    b[static_cast<std::size_t>(stages + 2)] = 1.2;
    return b;
}

}  // namespace

TEST_CASE("pnjlim: SPICE3 DEVpnjlim branches") {
    const double vc = vcrit(kVt, 1e-14);
    CHECK(vc == doctest::Approx(0.025 * std::log(0.025 / (std::sqrt(2.0) * 1e-14))).epsilon(1e-15));
    // Below the critical voltage, or a step of at most 2 vt: unchanged.
    CHECK(pnjlim(0.5, 0.0, kVt, vc) == 0.5);
    CHECK(pnjlim(0.74, 0.70, kVt, vc) == 0.74);
    CHECK(pnjlim(-5.0, 0.7, kVt, vc) == -5.0);  // reverse steps are never limited
    // vold > 0: logarithmic step.
    CHECK(pnjlim(5.0, 0.6, kVt, vc) == 0.6 + kVt * std::log(1.0 + (5.0 - 0.6) / kVt));
    // The limited exponential equals the linear prediction at vold.
    const double vold = 0.6;
    const double vnew = pnjlim(1.6, vold, kVt, vc);
    CHECK(std::exp((vnew - vold) / kVt) == doctest::Approx(1.0 + (1.6 - vold) / kVt).epsilon(1e-12));
    // vold <= 0: vt ln(vnew / vt).
    CHECK(pnjlim(5.0, 0.0, kVt, vc) == kVt * std::log(5.0 / kVt));
    CHECK(pnjlim(5.0, -1.0, kVt, vc) == kVt * std::log(5.0 / kVt));
    // A large step down that stays above vcrit: arg <= 0 gives vcrit.
    CHECK(pnjlim(1.0, 2.0, kVt, vc) == vc);
    // NaN passes through.
    CHECK(std::isnan(pnjlim(std::nan(""), 0.5, kVt, vc)));
}

TEST_CASE("Circuit::junctions lists diodes, then each BJT's base-emitter and base-collector junction") {
    CircuitSpec s;
    s.n = 4;
    s.diodes = {{0, -1, 1, -1}, {1, 2, -1, -1}, {1e-14, 2e-14, 0.0, 1e-14}};  // the last two are left out
    s.bjt = {{1, 2}, {2, 3}, {3, -1}, {1e-15, 2e-15}, {150, 150}, {3, 3}, {100, 100}, {25, 25}, {1, -1}};
    const Circuit c(s);
    const mna::circuit::Junctions& j = c.junctions();
    REQUIRE(j.size() == 6);
    // D1 0 -> 1, D2 ground -> 2
    CHECK(j.anode[0] == 0);
    CHECK(j.cathode[0] == 1);
    CHECK(j.anode[1] == -1);
    CHECK(j.cathode[1] == 2);
    CHECK(j.vt[0] == mna::devices::kDiodeVt);
    CHECK(j.vcrit[1] == doctest::Approx(vcrit(0.025, 2e-14)).epsilon(1e-15));
    // NPN (c=1, b=2, e=3): (b, e), (b, c)
    CHECK(j.anode[2] == 2);
    CHECK(j.cathode[2] == 3);
    CHECK(j.anode[3] == 2);
    CHECK(j.cathode[3] == 1);
    CHECK(j.vt[2] == mna::devices::kBjtVt);
    CHECK(j.is[2] == 1e-15);
    // PNP (c=2, b=3, e=ground): (e, b), (c, b)
    CHECK(j.anode[4] == -1);
    CHECK(j.cathode[4] == 3);
    CHECK(j.anode[5] == 2);
    CHECK(j.cathode[5] == 3);
    CHECK(j.vcrit[5] == doctest::Approx(vcrit(0.02585, 2e-15)).epsilon(1e-15));
}

TEST_CASE("limit_junctions moves one node per junction") {
    SUBCASE("ground-referenced junctions move their node") {
        CircuitSpec s;
        s.n = 2;
        s.diodes = {{0, -1}, {-1, 1}, {1e-14, 1e-14}};
        const Circuit c(s);
        const double x_old[2] = {0.0, 0.3};
        double x_new[2] = {5.0, -4.0};  // D2 (ground -> 1): v = -x[1] = 4
        CHECK(limit_junctions(c, x_old, x_new) == 2);
        CHECK(x_new[0] == kVt * std::log(5.0 / kVt));
        const double v2 = pnjlim(4.0, -0.3, kVt, c.junctions().vcrit[1]);
        CHECK(x_new[1] == -v2);
    }
    SUBCASE("floating junctions move the terminal with the larger update; ties move the anode") {
        CircuitSpec s;
        s.n = 2;
        s.diodes = {{0}, {1}, {1e-14}};
        const Circuit c(s);
        const double x_old[2] = {0.7, 0.0};
        double x_new[2] = {3.0, 0.5};  // anode moved more
        CHECK(limit_junctions(c, x_old, x_new) == 1);
        const double v = pnjlim(2.5, 0.7, kVt, c.junctions().vcrit[0]);
        CHECK(x_new[1] == 0.5);
        CHECK(x_new[0] == 0.5 + v);
        double y_new[2] = {0.9, -2.0};  // cathode moved more
        CHECK(limit_junctions(c, x_old, y_new) == 1);
        CHECK(y_new[0] == 0.9);
        CHECK(y_new[1] == 0.9 - pnjlim(2.9, 0.7, kVt, c.junctions().vcrit[0]));
        double tie[2] = {1.7, -1.0};
        CHECK(limit_junctions(c, x_old, tie) == 1);
        CHECK(tie[1] == -1.0);
    }
    SUBCASE("small steps and reverse bias are left alone") {
        CircuitSpec s;
        s.n = 1;
        s.diodes = {{0}, {-1}, {1e-14}};
        const Circuit c(s);
        const double x_old[1] = {0.65};
        double x_new[1] = {0.69};
        CHECK(limit_junctions(c, x_old, x_new) == 0);
        CHECK(x_new[0] == 0.69);
        double rev[1] = {-30.0};
        CHECK(limit_junctions(c, x_old, rev) == 0);
        CHECK(rev[0] == -30.0);
    }
}

TEST_CASE("node_count adds one unit node-to-ground gmin stamp per node unknown") {
    // V1 a 0 (branch 2), R1 a b, D1 b 0: nodes a, b.
    CircuitSpec s;
    s.n = 3;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    diode(s, 1, -1, 1e-14);
    const Circuit without(s);
    s.node_count = 2;
    const Circuit with(s);
    CHECK(with.node_count() == 2);
    CHECK(without.node_count() == 0);
    const std::vector<double> g0 = without.to_dense(mna::circuit::MatrixKind::Gmin);
    const std::vector<double> g1 = with.to_dense(mna::circuit::MatrixKind::Gmin);
    CHECK(g0[0] == 0.0);   // (a, a): no device stamp on a
    CHECK(g1[0] == 1.0);   // node-to-ground
    CHECK(g0[4] == 1.0);   // (b, b): the diode stamp
    CHECK(g1[4] == 2.0);   // plus node-to-ground
    CHECK(g1[8] == 0.0);   // the branch row never gets one
    // gmin = 0 leaves G untouched.
    CHECK(bitwise_equal(with.g_effective(0.0), with.values(mna::circuit::MatrixKind::G)));
    const std::vector<double> eff = with.g_effective(1e-9);
    CHECK(eff[static_cast<std::size_t>(with.slot(0, 0))] == 1e-3 + 1e-9);
    CircuitSpec bad = s;
    bad.node_count = 4;
    CHECK_THROWS_AS(Circuit{bad}, mna::Error);
    bad.node_count = -1;
    CHECK_THROWS_AS(Circuit{bad}, mna::Error);
}

TEST_CASE("junction strategy: a diode across an ideal source converges within max_iter") {
    const Circuit c = clamped_diode();
    for (const double volts : {0.8, 1.0, 5.0}) {
        const double b[2] = {0.0, volts};
        NewtonOptions o;
        o.max_iter = 200;
        o.gmin_continuation = false;
        o.junction_limiting = false;  // the reference iteration creeps ~vt per iteration
        if (volts > 0.8) {
            CHECK_THROWS_AS((void)mna::analyses::op(c, b, nullptr, 0.0, o), mna::Error);
        }
        o.junction_limiting = true;
        const mna::analyses::OpResult r = mna::analyses::op(c, b, nullptr, 0.0, o);
        CHECK(r.iterations <= 200);
        CHECK(r.x[0] == volts);
        CHECK(r.x[1] == doctest::Approx(-1e-14 * std::expm1(volts / 0.025)).epsilon(1e-12));
        if (volts > 0.8) {
            CHECK(r.junction_switches == 1);
            CHECK(r.limited_steps > 0);
        }
    }
    // 20 V: exp(800) overflows, the strategy cannot help and the error is still NR_CONVERGENCE.
    const double b20[2] = {0.0, 20.0};
    NewtonOptions o;
    o.max_iter = 200;
    try {
        (void)mna::analyses::op(c, b20, nullptr, 0.0, o);
        FAIL("20 V across a diode must not converge");
    } catch (const mna::Error& e) {
        CHECK(e.code() == mna::ErrorCode::NrConvergence);
    }
}

TEST_CASE("junction strategy: runs the reference iteration converges are bit for bit unchanged") {
    // V1 a 0 DC 5, R1 a b 1k, D1 b 0: the reference iteration converges without stalling.
    CircuitSpec s;
    s.n = 3;
    s.node_count = 2;
    voltage_source(s, 2, 0, -1);
    conductance(s, 0, 1, 1e-3);
    diode(s, 1, -1, 1e-14);
    const Circuit c(s);
    const double b[3] = {0.0, 0.0, 5.0};
    NewtonOptions on;
    on.max_iter = 200;
    NewtonOptions off = on;
    off.junction_limiting = false;
    const mna::analyses::OpResult r_on = mna::analyses::op(c, b, nullptr, 0.0, on);
    const mna::analyses::OpResult r_off = mna::analyses::op(c, b, nullptr, 0.0, off);
    CHECK(bitwise_equal(r_on.x, r_off.x));
    CHECK(r_on.iterations == r_off.iterations);
    CHECK(r_on.junction_switches == 0);
    CHECK(r_on.limited_steps == 0);
}

TEST_CASE("gmin continuation: CMOS inverter chains converge at gmin 0") {
    for (const int stages : {4, 6, 8}) {
        const Circuit c = inverter_chain(stages);
        const std::vector<double> b = chain_rhs(stages);
        NewtonOptions o;
        o.max_iter = 200;
        o.gmin_continuation = false;
        try {
            (void)mna::analyses::op(c, b.data(), nullptr, 0.0, o);
            FAIL("the reference iteration fails on a " << stages << "-stage chain");
        } catch (const mna::Error& e) {
            CHECK(e.code() == mna::ErrorCode::NrConvergence);
            CHECK(std::string(e.what()) == "Newton step could not reduce residual at iteration 1.");
        }
        o.gmin_continuation = true;
        const mna::analyses::OpResult r = mna::analyses::op(c, b.data(), nullptr, 0.0, o);
        CHECK(r.continuation_used);
        CHECK(r.continuation_levels > 0);
        CHECK(r.iterations <= 200);
        for (int k = 1; k <= stages; ++k) {
            CHECK(r.x[static_cast<std::size_t>(k)] == doctest::Approx(k % 2 == 1 ? 1.2 : 0.0).epsilon(1e-6).scale(1.0));
        }
        // The iteration budget is shared: max_iter = 2 leaves nothing for the continuation.
        NewtonOptions tight = o;
        tight.max_iter = 2;
        CHECK_THROWS_AS((void)mna::analyses::op(c, b.data(), nullptr, 0.0, tight), mna::Error);
    }
}

TEST_CASE("gmin continuation is not used when the first attempt converges or the system is singular") {
    // A floating resistor island stays singular at gmin = 0 whatever the continuation does: the
    // first attempt's error is raised unchanged.
    CircuitSpec s;
    s.n = 5;
    s.node_count = 4;
    voltage_source(s, 4, 0, -1);
    conductance(s, 0, 1, 1e-3);
    diode(s, 1, -1, 1e-14);
    conductance(s, 2, 3, 1e-3);
    const Circuit island(s);
    const double b[5] = {0.0, 0.0, 0.0, 0.0, 5.0};
    try {
        (void)mna::analyses::op(island, b, nullptr, 0.0, NewtonOptions{});
        FAIL("singular at gmin 0");
    } catch (const mna::Error& e) {
        CHECK(e.code() == mna::ErrorCode::JacobianSingular);
        CHECK(std::string(e.what()) == "Jacobian is singular at iteration 0.");
    }
    // The first gmin step regularizes it through the node-to-ground stamps of x2 and x3.
    const mna::analyses::OpResult r = mna::analyses::op(island, b, nullptr, 1e-12, NewtonOptions{});
    CHECK_FALSE(r.continuation_used);
    CHECK(r.x[2] == 0.0);
    CHECK(r.x[3] == 0.0);
}
