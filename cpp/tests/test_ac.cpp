// AC sweep: linearization, complex direct and Krylov paths, thread determinism, errors.
#include <doctest/doctest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mna/analyses/ac.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::analyses::AcOptions;
using mna::analyses::AcResult;
using mna::analyses::ac_sweep;
using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;
using Complex = std::complex<double>;

namespace {

constexpr double kPi = 3.14159265358979323846;

void add(mna::circuit::TripletList& t, std::int64_t r, std::int64_t c, double v) {
    if (r >= 0 && c >= 0) {
        t.rows.push_back(r);
        t.cols.push_back(c);
        t.values.push_back(v);
    }
}

// Two-terminal admittance stamp between j and k (-1 = ground).
void stamp(mna::circuit::TripletList& t, std::int64_t j, std::int64_t k, double v) {
    add(t, j, j, v);
    add(t, k, k, v);
    add(t, j, k, -v);
    add(t, k, j, -v);
}

// Voltage source from node p to ground on branch row br.
void vsource(CircuitSpec& s, std::int64_t p, std::int64_t br) {
    add(s.g, p, br, 1.0);
    add(s.g, br, p, 1.0);
}

// V1 in 0 AC 1; R1 in out R; C1 out 0 C. Unknowns: in, out, I(V1).
CircuitSpec rc_lowpass(double r, double c) {
    CircuitSpec s;
    s.n = 3;
    vsource(s, 0, 2);
    stamp(s.g, 0, 1, 1.0 / r);
    stamp(s.c, 1, -1, c);
    return s;
}

// AC-driven RLC ladder: m series resistors, shunt capacitors, and shunt
// inductors (with branch currents) on every other node. Unknowns: nodes
// 0..m, the source branch, then the inductor branches.
CircuitSpec rlc_ladder(int m) {
    CircuitSpec s;
    const std::int64_t nodes = m + 1;
    std::int64_t next = nodes;
    const std::int64_t vbr = next++;
    vsource(s, 0, vbr);
    for (int k = 0; k < m; ++k) {
        stamp(s.g, k, k + 1, 1.0 / (100.0 + 7.0 * k));
        stamp(s.c, k + 1, -1, 1e-9 * (1.0 + 0.1 * (k % 5)));
        if (k % 2 == 1) {
            const std::int64_t br = next++;
            add(s.g, k + 1, br, 1.0);
            add(s.g, br, k + 1, 1.0);
            add(s.c, br, br, -1e-6 * (1.0 + 0.05 * k));
        }
        s.gmin_stamps.rows.push_back(k + 1);
        s.gmin_stamps.cols.push_back(k + 1);
        s.gmin_stamps.values.push_back(1.0);
    }
    s.n = next;
    return s;
}

std::vector<Complex> source_rhs(std::int64_t n, std::int64_t row) {
    std::vector<Complex> b(static_cast<std::size_t>(n), Complex{});
    b[static_cast<std::size_t>(row)] = Complex(1.0, 0.0);
    return b;
}

std::vector<double> linspace(double a, double b, int points) {
    std::vector<double> f(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        f[static_cast<std::size_t>(i)] = points == 1 ? a : a + (b - a) * i / (points - 1);
    }
    return f;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

std::string ac_error_message(const Circuit& circuit, const std::vector<Complex>& b, const std::vector<double>& f,
                             int threads) {
    try {
        ac_sweep(circuit, nullptr, b.data(), f.data(), static_cast<std::int64_t>(f.size()), 0.0, AcOptions{}, threads);
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::AcError);
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST_CASE("ac_failure_message formats the frequency like Python's {:.3e}") {
    CHECK(mna::analyses::ac_failure_message(1000.0) == "AC solve failed at frequency 1.000e+03 Hz.");
    CHECK(mna::analyses::ac_failure_message(0.0) == "AC solve failed at frequency 0.000e+00 Hz.");
    CHECK(mna::analyses::ac_failure_message(-12.5) == "AC solve failed at frequency -1.250e+01 Hz.");
    CHECK(mna::analyses::ac_failure_message(1e100) == "AC solve failed at frequency 1.000e+100 Hz.");
    CHECK(mna::analyses::ac_failure_message(123456.0) == "AC solve failed at frequency 1.235e+05 Hz.");
}

TEST_CASE("RC low-pass magnitudes match the analytic response, label-major layout") {
    const double r = 1e3;
    const double c = 1e-6;
    const Circuit circuit(rc_lowpass(r, c));
    const auto b = source_rhs(3, 2);
    const std::vector<double> f = {0.0, 10.0, 159.15494309189535, 1e4, -10.0};
    const AcResult res = ac_sweep(circuit, nullptr, b.data(), f.data(), 5, 0.0, AcOptions{}, 1);
    REQUIRE(res.n == 3);
    REQUIRE(res.points == 5);
    REQUIRE(res.magnitudes.size() == 15);
    CHECK(res.krylov_stats.empty());
    for (std::size_t i = 0; i < f.size(); ++i) {
        const double wrc = std::abs(2.0 * kPi * f[i] * r * c);  // |H(-f)| = |H(f)|
        CHECK(res.magnitudes[0 * 5 + i] == doctest::Approx(1.0).epsilon(1e-14));
        CHECK(res.magnitudes[1 * 5 + i] == doctest::Approx(1.0 / std::sqrt(1.0 + wrc * wrc)).epsilon(1e-12));
        // |I(V1)| = |V_in / (R + 1/(jwC))|
        const double expected_i = wrc / (r * std::sqrt(1.0 + wrc * wrc));
        CHECK(res.magnitudes[2 * 5 + i] == doctest::Approx(expected_i).epsilon(1e-12));
    }
    // f = 0 is a plain DC point (open capacitor): no current.
    CHECK(res.magnitudes[2 * 5 + 0] == 0.0);
}

TEST_CASE("one point, and points <= 0 is AC_ERROR") {
    const Circuit circuit(rc_lowpass(1e3, 1e-6));
    const auto b = source_rhs(3, 2);
    const std::vector<double> f = {100.0};
    const AcResult res = ac_sweep(circuit, nullptr, b.data(), f.data(), 1, 0.0, AcOptions{}, 4);
    CHECK(res.magnitudes.size() == 3);
    CHECK(res.threads_used == 1);
    for (std::int64_t points : {std::int64_t{0}, std::int64_t{-3}}) {
        try {
            ac_sweep(circuit, nullptr, b.data(), f.data(), points, 0.0, AcOptions{}, 1);
            FAIL("expected AC_ERROR");
        } catch (const Error& e) {
            CHECK(e.code() == ErrorCode::AcError);
        }
    }
}

TEST_CASE("results are bitwise identical for every thread count (direct and Krylov)") {
    const CircuitSpec spec = rlc_ladder(40);
    const Circuit circuit(spec);
    const auto b = source_rhs(spec.n, 41);
    const std::vector<double> f = linspace(1e3, 5e7, 37);
    for (bool krylov : {false, true}) {
        AcOptions options;
        options.use_krylov = krylov;
        options.krylov_rank = 30;
        const AcResult serial = ac_sweep(circuit, nullptr, b.data(), f.data(), 37, 1e-9, options, 1);
        CHECK(serial.threads_used == 1);
        for (int threads : {2, 3, 4, 8}) {
            const AcResult par = ac_sweep(circuit, nullptr, b.data(), f.data(), 37, 1e-9, options, threads);
            CHECK(par.threads_used == threads);
            CHECK(bitwise_equal(serial.magnitudes, par.magnitudes));
            REQUIRE(par.krylov_stats.size() == serial.krylov_stats.size());
            for (std::size_t i = 0; i < par.krylov_stats.size(); ++i) {
                CHECK(par.krylov_stats[i].info.iterations == serial.krylov_stats[i].info.iterations);
                CHECK(par.krylov_stats[i].used_direct_fallback == serial.krylov_stats[i].used_direct_fallback);
            }
        }
        // points < 2 * threads: the sweep stays on the caller with the same answer.
        const AcResult few = ac_sweep(circuit, nullptr, b.data(), f.data(), 3, 1e-9, options, 4);
        CHECK(few.threads_used == 1);
        for (std::int64_t r = 0; r < spec.n; ++r) {
            for (std::size_t i = 0; i < 3; ++i) {
                CHECK(few.magnitudes[static_cast<std::size_t>(r) * 3 + i] ==
                      serial.magnitudes[static_cast<std::size_t>(r) * 37 + i]);
            }
        }
    }
}

TEST_CASE("Krylov path: accepted answers agree with the direct solve; failures fall back") {
    const CircuitSpec spec = rlc_ladder(30);
    const Circuit circuit(spec);
    const auto b = source_rhs(spec.n, 31);
    const std::vector<double> f = linspace(1e4, 1e7, 9);
    const AcResult direct = ac_sweep(circuit, nullptr, b.data(), f.data(), 9, 0.0, AcOptions{}, 1);

    AcOptions krylov;
    krylov.use_krylov = true;
    krylov.krylov_rank = static_cast<std::int64_t>(spec.n);
    const AcResult accepted = ac_sweep(circuit, nullptr, b.data(), f.data(), 9, 0.0, krylov, 2);
    REQUIRE(accepted.krylov_stats.size() == 9);
    double scale = 0.0;
    for (double v : direct.magnitudes) {
        scale = std::max(scale, v);
    }
    for (std::size_t i = 0; i < 9; ++i) {
        const auto& st = accepted.krylov_stats[i];
        CHECK(st.matrix_kind == mna::linear::MatrixKind::General);  // complex symmetric is not Hermitian
        CHECK(st.info.route == mna::linear::KrylovRoute::General);
        CHECK(std::string(st.info.method) == "arnoldi_gmres");
        CHECK(st.matrix_nnz > 0);
        CHECK_FALSE(st.used_direct_fallback);
    }
    for (std::size_t k = 0; k < direct.magnitudes.size(); ++k) {
        CHECK(std::abs(accepted.magnitudes[k] - direct.magnitudes[k]) <= 1e-7 * scale);
    }

    AcOptions starved = krylov;
    starved.krylov_rank = 1;
    starved.krylov_max_iter = 1;
    const AcResult fallback = ac_sweep(circuit, nullptr, b.data(), f.data(), 9, 0.0, starved, 1);
    for (const auto& st : fallback.krylov_stats) {
        CHECK_FALSE(st.info.converged);
        CHECK(st.used_direct_fallback);
    }
    CHECK(bitwise_equal(fallback.magnitudes, direct.magnitudes));

    AcOptions unknown = krylov;
    unknown.krylov_method = "bicgstab";
    CHECK_THROWS_AS(ac_sweep(circuit, nullptr, b.data(), f.data(), 9, 0.0, unknown, 1), std::invalid_argument);
    AcOptions bad_rank = krylov;
    bad_rank.krylov_rank = 0;
    try {
        ac_sweep(circuit, nullptr, b.data(), f.data(), 9, 0.0, bad_rank, 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
}

TEST_CASE("the lowest singular frequency is reported for every thread count") {
    // V1 a 0 AC 1; R1 a b 1k; C1 b c 1u; C2 c 0 1u: node c is floating at DC only.
    CircuitSpec s;
    s.n = 4;  // a, b, c, I(V1)
    vsource(s, 0, 3);
    stamp(s.g, 0, 1, 1e-3);
    stamp(s.c, 1, 2, 1e-6);
    stamp(s.c, 2, -1, 1e-6);
    const Circuit circuit(s);
    const auto b = source_rhs(4, 3);
    const std::vector<double> f = linspace(-70.0, 130.0, 21);  // f[7] == 0
    REQUIRE(f[7] == 0.0);
    for (int threads : {1, 2, 4, 8}) {
        CHECK(ac_error_message(circuit, b, f, threads) == "AC solve failed at frequency 0.000e+00 Hz.");
    }
    const std::vector<double> ac_only = {10.0, 20.0};
    const AcResult ok = ac_sweep(circuit, nullptr, b.data(), ac_only.data(), 2, 0.0, AcOptions{}, 1);
    // V(b) = Zc / (R + Zc) with the two capacitors in series (0.5 uF).
    const double zc = 1.0 / (2.0 * kPi * 10.0 * 0.5e-6);
    CHECK(ok.magnitudes[1 * 2 + 0] == doctest::Approx(zc / std::sqrt(1e6 + zc * zc)).epsilon(1e-12));

    // A node with no stamp at all is singular at every frequency: the first one is reported.
    CircuitSpec island;
    island.n = 3;  // a, x (nothing connected), I(V1)
    vsource(island, 0, 2);
    stamp(island.g, 0, -1, 1e-3);
    island.c.rows.push_back(1);  // keep x in the pattern with a zero entry
    island.c.cols.push_back(1);
    island.c.values.push_back(0.0);
    const Circuit floating(island);
    const auto bi = source_rhs(3, 2);
    const std::vector<double> sweep = linspace(5.0, 50.0, 16);
    for (int threads : {1, 4}) {
        CHECK(ac_error_message(floating, bi, sweep, threads) == "AC solve failed at frequency 5.000e+00 Hz.");
    }
}

TEST_CASE("linearization adds the device Jacobian at the operating point") {
    // V1 a 0 AC 1; R1 a d 1k; D1 d 0 Is. Unknowns: a, d, I(V1).
    CircuitSpec s;
    s.n = 3;
    vsource(s, 0, 2);
    stamp(s.g, 0, 1, 1e-3);
    s.diodes.anode.push_back(1);
    s.diodes.cathode.push_back(-1);
    s.diodes.is.push_back(1e-14);
    const Circuit circuit(s);
    const std::vector<double> x_op = {5.0, 0.6, -4.4e-3};
    const double gd = 1e-14 / 0.025 * std::exp(0.6 / 0.025);

    const std::vector<double> g0 = mna::analyses::linearized_g(circuit, nullptr, 0.0, 1);
    const std::vector<double> g1 = mna::analyses::linearized_g(circuit, x_op.data(), 0.0, 1);
    const std::int32_t dd = circuit.slot(1, 1);
    REQUIRE(dd >= 0);
    CHECK(g0[static_cast<std::size_t>(dd)] == 1e-3);
    CHECK(g1[static_cast<std::size_t>(dd)] == 1e-3 + gd);

    const auto b = source_rhs(3, 2);
    const std::vector<double> f = {1.0, 1e3};
    const AcResult res = ac_sweep(circuit, x_op.data(), b.data(), f.data(), 2, 0.0, AcOptions{}, 1);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(res.magnitudes[1 * 2 + i] == doctest::Approx(1.0 / (1.0 + 1e3 * gd)).epsilon(1e-12));
    }
    // Without x_op the diode is ignored (open): V(d) follows V(a).
    const AcResult open = ac_sweep(circuit, nullptr, b.data(), f.data(), 2, 0.0, AcOptions{}, 1);
    CHECK(open.magnitudes[1 * 2 + 0] == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("gmin stamps enter G_lin") {
    const CircuitSpec spec = rlc_ladder(4);
    const Circuit circuit(spec);
    const std::vector<double> g0 = mna::analyses::linearized_g(circuit, nullptr, 0.0, 1);
    const std::vector<double> g1 = mna::analyses::linearized_g(circuit, nullptr, 1e-3, 1);
    const std::int32_t slot = circuit.slot(1, 1);
    CHECK(g1[static_cast<std::size_t>(slot)] == g0[static_cast<std::size_t>(slot)] + 1e-3);
    CHECK(bitwise_equal(circuit.g_effective(1e-3), g1));
}

TEST_CASE("empty circuit gives an empty (0, points) result") {
    CircuitSpec s;
    s.n = 0;
    const Circuit circuit(s);
    const std::vector<double> f = {1.0, 2.0, 3.0, 4.0};
    for (bool krylov : {false, true}) {
        AcOptions options;
        options.use_krylov = krylov;
        options.krylov_rank = 1;
        const AcResult res = ac_sweep(circuit, nullptr, nullptr, f.data(), 4, 0.0, options, 2);
        CHECK(res.n == 0);
        CHECK(res.magnitudes.empty());
        CHECK(res.krylov_stats.size() == (krylov ? 4u : 0u));
    }
}
