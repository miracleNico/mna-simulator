// Device model boundaries: Level-1 MOS, Level-1 BJT, diode.
#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <limits>

#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/devices/mos_level1.hpp"

using namespace mna::devices;

namespace {
constexpr double kBeta = 1e-3;
constexpr double kVth = 0.4;
constexpr double kLambda = 0.02;
constexpr double kInf = std::numeric_limits<double>::infinity();

MosEval nmos(double vd, double vg, double vs, double lambda = kLambda) {
    return mos_level1(1, kBeta, kVth, lambda, vd, vg, vs);
}
MosEval pmos(double vd, double vg, double vs, double lambda = kLambda) {
    return mos_level1(-1, kBeta, kVth, lambda, vd, vg, vs);
}
}  // namespace

TEST_CASE("MOS: vgs == vth is off with zero derivatives") {
    const MosEval r = nmos(1.0, 0.4, 0.0);
    CHECK(r.ids == 0.0);
    CHECK(r.d_drain == 0.0);
    CHECK(r.d_gate == 0.0);
    CHECK(r.d_source == 0.0);
    CHECK(mos_level1_region(1, 1.0, 0.4, 0.0, kVth) == MosRegion::Off);
    CHECK(nmos(1.0, 0.4 + 1e-12, 0.0).ids > 0.0);
}

TEST_CASE("MOS: triode/saturation boundary is saturation and C1 continuous") {
    // vds == vgs - vth exactly (0.8 in decimal, 1.2 - 0.4 in binary)
    const double vb = 1.2 - 0.4;
    const MosEval at = nmos(vb, 1.2, 0.0);
    const double sat = 0.5 * kBeta * vb * vb * (1.0 + kLambda * vb);
    CHECK(at.ids == sat);
    CHECK(mos_level1_region(1, vb, 1.2, 0.0, kVth) == MosRegion::Saturation);
    const MosEval below = nmos(vb - 1e-9, 1.2, 0.0);
    CHECK(mos_level1_region(1, vb - 1e-9, 1.2, 0.0, kVth) == MosRegion::Triode);
    CHECK(below.ids == doctest::Approx(at.ids).epsilon(1e-8));
    CHECK(below.d_drain == doctest::Approx(at.d_drain).epsilon(1e-6));
    CHECK(below.d_gate == doctest::Approx(at.d_gate).epsilon(1e-8));
    // Triode formula, verbatim operation order.
    const double vov = 1.2 - 0.4;
    const double vds = 0.3;
    const double channel = vov * vds - 0.5 * vds * vds;
    const double modulation = 1.0 + kLambda * vds;
    const MosEval tri = nmos(0.3, 1.2, 0.0);
    CHECK(tri.ids == kBeta * channel * modulation);
    CHECK(tri.d_gate == kBeta * vds * modulation);
    CHECK(tri.d_drain == kBeta * ((vov - vds) * modulation + channel * kLambda));
    CHECK(tri.d_source == -tri.d_gate - tri.d_drain);
}

TEST_CASE("MOS: vd == vs takes the forward branch; vd < vs swaps source and drain") {
    const MosEval eq = nmos(0.3, 1.2, 0.3);
    CHECK(eq.ids == 0.0);
    CHECK(eq.d_drain == doctest::Approx(5e-4).epsilon(1e-12));
    CHECK(eq.d_gate == 0.0);
    CHECK(eq.d_source == doctest::Approx(-5e-4).epsilon(1e-12));
    const MosEval rev = nmos(0.0, 1.2, 0.1);
    CHECK(rev.ids < 0.0);
    CHECK(mos_level1_region(1, 0.0, 1.2, 0.1, kVth) == MosRegion::Reverse);
    // Reverse mode equals the forward device with drain and source exchanged.
    const MosEval fwd = nmos(0.1, 1.2, 0.0);
    CHECK(rev.ids == -fwd.ids);
    CHECK(rev.d_drain == -fwd.d_source);
    CHECK(rev.d_gate == -fwd.d_gate);
    CHECK(rev.d_source == -fwd.d_drain);
}

TEST_CASE("MOS: PMOS mirrors NMOS and lambda = 0 removes modulation") {
    for (double vd : {0.2, 1.0, -0.3}) {
        for (double vg : {0.0, 0.5, -1.2}) {
            for (double vs : {1.2, 0.0}) {
                const MosEval p = pmos(vd, vg, vs);
                const MosEval n = nmos(-vd, -vg, -vs);
                CHECK(p.ids == -n.ids);
                CHECK(p.d_drain == n.d_drain);
                CHECK(p.d_gate == n.d_gate);
                CHECK(p.d_source == n.d_source);
            }
        }
    }
    CHECK(pmos(0.2, 0.0, 1.2).ids < 0.0);
    CHECK(mos_level1_region(-1, 0.2, 0.0, 1.2, kVth) == MosRegion::Saturation);
    const MosEval sat0 = nmos(1.2, 1.2, 0.0, 0.0);
    const double vov = 1.2 - 0.4;
    CHECK(sat0.ids == 0.5 * kBeta * vov * vov * (1.0 + 0.0 * 1.2));
    CHECK(sat0.d_drain == 0.0);
    CHECK(sat0.d_gate == kBeta * vov * 1.0);
}

namespace {
BjtEval npn(double vc, double vb, double ve, double vaf = 100.0, double var = 25.0, double bf = 150.0,
            double br = 3.0) {
    return bjt_level1(1, 1e-15, bf, br, vaf, var, vc, vb, ve);
}
void check_kirchhoff(const BjtEval& r) {
    CHECK(r.i[2] == -(r.i[0] + r.i[1]));
    for (int col = 0; col < 3; ++col) {
        CHECK(r.d[2][col] == -(r.d[0][col] + r.d[1][col]));
    }
}
}  // namespace

TEST_CASE("BJT: regions, Kirchhoff sums and pinned reference values") {
    const BjtEval cutoff = npn(5.0, -5.0, 0.0);
    check_kirchhoff(cutoff);
    CHECK(cutoff.i[0] == doctest::Approx(-2.4304635761589396e-16).epsilon(1e-12));
    CHECK(cutoff.i[1] == doctest::Approx(-2.0662251655629145e-16).epsilon(1e-12));
    CHECK(cutoff.i[2] == doctest::Approx(4.496688741721854e-16).epsilon(1e-12));
    CHECK(bjt_level1_region(1, 5.0, -5.0, 0.0) == BjtRegion::Cutoff);

    const BjtEval fwd = npn(5.0, 0.7, 0.0);
    check_kirchhoff(fwd);
    CHECK(fwd.i[0] > 0.0);
    CHECK(fwd.i[0] / fwd.i[1] == doctest::Approx(150.0 * (1.0 + 5.0 / 100.0)).epsilon(1e-6));
    CHECK(bjt_level1_region(1, 5.0, 0.7, 0.0) == BjtRegion::ForwardActive);

    const BjtEval sat = npn(0.1, 0.7, 0.0);
    check_kirchhoff(sat);
    CHECK(bjt_level1_region(1, 0.1, 0.7, 0.0) == BjtRegion::Saturation);
    const BjtEval rev = npn(0.0, 0.7, 5.0);
    check_kirchhoff(rev);
    CHECK(rev.i[0] < 0.0);
    CHECK(bjt_level1_region(1, 0.0, 0.7, 5.0) == BjtRegion::ReverseActive);
}

TEST_CASE("BJT: exponent continues linearly beyond |v/vt| = 80, consistent with its derivative (P3b)") {
    // bjt_exp: value and slope are continuous at both clip points.
    const double e80 = std::exp(80.0);
    const double em80 = std::exp(-80.0);
    CHECK(bjt_exp(80.0).value == e80);
    CHECK(bjt_exp(80.0).slope == e80);
    const BjtExp just_above = bjt_exp(std::nextafter(80.0, 200.0));
    CHECK(just_above.value == doctest::Approx(e80).epsilon(1e-12));
    CHECK(just_above.slope == e80);
    CHECK(bjt_exp(100.0).value == e80 * (1.0 + 20.0));
    CHECK(bjt_exp(100.0).slope == e80);
    CHECK(bjt_exp(-80.0).value == em80);
    const BjtExp just_below = bjt_exp(std::nextafter(-80.0, -200.0));
    CHECK(just_below.value == doctest::Approx(em80).epsilon(1e-12));
    CHECK(just_below.slope == em80);
    CHECK(bjt_exp(-100.0).value == em80 * (1.0 - 20.0));
    CHECK(std::isnan(bjt_exp(std::nan("")).value));

    // At the clip point the device keeps the reference values; just beyond it currents and
    // derivatives continue without a jump.
    const double v_clip = 80 * kBjtVt;
    const BjtEval at = npn(5.0, v_clip, 0.0);
    CHECK(at.i[0] == doctest::Approx(5.779125996966741e19).epsilon(1e-12));
    CHECK(at.d[0][1] == doctest::Approx(2.2356386835461277e21).epsilon(1e-12));
    const BjtEval beyond = npn(5.0, std::nextafter(v_clip, 10.0), 0.0);
    for (int i = 0; i < 3; ++i) {
        CHECK(beyond.i[i] == doctest::Approx(at.i[i]).epsilon(1e-12));
        for (int j = 0; j < 3; ++j) {
            CHECK(beyond.d[i][j] == doctest::Approx(at.d[i][j]).epsilon(1e-12));
        }
    }
    // Beyond the clip the current grows linearly (was: flat) and the Jacobian is its slope.
    for (double vb : {3.0, 6.0}) {
        const double h = 1e-3;
        const BjtEval r = npn(5.0, vb, 0.0);
        const BjtEval up = npn(5.0, vb + h, 0.0);
        const BjtEval down = npn(5.0, vb - h, 0.0);
        CHECK(up.i[0] > r.i[0]);
        for (int i = 0; i < 3; ++i) {
            CHECK((up.i[i] - down.i[i]) / (2.0 * h) == doctest::Approx(r.d[i][1]).epsilon(1e-6));
        }
    }
    // Deep reverse bias: reverse saturation currents (the exp term is ~1e-35).
    const BjtEval lo = npn(5.0, -30.0, 0.0);
    CHECK(lo.i[1] == doctest::Approx(-1e-15 * (1.0 / 151.0 + 1.0 / 4.0)).epsilon(1e-12));
}

TEST_CASE("BJT: vaf = 0, var = 0, bf = 0 and the PNP mirror") {
    const BjtEval no_early = npn(5.0, 0.7, 0.0, 0.0, 0.0);
    check_kirchhoff(no_early);
    CHECK(no_early.d[0][0] == doctest::Approx(no_early.d[0][0]));
    const double exp_be = std::exp(0.7 / kBjtVt);
    const double ibe = 1e-15 * (exp_be - 1.0);
    const double exp_bc = bjt_exp((0.7 - 5.0) / kBjtVt).value;
    const double ibc = 1e-15 * (exp_bc - 1.0);
    CHECK(no_early.i[0] == (150.0 / 151.0) * ibe * 1.0 - ibc * 1.0);
    const BjtEval no_bf = npn(5.0, 0.7, 0.0, 100.0, 25.0, 0.0, 3.0);
    check_kirchhoff(no_bf);
    CHECK(no_bf.i[1] == (1.0 - 0.0) * ibe + (1.0 - 3.0 / 4.0) * ibc * (1.0 + (0.0 - 5.0) / 25.0));
    for (double vc : {-5.0, 0.1}) {
        for (double vb : {-0.7, 0.3}) {
            const BjtEval p = bjt_level1(-1, 1e-15, 150.0, 3.0, 100.0, 25.0, vc, vb, 0.0);
            const BjtEval n = npn(-vc, -vb, -0.0);
            for (int i = 0; i < 3; ++i) {
                CHECK(p.i[i] == -n.i[i]);
                for (int j = 0; j < 3; ++j) {
                    CHECK(p.d[i][j] == n.d[i][j]);
                }
            }
        }
    }
    CHECK(bjt_level1_region(-1, -5.0, -0.7, 0.0) == BjtRegion::ForwardActive);
    CHECK(bjt_level1_region(-1, 0.0, 0.0, 0.0) == BjtRegion::Cutoff);
}

TEST_CASE("diode: reverse, zero bias and overflow") {
    const DiodeEval rev = diode(1e-14, -1.0);
    CHECK(rev.current == 1e-14 * (std::exp(-1.0 / 0.025) - 1.0));
    CHECK(rev.current == doctest::Approx(-1e-14).epsilon(1e-12));
    const DiodeEval zero = diode(1e-14, 0.0);
    CHECK(zero.current == 0.0);
    CHECK(zero.conductance == 1e-14 / 0.025);
    const DiodeEval fwd = diode(1e-14, 0.7);
    CHECK(fwd.conductance == 1e-14 / 0.025 * std::exp(0.7 / 0.025));
    const DiodeEval big = diode(1e-14, 20.0);  // exp(800) overflows, no clipping
    CHECK(big.current == kInf);
    CHECK(big.conductance == kInf);
}
