// Device model boundaries: Level-1 MOS, Level-1 BJT, diode; and the model-card kernels of
// phase M2 (SPICE diode, Gummel-Poon BJT, four-terminal MOS): derivative consistency by
// central differences, the limits that reduce to simpler forms, continuity at the region
// boundaries and exponent clips, NPN/PNP and NMOS/PMOS symmetry, and the compiled-circuit
// plumbing (tables, junctions, assembly, operating points).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/devices/gummel_poon.hpp"
#include "mna/devices/mos_card.hpp"
#include "mna/devices/mos_level1.hpp"
#include "mna/devices/spice_diode.hpp"
#include "mna/error.hpp"

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

// ==== model-card kernels (phase M2) ==========================================================

namespace {

/// |a - b| <= rtol * max(|a|, |b|, scale)
bool close(double a, double b, double rtol, double scale = 0.0) {
    return std::fabs(a - b) <= rtol * std::max({std::fabs(a), std::fabs(b), scale});
}

GummelPoonParams q2n2222() {
    GummelPoonParams p;
    p.is = 14.34e-15;
    p.bf = 255.9;
    p.nf = 1.0;
    p.br = 6.092;
    p.nr = 1.0;
    p.vaf = 74.03;
    p.ikf = 0.2847;
    p.ise = 14.34e-15;
    p.ne = 1.307;
    p.ikr = 0.5;
    p.isc = 1e-14;
    p.nc = 2.0;
    p.var = 20.0;
    return p;
}

void check_gp_derivatives(int sign, const GummelPoonParams& p, double vc, double vb, double ve) {
    const BjtEval r = gummel_poon(sign, p, vc, vb, ve);
    double scale = 0.0;
    for (const auto& row : r.d) {
        for (double value : row) {
            scale = std::max(scale, std::fabs(value));
        }
    }
    const double h = 1e-7;
    for (int col = 0; col < 3; ++col) {
        double up[3] = {vc, vb, ve};
        double down[3] = {vc, vb, ve};
        up[col] += h;
        down[col] -= h;
        const BjtEval a = gummel_poon(sign, p, up[0], up[1], up[2]);
        const BjtEval b = gummel_poon(sign, p, down[0], down[1], down[2]);
        double current_scale = 0.0;
        for (double value : r.i) {
            current_scale = std::max(current_scale, std::fabs(value));
        }
        for (int row = 0; row < 3; ++row) {
            const double numeric = (a.i[row] - b.i[row]) / (2.0 * h);
            const double rounding = 64.0 * 2.2e-16 * current_scale / h;
            CHECK(std::fabs(numeric - r.d[row][col]) <= 2e-6 * std::max(std::fabs(r.d[row][col]), scale) + rounding);
        }
    }
    // Kirchhoff: emitter current and row closes the sum.
    CHECK(r.i[2] == -(r.i[0] + r.i[1]));
}

}  // namespace

// ---- SPICE diode ----------------------------------------------------------------------------

TEST_CASE("SPICE diode: thermal voltage, forward law and emission coefficient") {
    CHECK(kCardVt == 8.617333262e-5 * 300.15);
    CHECK(kCardVt == doctest::Approx(0.025864).epsilon(1e-4));
    const double is = 2.52e-9;
    const double n = 1.752;
    const SpiceDiodeEval r = spice_diode(is, n, kInf, 0.6);
    const double vte = n * kCardVt;
    CHECK(r.current == is * (std::exp(0.6 / vte) - 1.0));
    CHECK(r.conductance == is * std::exp(0.6 / vte) / vte);
    CHECK(spice_diode(is, n, kInf, 0.0).current == 0.0);
    CHECK(spice_diode_region(n, kInf, 0.6) == DiodeRegion::Forward);
    CHECK(spice_diode_region(n, kInf, -0.01) == DiodeRegion::Off);
    CHECK(spice_diode_region(n, kInf, -10.0) == DiodeRegion::Reverse);
    CHECK(spice_diode_region(n, 50.0, -60.0) == DiodeRegion::Breakdown);
}

TEST_CASE("SPICE diode: reverse region joins the forward branch with value and slope at -3 n vt") {
    const double is = 1e-14;
    const double n = 1.3;
    const double vte = n * kCardVt;
    const double v0 = -3.0 * vte;
    const SpiceDiodeEval at = spice_diode(is, n, kInf, v0);  // forward branch (v >= -3 vte)
    const SpiceDiodeEval below = spice_diode(is, n, kInf, std::nextafter(v0, -1.0));
    CHECK(close(at.current, below.current, 1e-12));
    CHECK(close(at.conductance, below.conductance, 1e-12));
    CHECK(at.current == doctest::Approx(-is * (1.0 - std::exp(-3.0))).epsilon(1e-12));
    // Deep reverse: -is (1 + (3 vte / (e v))^3) -> -is.
    const SpiceDiodeEval deep = spice_diode(is, n, kInf, -50.0);
    CHECK(deep.current == doctest::Approx(-is).epsilon(1e-6));
    CHECK(deep.conductance > 0.0);
}

TEST_CASE("SPICE diode: breakdown voltage is shifted so the current at -BV is IBV") {
    // 1N4148-like: ibv above is * bv / vt, the fixed-point iteration runs.
    const double is = 2.52e-9;
    const double bv = 100.0;
    const double ibv = 100e-6;
    const double xbv = spice_diode_breakdown_voltage(is, bv, ibv);
    CHECK(xbv < bv);
    CHECK(xbv > bv - 1.0);
    const double matched = is * (std::exp((bv - xbv) / kCardVt) - 1.0 + xbv / kCardVt);
    CHECK(std::fabs(matched - ibv) <= kCardReltol * ibv);
    // The area factor (is and ibv scaled together) does not move it.
    CHECK(close(spice_diode_breakdown_voltage(3.0 * is, bv, 3.0 * ibv), xbv, 1e-12));
    // ibv below is * bv / vt: bv kept; no breakdown: +inf.
    CHECK(spice_diode_breakdown_voltage(is, bv, 1e-7) == bv);
    CHECK(spice_diode_breakdown_voltage(is, kInf, ibv) == kInf);
    // Breakdown branch: -is exp(-(xbv + v) / vt); continuous to within is (3 vte / (e xbv))^3 at -xbv.
    const double n = 1.752;
    const SpiceDiodeEval at = spice_diode(is, n, xbv, -xbv);  // reverse branch (v >= -bv_eff)
    const SpiceDiodeEval beyond = spice_diode(is, n, xbv, std::nextafter(-xbv, -200.0));
    const double jump = is * std::pow(3.0 * n * kCardVt / (kEuler * xbv), 3);
    CHECK(std::fabs(at.current - beyond.current) <= 2.0 * jump + 1e-12 * is);
    CHECK(beyond.conductance == doctest::Approx(is / kCardVt).epsilon(1e-9));
    const SpiceDiodeEval deeper = spice_diode(is, n, xbv, -xbv - 0.2);
    CHECK(deeper.current == doctest::Approx(-is * std::exp(0.2 / kCardVt)).epsilon(1e-12));
}

TEST_CASE("SPICE diode: exponent clips continue linearly with continuous value and slope") {
    const double is = 1e-14;
    const double n = 1.0;
    const double vte = n * kCardVt;
    const double v_clip = kBjtExpClip * vte;
    const SpiceDiodeEval at = spice_diode(is, n, kInf, v_clip);
    const SpiceDiodeEval beyond = spice_diode(is, n, kInf, std::nextafter(v_clip, 100.0));
    CHECK(close(at.current, beyond.current, 1e-12));
    CHECK(close(at.conductance, beyond.conductance, 1e-12));
    const SpiceDiodeEval far = spice_diode(is, n, kInf, 50.0);
    CHECK(std::isfinite(far.current));
    CHECK(far.conductance == is * std::exp(kBjtExpClip) / vte);
    // Breakdown clip at -(bv + v) / vt = 80.
    const double bv = 5.0;
    const double v_brk = -bv - kBjtExpClip * kCardVt;
    const SpiceDiodeEval b_at = spice_diode(is, n, bv, v_brk);
    const SpiceDiodeEval b_beyond = spice_diode(is, n, bv, std::nextafter(v_brk, -100.0));
    CHECK(close(b_at.current, b_beyond.current, 1e-12));
    CHECK(close(b_at.conductance, b_beyond.conductance, 1e-12));
    CHECK(std::isfinite(spice_diode(is, n, bv, -1e3).current));
}

TEST_CASE("SPICE diode: conductance matches central differences in every region") {
    const double is = 1e-14;
    const double n = 1.1;
    const double bv = spice_diode_breakdown_voltage(is, 6.2, 1e-3);
    for (double v : {-12.0, -bv - 0.3, -bv - 3.0, -bv + 0.5, -2.0, -0.2, -0.05, 0.0, 0.3, 0.7, 2.0, 4.0}) {
        const double h = 1e-7;
        const SpiceDiodeEval r = spice_diode(is, n, bv, v);
        const double numeric =
            (spice_diode(is, n, bv, v + h).current - spice_diode(is, n, bv, v - h).current) / (2.0 * h);
        const double rounding = 64.0 * 2.2e-16 * std::fabs(r.current) / h;
        CHECK(std::fabs(numeric - r.conductance) <= 2e-6 * std::max(std::fabs(r.conductance), is / kCardVt) + rounding);
    }
}

// ---- Gummel-Poon ---------------------------------------------------------------------------

TEST_CASE("Gummel-Poon: ISE = ISC = 0, IKF = IKR = inf, VAF = VAR = inf give the simple transport form") {
    GummelPoonParams p;
    p.is = 2e-15;
    p.bf = 150.0;
    p.br = 3.0;
    p.vaf = kInf;
    p.var = kInf;
    p.ikf = kInf;
    p.ikr = kInf;
    for (double vbe : {-0.4, 0.0, 0.62, 0.7}) {
        for (double vbc : {-3.0, 0.0, 0.5}) {
            const BjtEval r = npn_gummel_poon(vbe - vbc, vbe, 0.0, p);
            const double cbe = p.is * (std::exp(vbe / kCardVt) - 1.0);
            const double cbc = p.is * (std::exp(vbc / kCardVt) - 1.0);
            CHECK(r.i[0] == (cbe - cbc) / 1.0 - cbc / p.br - 0.0);
            CHECK(r.i[1] == cbe / p.bf + 0.0 + cbc / p.br + 0.0);
        }
    }
    // SPICE: 0 means "not given" for VAF, VAR, IKF, IKR (bit-identical to +inf).
    GummelPoonParams zero = p;
    zero.vaf = 0.0;
    zero.var = 0.0;
    zero.ikf = 0.0;
    zero.ikr = 0.0;
    const BjtEval a = npn_gummel_poon(3.0, 0.7, 0.0, p);
    const BjtEval b = npn_gummel_poon(3.0, 0.7, 0.0, zero);
    for (int i = 0; i < 3; ++i) {
        CHECK(a.i[i] == b.i[i]);
        for (int j = 0; j < 3; ++j) {
            CHECK(a.d[i][j] == b.d[i][j]);
        }
    }
    // Large but finite IKF / VAF approach the limit.
    GummelPoonParams near = p;
    near.ikf = 1e12;
    near.vaf = 1e12;
    const BjtEval c = npn_gummel_poon(3.0, 0.7, 0.0, near);
    CHECK(close(c.i[0], a.i[0], 1e-9));
}

TEST_CASE("Gummel-Poon: Early effect, high injection and leakage move the currents as expected") {
    const GummelPoonParams p = q2n2222();
    // Forward active: Ic/Ib below BF (leakage, high injection) and Ic grows with vce (VAF).
    const BjtEval low = npn_gummel_poon(2.0, 0.65, 0.0, p);
    const BjtEval high = npn_gummel_poon(10.0, 0.65, 0.0, p);
    CHECK(high.i[0] > low.i[0]);
    CHECK(low.i[0] / low.i[1] < p.bf);
    CHECK(bjt_level1_region(1, 2.0, 0.65, 0.0) == BjtRegion::ForwardActive);
    // High injection: at 0.9 V the collector current is far below the ideal transport current.
    const BjtEval hi = npn_gummel_poon(5.0, 0.9, 0.0, p);
    CHECK(hi.i[0] < 0.5 * p.is * std::exp(0.9 / kCardVt));
}

TEST_CASE("Gummel-Poon: Jacobian matches central differences, NPN and PNP") {
    const GummelPoonParams p = q2n2222();
    GummelPoonParams simple;
    simple.is = 1e-15;
    for (const GummelPoonParams& params : {p, simple}) {
        for (double vbe : {-2.0, -0.3, 0.0, 0.62, 0.75, 0.95}) {
            for (double vbc : {-5.0, -0.4, 0.3, 0.7}) {
                check_gp_derivatives(1, params, vbe - vbc + 0.1, vbe + 0.1, 0.1);
                check_gp_derivatives(-1, params, -(vbe - vbc + 0.1), -(vbe + 0.1), -0.1);
            }
        }
    }
}

TEST_CASE("Gummel-Poon: PNP mirrors NPN; exponent clips are continuous") {
    const GummelPoonParams p = q2n2222();
    for (double vc : {-5.0, 0.1, 0.8}) {
        for (double vb : {-0.7, 0.3, 0.75}) {
            const BjtEval pnp = gummel_poon(-1, p, vc, vb, 0.0);
            const BjtEval n_side = npn_gummel_poon(-vc, -vb, -0.0, p);
            for (int i = 0; i < 3; ++i) {
                CHECK(pnp.i[i] == -n_side.i[i]);
                for (int j = 0; j < 3; ++j) {
                    CHECK(pnp.d[i][j] == n_side.d[i][j]);
                }
            }
        }
    }
    const double v_clip = kBjtExpClip * p.nf * kCardVt;
    const BjtEval at = npn_gummel_poon(5.0, v_clip, 0.0, p);
    const BjtEval beyond = npn_gummel_poon(5.0, std::nextafter(v_clip, 10.0), 0.0, p);
    for (int i = 0; i < 3; ++i) {
        CHECK(close(beyond.i[i], at.i[i], 1e-12));
        for (int j = 0; j < 3; ++j) {
            CHECK(close(beyond.d[i][j], at.d[i][j], 1e-12, std::fabs(at.d[0][1])));
        }
    }
    const BjtEval far = npn_gummel_poon(5.0, 8.0, 0.0, p);
    for (double value : far.i) {
        CHECK(std::isfinite(value));
    }
    check_gp_derivatives(1, p, 5.0, 2.5, 0.0);   // beyond the clip
    check_gp_derivatives(1, p, -9.0, -3.0, 0.0); // both junctions below -80 vt
}

// ---- card MOS ------------------------------------------------------------------------------

TEST_CASE("MOS card: GAMMA = 0 is bit-identical to mos_level1 with vth = type * VTO") {
    for (int sign : {1, -1}) {
        const double vto = sign * 0.7;
        for (double vd : {-1.0, 0.0, 0.3, 2.0}) {
            for (double vg : {-2.0, 0.0, 0.7, 1.5, 3.0}) {
                for (double vs : {-0.4, 0.0, 0.3}) {
                    const Mos4Eval c = mos_card(sign, 1e-3, vto, 0.02, 0.0, 0.6, vd, vg, vs, -1.0);
                    const MosEval l = mos_level1(sign, 1e-3, 0.7, 0.02, vd, vg, vs);
                    CHECK(c.ids == l.ids);
                    CHECK(c.d_drain == l.d_drain);
                    CHECK(c.d_gate == l.d_gate);
                    CHECK(c.d_source == l.d_source);
                    CHECK(c.d_bulk == 0.0);
                    CHECK(mos_card_region(sign, vto, 0.0, 0.6, vd, vg, vs, -1.0) ==
                          mos_level1_region(sign, vd, vg, vs, 0.7));
                }
            }
        }
    }
}

TEST_CASE("MOS card: body effect shifts the threshold by gamma (sqrt(phi - vbs) - sqrt(phi))") {
    const double gamma = 0.5;
    const double phi = 0.7;
    double vth = 0.0;
    double g = 0.0;
    mos_card_threshold(-2.0, 0.8, gamma, phi, vth, g);
    CHECK(vth == 0.8 + gamma * (std::sqrt(phi + 2.0) - std::sqrt(phi)));
    CHECK(g == gamma / (2.0 * std::sqrt(phi + 2.0)));
    mos_card_threshold(0.3, 0.8, gamma, phi, vth, g);  // forward body bias: held at vbs = 0
    CHECK(vth == 0.8);
    CHECK(g == 0.0);
    mos_card_threshold(0.0, 0.8, gamma, phi, vth, g);
    CHECK(vth == 0.8);
    CHECK(g == gamma / (2.0 * std::sqrt(phi)));
    // Saturation current with the shifted threshold; bulk derivative = gm * g.
    const double vb = -2.0;
    const Mos4Eval r = mos_card(1, 1e-3, 0.8, 0.0, gamma, phi, 3.0, 2.0, 0.0, vb);
    mos_card_threshold(vb, 0.8, gamma, phi, vth, g);
    CHECK(r.ids == 0.5 * 1e-3 * (2.0 - vth) * (2.0 - vth) * (1.0 + 0.0 * 3.0));
    CHECK(close(r.d_bulk, r.d_gate * g, 1e-15));
    CHECK(mos_card_vth(1, 0.8, gamma, phi, 3.0, 0.0, vb) == vth);
    // Reverse mode measures the body voltage from the drain (the lower terminal).
    const Mos4Eval rev = mos_card(1, 1e-3, 0.8, 0.0, gamma, phi, 0.0, 2.0, 3.0, vb);
    CHECK(rev.ids == -r.ids);
    CHECK(rev.d_bulk == -r.d_bulk);
}

TEST_CASE("MOS card: gradient matches central differences; PMOS mirrors NMOS") {
    for (int sign : {1, -1}) {
        const double s = sign;
        for (double vd : {0.05, 0.4, 1.3, 3.0}) {
            for (double vg : {0.0, 1.1, 2.05, 3.5}) {
                for (double vs : {0.0, 0.2}) {
                    for (double vb : {-2.0, -0.6, 0.1}) {
                        if (std::min(std::fabs(vb - vs), std::fabs(vb - vd)) < 1e-3) {
                            continue;
                        }
                        const double v[4] = {s * vd, s * vg, s * vs, s * vb};
                        const double vto = s * 0.7;
                        const Mos4Eval r = mos_card(sign, 2e-4, vto, 0.02, 0.45, 0.65, v[0], v[1], v[2], v[3]);
                        const double grad[4] = {r.d_drain, r.d_gate, r.d_source, r.d_bulk};
                        double scale = 2e-7;
                        for (double value : grad) {
                            scale = std::max(scale, std::fabs(value));
                        }
                        for (int col = 0; col < 4; ++col) {
                            const double h = 1e-7;
                            double up[4] = {v[0], v[1], v[2], v[3]};
                            double down[4] = {v[0], v[1], v[2], v[3]};
                            up[col] += h;
                            down[col] -= h;
                            const double numeric =
                                (mos_card(sign, 2e-4, vto, 0.02, 0.45, 0.65, up[0], up[1], up[2], up[3]).ids -
                                 mos_card(sign, 2e-4, vto, 0.02, 0.45, 0.65, down[0], down[1], down[2], down[3]).ids) /
                                (2.0 * h);
                            CHECK(std::fabs(numeric - grad[col]) <= 2e-6 * std::max(std::fabs(grad[col]), scale));
                        }
                        if (sign < 0) {
                            const Mos4Eval n = mos_card(1, 2e-4, 0.7, 0.02, 0.45, 0.65, vd, vg, vs, vb);
                            CHECK(r.ids == -n.ids);
                            CHECK(r.d_drain == n.d_drain);
                            CHECK(r.d_gate == n.d_gate);
                            CHECK(r.d_source == n.d_source);
                            CHECK(r.d_bulk == n.d_bulk);
                        }
                    }
                }
            }
        }
    }
}

// ---- compiled circuit ----------------------------------------------------------------------

TEST_CASE("compiled circuit: model-card tables, junctions and assembly") {
    using mna::circuit::Circuit;
    using mna::circuit::CircuitSpec;
    // Unknowns: 0 = anode, 1 = internal anode (series RS in G), 2 = collector, 3 = base,
    // 4 = drain, 5 = bulk. A SPICE diode 1 -> ground, an NPN (2, 3, ground), an NMOS
    // (4, gate 3, source ground, bulk 5).
    CircuitSpec spec;
    spec.n = 6;
    spec.node_count = 6;
    spec.g.rows = {0, 1, 0, 1};
    spec.g.cols = {0, 1, 1, 0};
    spec.g.values = {2.0, 2.0, -2.0, -2.0};
    spec.spice_diodes.anode = {1};
    spec.spice_diodes.cathode = {-1};
    spec.spice_diodes.is = {2.52e-9};
    spec.spice_diodes.n = {1.752};
    spec.spice_diodes.bv = {100.0};
    spec.spice_diodes.ibv = {100e-6};
    const GummelPoonParams p = q2n2222();
    auto& gt = spec.gummel_poon;
    gt.collector = {2};
    gt.base = {3};
    gt.emitter = {-1};
    gt.is = {p.is};
    gt.bf = {p.bf};
    gt.nf = {p.nf};
    gt.br = {p.br};
    gt.nr = {p.nr};
    gt.vaf = {p.vaf};
    gt.var = {p.var};
    gt.ikf = {p.ikf};
    gt.ikr = {p.ikr};
    gt.ise = {p.ise};
    gt.ne = {p.ne};
    gt.isc = {p.isc};
    gt.nc = {p.nc};
    gt.sign = {1};
    auto& mt = spec.mos4;
    mt.drain = {4};
    mt.gate = {3};
    mt.source = {-1};
    mt.bulk = {5};
    mt.beta = {1e-3};
    mt.vto = {0.7};
    mt.lambda = {0.01};
    mt.gamma = {0.4};
    mt.phi = {0.65};
    mt.sign = {1};
    const Circuit c(spec);
    CHECK(c.device_counts().spice_diodes == 1);
    CHECK(c.device_counts().gummel_poon == 1);
    CHECK(c.device_counts().mos4 == 1);
    CHECK(c.device_counts().total() == 3);
    CHECK(c.assembly().groups_present == 1);
    const double xbv = spice_diode_breakdown_voltage(2.52e-9, 100.0, 100e-6);
    CHECK(c.spice_diodes().bv_eff[0] == xbv);
    // Junctions: the diode (vt = n vt_card), then the NPN's (b, e) and (b, c).
    const auto& j = c.junctions();
    REQUIRE(j.size() == 3);
    CHECK(j.anode[0] == 1);
    CHECK(j.cathode[0] == -1);
    CHECK(j.vt[0] == 1.752 * kCardVt);
    CHECK(j.anode[1] == 3);
    CHECK(j.cathode[1] == -1);
    CHECK(j.anode[2] == 3);
    CHECK(j.cathode[2] == 2);
    CHECK(j.vt[1] == kCardVt);
    // Bulk column present for the drain row.
    CHECK(c.slot(4, 5) >= 0);

    const double x[6] = {0.8, 0.62, 3.0, 0.66, 1.0, -1.0};
    std::vector<double> f(6);
    std::vector<double> dense(36);
    mna::assembly::evaluate_dense_jacobian(c, x, 0.0, 1, f.data(), dense.data());
    const SpiceDiodeEval d = spice_diode(2.52e-9, 1.752, xbv, 0.62);
    const BjtEval q = gummel_poon(1, p, 3.0, 0.66, 0.0);
    const Mos4Eval m = mos_card(1, 1e-3, 0.7, 0.01, 0.4, 0.65, 1.0, 0.66, 0.0, -1.0);
    CHECK(f[0] == 0.0);
    CHECK(f[1] == d.current);
    CHECK(f[2] == q.i[0]);
    CHECK(f[3] == q.i[1]);
    CHECK(f[4] == m.ids);
    CHECK(f[5] == 0.0);
    CHECK(dense[1 * 6 + 1] == d.conductance);
    CHECK(dense[2 * 6 + 3] == q.d[0][1]);
    CHECK(dense[4 * 6 + 3] == m.d_gate);
    CHECK(dense[4 * 6 + 5] == m.d_bulk);
    CHECK(dense[4 * 6 + 4] == m.d_drain);

    // Same results on several threads.
    std::vector<double> f4(6);
    std::vector<double> dense4(36);
    mna::assembly::evaluate_dense_jacobian(c, x, 0.0, 4, f4.data(), dense4.data());
    CHECK(f4 == f);
    CHECK(dense4 == dense);

    const mna::assembly::OperatingPoints op = mna::assembly::operating_points(c, x);
    CHECK(op.spice_diode_id[0] == d.current);
    CHECK(op.spice_diode_region[0] == static_cast<int>(DiodeRegion::Forward));
    CHECK(op.gp_ic[0] == q.i[0]);
    CHECK(op.gp_region[0] == static_cast<int>(BjtRegion::ForwardActive));
    CHECK(op.mos4_gmbs[0] == m.d_bulk);
    CHECK(op.mos4_region[0] == static_cast<int>(MosRegion::Off));  // vg = 0.66 V < vth(vbs = -1 V) = 0.89 V
    CHECK(op.mos4_vth[0] == doctest::Approx(0.7 + 0.4 * (std::sqrt(1.65) - std::sqrt(0.65))).epsilon(1e-15));
}

TEST_CASE("compiled circuit: invalid model-card parameters are NETLIST_FATAL") {
    using mna::circuit::Circuit;
    using mna::circuit::CircuitSpec;
    CircuitSpec spec;
    spec.n = 1;
    spec.spice_diodes.anode = {0};
    spec.spice_diodes.cathode = {-1};
    spec.spice_diodes.is = {-1.0};
    spec.spice_diodes.n = {1.0};
    spec.spice_diodes.bv = {kInf};
    spec.spice_diodes.ibv = {1e-3};
    CHECK_THROWS_AS(Circuit{spec}, mna::Error);
    spec.spice_diodes.is = {1e-14};
    spec.spice_diodes.ibv = {};
    CHECK_THROWS_AS(Circuit{spec}, mna::Error);
    spec.spice_diodes.ibv = {1e-3};
    CHECK_NOTHROW(Circuit{spec});
    spec.mos4.drain = {0};
    spec.mos4.gate = {0};
    spec.mos4.source = {-1};
    spec.mos4.bulk = {1};  // outside [-1, n)
    spec.mos4.beta = {1e-3};
    spec.mos4.vto = {0.5};
    spec.mos4.lambda = {0.0};
    spec.mos4.gamma = {0.0};
    spec.mos4.phi = {0.6};
    spec.mos4.sign = {1};
    CHECK_THROWS_AS(Circuit{spec}, mna::Error);
    spec.mos4.bulk = {-1};
    spec.mos4.phi = {0.0};
    CHECK_THROWS_AS(Circuit{spec}, mna::Error);
}
