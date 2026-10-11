// Level-1 (Ebers-Moll-lite) BJT, ported from the Python core
// (solvers._bjt_level1_npn_currents_and_derivatives,
// _level1_bjt_currents_and_derivatives): transport currents, finite forward /
// reverse beta, simple Early modulation, vt = 0.02585.
//
// Exponent beyond +-80 (CPP_MIGRATION_PLAN.md 2.4.1, P3b): the junction
// exponential continues linearly past |v / vt| = 80 with the slope it has at
// the clip point, exp(80) * (1 + x - 80) above and exp(-80) * (1 + x + 80)
// below (bjt_exp), so current and Jacobian stay consistent and the Newton
// matrix stays nonsingular. Was (the reference): the argument clipped to
// [-80, 80], flat currents beyond the clip but the derivative kept its clip
// value. A zero slope was rejected: it made the HB Newton Jacobian singular
// when an iterate crossed the clip.
//
// Inside the clip the arithmetic is identical to the Python code; results
// differ from it only where NumPy's exp (SIMD implementation) and the C
// library exp disagree in the last bit.
#pragma once

#include "mna/devices/fp_contract.hpp"

#include <cmath>

namespace mna::devices {

inline constexpr double kBjtVt = 0.02585;
/// |v / vt| beyond which the junction exponential is extrapolated linearly.
inline constexpr double kBjtExpClip = 80.0;

/// exp(x) and its derivative d exp(x) / dx, continued linearly beyond
/// |x| > kBjtExpClip (value and slope continuous at the clip points).
struct BjtExp {
    double value;
    double slope;
};

inline BjtExp bjt_exp(double x) noexcept {
    MNA_NO_FP_CONTRACT
    if (x > kBjtExpClip) {
        const double e = std::exp(kBjtExpClip);
        return {e * (1.0 + (x - kBjtExpClip)), e};
    }
    if (x < -kBjtExpClip) {
        const double e = std::exp(-kBjtExpClip);
        return {e * (1.0 + (x + kBjtExpClip)), e};
    }
    const double e = std::exp(x);
    return {e, e};
}

/// Terminal currents (positive into the device) and d I_row / d V_col with
/// rows and columns ordered collector, base, emitter.
struct BjtEval {
    double i[3];
    double d[3][3];
};

inline BjtEval npn_level1(double vc, double vb, double ve, double is, double bf, double br, double vaf,
                          double var) noexcept {
    MNA_NO_FP_CONTRACT
    const double vt = kBjtVt;
    const double alpha_f = bf > 0.0 ? bf / (bf + 1.0) : 0.0;
    const double alpha_r = br > 0.0 ? br / (br + 1.0) : 0.0;
    const double vbe = vb - ve;
    const double vbc = vb - vc;
    const double vce = vc - ve;
    const double vec = ve - vc;

    const BjtExp exp_be = bjt_exp(vbe / vt);
    const BjtExp exp_bc = bjt_exp(vbc / vt);
    const double ibe = is * (exp_be.value - 1.0);
    const double ibc = is * (exp_bc.value - 1.0);
    const double gbe = is * exp_be.slope / vt;
    const double gbc = is * exp_bc.slope / vt;

    const double fwd_mod = vaf > 0.0 ? 1.0 + vce / vaf : 1.0;
    const double rev_mod = var > 0.0 ? 1.0 + vec / var : 1.0;
    const double df_dvc = vaf > 0.0 ? 1.0 / vaf : 0.0;
    const double df_dve = -df_dvc;
    const double dr_dvc = var > 0.0 ? -1.0 / var : 0.0;
    const double dr_dve = -dr_dvc;

    const double ic = alpha_f * ibe * fwd_mod - ibc * rev_mod;
    const double ib = (1.0 - alpha_f) * ibe + (1.0 - alpha_r) * ibc * rev_mod;
    const double ie = -(ic + ib);

    const double dic_dvc = alpha_f * ibe * df_dvc + gbc * rev_mod - ibc * dr_dvc;
    const double dic_dvb = alpha_f * gbe * fwd_mod - gbc * rev_mod;
    const double dic_dve = alpha_f * (-gbe * fwd_mod + ibe * df_dve) - ibc * dr_dve;

    const double dib_dvc = (1.0 - alpha_r) * (-gbc * rev_mod + ibc * dr_dvc);
    const double dib_dvb = (1.0 - alpha_f) * gbe + (1.0 - alpha_r) * gbc * rev_mod;
    const double dib_dve = -(1.0 - alpha_f) * gbe + (1.0 - alpha_r) * ibc * dr_dve;

    const double die_dvc = -(dic_dvc + dib_dvc);
    const double die_dvb = -(dic_dvb + dib_dvb);
    const double die_dve = -(dic_dve + dib_dve);

    return {{ic, ib, ie},
            {{dic_dvc, dic_dvb, dic_dve}, {dib_dvc, dib_dvb, dib_dve}, {die_dvc, die_dvb, die_dve}}};
}

/// sign = +1 for NPN, -1 for PNP: I_pnp(v) = -I_npn(-v), derivatives unchanged.
inline BjtEval bjt_level1(int sign, double is, double bf, double br, double vaf, double var, double vc, double vb,
                          double ve) noexcept {
    MNA_NO_FP_CONTRACT
    if (sign < 0) {
        BjtEval r = npn_level1(-vc, -vb, -ve, is, bf, br, vaf, var);
        for (double& current : r.i) {
            current = -current;
        }
        return r;
    }
    return npn_level1(vc, vb, ve, is, bf, br, vaf, var);
}

/// Region codes of core_basic._bjt_level1_region.
enum class BjtRegion : int { Cutoff = 0, ForwardActive = 1, ReverseActive = 2, Saturation = 3 };

inline BjtRegion bjt_level1_region(int sign, double vc, double vb, double ve) noexcept {
    MNA_NO_FP_CONTRACT
    double v1 = 0.0;  // vbe (NPN) / veb (PNP)
    double v2 = 0.0;  // vbc (NPN) / vcb (PNP)
    if (sign < 0) {
        v1 = ve - vb;
        v2 = vc - vb;
    } else {
        v1 = vb - ve;
        v2 = vb - vc;
    }
    if (v1 <= 0.0 && v2 <= 0.0) {
        return BjtRegion::Cutoff;
    }
    if (v1 > 0.0 && v2 <= 0.0) {
        return BjtRegion::ForwardActive;
    }
    if (v1 <= 0.0 && v2 > 0.0) {
        return BjtRegion::ReverseActive;
    }
    return BjtRegion::Saturation;
}

}  // namespace mna::devices
