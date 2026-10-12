// Four-terminal Level-1 MOSFET from a model card (HIERARCHY_AND_MODELS_PLAN.md
// 3.2, phase M2): the Shichman-Hodges equations of mos_level1.hpp with the
// body effect of SPICE3f5's mos1load.c, for cards of LEVEL 1, 2 and 3 (2 and 3
// are mapped to these equations by the builder).
//
// Parameters: beta = KP * W / L * M, vto = VTO as on the card, lambda, gamma,
// phi (> 0). Everything is evaluated in the n-channel frame of the sign trick
// (PMOS: all terminal voltages negated, current negated, derivatives
// unchanged), where the zero-bias threshold is vth0 = type * VTO (SPICE's
// convention: an enhancement PMOS card has VTO < 0, so vth0 = |VTO| for every
// enhancement device; a depletion card keeps its sign).
//
// Body effect, with vbs the bulk-source voltage in that frame:
//   vbs <= 0: vth = vth0 + gamma * (sqrt(phi - vbs) - sqrt(phi)),
//             dvth/dvbs = -gamma / (2 sqrt(phi - vbs))
//   vbs >  0: vth = vth0 (held at its vbs = 0 value; SPICE3f5's forward-bias
//             approximation is phase M4), dvth/dvbs = 0
// The value is continuous at vbs = 0; the slope jumps from
// -gamma / (2 sqrt(phi)) to 0 there. gamma = 0 skips the body term, so the
// device is then bit-identical to mos_level1 with vth = vth0 and a zero bulk
// derivative.
//
// Drain/source symmetry as nmos_level1: when vd < vs the terminals swap and
// the body voltage is measured from the drain (vbd). The bulk carries no
// current (no bulk junction diodes before phase M4), so only the drain and
// source rows have entries: d Ids / d(vd, vg, vs, vb).
#pragma once

#include "mna/devices/fp_contract.hpp"
#include "mna/devices/mos_level1.hpp"

#include <cmath>

namespace mna::devices {

struct Mos4Eval {
    double ids;
    double d_drain;
    double d_gate;
    double d_source;
    double d_bulk;
};

/// Threshold in the n-channel frame and g = -dvth/dvbs (>= 0).
inline void mos_card_threshold(double vbs, double vth0, double gamma, double phi, double& vth, double& g) noexcept {
    MNA_NO_FP_CONTRACT
    if (gamma != 0.0 && vbs <= 0.0) {
        const double s = std::sqrt(phi - vbs);
        vth = vth0 + gamma * (s - std::sqrt(phi));
        g = gamma / (2.0 * s);
        return;
    }
    vth = vth0;
    g = 0.0;
}

/// NMOS with vds >= 0: Ids and its derivatives with respect to vgs, vds, vbs.
inline void nmos_card_forward(double vgs, double vds, double vbs, double beta, double vth0, double lambda,
                              double gamma, double phi, double& current, double& d_vgs, double& d_vds,
                              double& d_vbs) noexcept {
    MNA_NO_FP_CONTRACT
    double vth = 0.0;
    double g = 0.0;
    mos_card_threshold(vbs, vth0, gamma, phi, vth, g);
    nmos_level1_forward(vgs, vds, beta, vth, lambda, current, d_vgs, d_vds);
    // Ids depends on vth only through vgs - vth in both regions: dIds/dvth = -dIds/dvgs.
    d_vbs = d_vgs * g;
}

/// Symmetric four-terminal NMOS: source and drain swap when vd < vs.
inline Mos4Eval nmos_card(double vd, double vg, double vs, double vb, double beta, double vth0, double lambda,
                          double gamma, double phi) noexcept {
    MNA_NO_FP_CONTRACT
    double current = 0.0;
    double d_vgs = 0.0;
    double d_vds = 0.0;
    double d_vbs = 0.0;
    if (vd >= vs) {
        nmos_card_forward(vg - vs, vd - vs, vb - vs, beta, vth0, lambda, gamma, phi, current, d_vgs, d_vds, d_vbs);
        return {current, d_vds, d_vgs, -d_vgs - d_vds - d_vbs, d_vbs};
    }
    nmos_card_forward(vg - vd, vs - vd, vb - vd, beta, vth0, lambda, gamma, phi, current, d_vgs, d_vds, d_vbs);
    return {-current, d_vgs + d_vds + d_vbs, -d_vgs, -d_vds, -d_vbs};
}

/// sign = +1 for NMOS, -1 for PMOS; vto is the card's VTO (vth0 = sign * vto).
inline Mos4Eval mos_card(int sign, double beta, double vto, double lambda, double gamma, double phi, double vd,
                         double vg, double vs, double vb) noexcept {
    MNA_NO_FP_CONTRACT
    if (sign < 0) {
        Mos4Eval r = nmos_card(-vd, -vg, -vs, -vb, beta, -vto, lambda, gamma, phi);
        r.ids = -r.ids;
        return r;
    }
    return nmos_card(vd, vg, vs, vb, beta, vto, lambda, gamma, phi);
}

/// Body-adjusted threshold in the n-channel frame at the operating point,
/// measured from the terminal that acts as the source (the lower one).
inline double mos_card_vth(int sign, double vto, double gamma, double phi, double vd, double vs, double vb) noexcept {
    MNA_NO_FP_CONTRACT
    const double s = sign < 0 ? -1.0 : 1.0;
    const double vd_f = s * vd;
    const double vs_f = s * vs;
    const double vb_f = s * vb;
    const double source = vd_f >= vs_f ? vs_f : vd_f;
    double vth = 0.0;
    double g = 0.0;
    mos_card_threshold(vb_f - source, s * vto, gamma, phi, vth, g);
    return vth;
}

/// Region codes of mos_level1_region with the body-adjusted threshold.
inline MosRegion mos_card_region(int sign, double vto, double gamma, double phi, double vd, double vg, double vs,
                                 double vb) noexcept {
    MNA_NO_FP_CONTRACT
    return mos_level1_region(sign, vd, vg, vs, mos_card_vth(sign, vto, gamma, phi, vd, vs, vb));
}

}  // namespace mna::devices
