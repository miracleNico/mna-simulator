// SPICE diode from a model card (HIERARCHY_AND_MODELS_PLAN.md 3.2, phase M2):
// the DC current of SPICE3f5's dioload.c with the breakdown voltage of
// diotemp.c, at the nominal temperature (no TNOM/TEMP, EG, XTI terms yet).
//
// Parameters (already scaled by the instance area, plan 3.4): is = area * IS,
// n = N, bv = BV (+inf: no breakdown branch), ibv = area * IBV.
// vte = n * vt with vt = k T / q at T = 300.15 K (kCardVt, plan 3.4).
//
//   v >= -3 vte          I = is * (exp(v / vte) - 1)
//   -bv_eff <= v < -3 vte I = -is * (1 + a^3),   a = 3 vte / (e v)
//   v < -bv_eff          I = -is * exp(-(bv_eff + v) / vt)
//
// The middle branch is SPICE3f5's smooth reverse region (value and slope
// continuous with the forward branch at -3 vte); with bv = +inf it extends to
// -inf. bv_eff is the breakdown voltage SPICE3f5 shifts so that the
// breakdown current at v = -BV matches IBV (spice_diode_breakdown_voltage).
// At v = -bv_eff the breakdown branch equals -is while the reverse branch is
// -is * (1 + a^3): the value jumps by is * (3 vte / (e bv_eff))^3 (1e-10
// relative for BV = 100 V) and the slope from about 0 to is / vt, exactly as in
// SPICE3f5. The breakdown exponent uses vt, not n * vt (SPICE3f5; later SPICE
// versions add a separate NBV).
//
// Exponent overflow: both exponentials go through bjt_exp, i.e. they continue
// linearly beyond |x| = kBjtExpClip = 80 with the slope at the clip point, so
// current and conductance stay finite and consistent (the P3b treatment of the
// Level-1 BJT, CPP_MIGRATION_PLAN.md 2.4.1).
//
// Not modelled here (the builder handles them): RS (a linear conductance to the
// internal node <name>#a), CJO (a constant capacitor), the gmin of SPICE's
// diode load (gmin stamps across the junction, as for the ideal diode).
// SPICE3f5's breakdown-side junction limiting is not applied; the forward
// junction is limited by pnjlim with vte and vcrit = vte ln(vte / (sqrt(2) is)).
#pragma once

#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/fp_contract.hpp"

#include <cmath>
#include <limits>

namespace mna::devices {

/// Boltzmann constant over the electron charge in V/K, CODATA 2018.
inline constexpr double kBoltzmannOverQ = 8.617333262e-5;
/// Nominal temperature of the model-card equations, 27 C.
inline constexpr double kCardTemperature = 300.15;
/// k T / q at 300.15 K (about 0.025865 V) for every model-card kernel.
inline constexpr double kCardVt = kBoltzmannOverQ * kCardTemperature;
/// SPICE's CONSTe (Euler's number) of the reverse-region cubic.
inline constexpr double kEuler = 2.718281828459045235360287;
/// SPICE's default RELTOL, the tolerance of the breakdown-voltage iteration.
inline constexpr double kCardReltol = 1e-3;

struct SpiceDiodeEval {
    double current;      ///< anode -> cathode through the junction
    double conductance;  ///< dI/dv
};

/// Effective breakdown voltage of SPICE3f5's diotemp.c for a junction with
/// saturation current `is`, breakdown voltage `bv` and breakdown current
/// `ibv` (is and ibv may both carry the same area factor: the result does not
/// depend on it). +inf when bv is not finite (no breakdown). When ibv is
/// below is * bv / vt SPICE raises it to that value and keeps bv; otherwise
/// the fixed-point iteration
///     xbv = bv - vt ln(ibv / is + 1 - xbv / vt)
/// starting from bv - vt ln(1 + ibv / is) runs until
/// |is (exp((bv - xbv) / vt) - 1 + xbv / vt) - ibv| <= 1e-3 ibv, at most 25
/// times (SPICE then keeps the last iterate with a warning).
inline double spice_diode_breakdown_voltage(double is, double bv, double ibv) noexcept {
    MNA_NO_FP_CONTRACT
    if (!(bv < std::numeric_limits<double>::infinity())) {
        return std::numeric_limits<double>::infinity();
    }
    const double vt = kCardVt;
    const double cbv = ibv;
    if (cbv < is * bv / vt) {
        return bv;
    }
    const double tol = kCardReltol * cbv;
    double xbv = bv - vt * std::log(1.0 + cbv / is);
    for (int iter = 0; iter < 25; ++iter) {
        xbv = bv - vt * std::log(cbv / is + 1.0 - xbv / vt);
        const double xcbv = is * (std::exp((bv - xbv) / vt) - 1.0 + xbv / vt);
        if (std::fabs(xcbv - cbv) <= tol) {
            break;
        }
    }
    return xbv;
}

/// Junction current and conductance at v = V(anode) - V(cathode); bv_eff from
/// spice_diode_breakdown_voltage (+inf: no breakdown).
inline SpiceDiodeEval spice_diode(double is, double n, double bv_eff, double v) noexcept {
    MNA_NO_FP_CONTRACT
    const double vte = n * kCardVt;
    if (v >= -3.0 * vte) {
        const BjtExp e = bjt_exp(v / vte);
        return {is * (e.value - 1.0), is * e.slope / vte};
    }
    if (v >= -bv_eff) {
        double arg = 3.0 * vte / (v * kEuler);
        arg = arg * arg * arg;
        return {-is * (1.0 + arg), is * 3.0 * arg / v};
    }
    const BjtExp e = bjt_exp(-(bv_eff + v) / kCardVt);
    return {-is * e.value, is * e.slope / kCardVt};
}

/// Region codes reported by operating_points for a SPICE diode.
enum class DiodeRegion : int { Off = 0, Forward = 1, Reverse = 2, Breakdown = 3 };

/// Forward (v > 0), Off (-3 vte <= v <= 0), Reverse (-bv_eff <= v < -3 vte),
/// Breakdown (v < -bv_eff), following the branches of spice_diode.
inline DiodeRegion spice_diode_region(double n, double bv_eff, double v) noexcept {
    MNA_NO_FP_CONTRACT
    const double vte = n * kCardVt;
    if (v > 0.0) {
        return DiodeRegion::Forward;
    }
    if (v >= -3.0 * vte) {
        return DiodeRegion::Off;
    }
    return v >= -bv_eff ? DiodeRegion::Reverse : DiodeRegion::Breakdown;
}

}  // namespace mna::devices
