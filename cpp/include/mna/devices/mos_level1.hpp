// Level-1 (Shichman-Hodges) MOSFET, ported verbatim from the Python core
// (solvers._nmos_level1_forward, _nmos_level1_current_and_derivatives,
// _level1_mos_current_and_derivatives). Every expression keeps the Python
// operation order, so results are bit-identical to the reference.
#pragma once

#include "mna/devices/fp_contract.hpp"

namespace mna::devices {

/// Drain-to-source current and its derivatives with respect to the drain,
/// gate and source voltages.
struct MosEval {
    double ids;
    double d_drain;
    double d_gate;
    double d_source;
};

/// NMOS with VDS >= 0: IDS, dIDS/dVGS, dIDS/dVDS. Off when vgs <= vth, triode
/// when vds < vgs - vth, saturation otherwise (the boundary is saturation).
inline void nmos_level1_forward(double vgs, double vds, double beta, double vth, double lambda, double& current,
                                double& d_vgs, double& d_vds) noexcept {
    MNA_NO_FP_CONTRACT
    if (vgs <= vth) {
        current = 0.0;
        d_vgs = 0.0;
        d_vds = 0.0;
        return;
    }
    const double vov = vgs - vth;
    if (vds < vov) {
        const double channel = vov * vds - 0.5 * vds * vds;
        const double modulation = 1.0 + lambda * vds;
        current = beta * channel * modulation;
        d_vgs = beta * vds * modulation;
        d_vds = beta * ((vov - vds) * modulation + channel * lambda);
        return;
    }
    current = 0.5 * beta * vov * vov * (1.0 + lambda * vds);
    d_vgs = beta * vov * (1.0 + lambda * vds);
    d_vds = 0.5 * beta * vov * vov * lambda;
}

/// Symmetric NMOS: source and drain swap when vd < vs (vd == vs is forward).
inline MosEval nmos_level1(double vd, double vg, double vs, double beta, double vth, double lambda) noexcept {
    MNA_NO_FP_CONTRACT
    double current = 0.0;
    double d_vgs = 0.0;
    double d_vds = 0.0;
    if (vd >= vs) {
        nmos_level1_forward(vg - vs, vd - vs, beta, vth, lambda, current, d_vgs, d_vds);
        return {current, d_vds, d_vgs, -d_vgs - d_vds};
    }
    nmos_level1_forward(vg - vd, vs - vd, beta, vth, lambda, current, d_vgs, d_vds);
    return {-current, d_vgs + d_vds, -d_vgs, -d_vds};
}

/// sign = +1 for NMOS, -1 for PMOS: I_pmos(vd, vg, vs) = -I_nmos(-vd, -vg, -vs),
/// derivatives unchanged.
inline MosEval mos_level1(int sign, double beta, double vth, double lambda, double vd, double vg, double vs) noexcept {
    MNA_NO_FP_CONTRACT
    if (sign < 0) {
        MosEval r = nmos_level1(-vd, -vg, -vs, beta, vth, lambda);
        r.ids = -r.ids;
        return r;
    }
    return nmos_level1(vd, vg, vs, beta, vth, lambda);
}

/// Region codes of core_basic._mos_level1_region.
enum class MosRegion : int { Off = 0, Triode = 1, Saturation = 2, Reverse = 3 };

inline MosRegion mos_level1_region(int sign, double vd, double vg, double vs, double vth) noexcept {
    MNA_NO_FP_CONTRACT
    double vgs_eff = 0.0;
    double vds_eff = 0.0;
    if (sign < 0) {
        vgs_eff = vs - vg;
        vds_eff = vs - vd;
    } else {
        vgs_eff = vg - vs;
        vds_eff = vd - vs;
    }
    if (vds_eff < 0) {
        return MosRegion::Reverse;
    }
    if (vgs_eff <= vth) {
        return MosRegion::Off;
    }
    return vds_eff < (vgs_eff - vth) ? MosRegion::Triode : MosRegion::Saturation;
}

}  // namespace mna::devices
