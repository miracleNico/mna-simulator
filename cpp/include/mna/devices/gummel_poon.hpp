// Gummel-Poon BJT, DC part, from a model card (HIERARCHY_AND_MODELS_PLAN.md
// 3.2, phase M2), following SPICE3f5's bjtload.c at the nominal temperature.
//
// Parameters (GummelPoonParams) are the intrinsic-device values already scaled
// by the instance area (plan 3.4): is, ise, isc, ikf, ikr multiplied by the
// area; bf, nf, br, nr, ne, nc, vaf, var as on the card. vaf, var, ikf, ikr
// equal to +inf or to 0 mean "not given" (SPICE: 0 is infinite), so their
// inverses are 0. vt = kCardVt (k T / q at 300.15 K). NPN, with vbe = vb - ve
// and vbc = vb - vc:
//
//   cbe  = is  (exp(vbe / (nf vt)) - 1)     cbc  = is  (exp(vbc / (nr vt)) - 1)
//   cben = ise (exp(vbe / (ne vt)) - 1)     cbcn = isc (exp(vbc / (nc vt)) - 1)
//   q1   = 1 / (1 - vbc / vaf - vbe / var)
//   q2   = cbe / ikf + cbc / ikr
//   qb   = q1 (1 + sqrt(max(0, 1 + 4 q2))) / 2   (qb = q1 when ikf = ikr = inf)
//   Ic   = (cbe - cbc) / qb - cbc / br - cbcn
//   Ib   = cbe / bf + cben + cbc / br + cbcn
//   Ie   = -(Ic + Ib)
//
// with analytic derivatives (the dqb/dvbe, dqb/dvbc of bjtload.c). PNP by the
// sign trick of bjt_level1: I_pnp(v) = -I_npn(-v), derivatives unchanged.
//
// Differences from bjtload.c, all chosen so the currents and the Jacobian
// stay consistent (the derivative is the derivative of the current):
//   * every exponential goes through bjt_exp (linear continuation beyond
//     |x| = 80, as for the Level-1 BJT) instead of SPICE's reverse-bias
//     linearization below -5 nf vt (a constant -is current with the secant
//     conductance -is / vbe);
//   * sqrt(max(0, 1 + 4 q2)) is used as is (SPICE takes 1 when the argument is
//     exactly 0, a jump), and its derivative is 0 where the argument is
//     clamped;
//   * no gmin inside the junction currents: the builder stamps gmin across
//     the base-emitter and base-collector junctions as for the Level-1 BJT.
// Not modelled here (the builder): RB, RE, RC (linear conductances to the
// internal nodes <name>#b, <name>#e, <name>#c), CJE, CJC (constant zero-bias
// capacitors), the substrate node (ignored). Transit times, excess phase,
// RBM/IRB and temperature terms are phase M4.
#pragma once

#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/fp_contract.hpp"
#include "mna/devices/spice_diode.hpp"

#include <cmath>

namespace mna::devices {

struct GummelPoonParams {
    double is = 1e-16;
    double bf = 100.0;
    double nf = 1.0;
    double br = 1.0;
    double nr = 1.0;
    double vaf = 0.0;  ///< 0 or +inf: no forward Early effect
    double var = 0.0;  ///< 0 or +inf: no reverse Early effect
    double ikf = 0.0;  ///< 0 or +inf: no forward high injection
    double ikr = 0.0;  ///< 0 or +inf: no reverse high injection
    double ise = 0.0;
    double ne = 1.5;
    double isc = 0.0;
    double nc = 2.0;
};

/// 1 / x, or 0 when x is 0 (SPICE's "not given") or infinite.
inline double gp_inverse(double x) noexcept {
    MNA_NO_FP_CONTRACT
    return x != 0.0 ? 1.0 / x : 0.0;
}

/// NPN terminal currents (positive into the device) and d I_row / d V_col,
/// rows and columns ordered collector, base, emitter (as BjtEval).
inline BjtEval npn_gummel_poon(double vc, double vb, double ve, const GummelPoonParams& p) noexcept {
    MNA_NO_FP_CONTRACT
    const double vt = kCardVt;
    const double vbe = vb - ve;
    const double vbc = vb - vc;

    const double vtf = p.nf * vt;
    const BjtExp ebe = bjt_exp(vbe / vtf);
    const double cbe = p.is * (ebe.value - 1.0);
    const double gbe = p.is * ebe.slope / vtf;
    double cben = 0.0;
    double gben = 0.0;
    if (p.ise != 0.0) {
        const double vte = p.ne * vt;
        const BjtExp e = bjt_exp(vbe / vte);
        cben = p.ise * (e.value - 1.0);
        gben = p.ise * e.slope / vte;
    }

    const double vtr = p.nr * vt;
    const BjtExp ebc = bjt_exp(vbc / vtr);
    const double cbc = p.is * (ebc.value - 1.0);
    const double gbc = p.is * ebc.slope / vtr;
    double cbcn = 0.0;
    double gbcn = 0.0;
    if (p.isc != 0.0) {
        const double vtc = p.nc * vt;
        const BjtExp e = bjt_exp(vbc / vtc);
        cbcn = p.isc * (e.value - 1.0);
        gbcn = p.isc * e.slope / vtc;
    }

    const double inv_vaf = gp_inverse(p.vaf);
    const double inv_var = gp_inverse(p.var);
    const double oik = gp_inverse(p.ikf);
    const double oikr = gp_inverse(p.ikr);
    const double q1 = 1.0 / (1.0 - inv_vaf * vbc - inv_var * vbe);
    double qb = 0.0;
    double dqbdve = 0.0;
    double dqbdvc = 0.0;
    if (oik == 0.0 && oikr == 0.0) {
        qb = q1;
        dqbdve = q1 * qb * inv_var;
        dqbdvc = q1 * qb * inv_vaf;
    } else {
        const double q2 = oik * cbe + oikr * cbc;
        const double arg = 1.0 + 4.0 * q2;
        const double sqarg = arg > 0.0 ? std::sqrt(arg) : 0.0;
        qb = q1 * (1.0 + sqarg) / 2.0;
        const double dse = arg > 0.0 ? oik * gbe / sqarg : 0.0;
        const double dsc = arg > 0.0 ? oikr * gbc / sqarg : 0.0;
        dqbdve = q1 * (qb * inv_var + dse);
        dqbdvc = q1 * (qb * inv_vaf + dsc);
    }

    const double cex = cbe - cbc;
    const double ic = cex / qb - cbc / p.br - cbcn;
    const double ib = cbe / p.bf + cben + cbc / p.br + cbcn;
    const double ie = -(ic + ib);

    const double dic_dvbe = (gbe - cex * dqbdve / qb) / qb;
    const double dic_dvbc = -(gbc + cex * dqbdvc / qb) / qb - gbc / p.br - gbcn;
    const double dib_dvbe = gbe / p.bf + gben;
    const double dib_dvbc = gbc / p.br + gbcn;

    // vbe = vb - ve, vbc = vb - vc: d/dvc = -d/dvbc, d/dvb = d/dvbe + d/dvbc, d/dve = -d/dvbe.
    const double dic_dvc = -dic_dvbc;
    const double dic_dvb = dic_dvbe + dic_dvbc;
    const double dic_dve = -dic_dvbe;
    const double dib_dvc = -dib_dvbc;
    const double dib_dvb = dib_dvbe + dib_dvbc;
    const double dib_dve = -dib_dvbe;

    return {{ic, ib, ie},
            {{dic_dvc, dic_dvb, dic_dve},
             {dib_dvc, dib_dvb, dib_dve},
             {-(dic_dvc + dib_dvc), -(dic_dvb + dib_dvb), -(dic_dve + dib_dve)}}};
}

/// sign = +1 for NPN, -1 for PNP: I_pnp(v) = -I_npn(-v), derivatives unchanged.
inline BjtEval gummel_poon(int sign, const GummelPoonParams& p, double vc, double vb, double ve) noexcept {
    MNA_NO_FP_CONTRACT
    if (sign < 0) {
        BjtEval r = npn_gummel_poon(-vc, -vb, -ve, p);
        for (double& current : r.i) {
            current = -current;
        }
        return r;
    }
    return npn_gummel_poon(vc, vb, ve, p);
}

}  // namespace mna::devices
