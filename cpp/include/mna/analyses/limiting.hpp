// Junction voltage-step limiting for the Newton drivers (CPP_MIGRATION_PLAN.md
// 2.4.1, P3b: "improve, not parity").
//
// pnjlim is SPICE3's DEVpnjlim: a forward-biased junction whose new voltage
// lies above the critical voltage vcrit = vt ln(vt / (sqrt(2) is)) and moves
// by more than 2 vt is moved logarithmically instead of linearly, so the
// exponential current at the limited voltage equals the current the
// linearization at the old voltage predicted:
//
//     vold > 0:  vnew <- vold + vt ln(1 + (vnew - vold) / vt)   (vcrit when the argument is <= 0)
//     vold <= 0: vnew <- vt ln(vnew / vt)
//
// limit_junctions applies it to every junction of a compiled circuit
// (Circuit::junctions()) on a Newton update: the junction voltage of the new
// iterate is limited against the previous iterate and written back by moving
// one node, the non-ground terminal of a ground-referenced junction, or, for
// a floating junction, the terminal whose update is larger in magnitude (the
// anode on a tie). Junctions are processed in table order, each against the
// iterate as modified by the junctions before it.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "mna/circuit/circuit.hpp"
#include "mna/devices/fp_contract.hpp"

namespace mna::analyses {

/// SPICE3 DEVpnjlim. Returns vnew unchanged unless vnew > vcrit and
/// |vnew - vold| > 2 vt (NaN inputs are returned unchanged).
inline double pnjlim(double vnew, double vold, double vt, double vcrit) noexcept {
    MNA_NO_FP_CONTRACT
    if (vnew > vcrit && std::fabs(vnew - vold) > vt + vt) {
        if (vold > 0.0) {
            const double arg = 1.0 + (vnew - vold) / vt;
            return arg > 0.0 ? vold + vt * std::log(arg) : vcrit;
        }
        return vt * std::log(vnew / vt);
    }
    return vnew;
}

/// Limits every junction of `circuit` in the new iterate `x_new` against
/// `x_old` (n values each; ground is 0 V) as described above. Returns the
/// number of junctions whose voltage was changed.
std::size_t limit_junctions(const circuit::Circuit& circuit, const double* x_old, double* x_new) noexcept;

}  // namespace mna::analyses
