// Shockley diode exactly as device_models.diode_current_expression builds it:
// I = Is * (exp(v / 0.025) - 1) with no exponent clipping (exp overflows to
// inf, as in the Python core), dI/dv = Is / 0.025 * exp(v / 0.025).
#pragma once

#include "mna/devices/fp_contract.hpp"

#include <cmath>

namespace mna::devices {

inline constexpr double kDiodeVt = 0.025;

struct DiodeEval {
    double current;
    double conductance;
};

inline DiodeEval diode(double is, double v) noexcept {
    MNA_NO_FP_CONTRACT
    const double e = std::exp(v / kDiodeVt);
    return {is * (e - 1.0), is / kDiodeVt * e};
}

}  // namespace mna::devices
