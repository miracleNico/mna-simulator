#include "mna/analyses/limiting.hpp"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::analyses {

namespace {

inline double node_voltage(const double* x, std::int32_t node) noexcept { return node < 0 ? 0.0 : x[node]; }

}  // namespace

std::size_t limit_junctions(const circuit::Circuit& circuit, const double* x_old, double* x_new) noexcept {
    MNA_NO_FP_CONTRACT
    const circuit::Junctions& junctions = circuit.junctions();
    std::size_t limited = 0;
    for (std::size_t j = 0; j < junctions.size(); ++j) {
        const std::int32_t a = junctions.anode[j];
        const std::int32_t k = junctions.cathode[j];
        const double vold = node_voltage(x_old, a) - node_voltage(x_old, k);
        const double vnew = node_voltage(x_new, a) - node_voltage(x_new, k);
        const double v = pnjlim(vnew, vold, junctions.vt[j], junctions.vcrit[j]);
        if (v == vnew) {
            continue;
        }
        ++limited;
        if (k < 0) {
            x_new[a] = v;
        } else if (a < 0) {
            x_new[k] = -v;
        } else if (std::fabs(x_new[a] - x_old[a]) >= std::fabs(x_new[k] - x_old[k])) {
            x_new[a] = x_new[k] + v;
        } else {
            x_new[k] = x_new[a] - v;
        }
    }
    return limited;
}

}  // namespace mna::analyses
