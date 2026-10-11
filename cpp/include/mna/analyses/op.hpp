// DC operating point (.op) on the compiled circuit.
//
// G_eff = G + gmin * gmin_stamps (CPP_MIGRATION_PLAN.md 2.4: one compiled
// circuit serves every gmin step). Without nonlinear devices the operating
// point is one linear solve of G_eff x = b (Krylov first when enabled, as the
// solve_linear facade does); otherwise the damped Newton driver runs from x0
// (zeros when x0 is null).
//
// gmin continuation (CPP_MIGRATION_PLAN.md 2.4.1, P3b; NewtonOptions::
// gmin_continuation): when that Newton solve fails, the core retries inside
// the same call with SPICE-style gmin stepping: levels g = 1e-2, 1e-3, ...
// solve G + g * gmin_stamps (device-terminal and node-to-ground gmin) each
// from the previous level's solution, down to the requested gmin, the final
// level. A failed level is retried closer to the last converged one (factor
// 10 -> sqrt(10) -> ..., at most kGminContinuationRetries times). The first
// attempt and the continuation share the max_iter budget of Jacobian
// solves; when the final level does not converge, the first attempt's error
// is raised unchanged. Circuits whose first attempt converges never reach
// the continuation, so their results are untouched.
//
// Errors (mna::Error), with the reference texts:
//   NR_CONVERGENCE     "Linear system is singular (G matrix): Singular matrix"
//                      "Newton step could not reduce residual at iteration {i}."
//                      "Solver did not converge after {max_iter} iterations."
//                      "non-finite solution" (2.4.1: was returned with status ok)
//   JACOBIAN_SINGULAR  "Jacobian is singular at iteration {i}."
#pragma once

#include <cstdint>
#include <vector>

#include "mna/analyses/newton.hpp"
#include "mna/circuit/circuit.hpp"

namespace mna::analyses {

/// gmin continuation levels: first level, geometric factor, smallest level
/// before the final solve at the requested gmin, retries after failed levels.
inline constexpr double kGminContinuationStart = 1e-2;
inline constexpr double kGminContinuationFactor = 10.0;
inline constexpr double kGminContinuationFloor = 1e-12;
inline constexpr int kGminContinuationRetries = 8;

struct OpResult {
    std::vector<double> x;
    std::int64_t iterations = 0;      ///< linear solves (1 for a linear circuit), continuation included
    std::int64_t factorizations = 0;  ///< numeric factorizations of the direct solver
    std::vector<KrylovStat> krylov_stats;
    std::int64_t junction_switches = 0;     ///< Newton solves that switched to the junction strategy
    std::int64_t limited_steps = 0;         ///< Newton iterations changed by the junction limiter
    std::int64_t monotonicity_accepts = 0;  ///< damping trials accepted by the monotonicity test
    std::int64_t continuation_levels = 0;   ///< Newton solves run by the gmin continuation
    bool continuation_used = false;         ///< the solution comes from the gmin continuation
};

/// b: n right-hand side values; x0: n starting values or nullptr (zeros),
/// ignored for a linear circuit.
OpResult op(const circuit::Circuit& circuit, const double* b, const double* x0, double gmin,
            const NewtonOptions& options);

/// True when every value is finite.
bool all_finite(const std::vector<double>& values) noexcept;

}  // namespace mna::analyses
