// DC operating point (.op) on the compiled circuit.
//
// G_eff = G + gmin * gmin_stamps (CPP_MIGRATION_PLAN.md 2.4: one compiled
// circuit serves every gmin step). Without nonlinear devices the operating
// point is one linear solve of G_eff x = b (Krylov first when enabled, as the
// solve_linear facade does); otherwise the damped Newton driver runs from x0
// (zeros when x0 is null).
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

struct OpResult {
    std::vector<double> x;
    std::int64_t iterations = 0;      ///< linear solves (1 for a linear circuit)
    std::int64_t factorizations = 0;  ///< numeric factorizations of the direct solver
    std::vector<KrylovStat> krylov_stats;
};

/// b: n right-hand side values; x0: n starting values or nullptr (zeros),
/// ignored for a linear circuit.
OpResult op(const circuit::Circuit& circuit, const double* b, const double* x0, double gmin,
            const NewtonOptions& options);

/// True when every value is finite.
bool all_finite(const std::vector<double>& values) noexcept;

}  // namespace mna::analyses
