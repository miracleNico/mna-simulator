#include "mna/analyses/op.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "mna/error.hpp"

namespace mna::analyses {

bool all_finite(const std::vector<double>& values) noexcept {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

OpResult op(const circuit::Circuit& circuit, const double* b, const double* x0, double gmin,
            const NewtonOptions& options) {
    const auto n = static_cast<std::size_t>(circuit.n());
    const std::vector<double> g_eff = circuit.g_effective(gmin);
    OpResult result;

    if (!circuit.has_nonlinear()) {
        SystemSolver system(circuit, options.krylov, options.num_threads, &result.krylov_stats);
        result.x.assign(n, 0.0);
        result.iterations = 1;
        if (!system.solve(g_eff.data(), b, result.x.data(), nullptr, false, FactorMode::Factor)) {
            throw Error(ErrorCode::NrConvergence, "Linear system is singular (G matrix): Singular matrix");
        }
        result.factorizations = system.factorizations();
    } else {
        NewtonDriver newton(circuit, options, &result.krylov_stats);
        if (x0 != nullptr) {
            result.x.assign(x0, x0 + n);
        } else {
            result.x.assign(n, 0.0);
        }
        const NewtonOutcome outcome = newton.solve(g_eff.data(), b, 0.0, result.x);
        result.iterations = outcome.iterations;
        result.factorizations = newton.factorizations();
        switch (outcome.status) {
            case NewtonStatus::Converged:
                break;
            case NewtonStatus::Singular:
                throw Error(ErrorCode::JacobianSingular,
                            "Jacobian is singular at iteration " + std::to_string(outcome.iteration) + ".");
            case NewtonStatus::NoReduction:
                throw Error(ErrorCode::NrConvergence, "Newton step could not reduce residual at iteration " +
                                                          std::to_string(outcome.iteration) + ".");
            case NewtonStatus::MaxIterations:
                throw Error(ErrorCode::NrConvergence,
                            "Solver did not converge after " + std::to_string(options.max_iter) + " iterations.");
        }
    }
    if (!all_finite(result.x)) {
        throw Error(ErrorCode::NrConvergence, "non-finite solution");
    }
    return result;
}

}  // namespace mna::analyses
