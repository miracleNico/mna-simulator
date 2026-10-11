#include "mna/analyses/op.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "mna/error.hpp"

namespace mna::analyses {

bool all_finite(const std::vector<double>& values) noexcept {
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

namespace {

[[noreturn]] void throw_newton_failure(const NewtonOutcome& outcome, const NewtonOptions& options) {
    switch (outcome.status) {
        case NewtonStatus::Singular:
            throw Error(ErrorCode::JacobianSingular,
                        "Jacobian is singular at iteration " + std::to_string(outcome.iteration) + ".");
        case NewtonStatus::NoReduction:
            throw Error(ErrorCode::NrConvergence, "Newton step could not reduce residual at iteration " +
                                                      std::to_string(outcome.iteration) + ".");
        case NewtonStatus::MaxIterations:
        case NewtonStatus::Converged:
            break;
    }
    throw Error(ErrorCode::NrConvergence,
                "Solver did not converge after " + std::to_string(options.max_iter) + " iterations.");
}

}  // namespace

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
        std::vector<double> start;
        if (x0 != nullptr) {
            start.assign(x0, x0 + n);
        } else {
            start.assign(n, 0.0);
        }
        result.x = start;
        const NewtonOutcome outcome = newton.solve(g_eff.data(), b, 0.0, result.x);
        result.iterations = outcome.iterations;
        result.junction_switches = outcome.junction_switch >= 0 ? 1 : 0;
        result.limited_steps = outcome.limited_steps;
        result.monotonicity_accepts = outcome.monotonicity_accepts;
        bool converged = outcome.status == NewtonStatus::Converged;

        // gmin continuation (P3b): SPICE-style gmin stepping inside one .op. Every
        // level solves G + g * gmin_stamps (the builder's gmin pattern plus node-to-
        // ground gmin) from the previous level's solution, g falling geometrically
        // from kGminContinuationStart; the last solve is at the requested gmin. A
        // failed level retries closer to the last good one (the factor's square
        // root). One iteration budget (max_iter) covers the first attempt and the
        // continuation, and only a converged final solve replaces the first
        // attempt's failure, so whatever converged before is untouched. Without any
        // gmin stamp every level would be the same system: no continuation.
        const std::vector<double>& gmin_stamps = circuit.values(circuit::MatrixKind::Gmin);
        const bool regularizable =
            std::any_of(gmin_stamps.begin(), gmin_stamps.end(), [](double v) { return v != 0.0; });
        if (!converged && options.gmin_continuation && regularizable && gmin < kGminContinuationStart) {
            std::int64_t budget = options.max_iter - outcome.iterations;
            std::vector<double> good = start;
            bool have_good = false;
            double g_good = 0.0;
            double factor = kGminContinuationFactor;
            int retries = 0;
            double g = kGminContinuationStart;
            while (budget > 0) {
                const bool final_level = g <= gmin || g < kGminContinuationFloor;
                const double level = final_level ? gmin : g;
                const std::vector<double> a_level = final_level ? g_eff : circuit.g_effective(level);
                std::vector<double> x = good;
                const NewtonOutcome step = newton.solve(a_level.data(), b, 0.0, x, budget);
                budget -= step.iterations;
                result.iterations += step.iterations;
                result.junction_switches += step.junction_switch >= 0 ? 1 : 0;
                result.limited_steps += step.limited_steps;
                result.monotonicity_accepts += step.monotonicity_accepts;
                ++result.continuation_levels;
                // The final level must also leave a nonsingular Newton matrix: a structurally
                // singular circuit (a floating island) can otherwise "converge" without a solve
                // at the gmin limit of the previous level.
                if (step.status == NewtonStatus::Converged && all_finite(x) &&
                    !(final_level && newton.jacobian_singular(a_level.data(), 0.0, x))) {
                    good = std::move(x);
                    if (final_level) {
                        converged = true;
                        result.x = good;
                        result.continuation_used = true;
                        break;
                    }
                    have_good = true;
                    g_good = level;
                    g = level / factor;
                    continue;
                }
                if (!have_good || final_level || ++retries > kGminContinuationRetries) {
                    break;
                }
                factor = std::sqrt(factor);
                g = g_good / factor;
            }
        }
        result.factorizations = newton.factorizations();
        if (!converged) {
            throw_newton_failure(outcome, options);
        }
    }
    if (!all_finite(result.x)) {
        throw Error(ErrorCode::NrConvergence, "non-finite solution");
    }
    return result;
}

}  // namespace mna::analyses
