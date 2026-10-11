#include "mna/analyses/transient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "mna/analyses/op.hpp"
#include "mna/devices/fp_contract.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::analyses {

using circuit::Circuit;

namespace {

[[noreturn]] void bad_input(const std::string& message) {
    throw Error(ErrorCode::NetlistFatal, "transient: " + message);
}

void validate(const Circuit& circuit, const std::vector<double>& t_grid, const std::vector<TimeSource>& sources,
              const TranOptions& options) {
    if (t_grid.empty()) {
        bad_input("the time grid is empty");
    }
    for (std::size_t k = 1; k < t_grid.size(); ++k) {
        if (!(t_grid[k] > t_grid[k - 1])) {
            bad_input("the time grid must be strictly increasing (point " + std::to_string(k) + ")");
        }
    }
    for (const TimeSource& source : sources) {
        if (source.row < 0 || source.row >= circuit.n()) {
            bad_input("time source row " + std::to_string(source.row) + " outside [0, " +
                      std::to_string(circuit.n()) + ")");
        }
        if (!source.program) {
            bad_input("null time source program");
        }
        if (!source.program->unknowns().empty()) {
            bad_input("a time source program reads circuit unknowns");
        }
    }
    if (options.progress_every < 0) {
        bad_input("progress_every must be >= 0");
    }
}

}  // namespace

std::string format_time(double t) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2e", t);
    return buffer;
}

std::vector<double> source_vector(std::int32_t n, const double* b_dc, const std::vector<TimeSource>& sources,
                                  double t) {
    MNA_NO_FP_CONTRACT
    std::vector<double> b(b_dc, b_dc + n);
    for (const TimeSource& source : sources) {
        b[static_cast<std::size_t>(source.row)] += expr::evaluate(*source.program, nullptr, t);
    }
    return b;
}

TranResult transient(const Circuit& circuit, const double* x0, const std::vector<double>& t_grid, const double* b_dc,
                     const std::vector<TimeSource>& sources, const TranOptions& options, const ProgressFn& progress) {
    MNA_NO_FP_CONTRACT
    validate(circuit, t_grid, sources, options);
    const auto n = static_cast<std::size_t>(circuit.n());
    const auto nnz = static_cast<std::size_t>(circuit.nnz());
    const std::size_t points = t_grid.size();
    const NewtonOptions& newton_options = options.newton;
    const bool linear = !circuit.has_nonlinear();
    const bool krylov = newton_options.krylov.enabled;
    // Reference: sparse_linear_krylov uses G_sparse + C_sparse / h, where
    // SciPy divides a sparse matrix by multiplying with 1 / h.
    const bool reciprocal_step = linear && krylov && options.krylov_csr_operator;

    TranResult result;
    result.values.assign(n * points, 0.0);
    auto store = [&result, n, points](std::size_t k, const std::vector<double>& x) {
        for (std::size_t i = 0; i < n; ++i) {
            result.values[i * points + k] = x[i];
        }
    };

    std::vector<double> x_prev(x0, x0 + n);
    if (!all_finite(x_prev)) {
        throw Error(ErrorCode::NrConvergence, "non-finite solution at t=" + format_time(t_grid[0]) + " s");
    }
    store(0, x_prev);

    const std::vector<double> g_eff = circuit.g_effective(options.gmin);
    const std::vector<double>& c_vals = circuit.values(circuit::MatrixKind::C);
    std::vector<double> a_step(nnz);  // G_eff + C / h
    std::vector<double> c_step(nnz);  // C / h
    bool have_step = false;
    double cached_h = 0.0;

    std::unique_ptr<SystemSolver> linear_system;
    std::unique_ptr<NewtonDriver> newton;
    if (linear) {
        linear_system = std::make_unique<SystemSolver>(circuit, newton_options.krylov, newton_options.num_threads,
                                                       &result.krylov_stats);
    } else {
        newton = std::make_unique<NewtonDriver>(circuit, newton_options, &result.krylov_stats);
    }

    std::vector<double> rhs(n);
    std::vector<double> x(n);
    for (std::size_t k = 1; k < points; ++k) {
        const double t = t_grid[k];
        const double h = t_grid[k] - t_grid[k - 1];
        const std::vector<double> b_now = source_vector(circuit.n(), b_dc, sources, t);

        // The step matrix (and a linear run's factorization) is kept while h is bitwise
        // the previous step's h: the reference forms G + C / h with every step's exact h.
        if (!have_step || h != cached_h) {
            const double inverse = 1.0 / h;
            for (std::size_t s = 0; s < nnz; ++s) {
                c_step[s] = reciprocal_step ? c_vals[s] * inverse : c_vals[s] / h;
                a_step[s] = g_eff[s] + c_step[s];
            }
            cached_h = h;
            have_step = true;
            if (linear_system) {
                linear_system->invalidate();
            }
        }
        // b_be = (C / h) x_{k-1} + b(t_k)
        pattern_matvec(circuit, c_step.data(), x_prev.data(), rhs.data());
        for (std::size_t i = 0; i < n; ++i) {
            rhs[i] = rhs[i] + b_now[i];
        }

        if (linear) {
            const bool ok = linear_system->solve(a_step.data(), rhs.data(), x.data(), krylov ? x_prev.data() : nullptr,
                                                 options.krylov_csr_operator, FactorMode::Cached);
            if (!ok) {
                throw Error(ErrorCode::NrConvergence, "Linear system is singular at t=" + format_time(t) +
                                                          " s (G + C/h matrix): Singular matrix");
            }
        } else {
            x = x_prev;
            const NewtonOutcome outcome = newton->solve(a_step.data(), rhs.data(), t, x);
            result.newton_iterations += outcome.iterations;
            result.junction_switches += outcome.junction_switch >= 0 ? 1 : 0;
            result.limited_steps += outcome.limited_steps;
            result.monotonicity_accepts += outcome.monotonicity_accepts;
            switch (outcome.status) {
                case NewtonStatus::Converged:
                    break;
                case NewtonStatus::Singular:
                    throw Error(ErrorCode::JacobianSingular, "Jacobian singular at t=" + format_time(t) +
                                                                 ", iter=" + std::to_string(outcome.iteration));
                case NewtonStatus::NoReduction:
                    throw Error(ErrorCode::NrConvergence, "Newton step could not reduce residual at t=" +
                                                              format_time(t) +
                                                              ", iter=" + std::to_string(outcome.iteration));
                case NewtonStatus::MaxIterations:
                    throw Error(ErrorCode::NrConvergence, "NR failed to converge at t=" + format_time(t) + " s");
            }
        }
        if (!all_finite(x)) {
            throw Error(ErrorCode::NrConvergence, "non-finite solution at t=" + format_time(t) + " s");
        }
        store(k, x);
        std::swap(x_prev, x);

        if (progress && options.progress_every > 0 &&
            (k % static_cast<std::size_t>(options.progress_every) == 0 || k + 1 == points)) {
            progress(static_cast<std::int64_t>(k), t, x_prev.data());
            ++result.progress_calls;
        }
    }
    result.factorizations = linear_system ? linear_system->factorizations() : newton->factorizations();
    return result;
}

}  // namespace mna::analyses
