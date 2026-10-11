#include "mna/analyses/newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "mna/analyses/limiting.hpp"
#include "mna/assembly.hpp"
#include "mna/devices/fp_contract.hpp"
#include "mna/error.hpp"
#include "mna/threads.hpp"

// No fused multiply-add: every product and sum is rounded like NumPy's.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::analyses {

using circuit::Circuit;

int device_eval_threads(const Circuit& circuit, int requested) noexcept {
    if (circuit.device_counts().total() < kParallelDeviceMinCount) {
        return 1;
    }
    return effective_threads(requested);
}

double finite_norm(const double* v, std::size_t n) noexcept {
    MNA_NO_FP_CONTRACT
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) {
            return std::numeric_limits<double>::infinity();
        }
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

void pattern_matvec(const Circuit& circuit, const double* values, const double* x, double* y) noexcept {
    MNA_NO_FP_CONTRACT
    const auto n = static_cast<std::size_t>(circuit.n());
    const std::int32_t* col_ptr = circuit.col_ptr().data();
    const std::int32_t* row_idx = circuit.row_idx().data();
    std::fill(y, y + n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        const double xc = x[col];
        for (std::int32_t s = col_ptr[col]; s < col_ptr[col + 1]; ++s) {
            y[row_idx[s]] += values[s] * xc;
        }
    }
}

// ---------------------------------------------------------------------------
// SystemSolver
// ---------------------------------------------------------------------------

SystemSolver::SystemSolver(const Circuit& circuit, const KrylovOptions& krylov, int num_threads,
                           std::vector<KrylovStat>* stats)
    : circuit_(circuit), krylov_(krylov), num_threads_(effective_threads(num_threads)), stats_(stats) {
    pattern_.n = circuit.n();
    pattern_.col_ptr = circuit.col_ptr();
    pattern_.row_idx = circuit.row_idx();
}

SystemSolver::~SystemSolver() = default;

void SystemSolver::invalidate() noexcept {
    factor_valid_ = false;
    numeric_valid_ = false;
    krylov_valid_ = false;
}

bool SystemSolver::factor_check(const double* values) {
    const std::vector<double> zero(static_cast<std::size_t>(pattern_.n), 0.0);
    std::vector<double> x(zero.size());
    return direct(values, zero.data(), x.data(), FactorMode::Refactor);
}

bool SystemSolver::resolve(const double* rhs, double* x) {
    if (!numeric_valid_ || !direct_) {
        return false;
    }
    const auto n = static_cast<std::size_t>(pattern_.n);
    std::copy(rhs, rhs + n, x);
    direct_->solve(x, n, 1);
    return true;
}

void SystemSolver::build_krylov_operator(const double* values) {
    const std::int32_t n = pattern_.n;
    krylov_pattern_.n = n;
    krylov_pattern_.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    krylov_pattern_.row_idx.clear();
    krylov_values_.clear();
    for (std::int32_t col = 0; col < n; ++col) {
        for (std::int32_t s = pattern_.col_ptr[static_cast<std::size_t>(col)];
             s < pattern_.col_ptr[static_cast<std::size_t>(col) + 1]; ++s) {
            const double v = values[s];
            if (v != 0.0) {  // NaN is kept, as by numpy.count_nonzero and SciPy's conversions
                krylov_pattern_.row_idx.push_back(pattern_.row_idx[static_cast<std::size_t>(s)]);
                krylov_values_.push_back(v);
            }
        }
        krylov_pattern_.col_ptr[static_cast<std::size_t>(col) + 1] =
            static_cast<std::int32_t>(krylov_values_.size());
    }
    krylov_matrix_ =
        linear::CsrMatrix<double>::from_csc(krylov_pattern_, krylov_values_.data(), krylov_values_.size());
    krylov_kind_ = linear::classify_matrix(krylov_matrix_, krylov_pattern_, krylov_values_.data());
}

bool SystemSolver::solve(const double* values, const double* rhs, double* x, const double* x0, bool csr_operator,
                         FactorMode mode) {
    const auto n = static_cast<std::size_t>(pattern_.n);
    if (krylov_.enabled) {
        if (mode != FactorMode::Cached || !krylov_valid_) {
            build_krylov_operator(values);
            krylov_valid_ = mode == FactorMode::Cached;
        }
        const linear::KrylovRoute route = linear::krylov_route(krylov_kind_, krylov_.method);
        linear::KrylovSettings settings;
        settings.tol = krylov_.tol;
        settings.max_iter = krylov_.max_iter;
        settings.rank = krylov_.rank;
        settings.num_threads = num_threads_;
        if (x0 != nullptr) {
            std::copy(x0, x0 + n, x);
        } else {
            std::fill(x, x + n, 0.0);
        }
        const linear::KrylovInfo info = linear::krylov_solve(krylov_matrix_, rhs, x, route, settings);
        const bool accepted =
            info.converged || info.relative_residual <= (std::max)(krylov_.tol * 10.0, 1e-8);
        if (stats_ != nullptr) {
            KrylovStat stat;
            stat.info = info;
            stat.matrix_kind = krylov_kind_;
            stat.dimension = static_cast<std::int64_t>(n);
            stat.nnz = static_cast<std::int64_t>(krylov_values_.size());
            stat.csr_operator = csr_operator;
            stat.used_direct_fallback = !accepted;
            stats_->push_back(stat);
        }
        if (accepted) {
            numeric_valid_ = false;  // the direct factorization (if any) is of another matrix
            return true;
        }
    }
    return direct(values, rhs, x, mode);
}

bool SystemSolver::direct(const double* values, const double* rhs, double* x, FactorMode mode) {
    const auto n = static_cast<std::size_t>(pattern_.n);
    const auto nnz = static_cast<std::size_t>(pattern_.nnz());
    if (!direct_) {
        direct_ = linear::make_linear_solver<double>(pattern_);
    }
    try {
        switch (mode) {
            case FactorMode::Factor:
                factor_valid_ = false;
                direct_->factor(values, nnz);
                ++factorizations_;
                break;
            case FactorMode::Refactor:
                factor_valid_ = false;
                direct_->refactor(values, nnz);
                ++factorizations_;
                break;
            case FactorMode::Cached:
                if (!factor_valid_) {
                    direct_->factor(values, nnz);
                    ++factorizations_;
                    factor_valid_ = true;
                }
                break;
        }
    } catch (const Error& e) {
        numeric_valid_ = false;
        if (e.code() == ErrorCode::JacobianSingular) {
            factor_valid_ = false;
            return false;
        }
        throw;
    }
    numeric_valid_ = true;
    std::copy(rhs, rhs + n, x);
    direct_->solve(x, n, 1);
    return true;
}

// ---------------------------------------------------------------------------
// NewtonDriver
// ---------------------------------------------------------------------------

NewtonDriver::NewtonDriver(const Circuit& circuit, const NewtonOptions& options, std::vector<KrylovStat>* stats)
    : circuit_(circuit),
      options_(options),
      device_threads_(device_eval_threads(circuit, options.num_threads)),
      system_(circuit, options.krylov, options.num_threads, stats) {
    const auto n = static_cast<std::size_t>(circuit.n());
    const auto nnz = static_cast<std::size_t>(circuit.nnz());
    f_.resize(n);
    residual_.resize(n);
    trial_.resize(n);
    trial_f_.resize(n);
    trial_residual_.resize(n);
    best_.resize(n);
    rhs_.resize(n);
    delta_.resize(n);
    ax_.resize(n);
    jdev_.resize(nnz);
    trial_jdev_.resize(nnz);
    jacobian_.resize(nnz);
    if (!circuit.junctions().empty()) {
        limited_.resize(n);
        correction_.resize(n);
    }
}

NewtonDriver::~NewtonDriver() = default;

void NewtonDriver::evaluate_residual(const double* a_static, const double* b, double t, const double* x, double* f,
                                     double* jvals, double* residual) {
    MNA_NO_FP_CONTRACT
    assembly::evaluate(circuit_, x, t, device_threads_, f, jvals);
    pattern_matvec(circuit_, a_static, x, ax_.data());
    const auto n = static_cast<std::size_t>(circuit_.n());
    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = (ax_[i] + f[i]) - b[i];  // (G @ x) + f - b
    }
}

bool NewtonDriver::jacobian_singular(const double* a_static, double t, const std::vector<double>& x) {
    MNA_NO_FP_CONTRACT
    const auto nnz = static_cast<std::size_t>(circuit_.nnz());
    assembly::evaluate(circuit_, x.data(), t, device_threads_, nullptr, jdev_.data());
    for (std::size_t s = 0; s < nnz; ++s) {
        jacobian_[s] = a_static[s] + jdev_[s];
    }
    return !system_.factor_check(jacobian_.data());
}

NewtonDriver::Damping NewtonDriver::damp(const double* a_static, const double* b, double t,
                                          const std::vector<double>& x, const std::vector<double>& step,
                                          double norm_now, bool monotonicity) {
    MNA_NO_FP_CONTRACT
    const auto n = static_cast<std::size_t>(circuit_.n());
    // _damped_newton_step: first trial that does not increase the finite
    // residual norm; the best strictly smaller one as the fallback.
    Damping d;
    d.best_norm = norm_now;
    d.next_norm = norm_now;
    double step_norm = 0.0;
    if (monotonicity) {
        step_norm = finite_norm(step.data(), n);
    }
    double scale = 1.0;
    for (int trial = 0; trial < kMaxDampingTrials; ++trial) {
        for (std::size_t i = 0; i < n; ++i) {
            trial_[i] = x[i] + step[i] * scale;
        }
        evaluate_residual(a_static, b, t, trial_.data(), trial_f_.data(), trial_jdev_.data(), trial_residual_.data());
        const double trial_norm = finite_norm(trial_residual_.data(), n);
        const bool finite = std::isfinite(trial_norm);
        if (finite && trial_norm < d.best_norm) {
            best_ = trial_;
            d.best_norm = trial_norm;
            d.best_scale = scale;
        }
        if (finite && trial_norm <= norm_now) {
            d.accepted = true;
            d.next_norm = trial_norm;
            d.step_scale = scale;
            return d;
        }
        // Restricted natural monotonicity test with this iteration's factorization.
        if (monotonicity && finite && step_norm > 0.0 && std::isfinite(step_norm)) {
            for (std::size_t i = 0; i < n; ++i) {
                rhs_[i] = -trial_residual_[i];
            }
            if (system_.resolve(rhs_.data(), correction_.data())) {
                const double correction_norm = finite_norm(correction_.data(), n);
                if (correction_norm <= (1.0 - 0.25 * scale) * step_norm) {
                    d.accepted = true;
                    d.next_norm = trial_norm;
                    d.step_scale = scale;
                    d.by_monotonicity = true;
                    return d;
                }
            }
        }
        scale *= 0.5;
    }
    return d;
}

NewtonOutcome NewtonDriver::solve(const double* a_static, const double* b, double t, std::vector<double>& x,
                                  std::int64_t max_iter) {
    MNA_NO_FP_CONTRACT
    const auto n = static_cast<std::size_t>(circuit_.n());
    const auto nnz = static_cast<std::size_t>(circuit_.nnz());
    if (x.size() != n) {
        throw Error(ErrorCode::NetlistFatal, "newton: starting point has " + std::to_string(x.size()) +
                                                 " entries, expected " + std::to_string(n));
    }
    const std::int64_t iteration_limit = max_iter < 0 ? options_.max_iter : max_iter;
    // P3b junction strategy: only with exponential junctions, and only after the
    // reference iteration failed or stalled, so every run it solves without
    // stalling is unchanged.
    const bool junctions = options_.junction_limiting && !circuit_.junctions().empty();
    const bool monotonicity = junctions && !options_.krylov.enabled;
    bool junction_mode = false;
    int stalled = 0;
    NewtonOutcome outcome;
    bool have_residual = false;
    double norm_now = 0.0;

    // Junction-mode damping: the limited step first (when the limiter changed
    // the update), then the plain Newton step.
    auto junction_step = [&]() {
        bool limited = false;
        Damping d;
        for (std::size_t i = 0; i < n; ++i) {
            limited_[i] = x[i] + delta_[i];
        }
        if (limit_junctions(circuit_, x.data(), limited_.data()) > 0) {
            limited = true;
            ++outcome.limited_steps;
            for (std::size_t i = 0; i < n; ++i) {
                limited_[i] = limited_[i] - x[i];
            }
            d = damp(a_static, b, t, x, limited_, norm_now, monotonicity);
        }
        if (!d.accepted) {
            if (limited) {
                ++outcome.fallback_steps;
            }
            d = damp(a_static, b, t, x, delta_, norm_now, monotonicity);
        }
        if (d.by_monotonicity) {
            ++outcome.monotonicity_accepts;
        }
        return d;
    };

    for (std::int64_t iteration = 0; iteration < iteration_limit; ++iteration) {
        if (!have_residual) {
            evaluate_residual(a_static, b, t, x.data(), f_.data(), jdev_.data(), residual_.data());
            norm_now = finite_norm(residual_.data(), n);
            have_residual = true;
        }
        if (norm_now < options_.f_tol) {
            outcome.status = NewtonStatus::Converged;
            return outcome;
        }

        for (std::size_t s = 0; s < nnz; ++s) {
            jacobian_[s] = a_static[s] + jdev_[s];
        }
        for (std::size_t i = 0; i < n; ++i) {
            rhs_[i] = -residual_[i];
        }
        ++outcome.iterations;
        if (!system_.solve(jacobian_.data(), rhs_.data(), delta_.data(), nullptr, false, FactorMode::Refactor)) {
            outcome.status = NewtonStatus::Singular;
            outcome.iteration = iteration;
            return outcome;
        }

        Damping d;
        if (junction_mode) {
            d = junction_step();
        } else {
            d = damp(a_static, b, t, x, delta_, norm_now, false);
            if (junctions) {
                if (!d.accepted) {
                    // The reference iteration would stop here: retry this
                    // iteration's update with the junction strategy.
                    junction_mode = true;
                    outcome.junction_switch = iteration;
                    d = junction_step();
                } else if (d.step_scale <= kStallScale) {
                    if (++stalled >= kStallIterations) {
                        junction_mode = true;  // from the next iteration on
                        outcome.junction_switch = iteration + 1;
                    }
                } else {
                    stalled = 0;
                }
            }
        }

        double next_norm = norm_now;
        double step_scale = 0.0;
        double delta_sq = 0.0;
        if (d.accepted) {
            next_norm = d.next_norm;
            step_scale = d.step_scale;
            for (std::size_t i = 0; i < n; ++i) {
                const double diff = trial_[i] - x[i];
                delta_sq += diff * diff;
            }
            // The trial's evaluation is the next iteration's evaluation.
            std::swap(x, trial_);
            std::swap(f_, trial_f_);
            std::swap(jdev_, trial_jdev_);
            std::swap(residual_, trial_residual_);
            norm_now = next_norm;
            have_residual = true;
        } else if (d.best_scale > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                const double diff = best_[i] - x[i];
                delta_sq += diff * diff;
            }
            x = best_;
            next_norm = d.best_norm;
            step_scale = d.best_scale;
            have_residual = false;  // re-evaluated at the top: same point, same values
        }

        if (next_norm < options_.f_tol) {
            outcome.status = NewtonStatus::Converged;
            return outcome;
        }
        if (step_scale == 0.0) {
            outcome.status = NewtonStatus::NoReduction;
            outcome.iteration = iteration;
            return outcome;
        }
        if (std::sqrt(delta_sq) < options_.v_tol) {
            outcome.status = NewtonStatus::Converged;
            return outcome;
        }
    }
    outcome.status = NewtonStatus::MaxIterations;
    return outcome;
}

}  // namespace mna::analyses
