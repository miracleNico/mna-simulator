#include "mna/analyses/newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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
    krylov_valid_ = false;
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
        if (e.code() == ErrorCode::JacobianSingular) {
            factor_valid_ = false;
            return false;
        }
        throw;
    }
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

NewtonOutcome NewtonDriver::solve(const double* a_static, const double* b, double t, std::vector<double>& x) {
    MNA_NO_FP_CONTRACT
    const auto n = static_cast<std::size_t>(circuit_.n());
    const auto nnz = static_cast<std::size_t>(circuit_.nnz());
    if (x.size() != n) {
        throw Error(ErrorCode::NetlistFatal, "newton: starting point has " + std::to_string(x.size()) +
                                                 " entries, expected " + std::to_string(n));
    }
    NewtonOutcome outcome;
    bool have_residual = false;
    double norm_now = 0.0;

    for (std::int64_t iteration = 0; iteration < options_.max_iter; ++iteration) {
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

        // _damped_newton_step: first trial that does not increase the finite
        // residual norm; the best strictly smaller one as the fallback.
        double best_norm = norm_now;
        double best_scale = 0.0;
        double scale = 1.0;
        bool accepted = false;
        double next_norm = norm_now;
        double step_scale = 0.0;
        for (int trial = 0; trial < kMaxDampingTrials; ++trial) {
            for (std::size_t i = 0; i < n; ++i) {
                trial_[i] = x[i] + delta_[i] * scale;
            }
            evaluate_residual(a_static, b, t, trial_.data(), trial_f_.data(), trial_jdev_.data(),
                              trial_residual_.data());
            const double trial_norm = finite_norm(trial_residual_.data(), n);
            const bool finite = std::isfinite(trial_norm);
            if (finite && trial_norm < best_norm) {
                best_ = trial_;
                best_norm = trial_norm;
                best_scale = scale;
            }
            if (finite && trial_norm <= norm_now) {
                accepted = true;
                next_norm = trial_norm;
                step_scale = scale;
                break;
            }
            scale *= 0.5;
        }

        double delta_sq = 0.0;
        if (accepted) {
            for (std::size_t i = 0; i < n; ++i) {
                const double d = trial_[i] - x[i];
                delta_sq += d * d;
            }
            // The trial's evaluation is the next iteration's evaluation.
            std::swap(x, trial_);
            std::swap(f_, trial_f_);
            std::swap(jdev_, trial_jdev_);
            std::swap(residual_, trial_residual_);
            norm_now = next_norm;
            have_residual = true;
        } else if (best_scale > 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                const double d = best_[i] - x[i];
                delta_sq += d * d;
            }
            x = best_;
            next_norm = best_norm;
            step_scale = best_scale;
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
