#include "mna/linear/dense_solver.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"

namespace mna::linear {
namespace {

[[noreturn]] void fatal(const std::string& message) { throw Error(ErrorCode::NetlistFatal, message); }

template <class Scalar>
bool is_finite(const Scalar& v) noexcept {
    if constexpr (std::is_same_v<Scalar, double>) {
        return std::isfinite(v);
    } else {
        return std::isfinite(v.real()) && std::isfinite(v.imag());
    }
}

}  // namespace

template <class Scalar>
struct DenseSolver<Scalar>::Impl {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    std::int32_t n = 0;
    bool csc_mode = false;
    CscPattern pattern;
    Matrix work;  // dense copy handed to the LU
    Eigen::PartialPivLU<Matrix> lu;
    bool factored = false;
    double pivot_ratio = 1.0;

    void check_values(const Scalar* values, std::size_t size, const char* operation) const {
        const std::size_t expected = csc_mode ? static_cast<std::size_t>(pattern.nnz())
                                              : static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        if (size != expected) {
            fatal(std::string(operation) + ": got " + std::to_string(size) + " values, expected " +
                  std::to_string(expected) + (csc_mode ? " (pattern entries)" : " (n * n, column-major)"));
        }
        if (size > 0 && values == nullptr) {
            fatal(std::string(operation) + ": null value array");
        }
    }

    void factor(const Scalar* values, std::size_t size) {
        check_values(values, size, "dense factor");
        factored = false;
        if (n == 0) {
            pivot_ratio = 1.0;
            factored = true;
            return;
        }
        work.setZero(n, n);
        if (csc_mode) {
            for (std::int32_t j = 0; j < n; ++j) {
                for (std::int32_t p = pattern.col_ptr[static_cast<std::size_t>(j)];
                     p < pattern.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                    work(pattern.row_idx[static_cast<std::size_t>(p)], j) = values[p];
                }
            }
        } else {
            work = Eigen::Map<const Matrix>(values, n, n);
        }
        lu.compute(work);

        // Singularity rule (see the header): |u_ii| < eps * max|u_ij| or non-finite factors.
        const Matrix& packed = lu.matrixLU();
        double max_u = 0.0;
        bool finite = true;
        for (Eigen::Index j = 0; j < n; ++j) {
            for (Eigen::Index i = 0; i < n; ++i) {
                const Scalar v = packed(i, j);
                if (!is_finite(v)) {
                    finite = false;
                } else if (i <= j) {
                    max_u = std::max(max_u, static_cast<double>(std::abs(v)));
                }
            }
        }
        if (!finite) {
            throw Error(ErrorCode::JacobianSingular, "dense LU: matrix is singular (non-finite factor entries)");
        }
        const double threshold = std::numeric_limits<double>::epsilon() * max_u;
        double min_diag = std::numeric_limits<double>::infinity();
        double max_diag = 0.0;
        for (Eigen::Index k = 0; k < n; ++k) {
            const double d = std::abs(packed(k, k));
            if (!(d >= threshold) || max_u == 0.0) {
                throw Error(ErrorCode::JacobianSingular, "dense LU: matrix is singular (pivot " + std::to_string(k) +
                                                             " below eps * max|U|)");
            }
            min_diag = std::min(min_diag, d);
            max_diag = std::max(max_diag, d);
        }
        pivot_ratio = min_diag / max_diag;
        factored = true;
    }

    bool check_rhs(const Scalar* rhs, std::size_t size, std::int32_t nrhs, const char* operation) const {
        if (!factored) {
            fatal(std::string(operation) + " called before factor()");
        }
        if (nrhs < 0) {
            fatal(std::string(operation) + ": negative number of right-hand sides");
        }
        const std::size_t expected = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
        if (size != expected) {
            fatal(std::string(operation) + ": right-hand side has " + std::to_string(size) +
                  " values, expected n * nrhs = " + std::to_string(expected));
        }
        if (expected == 0) {
            return false;
        }
        if (rhs == nullptr) {
            fatal(std::string(operation) + ": null right-hand side");
        }
        return true;
    }
};

template <class Scalar>
DenseSolver<Scalar>::DenseSolver(std::int32_t n) : impl_(std::make_unique<Impl>()) {
    if (n < 0) {
        fatal("dense solver: negative dimension " + std::to_string(n));
    }
    impl_->n = n;
}

template <class Scalar>
DenseSolver<Scalar>::DenseSolver(CscPattern pattern) : impl_(std::make_unique<Impl>()) {
    if (pattern.n < 0 || pattern.col_ptr.size() != static_cast<std::size_t>(pattern.n) + 1) {
        fatal("dense solver: malformed CSC pattern");
    }
    impl_->n = pattern.n;
    impl_->csc_mode = true;
    impl_->pattern = std::move(pattern);
}

template <class Scalar>
DenseSolver<Scalar>::~DenseSolver() = default;
template <class Scalar>
DenseSolver<Scalar>::DenseSolver(DenseSolver&&) noexcept = default;
template <class Scalar>
DenseSolver<Scalar>& DenseSolver<Scalar>::operator=(DenseSolver&&) noexcept = default;

template <class Scalar>
void DenseSolver<Scalar>::factor(const Scalar* values, std::size_t size) {
    if (!impl_) {
        fatal("dense factor on a moved-from solver");
    }
    impl_->factor(values, size);
}

template <class Scalar>
FactorPath DenseSolver<Scalar>::refactor(const Scalar* values, std::size_t size) {
    factor(values, size);
    return FactorPath::FullFactor;
}

template <class Scalar>
void DenseSolver<Scalar>::solve(Scalar* rhs, std::size_t size, std::int32_t nrhs) {
    if (!impl_) {
        fatal("dense solve on a moved-from solver");
    }
    if (!impl_->check_rhs(rhs, size, nrhs, "dense solve")) {
        return;
    }
    using Matrix = typename Impl::Matrix;
    Eigen::Map<Matrix> b(rhs, impl_->n, nrhs);
    const Matrix x = impl_->lu.solve(b);
    b = x;
}

template <class Scalar>
void DenseSolver<Scalar>::tsolve(Scalar* rhs, std::size_t size, std::int32_t nrhs, bool conjugate) {
    if (!impl_) {
        fatal("dense tsolve on a moved-from solver");
    }
    if (!impl_->check_rhs(rhs, size, nrhs, "dense tsolve")) {
        return;
    }
    using Matrix = typename Impl::Matrix;
    Eigen::Map<Matrix> b(rhs, impl_->n, nrhs);
    Matrix x;
    if (conjugate) {
        x = impl_->lu.adjoint().solve(b);
    } else {
        x = impl_->lu.transpose().solve(b);
    }
    b = x;
}

template <class Scalar>
double DenseSolver<Scalar>::rcond() {
    if (!impl_ || !impl_->factored) {
        fatal("rcond called before factor()");
    }
    return impl_->pivot_ratio;
}

template <class Scalar>
std::int64_t DenseSolver<Scalar>::nnz_lu() const noexcept {
    if (!impl_ || !impl_->factored) {
        return 0;
    }
    const auto n = static_cast<std::int64_t>(impl_->n);
    return n * (n + 1);
}

template <class Scalar>
std::int32_t DenseSolver<Scalar>::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

template <class Scalar>
bool DenseSolver<Scalar>::factored() const noexcept {
    return impl_ && impl_->factored;
}

template <class Scalar>
bool DenseSolver<Scalar>::has_pattern() const noexcept {
    return impl_ && impl_->csc_mode;
}

template class DenseSolver<double>;
template class DenseSolver<Complex>;

// ----------------------------------------------------------------------------
// Policy and factory (linear_solver.hpp)
// ----------------------------------------------------------------------------

bool use_dense_solver(std::int32_t n) noexcept {
    const std::int32_t limit = kDenseMaxN;
    return n > 0 && n <= limit;
}

template <class Scalar>
std::unique_ptr<LinearSolver<Scalar>> make_linear_solver(const CscPattern& pattern, SolverKind kind) {
    const bool dense = kind == SolverKind::Dense || (kind == SolverKind::Auto && use_dense_solver(pattern.n));
    if (dense) {
        // Validate exactly like the KLU path does before scattering into a dense copy.
        CscPattern checked = CscPattern::from_arrays(pattern.n, pattern.col_ptr.data(), pattern.col_ptr.size(),
                                                     pattern.row_idx.data(), pattern.row_idx.size());
        return std::make_unique<DenseSolver<Scalar>>(std::move(checked));
    }
    auto klu = std::make_unique<KluSolverT<Scalar>>();
    klu->analyze(pattern);
    return klu;
}

template std::unique_ptr<LinearSolver<double>> make_linear_solver<double>(const CscPattern&, SolverKind);
template std::unique_ptr<LinearSolver<Complex>> make_linear_solver<Complex>(const CscPattern&, SolverKind);

}  // namespace mna::linear
