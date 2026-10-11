#include "mna/mor/safe_solve.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "mna/error.hpp"

namespace mna::mor {

const char* solve_path_name(SolvePath path) noexcept {
    switch (path) {
        case SolvePath::LU:
            return "lu";
        case SolvePath::LeastSquares:
            return "least_squares";
        case SolvePath::RegularizedLeastSquares:
            return "regularized_least_squares";
    }
    return "unknown";
}

template <class Scalar>
struct DenseSafeSolver<Scalar>::Impl {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    std::int32_t n = 0;
    bool factored = false;
    SolvePath path = SolvePath::LU;
    Eigen::PartialPivLU<Matrix> lu;
    Eigen::CompleteOrthogonalDecomposition<Matrix> cod;

    void least_squares(const Matrix& a) {
        // np.linalg.lstsq's default rcond: eps * max(m, n).
        cod.setThreshold(std::numeric_limits<double>::epsilon() * static_cast<double>((std::max)(n, 1)));
        cod.compute(a);
    }
};

template <class Scalar>
DenseSafeSolver<Scalar>::DenseSafeSolver() : impl_(std::make_unique<Impl>()) {}

template <class Scalar>
DenseSafeSolver<Scalar>::~DenseSafeSolver() = default;

template <class Scalar>
DenseSafeSolver<Scalar>::DenseSafeSolver(DenseSafeSolver&&) noexcept = default;

template <class Scalar>
DenseSafeSolver<Scalar>& DenseSafeSolver<Scalar>::operator=(DenseSafeSolver&&) noexcept = default;

template <class Scalar>
void DenseSafeSolver<Scalar>::factor(std::int32_t n, const Scalar* a, bool known_singular) {
    using Matrix = typename Impl::Matrix;
    if (n < 0) {
        throw Error(ErrorCode::NetlistFatal, "safe_solve: negative dimension");
    }
    Impl& s = *impl_;
    s.factored = false;
    s.n = n;
    if (n == 0) {
        s.path = SolvePath::LU;
        s.factored = true;
        return;
    }
    const Eigen::Map<const Matrix> A(a, n, n);
    if (!known_singular) {
        s.lu.compute(A);
        bool singular = false;
        const auto& lu = s.lu.matrixLU();
        for (std::int32_t i = 0; i < n; ++i) {
            if (lu(i, i) == Scalar(0)) {  // LAPACK getrf: info > 0 only for an exactly zero U(i, i)
                singular = true;
                break;
            }
        }
        if (!singular) {
            s.path = SolvePath::LU;
            s.factored = true;
            return;
        }
    }
    if (A.allFinite()) {
        s.least_squares(A);
        if (s.cod.matrixQTZ().allFinite()) {
            s.path = SolvePath::LeastSquares;
            s.factored = true;
            return;
        }
    }
    // np.linalg.lstsq raised: retry on A + I * 1e-12 * max(1, ||A||_inf). Python's
    // max(1.0, nan) is 1.0, and a non-finite A stays non-finite, so that retry raises too.
    const double norm_inf = A.cwiseAbs().rowwise().sum().maxCoeff();
    const double scale = norm_inf > 1.0 ? norm_inf : 1.0;
    Matrix regularized = A;
    regularized.diagonal().array() += Scalar(scale * 1e-12);
    if (regularized.allFinite()) {
        s.least_squares(regularized);
        if (s.cod.matrixQTZ().allFinite()) {
            s.path = SolvePath::RegularizedLeastSquares;
            s.factored = true;
            return;
        }
    }
    throw Error(ErrorCode::JacobianSingular, "SVD did not converge in Linear Least Squares");
}

template <class Scalar>
void DenseSafeSolver<Scalar>::solve(Scalar* b, std::int32_t nrhs) const {
    using Matrix = typename Impl::Matrix;
    const Impl& s = *impl_;
    if (!s.factored) {
        throw Error(ErrorCode::NetlistFatal, "safe_solve: solve called before factor");
    }
    if (s.n == 0 || nrhs <= 0) {
        return;
    }
    Eigen::Map<Matrix> B(b, s.n, nrhs);
    if (s.path == SolvePath::LU) {
        B = s.lu.solve(B).eval();
    } else {
        B = s.cod.solve(B).eval();
    }
}

template <class Scalar>
SolvePath DenseSafeSolver<Scalar>::path() const noexcept {
    return impl_->path;
}

template <class Scalar>
std::int32_t DenseSafeSolver<Scalar>::size() const noexcept {
    return impl_->n;
}

template <class Scalar>
bool DenseSafeSolver<Scalar>::factored() const noexcept {
    return impl_->factored;
}

template <class Scalar>
std::vector<Scalar> safe_solve(std::int32_t n, const Scalar* a, const Scalar* b, std::int32_t nrhs, SolvePath* path) {
    if (nrhs < 0) {
        throw Error(ErrorCode::NetlistFatal, "safe_solve: negative right-hand-side count");
    }
    DenseSafeSolver<Scalar> solver;
    solver.factor(n, a);
    std::vector<Scalar> x(b, b + static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
    solver.solve(x.data(), nrhs);
    if (path != nullptr) {
        *path = solver.path();
    }
    return x;
}

template class DenseSafeSolver<double>;
template class DenseSafeSolver<Complex>;
template std::vector<double> safe_solve<double>(std::int32_t, const double*, const double*, std::int32_t, SolvePath*);
template std::vector<Complex> safe_solve<Complex>(std::int32_t, const Complex*, const Complex*, std::int32_t,
                                                  SolvePath*);

}  // namespace mna::mor
