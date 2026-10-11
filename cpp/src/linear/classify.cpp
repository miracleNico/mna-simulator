#include "mna/linear/classify.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

#include "mna/error.hpp"

namespace mna::linear {
namespace {

inline double conj_of(double v) noexcept { return v; }
inline Complex conj_of(const Complex& v) noexcept { return std::conj(v); }

template <class Scalar>
bool cholesky_succeeds(const CscPattern& pattern, const Scalar* values) {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Index n = pattern.n;
    if (n == 0) {
        return true;
    }
    Matrix dense = Matrix::Zero(n, n);
    for (Eigen::Index j = 0; j < n; ++j) {
        for (std::int32_t p = pattern.col_ptr[static_cast<std::size_t>(j)];
             p < pattern.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            dense(pattern.row_idx[static_cast<std::size_t>(p)], j) = values[p];
        }
    }
    // Like the potrf('L') behind numpy.linalg.cholesky (NumPy ships OpenBLAS):
    // only the lower triangle is read and the factorization fails exactly when
    // the (real part of the) pivot is <= 0. A NaN pivot is not rejected; it
    // propagates into every later pivot. Eigen's LLT applies the same rule.
    Eigen::LLT<Matrix, Eigen::Lower> llt(dense);
    return llt.info() == Eigen::Success;
}

}  // namespace

const char* matrix_kind_name(MatrixKind kind) noexcept {
    switch (kind) {
        case MatrixKind::PositiveDefinite: return "positive_definite";
        case MatrixKind::Symmetric: return "symmetric";
        case MatrixKind::General: break;
    }
    return "general";
}

template <class Scalar>
MatrixKind classify_matrix(const CsrMatrix<Scalar>& A, const CscPattern& pattern, const Scalar* csc_values,
                           double symmetry_tol) {
    const auto un = static_cast<std::size_t>(pattern.n);
    if (A.n != pattern.n || A.nnz() != pattern.nnz()) {
        throw Error(ErrorCode::NetlistFatal, "classify_matrix: row view and CSC storage describe different matrices");
    }

    // max |a_ij - conj(a_ji)| over every stored (i, j). Column j of A is
    // compared with the conjugated row j scattered into `work`; a difference
    // at a position where only a_ji is stored has the same magnitude as the
    // one at the stored transposed position, so the stored entries cover the
    // whole union pattern.
    std::vector<Scalar> work(un);
    std::vector<std::int32_t> marker(un, -1);
    double max_diff = 0.0;
    bool saw_nan = false;
    for (std::size_t j = 0; j < un; ++j) {
        for (std::int32_t p = A.row_ptr[j]; p < A.row_ptr[j + 1]; ++p) {
            const auto k = static_cast<std::size_t>(A.col_idx[static_cast<std::size_t>(p)]);
            work[k] = conj_of(A.values[static_cast<std::size_t>(p)]);
            marker[k] = static_cast<std::int32_t>(j);
        }
        for (auto p = static_cast<std::size_t>(pattern.col_ptr[j]); p < static_cast<std::size_t>(pattern.col_ptr[j + 1]);
             ++p) {
            const auto i = static_cast<std::size_t>(pattern.row_idx[p]);
            const Scalar other = marker[i] == static_cast<std::int32_t>(j) ? work[i] : Scalar{};
            const double d = std::abs(csc_values[p] - other);
            if (std::isnan(d)) {
                saw_nan = true;
            } else if (d > max_diff) {
                max_diff = d;
            }
        }
    }
    // NumPy's max propagates NaN and "NaN > tol" is false: a NaN difference
    // never makes the matrix general.
    if (!saw_nan && max_diff > symmetry_tol) {
        return MatrixKind::General;
    }
    if (pattern.n <= kCholeskyMaxN && cholesky_succeeds(pattern, csc_values)) {
        return MatrixKind::PositiveDefinite;
    }
    return MatrixKind::Symmetric;
}

template <class Scalar>
MatrixKind classify_matrix(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                           std::size_t ai_size, const Scalar* Ax, std::size_t ax_size, double symmetry_tol) {
    CscPattern pattern = CscPattern::from_arrays(n, Ap, ap_size, Ai, ai_size);
    const CsrMatrix<Scalar> A = CsrMatrix<Scalar>::from_csc(pattern, Ax, ax_size);
    return classify_matrix(A, pattern, Ax, symmetry_tol);
}

template MatrixKind classify_matrix<double>(const CsrMatrix<double>&, const CscPattern&, const double*, double);
template MatrixKind classify_matrix<Complex>(const CsrMatrix<Complex>&, const CscPattern&, const Complex*, double);
template MatrixKind classify_matrix<double>(std::int32_t, const std::int32_t*, std::size_t, const std::int32_t*,
                                            std::size_t, const double*, std::size_t, double);
template MatrixKind classify_matrix<Complex>(std::int32_t, const std::int32_t*, std::size_t, const std::int32_t*,
                                             std::size_t, const Complex*, std::size_t, double);

}  // namespace mna::linear
