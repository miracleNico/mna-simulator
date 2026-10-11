// Matrix classification for Krylov method selection, the sparse rule of
// solvers.classify_krylov_matrix applied to every input:
//
//   general            unless max |a_ij - conj(a_ji)| <= symmetry_tol over the
//                      stored pattern (a NaN difference makes that maximum NaN,
//                      which, as with NumPy's max, does not count as "> tol")
//   positive_definite  when symmetric, n <= kCholeskyMaxN and a dense Cholesky
//                      factorization (Eigen LLT of the lower triangle) succeeds:
//                      no pivot <= 0, the rule of the OpenBLAS potrf behind
//                      numpy.linalg.cholesky (a NaN pivot passes and propagates)
//   symmetric          otherwise
//
// A 1x1 [[1e-3]] is positive_definite; n == 0 is positive_definite as well
// (NumPy's Cholesky of an empty matrix succeeds).
#pragma once

#include <cstddef>
#include <cstdint>

#include "mna/linear/linear_solver.hpp"
#include "mna/linear/sparse.hpp"

namespace mna::linear {

enum class MatrixKind { General, Symmetric, PositiveDefinite };

/// "general", "symmetric", "positive_definite".
const char* matrix_kind_name(MatrixKind kind) noexcept;

inline constexpr double kDefaultSymmetryTol = 1e-10;
/// Largest dimension for which the dense Cholesky test runs.
inline constexpr std::int32_t kCholeskyMaxN = 256;

/// Classifies a square matrix given in both storage orders of the same matrix:
/// `A` as its row view and the CSC arrays it was built from.
template <class Scalar>
MatrixKind classify_matrix(const CsrMatrix<Scalar>& A, const CscPattern& pattern, const Scalar* csc_values,
                           double symmetry_tol = kDefaultSymmetryTol);

/// Validates CSC arrays and classifies the matrix.
template <class Scalar>
MatrixKind classify_matrix(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                           std::size_t ai_size, const Scalar* Ax, std::size_t ax_size,
                           double symmetry_tol = kDefaultSymmetryTol);

extern template MatrixKind classify_matrix<double>(const CsrMatrix<double>&, const CscPattern&, const double*, double);
extern template MatrixKind classify_matrix<Complex>(const CsrMatrix<Complex>&, const CscPattern&, const Complex*,
                                                    double);
extern template MatrixKind classify_matrix<double>(std::int32_t, const std::int32_t*, std::size_t,
                                                   const std::int32_t*, std::size_t, const double*, std::size_t,
                                                   double);
extern template MatrixKind classify_matrix<Complex>(std::int32_t, const std::int32_t*, std::size_t,
                                                    const std::int32_t*, std::size_t, const Complex*, std::size_t,
                                                    double);

}  // namespace mna::linear
