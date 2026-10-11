// Dense solves with the semantics of mor/linear.py::_safe_solve, shared by the
// model-order-reduction kernels.
//
// _safe_solve(A, B) tried, in order:
//   1. np.linalg.solve(A, B)    LAPACK getrf/getrs; LinAlgError only for an
//                               exactly zero pivot of U
//   2. np.linalg.lstsq(A, B)    minimum-norm least squares, singular values
//                               <= eps * max(m, n) * s_max dropped; LinAlgError
//                               when the SVD does not converge (non-finite input)
//   3. lstsq(A + I * 1e-12 * max(1, ||A||_inf), B)
//
// Here: step 1 is Eigen::PartialPivLU with the same exact-zero-pivot test,
// step 2 is Eigen::CompleteOrthogonalDecomposition with the threshold
// eps * n (its minimum-norm solution is the pseudo-inverse solution whenever
// the numerical rank agrees with the SVD's), step 3 the same on the
// regularized matrix. "The SVD did not converge" is modelled as "the matrix
// has a non-finite entry or its decomposition overflows". A non-finite entry
// stays non-finite in the regularized matrix, so that case throws
// mna::Error(JacobianSingular, "SVD did not converge in Linear Least Squares")
// (the Python facade turns it back into numpy.linalg.LinAlgError, the
// exception _safe_solve let escape).
//
// An exactly singular matrix (a zero row or column, duplicated rows) meets an
// exactly zero pivot in every LU. A merely numerically singular one may or may
// not, depending on rounding (LAPACK's blocked FMA updates differ from Eigen's
// in the last bits), so such systems can take different stages here and in the
// reference; their solutions are rounding noise in both.
//
// Matrices are column-major n x n, right-hand sides column-major n x nrhs.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace mna::mor {

using Complex = std::complex<double>;

/// Which stage of _safe_solve produced the factorization.
enum class SolvePath : int {
    LU = 0,                       ///< partial-pivoting LU
    LeastSquares = 1,             ///< the system was singular: least squares
    RegularizedLeastSquares = 2,  ///< least squares on A + 1e-12 * max(1, ||A||_inf) * I
};

const char* solve_path_name(SolvePath path) noexcept;

/// A reusable dense factorization with _safe_solve semantics: factor() once,
/// solve() for any number of right-hand sides (the same answers _safe_solve
/// gives for that matrix). Not safe for concurrent use.
template <class Scalar>
class DenseSafeSolver {
public:
    DenseSafeSolver();
    ~DenseSafeSolver();
    DenseSafeSolver(DenseSafeSolver&&) noexcept;
    DenseSafeSolver& operator=(DenseSafeSolver&&) noexcept;
    DenseSafeSolver(const DenseSafeSolver&) = delete;
    DenseSafeSolver& operator=(const DenseSafeSolver&) = delete;

    /// Factors the n x n column-major matrix `a` (copied). With
    /// `known_singular` the LU stage is skipped (the caller's own LU, e.g.
    /// KLU on the sparse form, already reported the matrix singular).
    void factor(std::int32_t n, const Scalar* a, bool known_singular = false);

    /// Solves in place: `b` is column-major n x nrhs.
    void solve(Scalar* b, std::int32_t nrhs) const;

    SolvePath path() const noexcept;
    std::int32_t size() const noexcept;
    bool factored() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class DenseSafeSolver<double>;
extern template class DenseSafeSolver<Complex>;

/// One-shot _safe_solve(A, B): returns X (column-major n x nrhs). `path`
/// (optional) receives the stage that solved it.
template <class Scalar>
std::vector<Scalar> safe_solve(std::int32_t n, const Scalar* a, const Scalar* b, std::int32_t nrhs,
                               SolvePath* path = nullptr);

extern template std::vector<double> safe_solve<double>(std::int32_t, const double*, const double*, std::int32_t,
                                                       SolvePath*);
extern template std::vector<Complex> safe_solve<Complex>(std::int32_t, const Complex*, const Complex*, std::int32_t,
                                                         SolvePath*);

}  // namespace mna::mor
