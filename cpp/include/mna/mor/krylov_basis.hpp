// Output-aware rational Krylov basis of mor/linear.py::build_linear_krylov_basis
// on sparse operands.
//
// Operands: G and C as value arrays in ONE shared square CSC pattern (e.g. a
// compiled Circuit's pattern with values("G") / g_effective(gmin) /
// linearized_g and values("C")). Inputs are a dense n x m column block (m may
// be 0), the output seeds a dense n x p column block (selector^T), shifts a
// list of (value, is_complex) pairs: a real shift s builds the real system
// G + s C, a complex one the complex system G + s C (even when Im s == 0,
// like numpy.iscomplexobj on the Python shift).
//
// Acceptance semantics (identical to the Python loop):
//   * target = max(1, min(order, n));
//   * seeds: the input block (when m > 0), then the output block, each through
//     the block Gram-Schmidt below;
//   * per shift, in order: solve (G + sC) X = inputs (when m > 0), accept;
//     solve (G + sC)^T X = outputs (the adjoint), accept; stop as soon as the
//     basis holds `target` columns. Every block with accepted columns becomes a
//     "state block" (kind input/adjoint, its shift, its accepted columns);
//   * then repeat sweeps over the state blocks until the basis is full or a
//     whole sweep adds nothing (previous_columns stall test): input states
//     solve (G + sC) X = -(C B), adjoint states (G + sC)^T X = -(C^T B); a
//     state's block becomes the newly accepted columns when there are any,
//     otherwise it keeps its previous block;
//   * complex solutions are split into [Re X, Im X] (all real columns, then all
//     imaginary ones), Im X only when its Frobenius norm is > 0;
//   * Gram-Schmidt: classical, one pass, against the whole basis
//     (v - B (B^T v)), a column is accepted when its norm is > 1e-14 and is
//     normalized; the block stops early once the basis is full;
//   * an empty basis at the end becomes the unit vector e_0.
//
// Shifted solves: each shift's matrix is factored once with KluSolver (real
// shift) or KluSolverZ (complex shift) and reused for every block; adjoint
// solves use tsolve (plain transpose). When KLU reports the matrix singular
// (JACOBIAN_SINGULAR), the solves of that shift fall back to the dense least
// squares of _safe_solve (DenseSafeSolver with the LU stage skipped) on the
// dense A or A^T.
//
// Threads: with num_threads > 1 and n >= kMorParallelMinUnknowns the shift
// systems are factored in parallel before the sweep (each by one thread);
// Gram-Schmidt is sequential, so the basis is identical for every thread
// count.
//
// Errors (mna::Error): NETLIST_FATAL for malformed input (sizes, pattern,
// n == 0); JACOBIAN_SINGULAR when the least-squares fallback meets a
// non-finite matrix (numpy's "SVD did not converge").
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "mna/linear/linear_solver.hpp"

namespace mna::mor {

using Complex = std::complex<double>;

/// Unknown count from which the MOR kernels use more than one thread for the
/// per-circuit work (shift factorizations, TPWL device evaluation); below it a
/// thread start costs more than the work (see kParallelDeviceMinCount).
inline constexpr std::int32_t kMorParallelMinUnknowns = 2048;

struct Shift {
    Complex value;
    bool is_complex = false;
};

/// G and C values in one shared CSC pattern.
struct SparseOperands {
    const linear::CscPattern* pattern = nullptr;
    const double* g = nullptr;  ///< pattern->nnz() values
    const double* c = nullptr;  ///< pattern->nnz() values
};

struct KrylovBasisResult {
    std::int32_t n = 0;
    std::int32_t columns = 0;
    std::vector<double> basis;        ///< column-major n x columns, orthonormal
    std::int64_t factorizations = 0;  ///< shift systems factored
    std::int64_t solves = 0;          ///< block solves (each one a multi-column right-hand side)
    std::int64_t lstsq_shifts = 0;    ///< shift systems that took the least-squares fallback
    int threads_used = 1;
};

/// inputs: column-major n x input_cols (may be null when input_cols == 0);
/// outputs: column-major n x output_cols (selector^T; the C-order p x n
/// selector array has exactly this layout).
KrylovBasisResult krylov_basis(const SparseOperands& operands, const double* inputs, std::int32_t input_cols,
                               const double* outputs, std::int32_t output_cols, const std::vector<Shift>& shifts,
                               std::int64_t order, int num_threads);

/// The block Gram-Schmidt step (exposed for tests): appends the accepted
/// columns of `candidate` (column-major n x cols) to `basis` (column-major
/// n x basis_cols, grown in place) while basis_cols < max_columns. Returns the
/// number of accepted columns (they are the last ones of the basis).
std::int32_t orthogonalize_block(std::vector<double>& basis, std::int32_t n, std::int32_t& basis_cols,
                                 const double* candidate, std::int32_t cols, std::int32_t max_columns,
                                 double tolerance = 1e-14);

}  // namespace mna::mor
