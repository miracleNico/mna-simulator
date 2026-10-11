// Harmonic balance (CPP_MIGRATION_PLAN.md 2.6, phase A): a port of
// solvers.solve_harmonic_balance that keeps its algorithm and replaces the
// dense Kronecker products by block-structured sparse assembly.
//
// Unknowns: x_bar[i * K + s] with K = 2H + 1 and s = 0 (DC), 2h - 1 (cos h),
// 2h (sin h) for h = 1..H; t_k = k * (period / K), k = 0..K-1 (NumPy's
// linspace(0, period, K, endpoint=False)), period = 2 pi / omega0.
//
//   Gamma     (S x K): row 0 = 1/K; rows 2h-1 / 2h = (2/K) cos / sin(2 pi h k / K)
//   Gamma_inv (K x S): column 0 = 1; columns 2h-1 / 2h = cos / sin(2 pi h k / K)
//   b_bar[i*K + s]   = b_dc[i] [s == 0] + sum_k Gamma[s, k] b_time_i(t_k)
//   Y_bar            = G_eff kron I_K + C kron Omega, Omega[2h-1, 2h] = h omega0 = -Omega[2h, 2h-1]
//   F(x_bar)         = Y_bar x_bar + f_bar(x_bar) - b_bar, f_bar_i = Gamma f_i(x(t_k))
//   J(x_bar)         = Y_bar + blocks Gamma diag(g_rc(t_k)) Gamma_inv for (r, c) in the device pattern
//
// Storage: Y_bar is sparse (K diagonal entries per nonzero of G_eff, 2H
// rotation entries per nonzero of C); each device Jacobian slot (r, c) of the
// compiled circuit adds one dense K x K block. The union pattern is built once
// per run.
//
// Device blocks are computed, as in the reference, by a dense product per slot
// and Newton iteration: Gamma diag(g) Gamma_inv = D Gamma_inv^T diag(g) Gamma_inv
// with D = diag(1/K, 2/K, ..., 2/K), evaluating only the upper triangle of the
// symmetric middle factor (K^3 / 2 multiply-adds; hb_gram_columns). An
// O(K^2) evaluation through the discrete Fourier sums of g (product-to-sum
// identities, exact in exact arithmetic and more accurate in floating point)
// was implemented and rejected: on the close-tone diode mixer (H = 205) the
// undamped Newton then overflows at iteration 2 with every direct solver
// tried (LAPACK, KLU, this file's BTF solver), exactly like Newton with
// exactly solved steps, while the dense product converges in 36 iterations
// with all of them. The diode conductance spans 1e-16..1e12 S over the
// samples, so the Newton matrix has a condition number near 1/eps, and the
// reference's convergence relies on the unstructured rounding of the dense
// product (any summation order works: the reference's Gamma @ (g * Gamma_inv),
// einsum, (Gamma * g) @ Gamma_inv and this symmetric evaluation all converge in
// 36 iterations). An FFT-based Jacobian (phase B) would hit the same wall
// unless the Newton loop gains damping.
//
// Linear solver (BtfSolver): the union pattern is permuted once to block upper
// triangular form (SuiteSparse BTF, as KLU does internally); each diagonal
// block is factored on its own: KLU (symbolic analysis once, numeric
// factor/refactor per Newton iteration) for sparse blocks, Eigen's blocked
// partial-pivoting LU for blocks that are small or at least 40 % dense (the
// device blocks of a node driven only through devices and conductances, e.g.
// the mixer diode node), and a division for 1 x 1 blocks. A block is singular
// on an exact zero pivot (the LAPACK getrf / KLU rule) or a structural defect.
//
// Newton exactly as the reference: x_bar = 0, no damping, stop when
// ||F||_2 < tol (checked before each solve), at most max_iter solves. Errors
// (mna::Error, code HB_ERROR, texts of the reference):
//   "Harmonic balance solver did not converge within limit."  max_iter reached,
//                                                              or a non-finite ||F||
//   "Singular HB Jacobian."                                    singular Newton matrix
//   "Linear harmonic solve failed."                            singular Y_bar (linear circuit)
//   "HB system too large (N=..., nnz=...); reduce harmonics"   memory policy, raised before
//                                                              any allocation
//
// Determinism: device evaluation runs in parallel over the K samples (each
// sample writes its own slot), the Jacobian products over fixed column chunks
// of each block, and the diagonal blocks of the BTF form are factored in
// parallel; every output is computed by one thread in a fixed order, so the
// result is bitwise identical for any thread count. Loops with little work run
// inline (threads are started per parallel region).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mna/circuit/circuit.hpp"
#include "mna/expr/program.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"

namespace mna::analyses {

/// Largest HB dimension N = n * K accepted by hb_solve.
inline constexpr std::int64_t kHbMaxDimension = 1000000;
/// Default bound on the stored entries (MNA_HB_MAX_NNZ on the Python side).
inline constexpr std::int64_t kHbDefaultMaxNnz = 30000000;
/// Device-Jacobian blocks are multiplied in column chunks of this width (the
/// unit of parallel work; fixed, so results do not depend on the thread count).
inline constexpr std::int64_t kHbBlockChunk = 32;
/// BtfSolver factors a diagonal block densely when it has at most this many
/// rows or when nnz(block) >= kBtfDenseFraction * size^2.
inline constexpr std::int32_t kBtfDenseMaxSmall = 32;
inline constexpr double kBtfDenseFraction = 0.4;

// ---------------------------------------------------------------------------
// Fourier helpers (solvers.create_fourier_matrices / reconstruct_time_domain)
// ---------------------------------------------------------------------------

struct FourierMatrices {
    std::int64_t harmonics = 0;
    std::int64_t rows = 0;          ///< S = 2H + 1
    std::int64_t points = 0;        ///< K, the number of time points
    std::vector<double> gamma;      ///< S x K, row-major
    std::vector<double> gamma_inv;  ///< K x S, row-major
};

/// Port of create_fourier_matrices(H, time_points) for len(time_points) ==
/// num_points: same angle expression 2 * pi * h * k / K, same 1/K and 2/K
/// scalings (bitwise equal to the reference where std::cos/std::sin equal
/// NumPy's). Throws std::invalid_argument for H < 0 or num_points < 0 and
/// std::domain_error for num_points == 0 (the reference divides by K).
FourierMatrices hb_fourier_matrices(std::int64_t harmonics, std::int64_t num_points);

/// NumPy's linspace(0, 2 pi / omega0, num_points, endpoint=False).
std::vector<double> hb_time_points(double omega0, std::int64_t num_points);

/// Port of reconstruct_time_domain: out (n x num_t, row-major)
/// out[i, j] = x[i, 0] + sum_h (x[i, 2h-1] cos(h omega0 t_j) + x[i, 2h] sin(h omega0 t_j)),
/// accumulated over h in increasing order exactly as the reference (bitwise
/// equal where libm agrees). Parallel over time points; deterministic.
void hb_reconstruct(const double* x_bar, std::int64_t n, std::int64_t harmonics, double omega0, const double* t,
                    std::int64_t num_t, int num_threads, double* out);

/// Device-Jacobian block support. Gamma diag(g) Gamma_inv = D M with
/// D = diag(1/K, 2/K, ..., 2/K) and the symmetric M = Gamma_inv^T diag(g) Gamma_inv;
/// the block entry (a, b) is D[a] * M[min(a, b), max(a, b)]. This writes rows
/// [0, col_end) of columns [col_begin, col_end) of M (its upper triangle there)
/// into the column-major S x S array `gram`, as a dense product (Eigen GEMM) of
/// Gamma_inv[:, :col_end]^T and weighted = g[:, None] * Gamma_inv[:, col_begin:col_end]
/// (`scratch`: K * (col_end - col_begin) doubles). See the file comment for why
/// the dense product is kept.
void hb_gram_columns(const FourierMatrices& fm, const double* g, std::int64_t col_begin, std::int64_t col_end,
                     double* scratch, double* gram);

// ---------------------------------------------------------------------------
// Block triangular direct solver
// ---------------------------------------------------------------------------

/// Direct solver for a fixed square CSC pattern through its block upper
/// triangular form (see the file comment). factor() reuses the per-block
/// symbolic analysis and, for KLU blocks, the previous pivot order (refactor
/// with the KluSolver fallbacks). Singularity (structural, an exact zero
/// pivot in a dense or 1 x 1 block, or KLU's zero pivot) is reported as
/// mna::Error(JACOBIAN_SINGULAR) by factor(). Not safe for concurrent use.
class BtfSolver {
public:
    BtfSolver();
    ~BtfSolver();
    BtfSolver(BtfSolver&&) noexcept;
    BtfSolver& operator=(BtfSolver&&) noexcept;
    BtfSolver(const BtfSolver&) = delete;
    BtfSolver& operator=(const BtfSolver&) = delete;

    /// Validates and analyzes an n x n CSC pattern (sorted or unsorted rows,
    /// no duplicates). Throws mna::Error(NETLIST_FATAL) on malformed input.
    void analyze(std::int32_t n, const std::int32_t* col_ptr, const std::int32_t* row_idx);
    /// Numeric factorization of `values` (pattern order); diagonal blocks are
    /// factored in parallel on up to num_threads threads.
    void factor(const double* values, int num_threads);
    /// Solves A x = b in place (b has n entries).
    void solve(double* rhs);

    bool analyzed() const noexcept;
    bool factored() const noexcept;
    std::int32_t size() const noexcept;
    std::int32_t structural_rank() const noexcept;
    std::int32_t blocks() const noexcept;
    std::int32_t dense_blocks() const noexcept;
    std::int32_t klu_blocks() const noexcept;
    std::int32_t singleton_blocks() const noexcept;
    std::int32_t largest_block() const noexcept;
    /// Entries in the factors (dense blocks count size^2, 1 x 1 blocks 1, KLU
    /// blocks their L + U entries) plus the off-diagonal block entries.
    std::int64_t nnz_factors() const noexcept;
    /// Approximate bytes held by the factorization (values and indices).
    std::int64_t memory_bytes() const noexcept;
    /// Threads that took part in the last factor() (1 if inline).
    int threads_used() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// hb_solve
// ---------------------------------------------------------------------------

/// One time-dependent source row of b_time: b_time[row](t) = program(t).
struct HbTimeSource {
    std::int32_t row = 0;
    std::shared_ptr<const expr::Program> program;
};

struct HbOptions {
    double omega0 = 0.0;
    std::int64_t harmonics = 0;
    std::int64_t max_iter = 100;
    double tol = 1e-6;
    double gmin = 0.0;
    /// False: linear harmonic solve (devices ignored), as the reference does
    /// when solver_f_str is all "0" and there are no Level-1 devices.
    bool nonlinear = true;
    bool use_krylov = false;
    std::string krylov_method = "auto";
    double krylov_tol = 1e-9;
    std::int64_t krylov_max_iter = 2000;
    /// Resolved Krylov rank (solvers.resolve_krylov_rank on the HB dimension).
    std::int64_t krylov_rank = 80;
    int num_threads = 1;
    std::int64_t max_nnz = kHbDefaultMaxNnz;
};

/// One Krylov attempt (the stats entry solvers.solve_linear appends).
struct HbKrylovRecord {
    linear::KrylovInfo info;
    linear::MatrixKind matrix_kind = linear::MatrixKind::General;
    std::int64_t matrix_nnz = 0;  ///< nonzero values of the system matrix
    bool used_direct_fallback = false;
};

struct HbStats {
    bool linear = false;
    std::int64_t harmonics = 0;
    std::int64_t samples = 0;    ///< K
    std::int64_t dimension = 0;  ///< N = n K
    std::int64_t newton_iterations = 0;
    std::int64_t linear_solves = 0;
    double residual_norm = 0.0;  ///< ||F|| at the converged iterate (0 for linear)
    std::int64_t y_nnz = 0;            ///< entries of Y_bar (spec count)
    std::int64_t device_slots = 0;     ///< device Jacobian slots (dense K x K blocks)
    std::int64_t matrix_nnz = 0;       ///< entries of the assembled union pattern
    std::int64_t stored_nnz = 0;       ///< the memory-policy count (see hb_solve)
    std::int32_t btf_blocks = 0;
    std::int32_t dense_blocks = 0;
    std::int32_t klu_blocks = 0;
    std::int32_t singleton_blocks = 0;
    std::int32_t largest_block = 0;
    std::int64_t factor_nnz = 0;
    std::int64_t memory_bytes = 0;  ///< approximate peak bytes of the HB data structures
    std::int64_t factorizations = 0;
    int threads_used = 1;
    double setup_seconds = 0.0;
    double evaluate_seconds = 0.0;   ///< samples, device evaluation, residual
    double jacobian_seconds = 0.0;   ///< Fourier blocks and value assembly
    double factor_seconds = 0.0;
    double solve_seconds = 0.0;      ///< triangular solves and Krylov
    std::vector<HbKrylovRecord> krylov;
};

struct HbResult {
    std::vector<double> x_bar;  ///< N values
    HbStats stats;
};

/// Harmonic balance on a compiled circuit. b_dc has circuit.n() entries;
/// sources rows must be in [0, n) and their programs time dependent with no
/// unknowns (NETLIST_FATAL otherwise). Memory policy, checked before any
/// allocation: N = n K <= kHbMaxDimension and
///   stored = nnz(Y_bar) + device_slots * K^2 + 2 K^2 (the dense Fourier matrices) <= max_nnz,
/// else HB_ERROR "HB system too large (N=<N>, nnz=<stored>); reduce harmonics".
/// harmonics < 1 or a non-positive / non-finite omega0 are HB_ERROR too.
/// With use_krylov every linear solve first runs the native Krylov kernel on
/// the assembled matrix (always classified, then routed by krylov_method; an
/// unknown method is std::invalid_argument) and keeps the answer when it
/// converged or its relative residual is <= max(10 krylov_tol, 1e-8);
/// otherwise the direct solve runs (used_direct_fallback = true).
HbResult hb_solve(const circuit::Circuit& circuit, const double* b_dc, const std::vector<HbTimeSource>& sources,
                  const HbOptions& options);

}  // namespace mna::analyses
