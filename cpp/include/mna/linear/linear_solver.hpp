// Common pieces of the direct linear solvers: the square CSC sparsity pattern,
// the abstract LinearSolver interface shared by the sparse KLU solvers
// (klu_solver.hpp) and the dense Eigen LU (dense_solver.hpp), and the policy
// that picks between them.
//
// Errors are reported as mna::Error:
//   NETLIST_FATAL      malformed input (bad sizes, bad CSC structure, duplicate
//                      entries, calls out of order, problem too large)
//   JACOBIAN_SINGULAR  the matrix is singular
// and std::bad_alloc when memory runs out.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mna::linear {

using Complex = std::complex<double>;

/// Square sparsity pattern in compressed sparse column form.
struct CscPattern {
    std::int32_t n = 0;
    std::vector<std::int32_t> col_ptr;  // size n + 1, col_ptr[0] == 0, non-decreasing
    std::vector<std::int32_t> row_idx;  // size col_ptr[n], each in [0, n), no duplicates per column

    std::int64_t nnz() const noexcept { return col_ptr.empty() ? 0 : col_ptr.back(); }

    /// Copies and validates a pattern; throws mna::Error(NETLIST_FATAL).
    static CscPattern from_arrays(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size,
                                  const std::int32_t* Ai, std::size_t ai_size);
};

/// Which numeric path refactor() took.
enum class FactorPath {
    Refactor,    // the previous pivot order was reused (KLU klu_refactor) and is still well conditioned
    FullFactor,  // a factorization with fresh partial pivoting ran
};

/// "refactor" or "full_factor".
const char* factor_path_name(FactorPath path) noexcept;

/// refactor() keeps a reused pivot order only while the cheap pivot-ratio
/// estimate rcond() = min|u_ii| / max|u_ii| stays at or above this floor. Below
/// it the old order is treated as numerically unsafe for the new values and a
/// full factorization with fresh partial pivoting runs instead (whose own
/// rcond may legitimately be below the floor too; that is not an error).
inline constexpr double kRefactorRcondFloor = 1e-14;

/// Dense-vs-KLU policy: SolverKind::Auto uses the dense Eigen LU for systems
/// with 0 < n <= kDenseMaxN and KLU otherwise. 0 means "always KLU"; phase 3
/// tunes the threshold by benchmark (benchmarks/compare_backends.py).
inline constexpr std::int32_t kDenseMaxN = 0;

enum class SolverKind {
    Auto,   // use_dense_solver(n) ? Dense : Klu
    Klu,    // SuiteSparse KLU (real: klu_*, complex: klu_z_*)
    Dense,  // Eigen PartialPivLU on a dense copy
};

/// True when SolverKind::Auto picks the dense solver for an n x n system.
bool use_dense_solver(std::int32_t n) noexcept;

/// Direct solver for A X = B with a fixed n x n sparsity pattern.
///
/// Values are passed in the order of the pattern the solver was built for (CSC
/// order), right-hand sides as column-major n x nrhs arrays solved in place.
/// Implementations are not safe for concurrent use; separate instances are
/// independent and may be used from different threads.
template <class Scalar>
class LinearSolver {
public:
    using scalar_type = Scalar;

    virtual ~LinearSolver() = default;
    LinearSolver(const LinearSolver&) = delete;
    LinearSolver& operator=(const LinearSolver&) = delete;

    /// Numeric factorization with partial pivoting.
    virtual void factor(const Scalar* values, std::size_t size) = 0;
    /// Numeric refactorization for new values on the same pattern, reusing the
    /// previous pivot order where the implementation supports it; falls back to
    /// factor() when there is no previous factorization, the old order hits a
    /// zero pivot, or rcond() drops below kRefactorRcondFloor.
    virtual FactorPath refactor(const Scalar* values, std::size_t size) = 0;
    /// Solves A X = B in place; `rhs` is column-major n x nrhs.
    virtual void solve(Scalar* rhs, std::size_t size, std::int32_t nrhs) = 0;
    /// Solves A^T X = B (conjugate = false) or A^H X = B (conjugate = true) in
    /// place. `conjugate` has no effect for real matrices.
    virtual void tsolve(Scalar* rhs, std::size_t size, std::int32_t nrhs, bool conjugate) = 0;
    /// Pivot-ratio condition estimate min|u_ii| / max|u_ii| of the current
    /// factorization (0 for a zero or NaN pivot, 1.0 for n == 0).
    virtual double rcond() = 0;
    /// Entries stored in the L and U factors (both counted with their diagonal).
    virtual std::int64_t nnz_lu() const noexcept = 0;
    virtual std::int32_t size() const noexcept = 0;
    virtual bool factored() const noexcept = 0;
    /// "klu" or "dense".
    virtual const char* kind() const noexcept = 0;

protected:
    LinearSolver() = default;
    LinearSolver(LinearSolver&&) noexcept = default;
    LinearSolver& operator=(LinearSolver&&) noexcept = default;
};

/// Builds the solver `kind` selects for `pattern` (already analyzed for KLU).
template <class Scalar>
std::unique_ptr<LinearSolver<Scalar>> make_linear_solver(const CscPattern& pattern, SolverKind kind = SolverKind::Auto);

extern template std::unique_ptr<LinearSolver<double>> make_linear_solver<double>(const CscPattern&, SolverKind);
extern template std::unique_ptr<LinearSolver<Complex>> make_linear_solver<Complex>(const CscPattern&, SolverKind);

}  // namespace mna::linear
