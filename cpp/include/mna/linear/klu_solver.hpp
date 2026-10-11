// RAII wrappers over SuiteSparse KLU (32-bit indices) for square CSC matrices
// with a fixed sparsity pattern:
//   KluSolver   real values           (klu_*   API)
//   KluSolverZ  complex values        (klu_z_* API; std::complex<double> storage
//                                      is passed to KLU as interleaved re/im pairs)
//
// Typical Newton loop:
//     KluSolver lu;
//     lu.analyze(n, Ap, Ai);        // once per sparsity pattern
//     lu.factor(Ax);                // first numeric factorization (pivoting)
//     lu.solve(b);                  // b <- A^-1 b
//     ... values change, pattern does not ...
//     lu.refactor(Ax);              // reuse pivot order; falls back to factor()
//     lu.solve(b);
//     lu.tsolve(c);                 // c <- A^-T c (A^-H c with conjugate = true)
//
// refactor() falls back to a full factor() when there is no previous
// factorization, when klu_refactor reports a problem (a zero pivot in the old
// order), or when the pivot-ratio estimate rcond() of the refactored matrix is
// below kRefactorRcondFloor (1e-14): KLU's refactor never re-pivots, so a
// collapsed pivot ratio means the old order no longer suits the new values.
//
// Errors are reported as mna::Error:
//   NETLIST_FATAL      malformed input (bad sizes, bad CSC structure, duplicate
//                      entries, calls out of order, problem too large)
//   JACOBIAN_SINGULAR  the matrix is singular (KLU found a zero pivot)
// and std::bad_alloc when KLU runs out of memory. Both solvers share one
// KLU-status-to-error mapping.
//
// A solver is not safe for concurrent use; separate instances are independent
// and may be used from different threads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mna/linear/linear_solver.hpp"

namespace mna::linear {

template <class Scalar>
class KluSolverT final : public LinearSolver<Scalar> {
public:
    KluSolverT();
    ~KluSolverT() override;
    KluSolverT(KluSolverT&&) noexcept;
    KluSolverT& operator=(KluSolverT&&) noexcept;

    /// Symbolic analysis (BTF + AMD ordering) of an n x n CSC pattern. The
    /// pattern is copied. Discards any previous analysis and factorization.
    void analyze(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                 std::size_t ai_size);
    void analyze(std::int32_t n, const std::vector<std::int32_t>& Ap, const std::vector<std::int32_t>& Ai);
    void analyze(const CscPattern& pattern);

    /// Numeric factorization with partial pivoting. `Ax` holds nnz() values in
    /// the order of the analyzed pattern.
    void factor(const Scalar* Ax, std::size_t ax_size) override;
    void factor(const std::vector<Scalar>& Ax);

    /// Numeric refactorization reusing the pivot order of the last factor(); see
    /// the file comment for the fallbacks. Throws JACOBIAN_SINGULAR only if the
    /// fallback factor() fails too.
    FactorPath refactor(const Scalar* Ax, std::size_t ax_size) override;
    FactorPath refactor(const std::vector<Scalar>& Ax);

    /// Solves A X = B in place. `b` is column-major n x nrhs.
    void solve(Scalar* b, std::size_t b_size, std::int32_t nrhs) override;
    void solve(std::vector<Scalar>& b, std::int32_t nrhs = 1);

    /// Solves A^T X = B (or A^H X = B with `conjugate` for complex values) in
    /// place with klu_tsolve / klu_z_tsolve. `b` is column-major n x nrhs.
    void tsolve(Scalar* b, std::size_t b_size, std::int32_t nrhs, bool conjugate) override;
    void tsolve(std::vector<Scalar>& b, std::int32_t nrhs = 1, bool conjugate = false);

    /// Cheap reciprocal condition estimate min|u_ii| / max|u_ii| of the current
    /// factorization (klu_rcond / klu_z_rcond). 1.0 for n == 0.
    double rcond() override;

    /// Entries in the L and U factors (both counted with their diagonal),
    /// excluding the off-diagonal BTF blocks. 0 before factor() and for n == 0.
    std::int64_t nnz_lu() const noexcept override;
    /// Entries in the off-diagonal blocks of the BTF form (taken from A as is).
    std::int64_t nnz_offdiag() const noexcept;

    std::int32_t size() const noexcept override;
    std::int64_t nnz() const noexcept;
    bool analyzed() const noexcept;
    bool factored() const noexcept override;
    const char* kind() const noexcept override { return "klu"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using KluSolver = KluSolverT<double>;
using KluSolverZ = KluSolverT<Complex>;

extern template class KluSolverT<double>;
extern template class KluSolverT<Complex>;

}  // namespace mna::linear
