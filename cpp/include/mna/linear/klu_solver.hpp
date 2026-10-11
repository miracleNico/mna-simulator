// RAII wrapper over SuiteSparse KLU (real, 32-bit indices) for square CSC
// matrices with a fixed sparsity pattern.
//
// Typical Newton loop:
//     KluSolver lu;
//     lu.analyze(n, Ap, Ai);        // once per sparsity pattern
//     lu.factor(Ax);                // first numeric factorization (pivoting)
//     lu.solve(b);                  // b <- A^-1 b
//     ... values change, pattern does not ...
//     lu.refactor(Ax);              // reuse pivot order; falls back to factor()
//     lu.solve(b);
//
// Errors are reported as mna::Error:
//   NETLIST_FATAL      malformed input (bad sizes, bad CSC structure, duplicate
//                      entries, calls out of order, problem too large)
//   JACOBIAN_SINGULAR  the matrix is singular (KLU found a zero pivot)
// and std::bad_alloc when KLU runs out of memory.
//
// A KluSolver is not safe for concurrent use; separate instances are
// independent and may be used from different threads.
//
// The complex solver for AC analysis (klu_z_*) will be a sibling class with the
// same interface over std::complex<double> values, reusing CscPattern and the
// status mapping in klu_solver.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mna::linear {

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
    Refactor,    // klu_refactor with the previous pivot order succeeded
    FullFactor,  // KLU reported a problem (or no prior factorization): klu_factor ran
};

class KluSolver {
public:
    KluSolver();
    ~KluSolver();
    KluSolver(KluSolver&&) noexcept;
    KluSolver& operator=(KluSolver&&) noexcept;
    KluSolver(const KluSolver&) = delete;
    KluSolver& operator=(const KluSolver&) = delete;

    /// Symbolic analysis (BTF + AMD ordering) of an n x n CSC pattern. The
    /// pattern is copied. Discards any previous analysis and factorization.
    void analyze(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                 std::size_t ai_size);
    void analyze(std::int32_t n, const std::vector<std::int32_t>& Ap, const std::vector<std::int32_t>& Ai);

    /// Numeric factorization with partial pivoting. `Ax` holds nnz() values in
    /// the order of the analyzed pattern.
    void factor(const double* Ax, std::size_t ax_size);
    void factor(const std::vector<double>& Ax);

    /// Numeric refactorization reusing the pivot order of the last factor().
    /// Falls back to factor() when there is no previous factorization or KLU
    /// reports a problem (e.g. a zero pivot in the old order). Throws
    /// JACOBIAN_SINGULAR only if the fallback factor() fails too.
    FactorPath refactor(const double* Ax, std::size_t ax_size);
    FactorPath refactor(const std::vector<double>& Ax);

    /// Solves A X = B in place. `b` is column-major n x nrhs.
    void solve(double* b, std::size_t b_size, std::int32_t nrhs = 1);
    void solve(std::vector<double>& b, std::int32_t nrhs = 1);

    /// Cheap reciprocal condition estimate min|u_ii| / max|u_ii| of the current
    /// factorization (klu_rcond). 1.0 for n == 0.
    double rcond();

    /// Entries in the L and U factors (both counted with their diagonal),
    /// excluding the off-diagonal BTF blocks. 0 before factor() and for n == 0.
    std::int64_t nnz_lu() const noexcept;
    /// Entries in the off-diagonal blocks of the BTF form (taken from A as is).
    std::int64_t nnz_offdiag() const noexcept;

    std::int32_t size() const noexcept;
    std::int64_t nnz() const noexcept;
    bool analyzed() const noexcept;
    bool factored() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mna::linear
