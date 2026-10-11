// Dense direct solver over Eigen's PartialPivLU (real MatrixXd and complex
// MatrixXcd) behind the LinearSolver interface.
//
// Two ways to feed it:
//   DenseSolver<double> lu(pattern);   // CSC mode: factor() takes values in the
//                                      // pattern's order and scatters them
//   DenseSolver<double> lu(n);         // dense mode: factor() takes a
//                                      // column-major n x n array (n * n values)
//
// Singularity: PartialPivLU never fails, it just produces infinities. The
// factorization is reported as JACOBIAN_SINGULAR (the code the KLU solvers and
// the Python core use for a singular system) when any |u_ii| is below
// eps * max|u_ij| (eps = DBL_EPSILON, max over the whole upper factor), when
// U is entirely zero, or when the factors contain a non-finite value.
//
// refactor() has no pivot order to reuse: it always runs a full factorization
// and reports FactorPath::FullFactor. rcond() is the pivot ratio
// min|u_ii| / max|u_ii| (the same estimate as klu_rcond). nnz_lu() counts the
// dense factors the way KLU counts its own (L and U with their diagonals):
// n * (n + 1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mna/linear/linear_solver.hpp"

namespace mna::linear {

template <class Scalar>
class DenseSolver final : public LinearSolver<Scalar> {
public:
    /// Dense mode for an n x n system (n >= 0).
    explicit DenseSolver(std::int32_t n);
    /// CSC mode for a validated pattern.
    explicit DenseSolver(CscPattern pattern);
    ~DenseSolver() override;
    DenseSolver(DenseSolver&&) noexcept;
    DenseSolver& operator=(DenseSolver&&) noexcept;

    void factor(const Scalar* values, std::size_t size) override;
    FactorPath refactor(const Scalar* values, std::size_t size) override;
    void solve(Scalar* rhs, std::size_t size, std::int32_t nrhs) override;
    void tsolve(Scalar* rhs, std::size_t size, std::int32_t nrhs, bool conjugate) override;
    double rcond() override;
    std::int64_t nnz_lu() const noexcept override;
    std::int32_t size() const noexcept override;
    bool factored() const noexcept override;
    const char* kind() const noexcept override { return "dense"; }

    /// True in CSC mode (constructed from a pattern).
    bool has_pattern() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

extern template class DenseSolver<double>;
extern template class DenseSolver<Complex>;

}  // namespace mna::linear
