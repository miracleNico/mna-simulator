// Row-oriented (CSR) view of a square sparse matrix and its matrix-vector
// product, the operator of the Krylov solvers.
//
// The view is built from CSC input (the storage the direct solvers use). Each
// row lists its entries in increasing column order. spmv() runs rows in
// parallel only when n >= kParallelSpmvMinN and more than one thread is
// requested; rows are split into the item-count-only chunks of
// mna::parallel_for and each row is accumulated by exactly one thread in
// stored order, so the result is bitwise identical for every thread count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mna/linear/linear_solver.hpp"

namespace mna::linear {

/// Smallest dimension for which spmv() splits rows across threads.
inline constexpr std::int32_t kParallelSpmvMinN = 4096;

template <class Scalar>
struct CsrMatrix {
    std::int32_t n = 0;
    std::vector<std::int32_t> row_ptr{0};  // size n + 1
    std::vector<std::int32_t> col_idx;     // size nnz, increasing within each row
    std::vector<Scalar> values;            // size nnz

    std::int64_t nnz() const noexcept { return static_cast<std::int64_t>(col_idx.size()); }

    /// Transposes validated CSC storage into the row view. `values` holds
    /// pattern.nnz() entries; throws mna::Error(NETLIST_FATAL) otherwise.
    static CsrMatrix from_csc(const CscPattern& pattern, const Scalar* values, std::size_t size);
    /// Validates the CSC arrays (CscPattern::from_arrays) and transposes them.
    static CsrMatrix from_csc(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                              std::size_t ai_size, const Scalar* Ax, std::size_t ax_size);
};

/// y = A x (x and y must not alias). Returns the number of threads used.
template <class Scalar>
int spmv(const CsrMatrix<Scalar>& A, const Scalar* x, Scalar* y, int num_threads);

extern template struct CsrMatrix<double>;
extern template struct CsrMatrix<Complex>;
extern template int spmv<double>(const CsrMatrix<double>&, const double*, double*, int);
extern template int spmv<Complex>(const CsrMatrix<Complex>&, const Complex*, Complex*, int);

}  // namespace mna::linear
