#include "mna/linear/sparse.hpp"

#include <string>

#include "mna/error.hpp"
#include "mna/threads.hpp"

namespace mna::linear {

template <class Scalar>
CsrMatrix<Scalar> CsrMatrix<Scalar>::from_csc(const CscPattern& pattern, const Scalar* values, std::size_t size) {
    const auto nnz = static_cast<std::size_t>(pattern.nnz());
    if (size != nnz) {
        throw Error(ErrorCode::NetlistFatal, "CSR view: got " + std::to_string(size) + " values, the pattern has " +
                                                 std::to_string(nnz) + " entries");
    }
    if (nnz > 0 && values == nullptr) {
        throw Error(ErrorCode::NetlistFatal, "CSR view: null value array");
    }
    const auto un = static_cast<std::size_t>(pattern.n);
    CsrMatrix csr;
    csr.n = pattern.n;
    csr.row_ptr.assign(un + 1, 0);
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);
    for (std::size_t p = 0; p < nnz; ++p) {
        ++csr.row_ptr[static_cast<std::size_t>(pattern.row_idx[p]) + 1];
    }
    for (std::size_t i = 0; i < un; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }
    std::vector<std::int32_t> next(csr.row_ptr.begin(), csr.row_ptr.end() - 1);
    // Columns are visited in increasing order, so every row ends up column-sorted.
    for (std::size_t j = 0; j < un; ++j) {
        for (auto p = static_cast<std::size_t>(pattern.col_ptr[j]); p < static_cast<std::size_t>(pattern.col_ptr[j + 1]);
             ++p) {
            const auto slot = static_cast<std::size_t>(next[static_cast<std::size_t>(pattern.row_idx[p])]++);
            csr.col_idx[slot] = static_cast<std::int32_t>(j);
            csr.values[slot] = values[p];
        }
    }
    return csr;
}

template <class Scalar>
CsrMatrix<Scalar> CsrMatrix<Scalar>::from_csc(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size,
                                              const std::int32_t* Ai, std::size_t ai_size, const Scalar* Ax,
                                              std::size_t ax_size) {
    return from_csc(CscPattern::from_arrays(n, Ap, ap_size, Ai, ai_size), Ax, ax_size);
}

template <class Scalar>
int spmv(const CsrMatrix<Scalar>& A, const Scalar* x, Scalar* y, int num_threads) {
    const std::int32_t* rp = A.row_ptr.data();
    const std::int32_t* ci = A.col_idx.data();
    const Scalar* v = A.values.data();
    auto rows = [rp, ci, v, x, y](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            Scalar sum{};
            for (std::int32_t p = rp[i]; p < rp[i + 1]; ++p) {
                sum += v[p] * x[ci[p]];
            }
            y[i] = sum;
        }
    };
    if (A.n < kParallelSpmvMinN || num_threads <= 1) {
        rows(0, A.n);
        return 1;
    }
    return parallel_for(A.n, num_threads, rows);
}

template struct CsrMatrix<double>;
template struct CsrMatrix<Complex>;
template int spmv<double>(const CsrMatrix<double>&, const double*, double*, int);
template int spmv<Complex>(const CsrMatrix<Complex>&, const Complex*, Complex*, int);

}  // namespace mna::linear
