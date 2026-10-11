// Galerkin projections of the model-order-reduction kernels with a dense
// orthonormal basis V (column-major n x k) and sparse operators (values in a
// square CSC pattern), without ever forming a dense n x n matrix:
//   project_matrix:  V^T A V   (A V as a sparse-dense product, then V^T (A V))
//   project_vector:  V^T x
//   lift_vector:     V z
// Every result is column-major. Errors: NETLIST_FATAL for malformed input.
#pragma once

#include <cstdint>
#include <vector>

#include "mna/linear/linear_solver.hpp"

namespace mna::mor {

/// A V for a CSC matrix (values in pattern order): column-major n x k.
std::vector<double> sparse_times_dense(const linear::CscPattern& pattern, const double* values, const double* V,
                                       std::int32_t k);

/// V^T A V: column-major k x k.
std::vector<double> project_matrix(const linear::CscPattern& pattern, const double* values, const double* V,
                                   std::int32_t k);

/// out (k values) = V^T x for the column-major n x k basis V.
void project_vector(std::int32_t n, std::int32_t k, const double* V, const double* x, double* out);

/// out (n values) = V z.
void lift_vector(std::int32_t n, std::int32_t k, const double* V, const double* z, double* out);

/// y (n values) = A x for a CSC matrix.
void sparse_matvec(const linear::CscPattern& pattern, const double* values, const double* x, double* y);

/// G and C values in their union CSC pattern.
struct DenseUnionOperands {
    linear::CscPattern pattern;
    std::vector<double> g;
    std::vector<double> c;
};

/// The union pattern of two dense n x n matrices (an entry is stored when it is
/// nonzero, NaN included, in G or in C: scipy's csc_array(dense) rule applied to
/// both) with both value arrays. Element (i, j) of a matrix is at
/// data[i * row_stride + j * col_stride] (strides in elements).
DenseUnionOperands dense_union_operands(std::int32_t n, const double* g, std::int64_t g_row_stride,
                                        std::int64_t g_col_stride, const double* c, std::int64_t c_row_stride,
                                        std::int64_t c_col_stride);

}  // namespace mna::mor
