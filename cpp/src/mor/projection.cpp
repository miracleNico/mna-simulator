#include "mna/mor/projection.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <string>

#include "mna/error.hpp"

namespace mna::mor {
namespace {

void check(const linear::CscPattern& pattern, const double* values, const double* V, std::int32_t k, const char* fn) {
    if (k < 0) {
        throw Error(ErrorCode::NetlistFatal, std::string(fn) + ": negative basis width");
    }
    if (pattern.col_ptr.size() != static_cast<std::size_t>(pattern.n) + 1) {
        throw Error(ErrorCode::NetlistFatal, std::string(fn) + ": col_ptr must have n + 1 entries");
    }
    if ((pattern.nnz() > 0 && values == nullptr) || (k > 0 && pattern.n > 0 && V == nullptr)) {
        throw Error(ErrorCode::NetlistFatal, std::string(fn) + ": null input array");
    }
}

}  // namespace

std::vector<double> sparse_times_dense(const linear::CscPattern& pattern, const double* values, const double* V,
                                       std::int32_t k) {
    check(pattern, values, V, k, "sparse_times_dense");
    const auto n = static_cast<std::size_t>(pattern.n);
    std::vector<double> W(n * static_cast<std::size_t>(k), 0.0);
    for (std::size_t q = 0; q < static_cast<std::size_t>(k); ++q) {
        const double* v = V + q * n;
        double* w = W.data() + q * n;
        for (std::size_t col = 0; col < n; ++col) {
            const double vj = v[col];
            for (auto p = static_cast<std::size_t>(pattern.col_ptr[col]);
                 p < static_cast<std::size_t>(pattern.col_ptr[col + 1]); ++p) {
                w[static_cast<std::size_t>(pattern.row_idx[p])] += values[p] * vj;
            }
        }
    }
    return W;
}

std::vector<double> project_matrix(const linear::CscPattern& pattern, const double* values, const double* V,
                                   std::int32_t k) {
    const std::vector<double> W = sparse_times_dense(pattern, values, V, k);
    std::vector<double> out(static_cast<std::size_t>(k) * static_cast<std::size_t>(k), 0.0);
    if (k == 0 || pattern.n == 0) {
        return out;
    }
    const Eigen::Map<const Eigen::MatrixXd> Vm(V, pattern.n, k);
    const Eigen::Map<const Eigen::MatrixXd> Wm(W.data(), pattern.n, k);
    Eigen::Map<Eigen::MatrixXd> R(out.data(), k, k);
    R.noalias() = Vm.transpose() * Wm;
    return out;
}

void project_vector(std::int32_t n, std::int32_t k, const double* V, const double* x, double* out) {
    if (k <= 0) {
        return;
    }
    Eigen::Map<Eigen::VectorXd> o(out, k);
    if (n <= 0) {
        o.setZero();
        return;
    }
    const Eigen::Map<const Eigen::MatrixXd> Vm(V, n, k);
    const Eigen::Map<const Eigen::VectorXd> xv(x, n);
    o.noalias() = Vm.transpose() * xv;
}

void lift_vector(std::int32_t n, std::int32_t k, const double* V, const double* z, double* out) {
    if (n <= 0) {
        return;
    }
    Eigen::Map<Eigen::VectorXd> o(out, n);
    if (k <= 0) {
        o.setZero();
        return;
    }
    const Eigen::Map<const Eigen::MatrixXd> Vm(V, n, k);
    const Eigen::Map<const Eigen::VectorXd> zv(z, k);
    o.noalias() = Vm * zv;
}

void sparse_matvec(const linear::CscPattern& pattern, const double* values, const double* x, double* y) {
    const auto n = static_cast<std::size_t>(pattern.n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = 0.0;
    }
    for (std::size_t col = 0; col < n; ++col) {
        const double xj = x[col];
        for (auto p = static_cast<std::size_t>(pattern.col_ptr[col]);
             p < static_cast<std::size_t>(pattern.col_ptr[col + 1]); ++p) {
            y[static_cast<std::size_t>(pattern.row_idx[p])] += values[p] * xj;
        }
    }
}

DenseUnionOperands dense_union_operands(std::int32_t n, const double* g, std::int64_t g_row_stride,
                                        std::int64_t g_col_stride, const double* c, std::int64_t c_row_stride,
                                        std::int64_t c_col_stride) {
    if (n < 0) {
        throw Error(ErrorCode::NetlistFatal, "dense_union_operands: negative dimension");
    }
    if (n > 0 && (g == nullptr || c == nullptr)) {
        throw Error(ErrorCode::NetlistFatal, "dense_union_operands: null input array");
    }
    // One pass over both arrays in storage order collects the entries (rows increasing), a
    // counting sort by column then gives CSC with sorted rows. `x != 0.0` is true for NaN too.
    struct Entry {
        std::int32_t row, col;
        double g, c;
    };
    std::vector<Entry> entries;
    const auto un = static_cast<std::size_t>(n);
    std::vector<std::int64_t> counts(un + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const double* g_row = g + i * g_row_stride;
        const double* c_row = c + i * c_row_stride;
        for (std::int64_t j = 0; j < n; ++j) {
            const double gv = g_row[j * g_col_stride];
            const double cv = c_row[j * c_col_stride];
            if (gv != 0.0 || cv != 0.0) {
                entries.push_back({static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), gv, cv});
                ++counts[static_cast<std::size_t>(j) + 1];
            }
        }
    }
    if (entries.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw Error(ErrorCode::NetlistFatal, "dense_union_operands: too many entries for 32-bit indices");
    }
    for (std::size_t j = 0; j < un; ++j) {
        counts[j + 1] += counts[j];
    }
    DenseUnionOperands out;
    out.pattern.n = n;
    out.pattern.col_ptr.resize(un + 1);
    for (std::size_t j = 0; j <= un; ++j) {
        out.pattern.col_ptr[j] = static_cast<std::int32_t>(counts[j]);
    }
    const std::size_t nnz = entries.size();
    out.pattern.row_idx.resize(nnz);
    out.g.resize(nnz);
    out.c.resize(nnz);
    for (const Entry& e : entries) {
        const auto slot = static_cast<std::size_t>(counts[static_cast<std::size_t>(e.col)]++);
        out.pattern.row_idx[slot] = e.row;
        out.g[slot] = e.g;
        out.c[slot] = e.c;
    }
    return out;
}

}  // namespace mna::mor
