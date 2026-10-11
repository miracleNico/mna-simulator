// Proper orthogonal decomposition basis of mor/tpwl.py::_pod_basis.
//
//   U, s = thin SVD of the n x T snapshot matrix (Eigen::BDCSVD, thin U)
//   rank = #{ s_i > max(s_0, 1) * 1e-12 }
//   r    = max(1, min(order, rank, U.cols))
//   basis = U[:, :r]
//
// Branches kept from the Python function: an empty snapshot matrix (T == 0)
// gives the unit vector e_0 (n x 1); a matrix whose singular values are all
// zero gives the leading identity columns LAPACK returns for it (e_0 for
// r == 1). Column signs are implementation defined (compare subspaces).
//
// Errors: NETLIST_FATAL for n == 0 or malformed input; JACOBIAN_SINGULAR
// "SVD did not converge" for non-finite snapshots (numpy raises LinAlgError).
#pragma once

#include <cstdint>
#include <vector>

namespace mna::mor {

struct PodResult {
    std::int32_t n = 0;
    std::int32_t columns = 0;
    std::vector<double> basis;            ///< column-major n x columns
    std::vector<double> singular_values;  ///< all min(n, T) of them, descending
    std::int32_t rank = 0;                ///< numerical rank by the 1e-12 rule (before clipping)
};

/// snapshots: column-major n x T.
PodResult pod_basis(std::int32_t n, std::int64_t T, const double* snapshots, std::int64_t order);

}  // namespace mna::mor
