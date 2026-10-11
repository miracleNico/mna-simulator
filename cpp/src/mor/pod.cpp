#include "mna/mor/pod.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cstddef>
#include <string>

#include "mna/error.hpp"

namespace mna::mor {

PodResult pod_basis(std::int32_t n, std::int64_t T, const double* snapshots, std::int64_t order) {
    if (n <= 0) {
        throw Error(ErrorCode::NetlistFatal, "mor_pod_basis: the snapshot matrix has no rows");
    }
    if (T < 0) {
        throw Error(ErrorCode::NetlistFatal, "mor_pod_basis: negative snapshot count");
    }
    if (T > 0 && snapshots == nullptr) {
        throw Error(ErrorCode::NetlistFatal, "mor_pod_basis: null snapshot array");
    }
    PodResult result;
    result.n = n;
    const auto un = static_cast<std::size_t>(n);
    if (T == 0) {
        result.basis.assign(un, 0.0);
        result.basis[0] = 1.0;
        result.columns = 1;
        return result;
    }
    const Eigen::Map<const Eigen::MatrixXd> S(snapshots, n, static_cast<Eigen::Index>(T));
    if (!S.allFinite()) {
        throw Error(ErrorCode::JacobianSingular, "SVD did not converge");
    }
    const Eigen::Index thin = (std::min)(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(T));
    Eigen::MatrixXd U;
    Eigen::VectorXd s;
    if (S.cwiseAbs().maxCoeff() == 0.0) {
        // LAPACK's SVD of a zero matrix: U = leading identity columns, s = 0.
        U = Eigen::MatrixXd::Identity(n, thin);
        s = Eigen::VectorXd::Zero(thin);
    } else {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU);
        if (svd.info() != Eigen::Success) {
            throw Error(ErrorCode::JacobianSingular, "SVD did not converge");
        }
        U = svd.matrixU();
        s = svd.singularValues();
    }
    result.singular_values.assign(s.data(), s.data() + s.size());
    if (s.size() == 0) {
        // np.eye(n, min(order, n)): unreachable for a non-empty matrix, kept for parity.
        const auto cols = static_cast<std::int32_t>((std::min)(order, static_cast<std::int64_t>(n)));
        result.columns = (std::max)(cols, 0);
        result.basis.assign(un * static_cast<std::size_t>(result.columns), 0.0);
        for (std::int32_t c = 0; c < result.columns; ++c) {
            result.basis[static_cast<std::size_t>(c) * un + static_cast<std::size_t>(c)] = 1.0;
        }
        return result;
    }
    const double threshold = (s[0] > 1.0 ? s[0] : 1.0) * 1e-12;  // max(singular_values[0], 1.0) * 1e-12
    std::int32_t rank = 0;
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        rank += s[i] > threshold ? 1 : 0;
    }
    result.rank = rank;
    const std::int64_t clipped =
        (std::min)({order, static_cast<std::int64_t>(rank), static_cast<std::int64_t>(U.cols())});
    result.columns = static_cast<std::int32_t>((std::max)(std::int64_t{1}, clipped));
    result.basis.assign(U.data(), U.data() + un * static_cast<std::size_t>(result.columns));
    return result;
}

}  // namespace mna::mor
