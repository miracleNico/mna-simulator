// Rational Krylov basis (mna/mor/krylov_basis.hpp) and projections (mna/mor/projection.hpp):
// acceptance rules against a dense transcription of the Python loop, the least-squares fallback
// on a singular shift, the e_0 fallback, thread determinism and input validation.
#include <doctest/doctest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/linear_solver.hpp"
#include "mna/mor/krylov_basis.hpp"
#include "mna/mor/projection.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::linear::CscPattern;
using mna::mor::Complex;
using mna::mor::KrylovBasisResult;
using mna::mor::Shift;
using mna::mor::SparseOperands;

namespace {

using Matrix = Eigen::MatrixXd;
using CMatrix = Eigen::MatrixXcd;

/// G and C of a dense pair in their union CSC pattern.
struct Operands {
    CscPattern pattern;
    std::vector<double> g, c;
    SparseOperands view() const { return {&pattern, g.data(), c.data()}; }
};

Operands to_operands(const Matrix& G, const Matrix& C) {
    const auto n = static_cast<std::int32_t>(G.rows());
    const auto un = static_cast<std::size_t>(n);
    Operands ops;
    auto u = mna::mor::dense_union_operands(n, G.data(), 1, n, C.data(), 1, n);  // column-major input
    ops.pattern = u.pattern;
    ops.g = u.g;
    ops.c = u.c;
    CHECK(ops.pattern.col_ptr.size() == un + 1);
    return ops;
}

/// RC ladder: n nodes, R between neighbours, R_load at the last node, C at every node.
void rc_ladder(int n, Matrix& G, Matrix& C) {
    G = Matrix::Zero(n, n);
    C = Matrix::Zero(n, n);
    for (int i = 0; i + 1 < n; ++i) {
        const double g = 1.0 / (100.0 + 10.0 * i);
        G(i, i) += g;
        G(i + 1, i + 1) += g;
        G(i, i + 1) -= g;
        G(i + 1, i) -= g;
    }
    G(n - 1, n - 1) += 1.0 / 50.0;
    G(0, 0) += 1e-3;
    for (int i = 0; i < n; ++i) {
        C(i, i) = 1e-9 * (1.0 + 0.25 * i);
    }
}

// Dense transcription of mor/linear.py::build_linear_krylov_basis (np.linalg.solve -> Eigen LU).
struct DenseReference {
    Matrix basis;
    int target = 1;

    int orthogonalize(const Matrix& block) {
        int accepted = 0;
        for (Eigen::Index col = 0; col < block.cols(); ++col) {
            if (basis.cols() >= target) {
                break;
            }
            Eigen::VectorXd v = block.col(col);
            if (basis.cols() > 0) {
                const Eigen::VectorXd coefficients = basis.transpose() * v;
                v -= basis * coefficients;
            }
            const double norm = v.norm();
            if (norm <= 1e-14) {
                continue;
            }
            basis.conservativeResize(v.size(), basis.cols() + 1);
            basis.col(basis.cols() - 1) = v / norm;
            ++accepted;
        }
        return accepted;
    }

    static Matrix split(const CMatrix& x) {
        if (x.imag().norm() > 0.0) {
            Matrix out(x.rows(), 2 * x.cols());
            out << x.real(), x.imag();
            return out;
        }
        return x.real();
    }
};

Matrix reference_basis(const Matrix& G, const Matrix& C, const Matrix& inputs, const Matrix& outputs,
                       const std::vector<Shift>& shifts, int order) {
    DenseReference ref;
    const auto n = static_cast<int>(G.rows());
    ref.target = std::max(1, std::min(order, n));
    ref.basis = Matrix(n, 0);
    if (inputs.cols() > 0) {
        ref.orthogonalize(inputs);
    }
    ref.orthogonalize(outputs);
    struct State {
        bool adjoint;
        CMatrix system;
        Matrix block;
    };
    std::vector<State> states;
    auto solve = [](const CMatrix& A, const Matrix& rhs) -> Matrix {
        return DenseReference::split(A.partialPivLu().solve(rhs.cast<Complex>()));
    };
    for (const Shift& shift : shifts) {
        const CMatrix A = G.cast<Complex>() + shift.value * C.cast<Complex>();
        if (inputs.cols() > 0) {
            const int accepted = ref.orthogonalize(solve(A, inputs));
            if (accepted > 0) {
                states.push_back({false, A, ref.basis.rightCols(accepted)});
            }
        }
        const int accepted_adj = ref.orthogonalize(solve(A.transpose(), outputs));
        if (accepted_adj > 0) {
            states.push_back({true, A, ref.basis.rightCols(accepted_adj)});
        }
        if (ref.basis.cols() >= ref.target) {
            return ref.basis;
        }
    }
    Eigen::Index previous = -1;
    while (ref.basis.cols() < ref.target && ref.basis.cols() != previous) {
        previous = ref.basis.cols();
        for (State& state : states) {
            const Matrix rhs = state.adjoint ? Matrix(-(C.transpose() * state.block)) : Matrix(-(C * state.block));
            const int accepted =
                ref.orthogonalize(solve(state.adjoint ? CMatrix(state.system.transpose()) : state.system, rhs));
            if (accepted > 0) {
                state.block = ref.basis.rightCols(accepted);
            }
            if (ref.basis.cols() >= ref.target) {
                break;
            }
        }
    }
    return ref.basis;
}

/// The rc_ladder operands built sparse (for sizes where dense copies would be wasteful).
Operands rc_ladder_sparse(int n) {
    Operands ops;
    ops.pattern.n = n;
    ops.pattern.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = std::max(0, j - 1); i <= std::min(n - 1, j + 1); ++i) {
            double g = 0.0;
            if (i == j) {
                if (j > 0) {
                    g += 1.0 / (100.0 + 10.0 * (j - 1));
                }
                if (j + 1 < n) {
                    g += 1.0 / (100.0 + 10.0 * j);
                }
                g += (j == n - 1 ? 1.0 / 50.0 : 0.0) + (j == 0 ? 1e-3 : 0.0);
            } else {
                g = -1.0 / (100.0 + 10.0 * std::min(i, j));
            }
            ops.pattern.row_idx.push_back(i);
            ops.g.push_back(g);
            ops.c.push_back(i == j ? 1e-9 * (1.0 + 0.25 * j) : 0.0);
        }
        ops.pattern.col_ptr.push_back(static_cast<std::int32_t>(ops.pattern.row_idx.size()));
    }
    return ops;
}

Matrix as_matrix(const KrylovBasisResult& r) { return Eigen::Map<const Matrix>(r.basis.data(), r.n, r.columns); }

/// Sine of the largest principal angle between two orthonormal column spaces.
double subspace_distance(const Matrix& a, const Matrix& b) {
    const Matrix residual = b - a * (a.transpose() * b);
    Eigen::JacobiSVD<Matrix> svd(residual);
    return svd.singularValues().size() > 0 ? svd.singularValues()(0) : 0.0;
}

KrylovBasisResult run(const Operands& ops, const Matrix& inputs, const Matrix& outputs,
                      const std::vector<Shift>& shifts, std::int64_t order, int threads = 1) {
    return mna::mor::krylov_basis(ops.view(), inputs.size() ? inputs.data() : nullptr,
                                  static_cast<std::int32_t>(inputs.cols()), outputs.data(),
                                  static_cast<std::int32_t>(outputs.cols()), shifts, order, threads);
}

Matrix unit(int n, int i) {
    Matrix e = Matrix::Zero(n, 1);
    e(i, 0) = 1.0;
    return e;
}

}  // namespace

TEST_CASE("orthogonalize_block: accepts independent columns, drops dependent ones, stops when full") {
    const std::int32_t n = 3;
    std::vector<double> basis;
    std::int32_t cols = 0;
    // [e0, 2*e0, e0 + e1, e2]: the duplicate is dropped (exact zero residual).
    const std::vector<double> block = {1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1};
    CHECK(mna::mor::orthogonalize_block(basis, n, cols, block.data(), 4, 2) == 2);
    CHECK(cols == 2);
    REQUIRE(basis.size() == 6);
    CHECK(basis[0] == 1.0);
    CHECK(basis[3] == doctest::Approx(0.0));
    CHECK(basis[4] == doctest::Approx(1.0));
    // Full: nothing more is accepted.
    CHECK(mna::mor::orthogonalize_block(basis, n, cols, block.data() + 9, 1, 2) == 0);
    CHECK(cols == 2);
    // A residual of norm 1e-15 (<= 1e-14) is rejected, 1e-13 accepted.
    const std::vector<double> tiny = {0, 0, 1e-15};
    CHECK(mna::mor::orthogonalize_block(basis, n, cols, tiny.data(), 1, 3) == 0);
    const std::vector<double> small = {0, 0, 1e-13};
    CHECK(mna::mor::orthogonalize_block(basis, n, cols, small.data(), 1, 3) == 1);
    CHECK(basis[8] == doctest::Approx(1.0));
}

TEST_CASE("krylov_basis matches the dense transcription of the Python loop (real and complex shifts)") {
    Matrix G, C;
    rc_ladder(12, G, C);
    const Operands ops = to_operands(G, C);
    const Matrix inputs = unit(12, 0);
    const Matrix outputs = unit(12, 11);
    const std::vector<Shift> transient = {{Complex(0.0, 0.0), false}, {Complex(1e9, 0.0), false}};
    const std::vector<Shift> ac = {{Complex(0.0, 0.0), false}, {Complex(0.0, 2e6), true}, {Complex(0.0, 6e7), true}};
    for (int order : {1, 3, 5, 8, 12, 40}) {
        for (const auto* shifts : {&transient, &ac}) {
            const KrylovBasisResult native = run(ops, inputs, outputs, *shifts, order);
            const Matrix expected = reference_basis(G, C, inputs, outputs, *shifts, order);
            CAPTURE(order);
            REQUIRE(native.columns == expected.cols());
            const Matrix V = as_matrix(native);
            CHECK((V.transpose() * V - Matrix::Identity(V.cols(), V.cols())).cwiseAbs().maxCoeff() < 1e-8);
            CHECK(subspace_distance(expected, V) < 1e-8);
            CHECK(native.lstsq_shifts == 0);
        }
    }
}

TEST_CASE("krylov_basis without inputs and with several outputs; order clipped to n") {
    Matrix G, C;
    rc_ladder(6, G, C);
    const Operands ops = to_operands(G, C);
    Matrix outputs(6, 2);
    outputs << unit(6, 2), unit(6, 5);
    const std::vector<Shift> shifts = {{Complex(0.0, 0.0), false}, {Complex(0.0, 1e7), true}};
    const KrylovBasisResult r = run(ops, Matrix(6, 0), outputs, shifts, 100);
    CHECK(r.columns == 6);
    const Matrix expected = reference_basis(G, C, Matrix(6, 0), outputs, shifts, 100);
    CHECK(expected.cols() == 6);
    CHECK(subspace_distance(expected, as_matrix(r)) < 1e-8);
}

TEST_CASE("krylov_basis: a singular shifted system falls back to least squares") {
    // Node 2 hangs on a capacitor only: G is singular at shift 0, regular at 1/h.
    Matrix G = Matrix::Zero(3, 3);
    Matrix C = Matrix::Zero(3, 3);
    G(0, 0) = 2e-3;
    G(0, 1) = G(1, 0) = -1e-3;
    G(1, 1) = 2e-3;
    C(1, 1) = 1e-9;
    C(2, 2) = 1e-9;
    C(1, 2) = C(2, 1) = -0.5e-9;
    const Operands ops = to_operands(G, C);
    const std::vector<Shift> shifts = {{Complex(0.0, 0.0), false}, {Complex(1e9, 0.0), false}};
    const KrylovBasisResult r = run(ops, unit(3, 0), unit(3, 2), shifts, 3);
    CHECK(r.lstsq_shifts == 1);
    CHECK(r.columns == 3);
    const Matrix V = as_matrix(r);
    CHECK((V.transpose() * V - Matrix::Identity(3, 3)).cwiseAbs().maxCoeff() < 1e-8);
    // The least-squares solution of G x = e0 is the minimum-norm one: zero at the floating node.
    const KrylovBasisResult only_zero = run(ops, unit(3, 0), unit(3, 0), {{Complex(0.0, 0.0), false}}, 2);
    CHECK(only_zero.lstsq_shifts == 1);
    REQUIRE(only_zero.columns == 2);
    CHECK(std::abs(only_zero.basis[3 + 2]) < 1e-12);
}

TEST_CASE("krylov_basis: nothing accepted gives the unit vector e_0") {
    Matrix G, C;
    rc_ladder(4, G, C);
    const Operands ops = to_operands(G, C);
    const KrylovBasisResult r = run(ops, Matrix(4, 0), Matrix::Zero(4, 1), {{Complex(0.0, 0.0), false}}, 3);
    REQUIRE(r.columns == 1);
    CHECK(r.basis == std::vector<double>{1.0, 0.0, 0.0, 0.0});
}

TEST_CASE("krylov_basis: shift factorizations on several threads give the same basis") {
    // n >= kMorParallelMinUnknowns so the shifts really are factored in parallel.
    const int n = mna::mor::kMorParallelMinUnknowns + 16;
    const Operands ops = rc_ladder_sparse(n);
    {
        // The sparse builder is the dense ladder.
        Matrix G, C;
        rc_ladder(8, G, C);
        const Operands small = rc_ladder_sparse(8);
        const Operands dense = to_operands(G, C);
        CHECK(small.pattern.row_idx == dense.pattern.row_idx);
        for (std::size_t k = 0; k < small.g.size(); ++k) {
            CHECK(small.g[k] == doctest::Approx(dense.g[k]).epsilon(1e-14));
        }
    }
    const std::vector<Shift> shifts = {
        {Complex(0.0, 0.0), false}, {Complex(1e9, 0.0), false}, {Complex(0.0, 1e8), true}};
    const KrylovBasisResult one = run(ops, unit(n, 0), unit(n, n - 1), shifts, 10, 1);
    const KrylovBasisResult four = run(ops, unit(n, 0), unit(n, n - 1), shifts, 10, 4);
    CHECK(one.threads_used == 1);
    CHECK(one.columns == four.columns);
    CHECK(std::memcmp(one.basis.data(), four.basis.data(), one.basis.size() * sizeof(double)) == 0);
}

TEST_CASE("krylov_basis rejects malformed input") {
    Matrix G, C;
    rc_ladder(3, G, C);
    const Operands ops = to_operands(G, C);
    const Matrix out = unit(3, 0);
    try {
        mna::mor::krylov_basis(SparseOperands{}, nullptr, 0, out.data(), 1, {}, 2, 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
    try {
        mna::mor::krylov_basis(ops.view(), nullptr, 1, out.data(), 1, {}, 2, 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
    CscPattern empty;
    empty.n = 0;
    empty.col_ptr = {0};
    try {
        mna::mor::krylov_basis(SparseOperands{&empty, nullptr, nullptr}, nullptr, 0, nullptr, 0, {}, 2, 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
}

TEST_CASE("projections: V^T A V, V^T x, V z and the dense union pattern") {
    Matrix G, C;
    rc_ladder(5, G, C);
    C(0, 3) = 2e-9;  // an entry only C has
    const Operands ops = to_operands(G, C);
    CHECK(ops.pattern.nnz() == 14);
    Matrix V = Matrix::Random(5, 2);
    const std::vector<double> gr = mna::mor::project_matrix(ops.pattern, ops.g.data(), V.data(), 2);
    const Matrix expected = V.transpose() * G * V;
    for (int e = 0; e < 4; ++e) {
        CHECK(gr[static_cast<std::size_t>(e)] == doctest::Approx(expected.data()[e]).epsilon(1e-12));
    }
    const Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);
    std::vector<double> z(2), back(5), y(5);
    mna::mor::project_vector(5, 2, V.data(), x.data(), z.data());
    CHECK(z[1] == doctest::Approx((V.transpose() * x)(1)));
    mna::mor::lift_vector(5, 2, V.data(), z.data(), back.data());
    CHECK(back[4] == doctest::Approx((V * V.transpose() * x)(4)));
    mna::mor::sparse_matvec(ops.pattern, ops.c.data(), x.data(), y.data());
    CHECK(y[0] == doctest::Approx((C * x)(0)));
}
