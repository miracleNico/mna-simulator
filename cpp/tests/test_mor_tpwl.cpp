// POD basis (mna/mor/pod.hpp) and TPWL training / simulation (mna/mor/tpwl.hpp).
#include <doctest/doctest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/mor/pod.hpp"
#include "mna/mor/tpwl.hpp"

using mna::Error;
using mna::ErrorCode;
using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;

namespace {

using Matrix = Eigen::MatrixXd;

void add(mna::circuit::TripletList& t, std::int64_t r, std::int64_t c, double v) {
    if (r >= 0 && c >= 0) {
        t.rows.push_back(r);
        t.cols.push_back(c);
        t.values.push_back(v);
    }
}

void stamp(mna::circuit::TripletList& t, std::int64_t j, std::int64_t k, double v) {
    add(t, j, j, v);
    add(t, k, k, v);
    add(t, j, k, -v);
    add(t, k, j, -v);
}

/// Three nodes: R chain, capacitors to ground, a diode from node 1 to node 2.
CircuitSpec diode_net() {
    CircuitSpec s;
    s.n = 3;
    stamp(s.g, 0, 1, 1e-2);
    stamp(s.g, 2, -1, 1e-3);
    stamp(s.c, 0, -1, 1e-9);
    stamp(s.c, 1, -1, 1e-9);
    stamp(s.c, 2, -1, 1e-9);
    s.diodes = {{1}, {2}, {1e-14}};
    return s;
}

std::vector<std::int64_t> sorted_unique(std::vector<std::int64_t> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

}  // namespace

TEST_CASE("pod_basis: rank rule, order clipping and the empty / zero branches") {
    // Rank-2 snapshots (n = 5, T = 6): columns are combinations of two fixed vectors.
    Matrix a(5, 1), b(5, 1);
    a << 1, 2, 3, 4, 5;
    b << 0, 1, 0, -1, 2;
    Matrix S(5, 6);
    for (int t = 0; t < 6; ++t) {
        S.col(t) = std::cos(0.3 * t) * a + std::sin(0.7 * t) * b;
    }
    const auto r2 = mna::mor::pod_basis(5, 6, S.data(), 10);
    CHECK(r2.rank == 2);
    CHECK(r2.columns == 2);
    const Eigen::Map<const Matrix> U(r2.basis.data(), 5, 2);
    CHECK((U.transpose() * U - Matrix::Identity(2, 2)).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((S - U * (U.transpose() * S)).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(mna::mor::pod_basis(5, 6, S.data(), 1).columns == 1);
    CHECK(mna::mor::pod_basis(5, 6, S.data(), 0).columns == 1);  // max(1, ...)

    // Singular values below max(s0, 1) * 1e-12 do not count: s0 < 1 uses the absolute 1e-12.
    Matrix tiny = 1e-14 * S;
    const auto r_tiny = mna::mor::pod_basis(5, 6, tiny.data(), 10);
    CHECK(r_tiny.rank == 0);
    CHECK(r_tiny.columns == 1);

    // T == 0: e_0.
    const auto empty = mna::mor::pod_basis(4, 0, nullptr, 3);
    CHECK(empty.columns == 1);
    CHECK(empty.basis == std::vector<double>{1.0, 0.0, 0.0, 0.0});
    // All-zero snapshots: LAPACK's U of a zero matrix starts with e_0.
    const std::vector<double> zeros(12, 0.0);
    const auto zero = mna::mor::pod_basis(4, 3, zeros.data(), 3);
    CHECK(zero.rank == 0);
    CHECK(zero.columns == 1);
    CHECK(zero.basis == std::vector<double>{1.0, 0.0, 0.0, 0.0});
    // Non-finite snapshots: numpy's SVD raises (LinAlgError) -> JACOBIAN_SINGULAR.
    std::vector<double> bad(12, 1.0);
    bad[5] = std::numeric_limits<double>::infinity();
    try {
        mna::mor::pod_basis(4, 3, bad.data(), 3);
        FAIL("expected JACOBIAN_SINGULAR");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::JacobianSingular);
    }
}

TEST_CASE("training_indexes: linspace grid plus every step at or above the median delta") {
    // n = 1, T = 10: deltas 1, 1, 1, 5, 1, 1, 1, 1, 9 -> median 1: every step counts.
    const std::vector<double> x = {0, 1, 2, 3, 8, 9, 10, 11, 12, 21};
    auto idx = mna::mor::training_indexes(1, 10, x.data(), 1, 1);
    CHECK(idx.size() == 10);
    // Deltas 0.1 x 7, 5, 9 (T = 10): median 0.1, all steps >= it.
    const std::vector<double> y = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 5.7, 14.7};
    idx = mna::mor::training_indexes(1, 10, y.data(), 1, 1);
    CHECK(idx.front() == 0);
    // Constant trajectory: no positive delta, only the linspace grid (max(4, min(16, k + p)) points).
    const std::vector<double> flat(10, 3.0);
    idx = mna::mor::training_indexes(1, 10, flat.data(), 2, 1);
    CHECK(idx == std::vector<std::int64_t>{0, 3, 6, 9});  // floor(j * 9 / 3)
    idx = mna::mor::training_indexes(1, 10, flat.data(), 4, 1);
    CHECK(idx == std::vector<std::int64_t>{0, 2, 4, 6, 9});  // floor(j * 2.25), last = 9
    // A jump in one step only: grid + that step (median of positive deltas = the jump itself).
    std::vector<double> jump(10, 0.0);
    for (int t = 6; t < 10; ++t) {
        jump[static_cast<std::size_t>(t)] = 1.0;
    }
    idx = mna::mor::training_indexes(1, 10, jump.data(), 1, 1);
    CHECK(idx == std::vector<std::int64_t>{0, 3, 6, 9});
    // Short trajectories: T = 1 -> [0]; T = 2 -> [0, 1] (no delta rule below T = 3); T = 0 -> [].
    CHECK(mna::mor::training_indexes(1, 1, x.data(), 3, 1) == std::vector<std::int64_t>{0});
    CHECK(mna::mor::training_indexes(1, 2, x.data(), 3, 1) == std::vector<std::int64_t>{0, 1});
    CHECK(mna::mor::training_indexes(1, 0, nullptr, 3, 1).empty());
    // Several rows: the delta is the Euclidean norm over rows (column-major n x T).
    const std::vector<double> two = {0, 0, 3, 4, 3, 4, 3, 5};  // deltas 5, 0, 1 -> median of {5, 1} = 3
    idx = mna::mor::training_indexes(2, 4, two.data(), 0, 0);
    CHECK(idx == sorted_unique({0, 1, 2, 3, 1}));
}

TEST_CASE("tpwl_local_models: f, J from the device evaluation, V^T J V and V^T (J x - f)") {
    const Circuit circuit(diode_net());
    const std::int32_t n = circuit.n();
    // Snapshots: 4 states.
    Matrix S(3, 4);
    S << 0.0, 0.5, 0.9, 1.0,  //
        0.0, 0.4, 0.7, 0.75,  //
        0.0, 0.01, 0.05, 0.1;
    Matrix V(3, 2);
    V << 1, 0, 0, 1, 0, 0;  // first two unknowns
    const std::vector<std::int64_t> indexes = {1, 3};
    const auto models = mna::mor::tpwl_local_models(circuit, 4, S.data(), indexes, 2, V.data(), 1);
    REQUIRE(models.count == 2);
    for (std::size_t m = 0; m < 2; ++m) {
        const Eigen::VectorXd x = S.col(indexes[m]);
        std::vector<double> f(3), jv(static_cast<std::size_t>(circuit.nnz())), dense(9);
        mna::assembly::evaluate(circuit, x.data(), 0.0, 1, f.data(), jv.data());
        circuit.scatter_dense(jv.data(), dense.data());  // row-major
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(dense.data());
        for (int i = 0; i < n; ++i) {
            CHECK(models.f[m * 3 + static_cast<std::size_t>(i)] == f[static_cast<std::size_t>(i)]);
        }
        CHECK(std::memcmp(models.jvals.data() + m * jv.size(), jv.data(), jv.size() * sizeof(double)) == 0);
        const Matrix jr = V.transpose() * J * V;
        const Eigen::VectorXd q = V.transpose() * (J * x - Eigen::Map<const Eigen::VectorXd>(f.data(), 3));
        for (int e = 0; e < 4; ++e) {
            CHECK(models.jr[m * 4 + static_cast<std::size_t>(e)] == doctest::Approx(jr.data()[e]).epsilon(1e-12));
        }
        for (int i = 0; i < 2; ++i) {
            CHECK(models.q[m * 2 + static_cast<std::size_t>(i)] == doctest::Approx(q[i]).epsilon(1e-12));
            CHECK(models.z[m * 2 + static_cast<std::size_t>(i)] == x[i]);
        }
    }
    try {
        mna::mor::tpwl_local_models(circuit, 4, S.data(), {4}, 2, V.data(), 1);
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
}

TEST_CASE("tpwl_simulate: nearest model per step and the reduced Backward-Euler update") {
    // k = n = 1: Gr = g, Cr = c, two models with different slopes; the model nearest to z wins.
    const double g = 1.0, c = 1.0;
    const std::vector<double> gr = {g}, cr = {c}, V = {1.0}, sr = {1.0};
    const std::vector<double> model_z = {0.0, 10.0};
    const std::vector<double> model_jr = {0.5, 2.0};
    const std::vector<double> model_q = {0.1, -0.2};
    const std::vector<double> x0 = {4.0}, b_dc = {1.0};
    const std::vector<double> grid = {0.0, 1.0, 2.0, 3.0};
    const auto r = mna::mor::tpwl_simulate(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), 2, model_z.data(),
                                           model_jr.data(), model_q.data(), x0.data(), grid, b_dc.data(), {});
    double z = 4.0;
    CHECK(r.values[0] == 4.0);
    for (std::size_t k = 1; k < grid.size(); ++k) {
        const std::size_t m = std::abs(z - 0.0) <= std::abs(z - 10.0) ? 0 : 1;
        CHECK(r.model[k] == static_cast<std::int64_t>(m));
        z = (1.0 + c * z + model_q[m]) / ((g + c) + model_jr[m]);
        CHECK(r.values[k] == doctest::Approx(z).epsilon(1e-14));
    }
    // A tie picks the first model; a NaN distance wins (numpy.argmin).
    const std::vector<double> tie_z = {3.0, 5.0};
    const std::vector<double> x_tie = {4.0};
    const auto tie =
        mna::mor::tpwl_simulate(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), 2, tie_z.data(), model_jr.data(),
                                model_q.data(), x_tie.data(), {0.0, 1.0}, b_dc.data(), {});
    CHECK(tie.model[1] == 0);
    const std::vector<double> nan_z = {4.0, std::numeric_limits<double>::quiet_NaN()};
    const auto with_nan =
        mna::mor::tpwl_simulate(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), 2, nan_z.data(), model_jr.data(),
                                model_q.data(), x_tie.data(), {0.0, 1.0}, b_dc.data(), {});
    CHECK(with_nan.model[1] == 1);
    try {
        mna::mor::tpwl_simulate(1, 1, gr.data(), cr.data(), V.data(), 1, sr.data(), 0, nullptr, nullptr, nullptr,
                                x0.data(), grid, b_dc.data(), {});
        FAIL("expected NETLIST_FATAL");
    } catch (const Error& e) {
        CHECK(e.code() == ErrorCode::NetlistFatal);
    }
}
