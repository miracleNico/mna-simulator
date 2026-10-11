// Deterministic two-phase device assembly.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/devices/mos_level1.hpp"
#include "mna/expr/program.hpp"

using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;

namespace {

void add_mos(CircuitSpec& s, std::int64_t d, std::int64_t g, std::int64_t src, double beta, int sign) {
    s.mos.drain.push_back(d);
    s.mos.gate.push_back(g);
    s.mos.source.push_back(src);
    s.mos.beta.push_back(beta);
    s.mos.vth.push_back(0.4);
    s.mos.lambda.push_back(0.02);
    s.mos.sign.push_back(sign);
}

void add_bjt(CircuitSpec& s, std::int64_t c, std::int64_t b, std::int64_t e, int sign) {
    s.bjt.collector.push_back(c);
    s.bjt.base.push_back(b);
    s.bjt.emitter.push_back(e);
    s.bjt.is.push_back(1e-15);
    s.bjt.bf.push_back(150.0);
    s.bjt.br.push_back(3.0);
    s.bjt.vaf.push_back(100.0);
    s.bjt.var.push_back(25.0);
    s.bjt.sign.push_back(sign);
}

/// Random mixed circuit with `devices` devices of each kind on n nodes.
CircuitSpec random_circuit(std::int64_t n, int devices, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int64_t> node(-1, n - 1);
    CircuitSpec s;
    s.n = n;
    for (int i = 0; i < devices; ++i) {
        add_mos(s, node(rng), node(rng), node(rng), 1e-3 * (1 + i % 3), i % 2 == 0 ? 1 : -1);
        add_bjt(s, node(rng), node(rng), node(rng), i % 3 == 0 ? -1 : 1);
        s.diodes.anode.push_back(node(rng));
        s.diodes.cathode.push_back(node(rng));
        s.diodes.is.push_back(1e-14);
    }
    // A few behavioral sources: x[a] * x[b].
    for (int i = 0; i < 3; ++i) {
        const std::int32_t a = static_cast<std::int32_t>(i % n);
        const std::int32_t b = static_cast<std::int32_t>((i + 1) % n);
        const std::int32_t ops[3] = {1, 1, 5};
        const std::int32_t args[3] = {a, b, 0};
        s.behavioral.node_p.push_back(node(rng));
        s.behavioral.node_n.push_back(node(rng));
        s.behavioral.programs.push_back(
            std::make_shared<mna::expr::Program>(ops, args, 3, std::vector<double>{}, static_cast<std::int32_t>(n), false));
    }
    return s;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

}  // namespace

TEST_CASE("assembly matches a direct per-device stamp on a small circuit") {
    CircuitSpec s;
    s.n = 3;
    add_mos(s, 0, 1, -1, 1e-3, 1);       // NMOS d=0 g=1 s=gnd
    add_mos(s, 0, 1, 2, 2e-3, -1);       // PMOS d=0 g=1 s=2
    add_bjt(s, 2, 1, -1, 1);             // NPN c=2 b=1 e=gnd
    s.diodes = {{0}, {2}, {1e-14}};
    const Circuit c(s);
    const double x[3] = {0.6, 1.1, 0.9};
    std::vector<double> f(3);
    std::vector<double> jv(static_cast<std::size_t>(c.nnz()));
    mna::assembly::evaluate(c, x, 0.0, 1, f.data(), jv.data());

    const auto m1 = mna::devices::mos_level1(1, 1e-3, 0.4, 0.02, 0.6, 1.1, 0.0);
    const auto m2 = mna::devices::mos_level1(-1, 2e-3, 0.4, 0.02, 0.6, 1.1, 0.9);
    const auto q = mna::devices::bjt_level1(1, 1e-15, 150.0, 3.0, 100.0, 25.0, 0.9, 1.1, 0.0);
    const auto d = mna::devices::diode(1e-14, 0.6 - 0.9);
    // Groups: (diode) + (MOS) + (BJT).
    CHECK(f[0] == (0.0 + d.current) + ((0.0 + m1.ids) + m2.ids));
    CHECK(f[1] == 0.0 + q.i[1]);
    CHECK(f[2] == ((0.0 + -d.current) + (0.0 + -m2.ids)) + q.i[0]);
    std::vector<double> dense(9);
    c.scatter_dense(jv.data(), dense.data());
    CHECK(dense[0 * 3 + 0] == d.conductance + (m1.d_drain + m2.d_drain));
    CHECK(dense[0 * 3 + 1] == m1.d_gate + m2.d_gate);
    CHECK(dense[0 * 3 + 2] == -d.conductance + m2.d_source);
    CHECK(dense[2 * 3 + 1] == -m2.d_gate + q.d[0][1]);
    CHECK(dense[1 * 3 + 1] == q.d[1][1]);
    CHECK(dense[1 * 3 + 0] == 0.0);

    std::vector<double> f2(3);
    std::vector<double> dense2(9);
    mna::assembly::evaluate_dense_jacobian(c, x, 0.0, 1, f2.data(), dense2.data());
    CHECK(bitwise_equal(f, f2));
    CHECK(bitwise_equal(dense, dense2));
}

TEST_CASE("behavioral sources stamp value and gradient with the p/n signs") {
    CircuitSpec s;
    s.n = 2;
    // I = x0 * x1 from node 0 to node 1
    const std::int32_t ops[3] = {1, 1, 5};
    const std::int32_t args[3] = {0, 1, 0};
    s.behavioral.node_p = {0};
    s.behavioral.node_n = {1};
    s.behavioral.programs = {std::make_shared<mna::expr::Program>(ops, args, 3, std::vector<double>{}, 2, false)};
    const Circuit c(s);
    const double x[2] = {3.0, 5.0};
    std::vector<double> f(2);
    std::vector<double> dense(4);
    mna::assembly::evaluate_dense_jacobian(c, x, 0.0, 1, f.data(), dense.data());
    CHECK(f[0] == 15.0);
    CHECK(f[1] == -15.0);
    CHECK(dense == std::vector<double>{5.0, 3.0, -5.0, -3.0});
}

TEST_CASE("assembly is bitwise identical for 1, 2, 3, 4 and 7 threads") {
    for (unsigned seed : {1u, 2u}) {
        const Circuit c(random_circuit(97, 900, seed));  // 2703 devices: several chunks per phase
        std::mt19937 rng(seed + 10);
        std::uniform_real_distribution<double> volt(-1.5, 1.5);
        std::vector<double> x(97);
        for (double& v : x) {
            v = volt(rng);
        }
        std::vector<double> f1(97);
        std::vector<double> j1(static_cast<std::size_t>(c.nnz()));
        mna::assembly::evaluate(c, x.data(), 0.0, 1, f1.data(), j1.data());
        for (int threads : {2, 3, 4, 7}) {
            CAPTURE(threads);
            std::vector<double> f(97);
            std::vector<double> j(static_cast<std::size_t>(c.nnz()));
            mna::assembly::evaluate(c, x.data(), 0.0, threads, f.data(), j.data());
            CHECK(bitwise_equal(f, f1));
            CHECK(bitwise_equal(j, j1));
        }
    }
}

TEST_CASE("phase 1 chunks are independent of the split") {
    const Circuit c(random_circuit(20, 50, 7));
    const auto& m = c.assembly();
    const std::int64_t devices = c.device_counts().total();
    const double x[20] = {0.1, -0.2, 0.3, 0.7, 0.65, -1.0, 1.2, 0.0, 0.4, 0.5,
                          0.9, -0.3, 0.2, 0.25, 0.8, 1.1, -0.7, 0.33, 0.44, 0.55};
    std::vector<double> fa(static_cast<std::size_t>(m.f_offset.back()));
    std::vector<double> ja(static_cast<std::size_t>(m.j_offset.back()));
    mna::assembly::evaluate_device_range(c, x, 0.0, 0, devices, fa.data(), ja.data());
    std::vector<double> fb(fa.size());
    std::vector<double> jb(ja.size());
    for (std::int64_t begin = 0; begin < devices; begin += 7) {
        mna::assembly::evaluate_device_range(c, x, 0.0, begin, std::min(devices, begin + 7), fb.data(), jb.data());
    }
    CHECK(bitwise_equal(fa, fb));
    CHECK(bitwise_equal(ja, jb));
}

TEST_CASE("operating points report currents, small-signal values and regions") {
    CircuitSpec s;
    s.n = 3;
    add_mos(s, 0, 1, -1, 1e-3, 1);
    add_bjt(s, 2, 1, -1, 1);
    const Circuit c(s);
    const double x[3] = {0.3, 1.2, 5.0};
    const mna::assembly::OperatingPoints op = mna::assembly::operating_points(c, x);
    const auto m = mna::devices::mos_level1(1, 1e-3, 0.4, 0.02, 0.3, 1.2, 0.0);
    CHECK(op.mos_ids[0] == m.ids);
    CHECK(op.mos_gds[0] == m.d_drain);
    CHECK(op.mos_gm[0] == m.d_gate);
    CHECK(op.mos_d_source[0] == m.d_source);
    CHECK(op.mos_region[0] == static_cast<int>(mna::devices::MosRegion::Triode));
    const auto q = mna::devices::bjt_level1(1, 1e-15, 150.0, 3.0, 100.0, 25.0, 5.0, 1.2, 0.0);
    CHECK(op.bjt_ic[0] == q.i[0]);
    CHECK(op.bjt_gm[0] == q.d[0][1]);
    CHECK(op.bjt_go[0] == q.d[0][0]);
    CHECK(op.bjt_rpi[0] == 1.0 / q.d[1][1]);
    CHECK(op.bjt_region[0] == static_cast<int>(mna::devices::BjtRegion::ForwardActive));
}

TEST_CASE("null outputs skip the corresponding gather") {
    const Circuit c(random_circuit(10, 5, 3));
    std::vector<double> x(10, 0.3);
    std::vector<double> f(10, 99.0);
    mna::assembly::evaluate(c, x.data(), 0.0, 2, f.data(), nullptr);
    std::vector<double> j(static_cast<std::size_t>(c.nnz()), 99.0);
    mna::assembly::evaluate(c, x.data(), 0.0, 2, nullptr, j.data());
    std::vector<double> f_ref(10);
    std::vector<double> j_ref(static_cast<std::size_t>(c.nnz()));
    mna::assembly::evaluate(c, x.data(), 0.0, 1, f_ref.data(), j_ref.data());
    CHECK(bitwise_equal(f, f_ref));
    CHECK(bitwise_equal(j, j_ref));
}
