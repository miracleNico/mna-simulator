// Circuit compilation: union pattern, value summation, device slots, ground handling.
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"

using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;
using mna::circuit::MatrixKind;

namespace {

void stamp(mna::circuit::TripletList& t, std::int64_t j, std::int64_t k, double v) {
    // Mirrors mna_builder.stamp: (j,j) (k,k) (j,k) (k,j), ground dropped by the builder.
    const std::pair<std::int64_t, std::int64_t> entries[4] = {{j, j}, {k, k}, {j, k}, {k, j}};
    const double signs[4] = {1.0, 1.0, -1.0, -1.0};
    for (int e = 0; e < 4; ++e) {
        if (entries[e].first >= 0 && entries[e].second >= 0) {
            t.rows.push_back(entries[e].first);
            t.cols.push_back(entries[e].second);
            t.values.push_back(signs[e] * v);
        }
    }
}

std::shared_ptr<const mna::expr::Program> product_program(std::int32_t a, std::int32_t b, std::int32_t n) {
    // x[a] * x[b]
    const std::int32_t ops[3] = {1, 1, 5};
    const std::int32_t args[3] = {a, b, 0};
    return std::make_shared<mna::expr::Program>(ops, args, 3, std::vector<double>{}, n, false);
}

}  // namespace

TEST_CASE("empty circuit") {
    CircuitSpec spec;
    const Circuit c(spec);
    CHECK(c.n() == 0);
    CHECK(c.nnz() == 0);
    CHECK(c.col_ptr() == std::vector<std::int32_t>{0});
    CHECK_FALSE(c.has_nonlinear());
    CHECK(c.to_dense(MatrixKind::G).empty());
}

TEST_CASE("linear stamps: duplicates summed in order, gmin kept separate, CSC sorted") {
    CircuitSpec spec;
    spec.n = 3;
    stamp(spec.g, 0, 1, 1.0 / 3.0);   // R between 0 and 1
    stamp(spec.g, 1, -1, 0.1);        // R from 1 to ground
    stamp(spec.g, 0, 1, 1e-17);       // tiny parallel R: summed after 1/3
    stamp(spec.c, 1, 2, 2e-12);
    stamp(spec.gmin_stamps, 1, 2, 1.0);
    const Circuit c(spec);
    CHECK(c.n() == 3);
    // Pattern: (0,0) (1,0) | (0,1) (1,1) (2,1) | (1,2) (2,2)
    CHECK(c.col_ptr() == std::vector<std::int32_t>{0, 2, 5, 7});
    CHECK(c.row_idx() == std::vector<std::int32_t>{0, 1, 0, 1, 2, 1, 2});
    CHECK(c.nnz() == 7);
    const std::vector<double> G = c.to_dense(MatrixKind::G);
    CHECK(G[0 * 3 + 0] == 1.0 / 3.0 + 1e-17);
    CHECK(G[0 * 3 + 1] == -1.0 / 3.0 + -1e-17);
    CHECK(G[1 * 3 + 1] == (1.0 / 3.0 + 0.1) + 1e-17);
    CHECK(G[2 * 3 + 2] == 0.0);
    const std::vector<double> C = c.to_dense(MatrixKind::C);
    CHECK(C[1 * 3 + 2] == -2e-12);
    CHECK(C[2 * 3 + 2] == 2e-12);
    const std::vector<double> M = c.to_dense(MatrixKind::Gmin);
    CHECK(M[1 * 3 + 1] == 1.0);
    CHECK(M[1 * 3 + 2] == -1.0);
    CHECK(M[0] == 0.0);
    const std::vector<double> eff = c.g_effective(1e-9);
    CHECK(eff[static_cast<std::size_t>(c.slot(1, 1))] == G[4] + 1e-9);
    CHECK(eff[static_cast<std::size_t>(c.slot(2, 1))] == -1e-9);
    CHECK(c.slot(2, 0) == -1);
    CHECK(c.slot(-1, 0) == -1);
    CHECK(c.slot(0, 3) == -1);
    CHECK_FALSE(c.has_nonlinear());
}

TEST_CASE("device footprints join the pattern; ground entries get slot -1") {
    CircuitSpec spec;
    spec.n = 4;
    stamp(spec.g, 0, -1, 1e-3);
    // NMOS drain 1, gate 2, source ground.
    spec.mos.drain = {1};
    spec.mos.gate = {2};
    spec.mos.source = {-1};
    spec.mos.beta = {1e-3};
    spec.mos.vth = {0.4};
    spec.mos.lambda = {0.02};
    spec.mos.sign = {1};
    // Diode 0 -> 3.
    spec.diodes.anode = {0};
    spec.diodes.cathode = {3};
    spec.diodes.is = {1e-14};
    // NPN c=3 b=2 e=ground.
    spec.bjt.collector = {3};
    spec.bjt.base = {2};
    spec.bjt.emitter = {-1};
    spec.bjt.is = {1e-15};
    spec.bjt.bf = {150.0};
    spec.bjt.br = {3.0};
    spec.bjt.vaf = {100.0};
    spec.bjt.var = {25.0};
    spec.bjt.sign = {1};
    // Behavioral current x0 * x2 from node 1 to ground.
    spec.behavioral.node_p = {1};
    spec.behavioral.node_n = {-1};
    spec.behavioral.programs = {product_program(0, 2, 4)};
    const Circuit c(spec);
    CHECK(c.has_nonlinear());
    CHECK(c.device_counts().diodes == 1);
    CHECK(c.device_counts().mos == 1);
    CHECK(c.device_counts().bjt == 1);
    CHECK(c.device_counts().behavioral == 1);

    const auto& ms = c.mos().slots;  // rows (d=1, s=gnd) x cols (d=1, g=2, s=gnd)
    REQUIRE(ms.size() == 6);
    CHECK(ms[0] == c.slot(1, 1));
    CHECK(ms[1] == c.slot(1, 2));
    CHECK(ms[2] == -1);
    CHECK(ms[3] == -1);
    CHECK(ms[4] == -1);
    CHECK(ms[5] == -1);
    const auto& ds = c.diodes().slots;
    CHECK(ds == std::vector<std::int32_t>{c.slot(0, 0), c.slot(0, 3), c.slot(3, 0), c.slot(3, 3)});
    for (std::int32_t s : ds) {
        CHECK(s >= 0);
    }
    const auto& bs = c.bjt().slots;  // rows/cols (c=3, b=2, e=gnd)
    CHECK(bs[0] == c.slot(3, 3));
    CHECK(bs[1] == c.slot(3, 2));
    CHECK(bs[2] == -1);
    CHECK(bs[3] == c.slot(2, 3));
    CHECK(bs[4] == c.slot(2, 2));
    CHECK(bs[8] == -1);
    const auto& hs = c.behavioral().slots;  // row p=1 over unknowns {0, 2}, then row gnd
    CHECK(hs == std::vector<std::int32_t>{c.slot(1, 0), c.slot(1, 2), -1, -1});
    // Every (row, col) of the union appears exactly once.
    std::set<std::pair<int, int>> seen;
    for (int col = 0; col < c.n(); ++col) {
        for (int s = c.col_ptr()[static_cast<std::size_t>(col)]; s < c.col_ptr()[static_cast<std::size_t>(col) + 1];
             ++s) {
            CHECK(seen.insert({c.row_idx()[static_cast<std::size_t>(s)], col}).second);
        }
    }
    const std::set<std::pair<int, int>> expected = {{0, 0}, {1, 1}, {1, 2}, {0, 3}, {3, 0}, {3, 3}, {3, 2},
                                                    {2, 3}, {2, 2}, {1, 0}};
    CHECK(seen == expected);
    // G values live only on the resistor entry.
    const std::vector<double>& g = c.values(MatrixKind::G);
    for (std::int32_t s = 0; s < c.nnz(); ++s) {
        CHECK(g[static_cast<std::size_t>(s)] == (s == c.slot(0, 0) ? 1e-3 : 0.0));
    }
    // Assembly map: global order diodes, behavioral, MOS, BJT (then the empty model-card classes).
    const auto& m = c.assembly();
    CHECK(m.class_begin == std::array<std::int64_t, 8>{0, 1, 2, 3, 4, 4, 4, 4});
    CHECK(m.f_offset == std::vector<std::int64_t>{0, 2, 4, 6, 9});
    CHECK(m.j_offset == std::vector<std::int64_t>{0, 4, 8, 14, 23});
}

TEST_CASE("aliased terminals share slots") {
    CircuitSpec spec;
    spec.n = 2;
    // Diode-connected NMOS: drain == gate == 0, source 1.
    spec.mos.drain = {0};
    spec.mos.gate = {0};
    spec.mos.source = {1};
    spec.mos.beta = {1e-3};
    spec.mos.vth = {0.4};
    spec.mos.lambda = {0.0};
    spec.mos.sign = {-1};
    const Circuit c(spec);
    const auto& ms = c.mos().slots;
    CHECK(ms[0] == ms[1]);
    CHECK(ms[3] == ms[4]);
    CHECK(c.nnz() == 4);
    CHECK(c.mos().sign[0] == -1);
}

TEST_CASE("malformed specs are rejected with NETLIST_FATAL") {
    auto rejects = [](CircuitSpec spec) {
        try {
            const Circuit c(spec);
        } catch (const mna::Error& e) {
            CHECK(e.code() == mna::ErrorCode::NetlistFatal);
            return true;
        }
        return false;
    };
    CircuitSpec bad_n;
    bad_n.n = -1;
    CHECK(rejects(bad_n));
    CircuitSpec huge;
    huge.n = std::int64_t{1} << 31;
    CHECK(rejects(huge));
    CircuitSpec bad_index;
    bad_index.n = 2;
    bad_index.g = {{0}, {2}, {1.0}};
    CHECK(rejects(bad_index));
    CircuitSpec below_ground;
    below_ground.n = 2;
    below_ground.g = {{-2}, {0}, {1.0}};
    CHECK(rejects(below_ground));
    CircuitSpec length_mismatch;
    length_mismatch.n = 2;
    length_mismatch.c = {{0, 1}, {0}, {1.0, 2.0}};
    CHECK(rejects(length_mismatch));
    CircuitSpec mos_mismatch;
    mos_mismatch.n = 2;
    mos_mismatch.mos.drain = {0};
    CHECK(rejects(mos_mismatch));
    CircuitSpec null_program;
    null_program.n = 2;
    null_program.behavioral.node_p = {0};
    null_program.behavioral.node_n = {1};
    null_program.behavioral.programs = {nullptr};
    CHECK(rejects(null_program));
    CircuitSpec unknown_out_of_range;
    unknown_out_of_range.n = 2;
    unknown_out_of_range.behavioral.node_p = {0};
    unknown_out_of_range.behavioral.node_n = {1};
    unknown_out_of_range.behavioral.programs = {product_program(0, 4, 5)};
    CHECK(rejects(unknown_out_of_range));
    CircuitSpec ok;
    ok.n = 2;
    ok.g = {{-1, 0}, {0, 1}, {5.0, 1.0}};  // ground triplet entries are ignored
    CHECK_FALSE(rejects(ok));
}
