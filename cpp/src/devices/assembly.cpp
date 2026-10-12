#include "mna/assembly.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/devices/gummel_poon.hpp"
#include "mna/devices/mos_card.hpp"
#include "mna/devices/mos_level1.hpp"
#include "mna/devices/spice_diode.hpp"
#include "mna/expr/program.hpp"
#include "mna/threads.hpp"

// No fused multiply-add anywhere in this file (see mna/devices/fp_contract.hpp).
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::assembly {

namespace {

using circuit::Circuit;

inline double voltage(const double* x, std::int32_t index) noexcept { return index < 0 ? 0.0 : x[index]; }

void eval_diodes(const Circuit::Diodes& dv, const double* x, std::int64_t begin, std::int64_t end, std::int64_t d0,
                 const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double v = voltage(x, dv.anode[u]) - voltage(x, dv.cathode[u]);
        const devices::DiodeEval r = devices::diode(dv.is[u], v);
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        f[0] = r.current;
        f[1] = -r.current;
        j[0] = r.conductance;
        j[1] = -r.conductance;
        j[2] = -r.conductance;
        j[3] = r.conductance;
    }
}

void eval_behavioral(const Circuit::Behavioral& bv, const double* x, double t, std::int64_t begin, std::int64_t end,
                     std::int64_t d0, const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const expr::Program& program = *bv.programs[u];
        const std::size_t k = program.unknowns().size();
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        double value = 0.0;
        expr::evaluate_gradient_unknowns(program, x, t, &value, j);
        f[0] = value;
        f[1] = -value;
        for (std::size_t q = 0; q < k; ++q) {
            j[k + q] = -j[q];
        }
    }
}

void eval_mos(const Circuit::Mos& mv, const double* x, std::int64_t begin, std::int64_t end, std::int64_t d0,
              const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const devices::MosEval r = devices::mos_level1(mv.sign[u], mv.beta[u], mv.vth[u], mv.lambda[u],
                                                       voltage(x, mv.drain[u]), voltage(x, mv.gate[u]),
                                                       voltage(x, mv.source[u]));
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        f[0] = r.ids;
        f[1] = -r.ids;
        j[0] = r.d_drain;
        j[1] = r.d_gate;
        j[2] = r.d_source;
        j[3] = -r.d_drain;
        j[4] = -r.d_gate;
        j[5] = -r.d_source;
    }
}

void eval_bjt(const Circuit::Bjt& bv, const double* x, std::int64_t begin, std::int64_t end, std::int64_t d0,
              const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const devices::BjtEval r = devices::bjt_level1(bv.sign[u], bv.is[u], bv.bf[u], bv.br[u], bv.vaf[u], bv.var[u],
                                                       voltage(x, bv.collector[u]), voltage(x, bv.base[u]),
                                                       voltage(x, bv.emitter[u]));
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        for (int row = 0; row < 3; ++row) {
            f[row] = r.i[row];
            for (int col = 0; col < 3; ++col) {
                j[3 * row + col] = r.d[row][col];
            }
        }
    }
}

void eval_spice_diodes(const Circuit::SpiceDiodes& dv, const double* x, std::int64_t begin, std::int64_t end,
                       std::int64_t d0, const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double v = voltage(x, dv.anode[u]) - voltage(x, dv.cathode[u]);
        const devices::SpiceDiodeEval r = devices::spice_diode(dv.is[u], dv.n[u], dv.bv_eff[u], v);
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        f[0] = r.current;
        f[1] = -r.current;
        j[0] = r.conductance;
        j[1] = -r.conductance;
        j[2] = -r.conductance;
        j[3] = r.conductance;
    }
}

void eval_gummel_poon(const Circuit::GummelPoon& bv, const double* x, std::int64_t begin, std::int64_t end,
                      std::int64_t d0, const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const devices::BjtEval r =
            devices::gummel_poon(bv.sign[u], bv.params[u], voltage(x, bv.collector[u]), voltage(x, bv.base[u]),
                                 voltage(x, bv.emitter[u]));
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        for (int row = 0; row < 3; ++row) {
            f[row] = r.i[row];
            for (int col = 0; col < 3; ++col) {
                j[3 * row + col] = r.d[row][col];
            }
        }
    }
}

void eval_mos_card(const Circuit::MosCard& mv, const double* x, std::int64_t begin, std::int64_t end,
                   std::int64_t d0, const Circuit::AssemblyMap& m, double* fs, double* js) {
    for (std::int64_t i = begin; i < end; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const devices::Mos4Eval r = devices::mos_card(
            mv.sign[u], mv.beta[u], mv.vto[u], mv.lambda[u], mv.gamma[u], mv.phi[u], voltage(x, mv.drain[u]),
            voltage(x, mv.gate[u]), voltage(x, mv.source[u]), voltage(x, mv.bulk[u]));
        double* f = fs + m.f_offset[static_cast<std::size_t>(d0 + i)];
        double* j = js + m.j_offset[static_cast<std::size_t>(d0 + i)];
        f[0] = r.ids;
        f[1] = -r.ids;
        j[0] = r.d_drain;
        j[1] = r.d_gate;
        j[2] = r.d_source;
        j[3] = r.d_bulk;
        j[4] = -r.d_drain;
        j[5] = -r.d_gate;
        j[6] = -r.d_source;
        j[7] = -r.d_bulk;
    }
}

/// Phase 2 for the outputs [begin, end) of one gather plan. Each output is the
/// sequential (device-order) sum of its entries within each assembly group,
/// and the group sums are added in group order: (((0 + g0) + g1) + g2) + g3. With at
/// most one group present that is the plain sequential sum, computed by a
/// branch-free scatter. `acc` (kAssemblyGroups doubles per output) is only used when
/// several groups are present; chunks touch disjoint parts of out and acc.
void gather_range(const Circuit::Gather& g, int groups_present, const double* scratch, std::int64_t begin,
                  std::int64_t end, double* out, double* acc) {
    const std::int64_t e0 = g.ptr[static_cast<std::size_t>(begin)];
    const std::int64_t e1 = g.ptr[static_cast<std::size_t>(end)];
    const std::int64_t* src = g.src.data();
    const std::int32_t* target = g.target.data();
    if (groups_present <= 1) {
        std::fill(out + begin, out + end, 0.0);
        for (std::int64_t e = e0; e < e1; ++e) {
            out[target[e]] += scratch[src[e]];
        }
        return;
    }
    constexpr int G = circuit::kAssemblyGroups;
    const std::int8_t* group = g.group.data();
    std::fill(acc + G * begin, acc + G * end, 0.0);
    for (std::int64_t e = e0; e < e1; ++e) {
        acc[G * static_cast<std::int64_t>(target[e]) + group[e]] += scratch[src[e]];
    }
    for (std::int64_t o = begin; o < end; ++o) {
        double total = 0.0;
        for (int k = 0; k < G; ++k) {
            total += acc[G * o + k];
        }
        out[o] = total;
    }
}

/// Runs fn(begin, end) over [0, count): inline as a single range for one
/// thread (results never depend on how the range is split: every device and
/// every output is computed independently), otherwise on parallel chunks.
template <class Fn>
void run_chunks(std::int64_t count, int num_threads, Fn&& fn) {
    if (count <= 0) {
        return;
    }
    if (num_threads <= 1) {
        fn(std::int64_t{0}, count);
        return;
    }
    parallel_for_chunks(count, num_threads, [&fn](std::int64_t, std::int64_t begin, std::int64_t end) { fn(begin, end); });
}

}  // namespace

void evaluate_device_range(const Circuit& circuit, const double* x, double t, std::int64_t begin, std::int64_t end,
                           double* f_scratch, double* j_scratch) {
    const Circuit::AssemblyMap& m = circuit.assembly();
    const auto& cb = m.class_begin;
    auto clip = [begin, end](std::int64_t lo, std::int64_t hi, std::int64_t& b, std::int64_t& e) {
        b = (std::max)(begin, lo) - lo;
        e = (std::min)(end, hi) - lo;
        return b < e;
    };
    std::int64_t b = 0;
    std::int64_t e = 0;
    if (clip(cb[0], cb[1], b, e)) {
        eval_diodes(circuit.diodes(), x, b, e, cb[0], m, f_scratch, j_scratch);
    }
    if (clip(cb[1], cb[2], b, e)) {
        eval_behavioral(circuit.behavioral(), x, t, b, e, cb[1], m, f_scratch, j_scratch);
    }
    if (clip(cb[2], cb[3], b, e)) {
        eval_mos(circuit.mos(), x, b, e, cb[2], m, f_scratch, j_scratch);
    }
    if (clip(cb[3], cb[4], b, e)) {
        eval_bjt(circuit.bjt(), x, b, e, cb[3], m, f_scratch, j_scratch);
    }
    if (clip(cb[4], cb[5], b, e)) {
        eval_spice_diodes(circuit.spice_diodes(), x, b, e, cb[4], m, f_scratch, j_scratch);
    }
    if (clip(cb[5], cb[6], b, e)) {
        eval_gummel_poon(circuit.gummel_poon(), x, b, e, cb[5], m, f_scratch, j_scratch);
    }
    if (clip(cb[6], cb[7], b, e)) {
        eval_mos_card(circuit.mos4(), x, b, e, cb[6], m, f_scratch, j_scratch);
    }
}

void evaluate(const Circuit& circuit, const double* x, double t, int num_threads, double* f_out, double* jvals_out) {
    const Circuit::AssemblyMap& m = circuit.assembly();
    const std::int64_t devices = circuit.device_counts().total();
    // Every scratch entry is written by phase 1, so the buffers need no initialization.
    const auto f_size = static_cast<std::size_t>(m.f_offset.back());
    const auto j_size = static_cast<std::size_t>(m.j_offset.back());
    std::unique_ptr<double[]> scratch(new double[f_size + j_size + 1]);
    double* fs = scratch.get();
    double* js = scratch.get() + f_size;

    run_chunks(devices, num_threads, [&](std::int64_t begin, std::int64_t end) {
        evaluate_device_range(circuit, x, t, begin, end, fs, js);
    });
    const int groups = m.groups_present;
    std::unique_ptr<double[]> acc;
    if (groups > 1) {
        const std::size_t outputs = (std::max)(static_cast<std::size_t>(circuit.n()), static_cast<std::size_t>(circuit.nnz()));
        acc.reset(new double[circuit::kAssemblyGroups * outputs + 1]);
    }
    if (f_out != nullptr) {
        run_chunks(circuit.n(), num_threads, [&](std::int64_t begin, std::int64_t end) {
            gather_range(m.f, groups, fs, begin, end, f_out, acc.get());
        });
    }
    if (jvals_out != nullptr) {
        run_chunks(circuit.nnz(), num_threads, [&](std::int64_t begin, std::int64_t end) {
            gather_range(m.j, groups, js, begin, end, jvals_out, acc.get());
        });
    }
}

void evaluate_dense_jacobian(const Circuit& circuit, const double* x, double t, int num_threads, double* f_out,
                             double* j_dense_out) {
    std::unique_ptr<double[]> jvals(new double[static_cast<std::size_t>(circuit.nnz()) + 1]);
    evaluate(circuit, x, t, num_threads, f_out, jvals.get());
    circuit.scatter_dense(jvals.get(), j_dense_out);
}

OperatingPoints operating_points(const Circuit& circuit, const double* x) {
    OperatingPoints op;
    const Circuit::Mos& mv = circuit.mos();
    const std::size_t nm = mv.drain.size();
    op.mos_ids.resize(nm);
    op.mos_gds.resize(nm);
    op.mos_gm.resize(nm);
    op.mos_d_source.resize(nm);
    op.mos_region.resize(nm);
    for (std::size_t i = 0; i < nm; ++i) {
        const double vd = voltage(x, mv.drain[i]);
        const double vg = voltage(x, mv.gate[i]);
        const double vs = voltage(x, mv.source[i]);
        const devices::MosEval r = devices::mos_level1(mv.sign[i], mv.beta[i], mv.vth[i], mv.lambda[i], vd, vg, vs);
        op.mos_ids[i] = r.ids;
        op.mos_gds[i] = r.d_drain;
        op.mos_gm[i] = r.d_gate;
        op.mos_d_source[i] = r.d_source;
        op.mos_region[i] = static_cast<std::int32_t>(devices::mos_level1_region(mv.sign[i], vd, vg, vs, mv.vth[i]));
    }
    const Circuit::Bjt& bv = circuit.bjt();
    const std::size_t nb = bv.collector.size();
    op.bjt_ic.resize(nb);
    op.bjt_ib.resize(nb);
    op.bjt_ie.resize(nb);
    op.bjt_gm.resize(nb);
    op.bjt_go.resize(nb);
    op.bjt_rpi.resize(nb);
    op.bjt_region.resize(nb);
    for (std::size_t i = 0; i < nb; ++i) {
        const double vc = voltage(x, bv.collector[i]);
        const double vb = voltage(x, bv.base[i]);
        const double ve = voltage(x, bv.emitter[i]);
        const devices::BjtEval r =
            devices::bjt_level1(bv.sign[i], bv.is[i], bv.bf[i], bv.br[i], bv.vaf[i], bv.var[i], vc, vb, ve);
        op.bjt_ic[i] = r.i[0];
        op.bjt_ib[i] = r.i[1];
        op.bjt_ie[i] = r.i[2];
        op.bjt_gm[i] = r.d[0][1];
        op.bjt_go[i] = r.d[0][0];
        op.bjt_rpi[i] = std::fabs(r.d[1][1]) > 1e-30 ? 1.0 / r.d[1][1] : std::numeric_limits<double>::infinity();
        op.bjt_region[i] = static_cast<std::int32_t>(devices::bjt_level1_region(bv.sign[i], vc, vb, ve));
    }

    const Circuit::SpiceDiodes& sd = circuit.spice_diodes();
    const std::size_t nsd = sd.anode.size();
    op.spice_diode_vd.resize(nsd);
    op.spice_diode_id.resize(nsd);
    op.spice_diode_gd.resize(nsd);
    op.spice_diode_bv_eff = sd.bv_eff;
    op.spice_diode_region.resize(nsd);
    for (std::size_t i = 0; i < nsd; ++i) {
        const double v = voltage(x, sd.anode[i]) - voltage(x, sd.cathode[i]);
        const devices::SpiceDiodeEval r = devices::spice_diode(sd.is[i], sd.n[i], sd.bv_eff[i], v);
        op.spice_diode_vd[i] = v;
        op.spice_diode_id[i] = r.current;
        op.spice_diode_gd[i] = r.conductance;
        op.spice_diode_region[i] = static_cast<std::int32_t>(devices::spice_diode_region(sd.n[i], sd.bv_eff[i], v));
    }

    const Circuit::GummelPoon& gp = circuit.gummel_poon();
    const std::size_t ng = gp.collector.size();
    op.gp_ic.resize(ng);
    op.gp_ib.resize(ng);
    op.gp_ie.resize(ng);
    op.gp_gm.resize(ng);
    op.gp_go.resize(ng);
    op.gp_rpi.resize(ng);
    op.gp_region.resize(ng);
    for (std::size_t i = 0; i < ng; ++i) {
        const double vc = voltage(x, gp.collector[i]);
        const double vb = voltage(x, gp.base[i]);
        const double ve = voltage(x, gp.emitter[i]);
        const devices::BjtEval r = devices::gummel_poon(gp.sign[i], gp.params[i], vc, vb, ve);
        op.gp_ic[i] = r.i[0];
        op.gp_ib[i] = r.i[1];
        op.gp_ie[i] = r.i[2];
        op.gp_gm[i] = r.d[0][1];
        op.gp_go[i] = r.d[0][0];
        op.gp_rpi[i] = std::fabs(r.d[1][1]) > 1e-30 ? 1.0 / r.d[1][1] : std::numeric_limits<double>::infinity();
        op.gp_region[i] = static_cast<std::int32_t>(devices::bjt_level1_region(gp.sign[i], vc, vb, ve));
    }

    const Circuit::MosCard& mc = circuit.mos4();
    const std::size_t nc = mc.drain.size();
    op.mos4_ids.resize(nc);
    op.mos4_gds.resize(nc);
    op.mos4_gm.resize(nc);
    op.mos4_d_source.resize(nc);
    op.mos4_gmbs.resize(nc);
    op.mos4_vth.resize(nc);
    op.mos4_region.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const double vd = voltage(x, mc.drain[i]);
        const double vg = voltage(x, mc.gate[i]);
        const double vs = voltage(x, mc.source[i]);
        const double vb = voltage(x, mc.bulk[i]);
        const devices::Mos4Eval r = devices::mos_card(mc.sign[i], mc.beta[i], mc.vto[i], mc.lambda[i], mc.gamma[i],
                                                      mc.phi[i], vd, vg, vs, vb);
        op.mos4_ids[i] = r.ids;
        op.mos4_gds[i] = r.d_drain;
        op.mos4_gm[i] = r.d_gate;
        op.mos4_d_source[i] = r.d_source;
        op.mos4_gmbs[i] = r.d_bulk;
        op.mos4_vth[i] = devices::mos_card_vth(mc.sign[i], mc.vto[i], mc.gamma[i], mc.phi[i], vd, vs, vb);
        op.mos4_region[i] = static_cast<std::int32_t>(
            devices::mos_card_region(mc.sign[i], mc.vto[i], mc.gamma[i], mc.phi[i], vd, vg, vs, vb));
    }
    return op;
}

}  // namespace mna::assembly
