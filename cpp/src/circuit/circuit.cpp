#include "mna/circuit/circuit.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/diode.hpp"
#include "mna/devices/gummel_poon.hpp"
#include "mna/devices/spice_diode.hpp"
#include "mna/error.hpp"

namespace mna::circuit {

namespace {

[[noreturn]] void bad_spec(const std::string& message) {
    throw Error(ErrorCode::NetlistFatal, "build_circuit: " + message);
}

constexpr std::int64_t kMaxIndex = std::numeric_limits<std::int32_t>::max();

void check_length(const char* table, const char* column, std::size_t got, std::size_t expected) {
    if (got != expected) {
        bad_spec(std::string(table) + "." + column + " has " + std::to_string(got) + " entries, expected " +
                 std::to_string(expected));
    }
}

std::int32_t checked_index(std::int64_t index, std::int64_t n, const char* table, const char* column, std::size_t i) {
    if (index < -1 || index >= n) {
        bad_spec(std::string(table) + "." + column + "[" + std::to_string(i) + "] = " + std::to_string(index) +
                 " is outside [-1, " + std::to_string(n) + ")");
    }
    return static_cast<std::int32_t>(index);
}

void check_positive(const char* table, const char* column, double value, std::size_t i, bool allow_inf = false) {
    if (!(value > 0.0) || (!allow_inf && !std::isfinite(value))) {
        bad_spec(std::string(table) + "." + column + "[" + std::to_string(i) + "] = " + std::to_string(value) +
                 " must be a positive " + (allow_inf ? "number" : "finite number"));
    }
}

void check_non_negative(const char* table, const char* column, double value, std::size_t i) {
    if (!(value >= 0.0)) {
        bad_spec(std::string(table) + "." + column + "[" + std::to_string(i) + "] = " + std::to_string(value) +
                 " must not be negative");
    }
}

void check_finite(const char* table, const char* column, double value, std::size_t i) {
    if (!std::isfinite(value)) {
        bad_spec(std::string(table) + "." + column + "[" + std::to_string(i) + "] = " + std::to_string(value) +
                 " must be finite");
    }
}

void check_triplets(const TripletList& t, const char* name, std::int64_t n) {
    check_length(name, "cols", t.cols.size(), t.rows.size());
    check_length(name, "values", t.values.size(), t.rows.size());
    for (std::size_t i = 0; i < t.rows.size(); ++i) {
        checked_index(t.rows[i], n, name, "rows", i);
        checked_index(t.cols[i], n, name, "cols", i);
    }
}

}  // namespace

Circuit::Circuit(const CircuitSpec& spec) { build(spec); }

void Circuit::build(const CircuitSpec& spec) {
    if (spec.n < 0 || spec.n > kMaxIndex) {
        bad_spec("n = " + std::to_string(spec.n) + " is outside [0, 2^31 - 1]");
    }
    const std::int64_t n = spec.n;
    n_ = static_cast<std::int32_t>(n);
    gmin_ = spec.gmin;
    if (spec.node_count < 0 || spec.node_count > n) {
        bad_spec("node_count = " + std::to_string(spec.node_count) + " is outside [0, n = " + std::to_string(n) + "]");
    }
    node_count_ = static_cast<std::int32_t>(spec.node_count);

    check_triplets(spec.g, "G", n);
    check_triplets(spec.c, "C", n);
    check_triplets(spec.gmin_stamps, "gmin", n);

    // ---- device tables -------------------------------------------------------
    const DiodeTable& dt = spec.diodes;
    const std::size_t nd = dt.anode.size();
    check_length("diodes", "cathode", dt.cathode.size(), nd);
    check_length("diodes", "is", dt.is.size(), nd);
    diodes_.anode.resize(nd);
    diodes_.cathode.resize(nd);
    diodes_.is = dt.is;
    for (std::size_t i = 0; i < nd; ++i) {
        diodes_.anode[i] = checked_index(dt.anode[i], n, "diodes", "anode", i);
        diodes_.cathode[i] = checked_index(dt.cathode[i], n, "diodes", "cathode", i);
    }

    const MosTable& mt = spec.mos;
    const std::size_t nm = mt.drain.size();
    check_length("mos", "gate", mt.gate.size(), nm);
    check_length("mos", "source", mt.source.size(), nm);
    check_length("mos", "beta", mt.beta.size(), nm);
    check_length("mos", "vth", mt.vth.size(), nm);
    check_length("mos", "lambda", mt.lambda.size(), nm);
    check_length("mos", "sign", mt.sign.size(), nm);
    mos_.drain.resize(nm);
    mos_.gate.resize(nm);
    mos_.source.resize(nm);
    mos_.beta = mt.beta;
    mos_.vth = mt.vth;
    mos_.lambda = mt.lambda;
    mos_.sign.resize(nm);
    for (std::size_t i = 0; i < nm; ++i) {
        mos_.drain[i] = checked_index(mt.drain[i], n, "mos", "drain", i);
        mos_.gate[i] = checked_index(mt.gate[i], n, "mos", "gate", i);
        mos_.source[i] = checked_index(mt.source[i], n, "mos", "source", i);
        mos_.sign[i] = mt.sign[i] < 0 ? -1 : 1;
    }

    const BjtTable& bt = spec.bjt;
    const std::size_t nb = bt.collector.size();
    check_length("bjt", "base", bt.base.size(), nb);
    check_length("bjt", "emitter", bt.emitter.size(), nb);
    check_length("bjt", "is", bt.is.size(), nb);
    check_length("bjt", "bf", bt.bf.size(), nb);
    check_length("bjt", "br", bt.br.size(), nb);
    check_length("bjt", "vaf", bt.vaf.size(), nb);
    check_length("bjt", "var", bt.var.size(), nb);
    check_length("bjt", "sign", bt.sign.size(), nb);
    bjt_.collector.resize(nb);
    bjt_.base.resize(nb);
    bjt_.emitter.resize(nb);
    bjt_.is = bt.is;
    bjt_.bf = bt.bf;
    bjt_.br = bt.br;
    bjt_.vaf = bt.vaf;
    bjt_.var = bt.var;
    bjt_.sign.resize(nb);
    for (std::size_t i = 0; i < nb; ++i) {
        bjt_.collector[i] = checked_index(bt.collector[i], n, "bjt", "collector", i);
        bjt_.base[i] = checked_index(bt.base[i], n, "bjt", "base", i);
        bjt_.emitter[i] = checked_index(bt.emitter[i], n, "bjt", "emitter", i);
        bjt_.sign[i] = bt.sign[i] < 0 ? -1 : 1;
    }

    const BehavioralTable& ht = spec.behavioral;
    const std::size_t nh = ht.node_p.size();
    check_length("behavioral", "node_n", ht.node_n.size(), nh);
    check_length("behavioral", "programs", ht.programs.size(), nh);
    behavioral_.node_p.resize(nh);
    behavioral_.node_n.resize(nh);
    behavioral_.programs = ht.programs;
    behavioral_.slot_ptr.assign(nh + 1, 0);
    for (std::size_t i = 0; i < nh; ++i) {
        behavioral_.node_p[i] = checked_index(ht.node_p[i], n, "behavioral", "node_p", i);
        behavioral_.node_n[i] = checked_index(ht.node_n[i], n, "behavioral", "node_n", i);
        const ProgramPtr& program = ht.programs[i];
        if (!program) {
            bad_spec("behavioral.programs[" + std::to_string(i) + "] is null");
        }
        for (std::int32_t u : program->unknowns()) {
            if (u >= n) {
                bad_spec("behavioral.programs[" + std::to_string(i) + "] reads unknown " + std::to_string(u) +
                         " but the circuit has " + std::to_string(n) + " unknowns");
            }
        }
        behavioral_.slot_ptr[i + 1] =
            behavioral_.slot_ptr[i] + 2 * static_cast<std::int64_t>(program->unknowns().size());
    }

    // ---- model-card devices (phase M2) -----------------------------------------
    const SpiceDiodeTable& sdt = spec.spice_diodes;
    const std::size_t nsd = sdt.anode.size();
    check_length("spice_diodes", "cathode", sdt.cathode.size(), nsd);
    check_length("spice_diodes", "is", sdt.is.size(), nsd);
    check_length("spice_diodes", "n", sdt.n.size(), nsd);
    check_length("spice_diodes", "bv", sdt.bv.size(), nsd);
    check_length("spice_diodes", "ibv", sdt.ibv.size(), nsd);
    spice_diodes_.anode.resize(nsd);
    spice_diodes_.cathode.resize(nsd);
    spice_diodes_.is = sdt.is;
    spice_diodes_.n = sdt.n;
    spice_diodes_.bv = sdt.bv;
    spice_diodes_.ibv = sdt.ibv;
    spice_diodes_.bv_eff.resize(nsd);
    for (std::size_t i = 0; i < nsd; ++i) {
        spice_diodes_.anode[i] = checked_index(sdt.anode[i], n, "spice_diodes", "anode", i);
        spice_diodes_.cathode[i] = checked_index(sdt.cathode[i], n, "spice_diodes", "cathode", i);
        check_positive("spice_diodes", "is", sdt.is[i], i);
        check_positive("spice_diodes", "n", sdt.n[i], i);
        check_positive("spice_diodes", "bv", sdt.bv[i], i, true);
        check_positive("spice_diodes", "ibv", sdt.ibv[i], i);
        spice_diodes_.bv_eff[i] = devices::spice_diode_breakdown_voltage(sdt.is[i], sdt.bv[i], sdt.ibv[i]);
    }

    const GummelPoonTable& gt = spec.gummel_poon;
    const std::size_t ng = gt.collector.size();
    using Column = std::pair<const char*, const std::vector<double>*>;
    const Column gp_positive[] = {{"is", &gt.is}, {"bf", &gt.bf}, {"nf", &gt.nf}, {"br", &gt.br},
                                  {"nr", &gt.nr}, {"ne", &gt.ne}, {"nc", &gt.nc}};
    const Column gp_non_negative[] = {{"vaf", &gt.vaf}, {"var", &gt.var}, {"ikf", &gt.ikf},
                                      {"ikr", &gt.ikr}, {"ise", &gt.ise}, {"isc", &gt.isc}};
    check_length("gummel_poon", "base", gt.base.size(), ng);
    check_length("gummel_poon", "emitter", gt.emitter.size(), ng);
    for (const Column& column : gp_positive) {
        check_length("gummel_poon", column.first, column.second->size(), ng);
    }
    for (const Column& column : gp_non_negative) {
        check_length("gummel_poon", column.first, column.second->size(), ng);
    }
    check_length("gummel_poon", "sign", gt.sign.size(), ng);
    gummel_poon_.collector.resize(ng);
    gummel_poon_.base.resize(ng);
    gummel_poon_.emitter.resize(ng);
    gummel_poon_.params.resize(ng);
    gummel_poon_.sign.resize(ng);
    for (std::size_t i = 0; i < ng; ++i) {
        gummel_poon_.collector[i] = checked_index(gt.collector[i], n, "gummel_poon", "collector", i);
        gummel_poon_.base[i] = checked_index(gt.base[i], n, "gummel_poon", "base", i);
        gummel_poon_.emitter[i] = checked_index(gt.emitter[i], n, "gummel_poon", "emitter", i);
        for (const Column& column : gp_positive) {
            check_positive("gummel_poon", column.first, (*column.second)[i], i);
        }
        for (const Column& column : gp_non_negative) {
            check_non_negative("gummel_poon", column.first, (*column.second)[i], i);
        }
        check_finite("gummel_poon", "ise", gt.ise[i], i);
        check_finite("gummel_poon", "isc", gt.isc[i], i);
        devices::GummelPoonParams& p = gummel_poon_.params[i];
        p.is = gt.is[i];
        p.bf = gt.bf[i];
        p.nf = gt.nf[i];
        p.br = gt.br[i];
        p.nr = gt.nr[i];
        p.vaf = gt.vaf[i];
        p.var = gt.var[i];
        p.ikf = gt.ikf[i];
        p.ikr = gt.ikr[i];
        p.ise = gt.ise[i];
        p.ne = gt.ne[i];
        p.isc = gt.isc[i];
        p.nc = gt.nc[i];
        gummel_poon_.sign[i] = gt.sign[i] < 0 ? -1 : 1;
    }

    const MosCardTable& ct = spec.mos4;
    const std::size_t nc = ct.drain.size();
    check_length("mos4", "gate", ct.gate.size(), nc);
    check_length("mos4", "source", ct.source.size(), nc);
    check_length("mos4", "bulk", ct.bulk.size(), nc);
    check_length("mos4", "beta", ct.beta.size(), nc);
    check_length("mos4", "vto", ct.vto.size(), nc);
    check_length("mos4", "lambda", ct.lambda.size(), nc);
    check_length("mos4", "gamma", ct.gamma.size(), nc);
    check_length("mos4", "phi", ct.phi.size(), nc);
    check_length("mos4", "sign", ct.sign.size(), nc);
    mos4_.drain.resize(nc);
    mos4_.gate.resize(nc);
    mos4_.source.resize(nc);
    mos4_.bulk.resize(nc);
    mos4_.beta = ct.beta;
    mos4_.vto = ct.vto;
    mos4_.lambda = ct.lambda;
    mos4_.gamma = ct.gamma;
    mos4_.phi = ct.phi;
    mos4_.sign.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        mos4_.drain[i] = checked_index(ct.drain[i], n, "mos4", "drain", i);
        mos4_.gate[i] = checked_index(ct.gate[i], n, "mos4", "gate", i);
        mos4_.source[i] = checked_index(ct.source[i], n, "mos4", "source", i);
        mos4_.bulk[i] = checked_index(ct.bulk[i], n, "mos4", "bulk", i);
        check_finite("mos4", "beta", ct.beta[i], i);
        check_finite("mos4", "vto", ct.vto[i], i);
        check_finite("mos4", "lambda", ct.lambda[i], i);
        check_non_negative("mos4", "gamma", ct.gamma[i], i);
        check_finite("mos4", "gamma", ct.gamma[i], i);
        check_positive("mos4", "phi", ct.phi[i], i);
        mos4_.sign[i] = ct.sign[i] < 0 ? -1 : 1;
    }

    counts_.diodes = static_cast<std::int64_t>(nd);
    counts_.mos = static_cast<std::int64_t>(nm);
    counts_.bjt = static_cast<std::int64_t>(nb);
    counts_.behavioral = static_cast<std::int64_t>(nh);
    counts_.spice_diodes = static_cast<std::int64_t>(nsd);
    counts_.gummel_poon = static_cast<std::int64_t>(ng);
    counts_.mos4 = static_cast<std::int64_t>(nc);

    // ---- union pattern: CSC by a counting sort over columns ---------------------
    // Footprint entries (row, col), ground dropped.
    std::vector<std::int32_t> entry_rows;
    std::vector<std::int32_t> entry_cols;
    const std::size_t expected = spec.g.rows.size() + spec.c.rows.size() + spec.gmin_stamps.rows.size() +
                                 static_cast<std::size_t>(node_count_) + kDiodeJ * nd + kMosJ * nm + kBjtJ * nb +
                                 static_cast<std::size_t>(behavioral_.slot_ptr[nh]) + kSpiceDiodeJ * nsd +
                                 kGummelPoonJ * ng + kMosCardJ * nc;
    entry_rows.reserve(expected);
    entry_cols.reserve(expected);
    auto add_entry = [&entry_rows, &entry_cols](std::int64_t row, std::int64_t col) {
        if (row >= 0 && col >= 0) {
            entry_rows.push_back(static_cast<std::int32_t>(row));
            entry_cols.push_back(static_cast<std::int32_t>(col));
        }
    };
    for (const TripletList* t : {&spec.g, &spec.c, &spec.gmin_stamps}) {
        for (std::size_t i = 0; i < t->rows.size(); ++i) {
            add_entry(t->rows[i], t->cols[i]);
        }
    }
    for (std::int32_t i = 0; i < node_count_; ++i) {  // node-to-ground gmin
        add_entry(i, i);
    }
    for (std::size_t i = 0; i < nd; ++i) {
        const std::int32_t a = diodes_.anode[i];
        const std::int32_t k = diodes_.cathode[i];
        for (std::int32_t r : {a, k}) {
            for (std::int32_t c : {a, k}) {
                add_entry(r, c);
            }
        }
    }
    for (std::size_t i = 0; i < nh; ++i) {
        for (std::int32_t r : {behavioral_.node_p[i], behavioral_.node_n[i]}) {
            for (std::int32_t u : behavioral_.programs[i]->unknowns()) {
                add_entry(r, u);
            }
        }
    }
    for (std::size_t i = 0; i < nm; ++i) {
        const std::int32_t cols[3] = {mos_.drain[i], mos_.gate[i], mos_.source[i]};
        for (std::int32_t r : {mos_.drain[i], mos_.source[i]}) {
            for (std::int32_t c : cols) {
                add_entry(r, c);
            }
        }
    }
    for (std::size_t i = 0; i < nb; ++i) {
        const std::int32_t terms[3] = {bjt_.collector[i], bjt_.base[i], bjt_.emitter[i]};
        for (std::int32_t r : terms) {
            for (std::int32_t c : terms) {
                add_entry(r, c);
            }
        }
    }
    for (std::size_t i = 0; i < nsd; ++i) {
        const std::int32_t a = spice_diodes_.anode[i];
        const std::int32_t k = spice_diodes_.cathode[i];
        for (std::int32_t r : {a, k}) {
            for (std::int32_t c : {a, k}) {
                add_entry(r, c);
            }
        }
    }
    for (std::size_t i = 0; i < ng; ++i) {
        const std::int32_t terms[3] = {gummel_poon_.collector[i], gummel_poon_.base[i], gummel_poon_.emitter[i]};
        for (std::int32_t r : terms) {
            for (std::int32_t c : terms) {
                add_entry(r, c);
            }
        }
    }
    for (std::size_t i = 0; i < nc; ++i) {
        const std::int32_t cols[4] = {mos4_.drain[i], mos4_.gate[i], mos4_.source[i], mos4_.bulk[i]};
        for (std::int32_t r : {mos4_.drain[i], mos4_.source[i]}) {
            for (std::int32_t c : cols) {
                add_entry(r, c);
            }
        }
    }
    {
        const std::size_t ncols = static_cast<std::size_t>(n);
        std::vector<std::int64_t> start(ncols + 1, 0);
        for (std::int32_t c : entry_cols) {
            ++start[static_cast<std::size_t>(c) + 1];
        }
        for (std::size_t c = 0; c < ncols; ++c) {
            start[c + 1] += start[c];
        }
        std::vector<std::int32_t> rows_by_col(entry_rows.size());
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::size_t e = 0; e < entry_rows.size(); ++e) {
            rows_by_col[static_cast<std::size_t>(fill[static_cast<std::size_t>(entry_cols[e])]++)] = entry_rows[e];
        }
        col_ptr_.assign(ncols + 1, 0);
        row_idx_.clear();
        row_idx_.reserve(rows_by_col.size());
        for (std::size_t c = 0; c < ncols; ++c) {
            const auto first = rows_by_col.begin() + start[c];
            const auto last = rows_by_col.begin() + start[c + 1];
            std::sort(first, last);
            const auto end_unique = std::unique(first, last);
            row_idx_.insert(row_idx_.end(), first, end_unique);
            if (static_cast<std::int64_t>(row_idx_.size()) > kMaxIndex) {
                bad_spec("the pattern has more than 2^31 - 1 entries");
            }
            col_ptr_[c + 1] = static_cast<std::int32_t>(row_idx_.size());
        }
    }

    auto key_slot = [this](std::int64_t row, std::int64_t col) -> std::int32_t {
        return (row < 0 || col < 0) ? -1 : slot(static_cast<std::int32_t>(row), static_cast<std::int32_t>(col));
    };

    // ---- static values, summed in input order -----------------------------------
    const std::size_t nnz = row_idx_.size();
    auto accumulate = [&](const TripletList& t, std::vector<double>& out) {
        out.assign(nnz, 0.0);
        for (std::size_t i = 0; i < t.rows.size(); ++i) {
            const std::int32_t s = key_slot(t.rows[i], t.cols[i]);
            if (s >= 0) {
                out[static_cast<std::size_t>(s)] += t.values[i];
            }
        }
    };
    accumulate(spec.g, g_vals_);
    accumulate(spec.c, c_vals_);
    accumulate(spec.gmin_stamps, gmin_vals_);
    // Node-to-ground gmin (P3b): one unit stamp per node unknown, after the builder's stamps.
    for (std::int32_t i = 0; i < node_count_; ++i) {
        gmin_vals_[static_cast<std::size_t>(slot(i, i))] += 1.0;
    }

    // ---- exponential junctions (pnjlim) ------------------------------------------
    auto add_junction = [this](std::int32_t anode, std::int32_t cathode, double vt, double is) {
        if (anode < 0 && cathode < 0) {
            return;
        }
        if (!(is > 0.0) || !std::isfinite(is)) {
            return;
        }
        const double vcrit = vt * std::log(vt / (std::sqrt(2.0) * is));
        if (!std::isfinite(vcrit)) {
            return;
        }
        junctions_.anode.push_back(anode);
        junctions_.cathode.push_back(cathode);
        junctions_.vt.push_back(vt);
        junctions_.is.push_back(is);
        junctions_.vcrit.push_back(vcrit);
    };
    for (std::size_t i = 0; i < nd; ++i) {
        add_junction(diodes_.anode[i], diodes_.cathode[i], devices::kDiodeVt, diodes_.is[i]);
    }
    for (std::size_t i = 0; i < nb; ++i) {
        const std::int32_t c = bjt_.collector[i];
        const std::int32_t b = bjt_.base[i];
        const std::int32_t e = bjt_.emitter[i];
        if (bjt_.sign[i] < 0) {
            add_junction(e, b, devices::kBjtVt, bjt_.is[i]);
            add_junction(c, b, devices::kBjtVt, bjt_.is[i]);
        } else {
            add_junction(b, e, devices::kBjtVt, bjt_.is[i]);
            add_junction(b, c, devices::kBjtVt, bjt_.is[i]);
        }
    }
    // Model-card junctions (phase M2) after every Level-1 one, so the limiter visits the
    // junctions of existing circuits in the same order as before.
    for (std::size_t i = 0; i < nsd; ++i) {
        add_junction(spice_diodes_.anode[i], spice_diodes_.cathode[i], spice_diodes_.n[i] * devices::kCardVt,
                     spice_diodes_.is[i]);
    }
    for (std::size_t i = 0; i < ng; ++i) {
        const std::int32_t c = gummel_poon_.collector[i];
        const std::int32_t b = gummel_poon_.base[i];
        const std::int32_t e = gummel_poon_.emitter[i];
        const double is = gummel_poon_.params[i].is;
        if (gummel_poon_.sign[i] < 0) {
            add_junction(e, b, devices::kCardVt, is);
            add_junction(c, b, devices::kCardVt, is);
        } else {
            add_junction(b, e, devices::kCardVt, is);
            add_junction(b, c, devices::kCardVt, is);
        }
    }

    // ---- device slots ---------------------------------------------------------
    diodes_.slots.resize(kDiodeJ * nd);
    for (std::size_t i = 0; i < nd; ++i) {
        const std::int32_t a = diodes_.anode[i];
        const std::int32_t k = diodes_.cathode[i];
        std::int32_t* s = &diodes_.slots[kDiodeJ * i];
        s[0] = key_slot(a, a);
        s[1] = key_slot(a, k);
        s[2] = key_slot(k, a);
        s[3] = key_slot(k, k);
    }
    behavioral_.slots.resize(static_cast<std::size_t>(behavioral_.slot_ptr[nh]));
    for (std::size_t i = 0; i < nh; ++i) {
        std::int32_t* s = behavioral_.slots.data() + behavioral_.slot_ptr[i];
        const std::vector<std::int32_t>& unknowns = behavioral_.programs[i]->unknowns();
        const std::size_t k = unknowns.size();
        for (std::size_t j = 0; j < k; ++j) {
            s[j] = key_slot(behavioral_.node_p[i], unknowns[j]);
            s[k + j] = key_slot(behavioral_.node_n[i], unknowns[j]);
        }
    }
    mos_.slots.resize(kMosJ * nm);
    for (std::size_t i = 0; i < nm; ++i) {
        const std::int32_t cols[3] = {mos_.drain[i], mos_.gate[i], mos_.source[i]};
        const std::int32_t rows[2] = {mos_.drain[i], mos_.source[i]};
        std::int32_t* s = &mos_.slots[kMosJ * i];
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 3; ++c) {
                s[3 * r + c] = key_slot(rows[r], cols[c]);
            }
        }
    }
    bjt_.slots.resize(kBjtJ * nb);
    for (std::size_t i = 0; i < nb; ++i) {
        const std::int32_t terms[3] = {bjt_.collector[i], bjt_.base[i], bjt_.emitter[i]};
        std::int32_t* s = &bjt_.slots[kBjtJ * i];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                s[3 * r + c] = key_slot(terms[r], terms[c]);
            }
        }
    }

    spice_diodes_.slots.resize(kSpiceDiodeJ * nsd);
    for (std::size_t i = 0; i < nsd; ++i) {
        const std::int32_t a = spice_diodes_.anode[i];
        const std::int32_t k = spice_diodes_.cathode[i];
        std::int32_t* s = &spice_diodes_.slots[kSpiceDiodeJ * i];
        s[0] = key_slot(a, a);
        s[1] = key_slot(a, k);
        s[2] = key_slot(k, a);
        s[3] = key_slot(k, k);
    }
    gummel_poon_.slots.resize(kGummelPoonJ * ng);
    for (std::size_t i = 0; i < ng; ++i) {
        const std::int32_t terms[3] = {gummel_poon_.collector[i], gummel_poon_.base[i], gummel_poon_.emitter[i]};
        std::int32_t* s = &gummel_poon_.slots[kGummelPoonJ * i];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                s[3 * r + c] = key_slot(terms[r], terms[c]);
            }
        }
    }
    mos4_.slots.resize(kMosCardJ * nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const std::int32_t cols[4] = {mos4_.drain[i], mos4_.gate[i], mos4_.source[i], mos4_.bulk[i]};
        const std::int32_t rows[2] = {mos4_.drain[i], mos4_.source[i]};
        std::int32_t* s = &mos4_.slots[kMosCardJ * i];
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 4; ++c) {
                s[4 * r + c] = key_slot(rows[r], cols[c]);
            }
        }
    }

    // ---- assembly map ------------------------------------------------------------
    AssemblyMap& m = map_;
    const std::int64_t D = counts_.total();
    {
        const std::int64_t sizes[kDeviceClasses] = {counts_.diodes,       counts_.behavioral,   counts_.mos,
                                                    counts_.bjt,          counts_.spice_diodes, counts_.gummel_poon,
                                                    counts_.mos4};
        m.class_begin[0] = 0;
        for (std::size_t c = 0; c < static_cast<std::size_t>(kDeviceClasses); ++c) {
            m.class_begin[c + 1] = m.class_begin[c] + sizes[c];
        }
    }
    m.f_offset.assign(static_cast<std::size_t>(D) + 1, 0);
    m.j_offset.assign(static_cast<std::size_t>(D) + 1, 0);
    // Scratch targets in device order; -1 for ground.
    std::vector<std::int32_t> f_target;
    std::vector<std::int32_t> j_target;
    std::vector<std::int8_t> f_group;
    std::vector<std::int8_t> j_group;
    const std::size_t f_total = kDiodeF * nd + kBehavioralF * nh + kMosF * nm + kBjtF * nb + kSpiceDiodeF * nsd +
                                kGummelPoonF * ng + kMosCardF * nc;
    const std::size_t j_total = kDiodeJ * nd + static_cast<std::size_t>(behavioral_.slot_ptr[nh]) + kMosJ * nm +
                                kBjtJ * nb + kSpiceDiodeJ * nsd + kGummelPoonJ * ng + kMosCardJ * nc;
    f_target.reserve(f_total);
    f_group.reserve(f_total);
    j_target.reserve(j_total);
    j_group.reserve(j_total);
    std::int64_t d = 0;
    auto push_device = [&](DeviceClass cls, std::initializer_list<std::int32_t> rows, const std::int32_t* slots,
                           std::size_t nslots) {
        const auto g = static_cast<std::int8_t>(assembly_group(cls));
        for (std::int32_t r : rows) {
            f_target.push_back(r);
            f_group.push_back(g);
        }
        for (std::size_t s = 0; s < nslots; ++s) {
            j_target.push_back(slots[s]);
            j_group.push_back(g);
        }
        m.f_offset[static_cast<std::size_t>(d) + 1] = static_cast<std::int64_t>(f_target.size());
        m.j_offset[static_cast<std::size_t>(d) + 1] = static_cast<std::int64_t>(j_target.size());
        ++d;
    };
    for (std::size_t i = 0; i < nd; ++i) {
        push_device(DeviceClass::Diode, {diodes_.anode[i], diodes_.cathode[i]}, &diodes_.slots[kDiodeJ * i],
                    kDiodeJ);
    }
    for (std::size_t i = 0; i < nh; ++i) {
        push_device(DeviceClass::Behavioral, {behavioral_.node_p[i], behavioral_.node_n[i]},
                    behavioral_.slots.data() + behavioral_.slot_ptr[i],
                    static_cast<std::size_t>(behavioral_.slot_ptr[i + 1] - behavioral_.slot_ptr[i]));
    }
    for (std::size_t i = 0; i < nm; ++i) {
        push_device(DeviceClass::Mos, {mos_.drain[i], mos_.source[i]}, &mos_.slots[kMosJ * i], kMosJ);
    }
    for (std::size_t i = 0; i < nb; ++i) {
        push_device(DeviceClass::Bjt, {bjt_.collector[i], bjt_.base[i], bjt_.emitter[i]}, &bjt_.slots[kBjtJ * i],
                    kBjtJ);
    }
    for (std::size_t i = 0; i < nsd; ++i) {
        push_device(DeviceClass::SpiceDiode, {spice_diodes_.anode[i], spice_diodes_.cathode[i]},
                    &spice_diodes_.slots[kSpiceDiodeJ * i], kSpiceDiodeJ);
    }
    for (std::size_t i = 0; i < ng; ++i) {
        push_device(DeviceClass::GummelPoon, {gummel_poon_.collector[i], gummel_poon_.base[i], gummel_poon_.emitter[i]},
                    &gummel_poon_.slots[kGummelPoonJ * i], kGummelPoonJ);
    }
    for (std::size_t i = 0; i < nc; ++i) {
        push_device(DeviceClass::MosCard, {mos4_.drain[i], mos4_.source[i]}, &mos4_.slots[kMosCardJ * i], kMosCardJ);
    }

    // Stable counting sort of scratch entries by target. Scratch entries are in
    // device order and device classes are laid out in assembly-group order, so
    // within one target the entries end up sorted by (group, device order).
    auto build_gather = [](const std::vector<std::int32_t>& target, const std::vector<std::int8_t>& group,
                           std::size_t outputs, Gather& out) {
        out.ptr.assign(outputs + 1, 0);
        for (std::int32_t t : target) {
            if (t >= 0) {
                ++out.ptr[static_cast<std::size_t>(t) + 1];
            }
        }
        for (std::size_t o = 0; o < outputs; ++o) {
            out.ptr[o + 1] += out.ptr[o];
        }
        const std::size_t total = static_cast<std::size_t>(out.ptr[outputs]);
        out.src.resize(total);
        out.target.resize(total);
        out.group.resize(total);
        std::vector<std::int64_t> fill(out.ptr.begin(), out.ptr.end() - 1);
        for (std::size_t e = 0; e < target.size(); ++e) {
            if (target[e] >= 0) {
                const auto pos = static_cast<std::size_t>(fill[static_cast<std::size_t>(target[e])]++);
                out.src[pos] = static_cast<std::int64_t>(e);
                out.target[pos] = target[e];
                out.group[pos] = group[e];
            }
        }
    };
    build_gather(f_target, f_group, static_cast<std::size_t>(n), m.f);
    build_gather(j_target, j_group, nnz, m.j);
    m.groups_present = static_cast<int>(counts_.diodes + counts_.behavioral > 0) +
                       static_cast<int>(counts_.mos > 0) + static_cast<int>(counts_.bjt > 0) +
                       static_cast<int>(counts_.spice_diodes + counts_.gummel_poon + counts_.mos4 > 0);
}

std::int32_t Circuit::slot(std::int32_t row, std::int32_t col) const noexcept {
    if (row < 0 || col < 0 || row >= n_ || col >= n_) {
        return -1;
    }
    const auto begin = row_idx_.begin() + col_ptr_[static_cast<std::size_t>(col)];
    const auto end = row_idx_.begin() + col_ptr_[static_cast<std::size_t>(col) + 1];
    const auto it = std::lower_bound(begin, end, row);
    if (it == end || *it != row) {
        return -1;
    }
    return static_cast<std::int32_t>(it - row_idx_.begin());
}

const std::vector<double>& Circuit::values(MatrixKind which) const noexcept {
    switch (which) {
        case MatrixKind::G: return g_vals_;
        case MatrixKind::C: return c_vals_;
        case MatrixKind::Gmin: break;
    }
    return gmin_vals_;
}

std::vector<double> Circuit::g_effective(double gmin) const {
    std::vector<double> out(g_vals_.size());
    for (std::size_t s = 0; s < out.size(); ++s) {
        out[s] = g_vals_[s] + gmin * gmin_vals_[s];
    }
    return out;
}

void Circuit::scatter_dense(const double* pattern_values, double* dense) const {
    const std::size_t n = static_cast<std::size_t>(n_);
    std::fill(dense, dense + n * n, 0.0);
    for (std::size_t col = 0; col < n; ++col) {
        for (std::int32_t s = col_ptr_[col]; s < col_ptr_[col + 1]; ++s) {
            dense[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(s)]) * n + col] =
                pattern_values[static_cast<std::size_t>(s)];
        }
    }
}

std::vector<double> Circuit::to_dense(MatrixKind which) const {
    const std::size_t n = static_cast<std::size_t>(n_);
    std::vector<double> dense(n * n);
    scatter_dense(values(which).data(), dense.data());
    return dense;
}

}  // namespace mna::circuit
