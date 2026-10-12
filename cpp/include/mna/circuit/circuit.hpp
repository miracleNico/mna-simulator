// Compiled circuit: the union sparsity pattern of the linear matrices and of
// every device Jacobian footprint, the static values in pattern order, the
// device tables as structure-of-arrays with their pattern slots, and the
// precomputed assembly map used by mna::assembly::evaluate.
//
// Input (CircuitSpec, filled from Python by build_circuit):
//   * G, C: COO triplets of the linear stamps WITHOUT gmin; gmin_stamps: the
//     unit-valued (+-1) triplets of every place the builder stamps gmin, so
//     G_eff = G + gmin * gmin_stamps can be formed for any gmin without a
//     rebuild. Duplicates are summed in input order (the order the Python
//     builder stamps, so the sums equal its dense +=).
//   * node_count: unknowns [0, node_count) are node voltages (the builder
//     orders nodes before branch currents). Each gets a unit node-to-ground
//     stamp (i, i) appended to the gmin stamps (SPICE-style GMIN to ground,
//     CPP_MIGRATION_PLAN.md 2.4.1, P3b), so g_effective(gmin) also shunts
//     every node by gmin; branch-current rows never get one. 0 (the default)
//     adds none.
//   * Device tables: diodes (anode, cathode, Is), Level-1 MOS (drain, gate,
//     source, beta, vth, lambda, sign = +1 NMOS / -1 PMOS), Level-1 BJT
//     (collector, base, emitter, Is, Bf, Br, Vaf, Var, sign = +1 NPN / -1 PNP),
//     behavioral current sources (node_p, node_n, program; the current flows
//     from node_p to node_n through the source, i.e. f[p] += I, f[n] -= I).
//   * Model-card device tables (HIERARCHY_AND_MODELS_PLAN.md 3.2, phase M2),
//     every parameter already scaled by the instance area / multiplier:
//     SPICE diodes (anode, cathode, is, n, bv, ibv; mna/devices/spice_diode.hpp,
//     the effective breakdown voltage is computed here), Gummel-Poon BJTs
//     (collector, base, emitter, is, bf, nf, br, nr, vaf, var, ikf, ikr, ise,
//     ne, isc, nc, sign; mna/devices/gummel_poon.hpp) and four-terminal MOS
//     (drain, gate, source, bulk, beta, vto, lambda, gamma, phi, sign;
//     mna/devices/mos_card.hpp). The terminals are the intrinsic ones: the
//     builder's internal nodes when series resistances are present.
//   * Node / unknown indices are 0-based; -1 is ground.
//
// The pattern is stored in compressed sparse column form (KLU's format) with
// 32-bit indices; row indices are sorted within each column.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mna/devices/gummel_poon.hpp"
#include "mna/expr/program.hpp"

namespace mna::circuit {

using ProgramPtr = std::shared_ptr<const expr::Program>;

struct TripletList {
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> values;
};

struct DiodeTable {
    std::vector<std::int64_t> anode;
    std::vector<std::int64_t> cathode;
    std::vector<double> is;
};

struct MosTable {
    std::vector<std::int64_t> drain;
    std::vector<std::int64_t> gate;
    std::vector<std::int64_t> source;
    std::vector<double> beta;
    std::vector<double> vth;
    std::vector<double> lambda;
    std::vector<std::int32_t> sign;
};

struct BjtTable {
    std::vector<std::int64_t> collector;
    std::vector<std::int64_t> base;
    std::vector<std::int64_t> emitter;
    std::vector<double> is;
    std::vector<double> bf;
    std::vector<double> br;
    std::vector<double> vaf;
    std::vector<double> var;
    std::vector<std::int32_t> sign;
};

struct BehavioralTable {
    std::vector<std::int64_t> node_p;
    std::vector<std::int64_t> node_n;
    std::vector<ProgramPtr> programs;
};

struct SpiceDiodeTable {
    std::vector<std::int64_t> anode;
    std::vector<std::int64_t> cathode;
    std::vector<double> is;
    std::vector<double> n;
    std::vector<double> bv;   ///< +inf: no breakdown
    std::vector<double> ibv;
};

struct GummelPoonTable {
    std::vector<std::int64_t> collector;
    std::vector<std::int64_t> base;
    std::vector<std::int64_t> emitter;
    std::vector<double> is, bf, nf, br, nr, vaf, var, ikf, ikr, ise, ne, isc, nc;
    std::vector<std::int32_t> sign;
};

struct MosCardTable {
    std::vector<std::int64_t> drain;
    std::vector<std::int64_t> gate;
    std::vector<std::int64_t> source;
    std::vector<std::int64_t> bulk;
    std::vector<double> beta, vto, lambda, gamma, phi;
    std::vector<std::int32_t> sign;
};

struct CircuitSpec {
    std::int64_t n = 0;
    double gmin = 0.0;  ///< the builder's gmin (informational; values exclude it)
    std::int64_t node_count = 0;  ///< node-voltage unknowns [0, node_count) get node-to-ground gmin
    TripletList g;
    TripletList c;
    TripletList gmin_stamps;
    DiodeTable diodes;
    MosTable mos;
    BjtTable bjt;
    BehavioralTable behavioral;
    SpiceDiodeTable spice_diodes;
    GummelPoonTable gummel_poon;
    MosCardTable mos4;
};

enum class MatrixKind { G, C, Gmin };

struct DeviceCounts {
    std::int64_t diodes = 0;
    std::int64_t mos = 0;
    std::int64_t bjt = 0;
    std::int64_t behavioral = 0;
    std::int64_t spice_diodes = 0;
    std::int64_t gummel_poon = 0;
    std::int64_t mos4 = 0;
    std::int64_t total() const noexcept {
        return diodes + mos + bjt + behavioral + spice_diodes + gummel_poon + mos4;
    }
};

/// Device classes in global device order. Phase 2 of the assembly sums each
/// output in groups and adds the group totals in this order:
/// (diodes + behavioral) + MOS + BJT + model-card devices. The first three
/// reproduce the Python core's `f = f_symbolic; f += f_mos; f += f_bjt`
/// grouping; the model-card classes (phase M2) form a fourth group that is
/// empty for every circuit without them, so those circuits sum exactly as
/// before (an empty group adds +0.0 to a total that started at +0.0).
enum class DeviceClass : int {
    Diode = 0,
    Behavioral = 1,
    Mos = 2,
    Bjt = 3,
    SpiceDiode = 4,
    GummelPoon = 5,
    MosCard = 6
};
inline constexpr int kDeviceClasses = 7;
inline constexpr int kAssemblyGroups = 4;
inline constexpr int assembly_group(DeviceClass cls) noexcept {
    switch (cls) {
        case DeviceClass::Diode:
        case DeviceClass::Behavioral: return 0;
        case DeviceClass::Mos: return 1;
        case DeviceClass::Bjt: return 2;
        default: return 3;
    }
}

/// Scratch entries per device: f contributions and Jacobian contributions.
inline constexpr int kDiodeF = 2;
inline constexpr int kDiodeJ = 4;   // (a,a) (a,k) (k,a) (k,k)
inline constexpr int kMosF = 2;     // +I at drain, -I at source
inline constexpr int kMosJ = 6;     // rows (d, s) x cols (d, g, s)
inline constexpr int kBjtF = 3;     // c, b, e
inline constexpr int kBjtJ = 9;     // rows (c, b, e) x cols (c, b, e)
inline constexpr int kBehavioralF = 2;  // +I at p, -I at n; J: rows (p, n) x program unknowns
inline constexpr int kSpiceDiodeF = 2;
inline constexpr int kSpiceDiodeJ = 4;  // (a,a) (a,k) (k,a) (k,k)
inline constexpr int kGummelPoonF = 3;  // c, b, e
inline constexpr int kGummelPoonJ = 9;  // rows (c, b, e) x cols (c, b, e)
inline constexpr int kMosCardF = 2;     // +I at drain, -I at source
inline constexpr int kMosCardJ = 8;     // rows (d, s) x cols (d, g, s, b)

/// Exponential pn junctions of the compiled circuit, for junction voltage-step
/// limiting (SPICE pnjlim, mna/analyses/limiting.hpp). Junction j conducts
/// from anode[j] to cathode[j] (-1 = ground) with I = is * (exp(v / vt) - 1)
/// at v = V(anode) - V(cathode). Order: every diode (anode, cathode,
/// vt = 0.025), then per Level-1 BJT its base-emitter and base-collector
/// junctions (vt = 0.02585; NPN (b, e), (b, c); PNP (e, b), (c, b)), then
/// every SPICE diode (intrinsic anode, cathode, vt = n * kCardVt, is), then
/// per Gummel-Poon BJT its intrinsic base-emitter and base-collector
/// junctions (vt = kCardVt and is, as SPICE3f5's bjtload limits them).
/// Junctions with both terminals on ground, or whose is is not a positive
/// finite number (no critical voltage), are left out.
/// vcrit = vt * ln(vt / (sqrt(2) * is)).
struct Junctions {
    std::vector<std::int32_t> anode, cathode;
    std::vector<double> vt, is, vcrit;
    std::size_t size() const noexcept { return anode.size(); }
    bool empty() const noexcept { return anode.empty(); }
};

/// Immutable after construction; evaluation never mutates it, so one Circuit
/// can be used from several threads at once.
class Circuit : public std::enable_shared_from_this<Circuit> {
public:
    /// Validates the spec and builds pattern, values, device slots and the
    /// assembly map. Throws mna::Error(NetlistFatal) on malformed input
    /// (inconsistent table lengths, index outside [-1, n), n or nnz beyond
    /// 32-bit range, a null program or one reading an unknown >= n).
    explicit Circuit(const CircuitSpec& spec);

    std::int32_t n() const noexcept { return n_; }
    std::int32_t nnz() const noexcept { return static_cast<std::int32_t>(row_idx_.size()); }
    double gmin() const noexcept { return gmin_; }
    /// Node-voltage unknowns [0, node_count()) carrying a node-to-ground gmin stamp.
    std::int32_t node_count() const noexcept { return node_count_; }
    bool has_nonlinear() const noexcept { return counts_.total() > 0; }
    const DeviceCounts& device_counts() const noexcept { return counts_; }

    const std::vector<std::int32_t>& col_ptr() const noexcept { return col_ptr_; }
    const std::vector<std::int32_t>& row_idx() const noexcept { return row_idx_; }
    /// Pattern slot of (row, col), -1 if the entry is not in the pattern.
    std::int32_t slot(std::int32_t row, std::int32_t col) const noexcept;

    /// Values in pattern order; Gmin holds the unit stamps of the builder's
    /// gmin pattern plus one unit node-to-ground stamp per node unknown.
    const std::vector<double>& values(MatrixKind which) const noexcept;
    /// G + gmin * gmin_stamps in pattern order (gmin = 0 gives G exactly).
    std::vector<double> g_effective(double gmin) const;
    /// Row-major dense n x n copy of a value array in pattern order.
    void scatter_dense(const double* pattern_values, double* dense) const;
    std::vector<double> to_dense(MatrixKind which) const;

    struct Diodes {
        std::vector<std::int32_t> anode, cathode;
        std::vector<double> is;
        std::vector<std::int32_t> slots;  ///< kDiodeJ per device, -1 for ground entries
    };
    struct Mos {
        std::vector<std::int32_t> drain, gate, source;
        std::vector<double> beta, vth, lambda;
        std::vector<std::int32_t> sign;
        std::vector<std::int32_t> slots;  ///< kMosJ per device
    };
    struct Bjt {
        std::vector<std::int32_t> collector, base, emitter;
        std::vector<double> is, bf, br, vaf, var;
        std::vector<std::int32_t> sign;
        std::vector<std::int32_t> slots;  ///< kBjtJ per device
    };
    struct Behavioral {
        std::vector<std::int32_t> node_p, node_n;
        std::vector<ProgramPtr> programs;
        std::vector<std::int64_t> slot_ptr;  ///< device i: slots[slot_ptr[i] .. slot_ptr[i+1]) = 2 * k_i
        std::vector<std::int32_t> slots;     ///< row p over unknowns, then row n over unknowns
    };
    struct SpiceDiodes {
        std::vector<std::int32_t> anode, cathode;
        std::vector<double> is, n, bv, ibv;
        std::vector<double> bv_eff;       ///< devices::spice_diode_breakdown_voltage(is, bv, ibv)
        std::vector<std::int32_t> slots;  ///< kSpiceDiodeJ per device
    };
    struct GummelPoon {
        std::vector<std::int32_t> collector, base, emitter;
        std::vector<devices::GummelPoonParams> params;
        std::vector<std::int32_t> sign;
        std::vector<std::int32_t> slots;  ///< kGummelPoonJ per device
    };
    struct MosCard {
        std::vector<std::int32_t> drain, gate, source, bulk;
        std::vector<double> beta, vto, lambda, gamma, phi;
        std::vector<std::int32_t> sign;
        std::vector<std::int32_t> slots;  ///< kMosCardJ per device
    };
    const Diodes& diodes() const noexcept { return diodes_; }
    const Mos& mos() const noexcept { return mos_; }
    const Bjt& bjt() const noexcept { return bjt_; }
    const Behavioral& behavioral() const noexcept { return behavioral_; }
    const SpiceDiodes& spice_diodes() const noexcept { return spice_diodes_; }
    const GummelPoon& gummel_poon() const noexcept { return gummel_poon_; }
    const MosCard& mos4() const noexcept { return mos4_; }
    /// Exponential junctions (diodes, BJT base-emitter / base-collector).
    const Junctions& junctions() const noexcept { return junctions_; }

    /// Gather plan of one output vector (residual rows or pattern slots): the
    /// scratch entries that reach output o are src[ptr[o] .. ptr[o+1]), sorted
    /// by (group, device order); target[e] == o and group[e] is the assembly
    /// group of entry e.
    struct Gather {
        std::vector<std::int64_t> ptr;     ///< size outputs + 1
        std::vector<std::int64_t> src;     ///< scratch index of each entry
        std::vector<std::int32_t> target;  ///< output of each entry
        std::vector<std::int8_t> group;    ///< assembly group of each entry
    };
    /// Device d (global order: diodes, behavioral, MOS, BJT, SPICE diodes,
    /// Gummel-Poon BJTs, card MOS) writes its f contributions to
    /// f_scratch[f_offset[d] ..] and its Jacobian contributions to
    /// j_scratch[j_offset[d] ..]; `f` and `j` gather them.
    struct AssemblyMap {
        /// Global index of the first device of each class (DeviceClass order) and the end.
        std::array<std::int64_t, kDeviceClasses + 1> class_begin{};
        std::vector<std::int64_t> f_offset;  ///< size devices + 1
        std::vector<std::int64_t> j_offset;  ///< size devices + 1
        Gather f;
        Gather j;
        /// Number of assembly groups with at least one device (0..4). With at
        /// most one group the grouped sum reduces to a plain sequential sum.
        int groups_present = 0;
    };
    const AssemblyMap& assembly() const noexcept { return map_; }

private:
    void build(const CircuitSpec& spec);

    std::int32_t n_ = 0;
    std::int32_t node_count_ = 0;
    double gmin_ = 0.0;
    DeviceCounts counts_;
    std::vector<std::int32_t> col_ptr_;
    std::vector<std::int32_t> row_idx_;
    std::vector<double> g_vals_, c_vals_, gmin_vals_;
    Diodes diodes_;
    Mos mos_;
    Bjt bjt_;
    Behavioral behavioral_;
    SpiceDiodes spice_diodes_;
    GummelPoon gummel_poon_;
    MosCard mos4_;
    Junctions junctions_;
    AssemblyMap map_;
};

}  // namespace mna::circuit
