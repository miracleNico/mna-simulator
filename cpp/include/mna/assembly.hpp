// Deterministic evaluation of the nonlinear device currents f(x, t) and of
// their Jacobian values on the compiled circuit pattern.
//
// Phase 1 evaluates every device into its own scratch slots, in parallel over
// device chunks (mna::parallel_for_chunks). Phase 2 gathers the scratch into
// residual rows and pattern slots through the circuit's assembly map: each
// output is summed by one thread in a fixed device order, in four groups
// ((diodes + behavioral), MOS, BJT, model-card devices) whose totals are added
// in that order; the first three are the grouping of the Python core
// (`f = f_symbolic; f += f_mos; f += f_bjt`), the fourth (SPICE diodes,
// Gummel-Poon BJTs, card MOS, phase M2) is empty for every circuit without
// them. The result is bitwise identical for every thread count;
// num_threads <= 1 runs the same chunk functions inline.
#pragma once

#include <cstdint>
#include <vector>

#include "mna/circuit/circuit.hpp"

namespace mna::assembly {

/// f_out: n device currents (nullptr to skip). jvals_out: nnz Jacobian values
/// in pattern order, device contributions only (slots without a device entry
/// are 0; nullptr to skip). x holds the n unknowns; ground is 0 V.
void evaluate(const circuit::Circuit& circuit, const double* x, double t, int num_threads, double* f_out,
              double* jvals_out);

/// Same as evaluate with the Jacobian scattered into a row-major dense n x n
/// array (f_out may be nullptr).
void evaluate_dense_jacobian(const circuit::Circuit& circuit, const double* x, double t, int num_threads,
                             double* f_out, double* j_dense_out);

/// Phase 1 for the devices [begin, end) of the global device order; writes
/// f_scratch / j_scratch at the circuit's assembly offsets. Exposed for tests.
void evaluate_device_range(const circuit::Circuit& circuit, const double* x, double t, std::int64_t begin,
                           std::int64_t end, double* f_scratch, double* j_scratch);

/// Per-device operating point data for core_basic._device_operating_points.
/// MOS: ids, gds = dIds/dVd, gm = dIds/dVg, d_source = dIds/dVs, region
/// (mna::devices::MosRegion). BJT: ic, ib, ie, gm = dIc/dVb, go = dIc/dVc,
/// rpi = 1/(dIb/dVb) (inf when |dIb/dVb| <= 1e-30), region (BjtRegion).
/// SPICE diodes: junction voltage vd, current id, conductance gd, the
/// effective breakdown voltage, region (DiodeRegion). Gummel-Poon BJTs: as
/// the Level-1 BJT at the intrinsic terminals. Card MOS: as the Level-1 MOS
/// plus gmbs = dIds/dVb and the body-adjusted threshold vth (n-channel frame).
struct OperatingPoints {
    std::vector<double> mos_ids, mos_gds, mos_gm, mos_d_source;
    std::vector<std::int32_t> mos_region;
    std::vector<double> bjt_ic, bjt_ib, bjt_ie, bjt_gm, bjt_go, bjt_rpi;
    std::vector<std::int32_t> bjt_region;
    std::vector<double> spice_diode_vd, spice_diode_id, spice_diode_gd, spice_diode_bv_eff;
    std::vector<std::int32_t> spice_diode_region;
    std::vector<double> gp_ic, gp_ib, gp_ie, gp_gm, gp_go, gp_rpi;
    std::vector<std::int32_t> gp_region;
    std::vector<double> mos4_ids, mos4_gds, mos4_gm, mos4_d_source, mos4_gmbs, mos4_vth;
    std::vector<std::int32_t> mos4_region;
};

OperatingPoints operating_points(const circuit::Circuit& circuit, const double* x);

}  // namespace mna::assembly
