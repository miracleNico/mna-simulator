// Trajectory piecewise-linear (TPWL) model-order reduction of mor/tpwl.py.
//
// training_indexes  (_training_indexes): the snapshot indexes that get a local
//     model: floor(linspace(0, T - 1, m)) with m = min(T, max(4, min(16,
//     basis_columns + output_count))) (numpy's float evaluation: j * ((T - 1)
//     / (m - 1)), last point exactly T - 1), plus, when T > 2, every step
//     index i (1-based) whose snapshot delta ||x_i - x_{i-1}||_2 is >= the
//     median of the strictly positive deltas; sorted, unique. The deltas are
//     summed row by row in index order, numpy's axis-0 reduction, so they are
//     bitwise equal to the reference's and the comparisons agree exactly.
//
// tpwl_local_models: for every training index, the device currents f(x_i) and
//     Jacobian values J(x_i) (mna::assembly::evaluate on the compiled circuit,
//     t = 0, J kept in the circuit's CSC pattern) and the reduced quantities
//     the step loop needs: z_i = V^T x_i, Jr_i = V^T J_i V (sparse-dense) and
//     q_i = V^T (J_i x_i - f_i). Models are distributed over threads (one model
//     per thread at a time, each writing only its own slots) when
//     n >= kMorParallelMinUnknowns; the output is identical for any count.
//
// tpwl_simulate (the step loop of solve_transient_tpwl):
//     z_0 = V^T x0, values[:, 0] = Sr z_0
//     step k (h = t_k - t_{k-1}):
//         i   = nearest model (Euclidean distance of z_{k-1} to z_i, first
//               minimum; a NaN distance wins, like numpy.argmin)
//         z_k = _safe_solve((Gr + Cr / h) + Jr_i, V^T b(t_k) + (Cr / h) z_{k-1} + q_i)
//         values[:, k] = Sr z_k
//     which is V^T of the reference's rhs_full = b(t) + (C/h) V z - f_i + J_i x_i
//     and A = Gr + Cr/h + V^T J_i V. The factorization is reused while the model
//     and h are unchanged. No failure detection (the reference has none).
//
// Matrices column-major; snapshots column-major n x T.
// Errors: NETLIST_FATAL for malformed input; JACOBIAN_SINGULAR from the
// least-squares fallback on a non-finite system.
#pragma once

#include <cstdint>
#include <vector>

#include "mna/analyses/transient.hpp"
#include "mna/circuit/circuit.hpp"

namespace mna::mor {

std::vector<std::int64_t> training_indexes(std::int32_t n, std::int64_t T, const double* snapshots,
                                           std::int64_t basis_columns, std::int64_t output_count);

struct TpwlModels {
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::int32_t nnz = 0;
    std::int64_t count = 0;
    std::vector<double> z;      ///< count x k, model-major
    std::vector<double> f;      ///< count x n
    std::vector<double> jvals;  ///< count x nnz (device Jacobian in pattern order)
    std::vector<double> jr;     ///< count x (k x k column-major)
    std::vector<double> q;      ///< count x k
    int threads_used = 1;
};

/// snapshots: column-major n x T (n == circuit.n()); V: column-major n x k.
TpwlModels tpwl_local_models(const circuit::Circuit& circuit, std::int64_t T, const double* snapshots,
                             const std::vector<std::int64_t>& indexes, std::int32_t k, const double* V,
                             int num_threads);

struct TpwlSimulation {
    std::int32_t outputs = 0;
    std::int64_t points = 0;
    std::vector<double> values;       ///< row-major outputs x points
    std::vector<std::int64_t> model;  ///< model used at each step (-1 at step 0)
    std::int64_t factorizations = 0;
    std::int64_t lstsq_steps = 0;
};

/// models: z (count x k), jr (count x k x k), q (count x k) as in TpwlModels.
TpwlSimulation tpwl_simulate(std::int32_t n, std::int32_t k, const double* gr, const double* cr, const double* V,
                             std::int32_t outputs, const double* sr, std::int64_t count, const double* model_z,
                             const double* model_jr, const double* model_q, const double* x0,
                             const std::vector<double>& t_grid, const double* b_dc,
                             const std::vector<analyses::TimeSource>& sources);

}  // namespace mna::mor
