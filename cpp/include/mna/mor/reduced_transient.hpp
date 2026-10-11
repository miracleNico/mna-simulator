// Step loop of mor/linear.py::solve_transient_mor on the reduced system.
//
//     z_0 = V^T x0,  values[:, 0] = Sr z_0
//     step k (h = t_k - t_{k-1}):
//         Cr_h = Cr / h
//         z_k  = _safe_solve(Gr + Cr_h, V^T b(t_k) + Cr_h z_{k-1})
//         values[:, k] = Sr z_k
// with b(t) = b_dc + the time sources (one compiled program per right-hand-side
// row, mna::analyses::source_vector). Gr + Cr / h is factored once and reused
// while h is bitwise unchanged (the reference recomputes the identical matrix
// every step, so this is exact). A step whose z is non-finite or has
// ||z||_2 > 1e12 marks the run as failed: that column and every later one
// become NaN and the loop stops (`reduced_failed` of the Python loop).
//
// Matrices are column-major (Gr, Cr: k x k; V: n x k; Sr: p x k).
// Errors: NETLIST_FATAL for malformed input (sizes, grid, sources);
// JACOBIAN_SINGULAR from the least-squares fallback on a non-finite system.
#pragma once

#include <cstdint>
#include <vector>

#include "mna/analyses/transient.hpp"

namespace mna::mor {

struct ReducedTransientResult {
    std::int32_t outputs = 0;
    std::int64_t points = 0;
    std::vector<double> values;  ///< row-major outputs x points
    bool failed = false;
    std::int64_t failed_step = -1;    ///< first NaN-filled step when failed
    std::int64_t factorizations = 0;  ///< reduced systems factored
    std::int64_t lstsq_steps = 0;     ///< steps solved by the least-squares fallback
};

ReducedTransientResult reduced_transient(std::int32_t n, std::int32_t k, const double* gr, const double* cr,
                                         const double* V, std::int32_t outputs, const double* sr, const double* x0,
                                         const std::vector<double>& t_grid, const double* b_dc,
                                         const std::vector<analyses::TimeSource>& sources);

/// Shared input checks of the reduced step loops (grid, sources).
void check_reduced_inputs(const char* fn, std::int32_t n, std::int32_t k, const std::vector<double>& t_grid,
                          const std::vector<analyses::TimeSource>& sources);

}  // namespace mna::mor
