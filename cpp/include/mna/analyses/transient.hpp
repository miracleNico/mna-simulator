// Backward-Euler transient (.tran) on the compiled circuit, a port of the
// Python core's solve_transient stepping loop.
//
// Step k (h = t_k - t_{k-1}):
//     (G_eff + C / h) x_k + f(x_k) = (C / h) x_{k-1} + b(t_k)
// with G_eff = G + gmin * gmin_stamps and b(t) = b_dc + time sources, each
// source a compiled program evaluated at t_k (one program per right-hand-side
// row, as compile_time_source_func compiles them).
//
// * Linear circuits factor G_eff + C / h when the step size changes and reuse
//   the factorization while it does not. Every step uses its exact
//   h = t_k - t_{k-1}, like the reference; because the grid is accumulated, h
//   of a "constant-step" grid changes in its last bits a few times per binade
//   of t (40 x 1 ns: 7 factorizations, 100000 x 10 us: 18), so a run factors
//   O(log(steps)) times instead of once per step.
// * With Krylov enabled on a linear circuit the solve_linear semantics apply
//   per step (initial guess x_{k-1}, acceptance rule, direct fallback). When
//   `krylov_csr_operator` is set (the problem has SciPy G/C, i.e. the
//   reference's sparse_linear_krylov branch) the step matrix is formed as
//   G + C * (1 / h), the arithmetic of SciPy's sparse division, and the Krylov
//   stats report a CSR operator; otherwise C / h and a dense operator.
// * Nonlinear circuits run the damped Newton driver from x_{k-1} on every
//   step (with its junction strategy, newton.hpp); the direct solver
//   refactors with the pivot order of the previous iteration or step.
// * An optional progress callback runs after every `progress_every`-th step
//   and after the last one, with the step index, its time and the state.
//
// Errors (mna::Error):
//   NR_CONVERGENCE     "Linear system is singular at t={t:.2e} s (G + C/h matrix): Singular matrix"
//                      (2.4.1: the reference raised a raw numpy LinAlgError)
//                      "Newton step could not reduce residual at t={t:.2e}, iter={i}"
//                      "NR failed to converge at t={t:.2e} s"
//                      "non-finite solution at t={t:.2e} s" (2.4.1)
//   JACOBIAN_SINGULAR  "Jacobian singular at t={t:.2e}, iter={i}"
//   NETLIST_FATAL      malformed input (grid, sizes, source rows/programs)
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mna/analyses/newton.hpp"
#include "mna/circuit/circuit.hpp"

namespace mna::analyses {

/// One time-dependent right-hand-side contribution: b[row] += program(t).
struct TimeSource {
    std::int32_t row = 0;
    circuit::ProgramPtr program;
};

struct TranOptions {
    NewtonOptions newton;
    double gmin = 0.0;
    bool krylov_csr_operator = true;
    /// Call the progress callback every this many steps (0: never).
    std::int64_t progress_every = 0;
};

/// progress(step, t, x): step >= 1 is the index of the time point just
/// solved, x its n state values (valid during the call only).
using ProgressFn = std::function<void(std::int64_t step, double t, const double* x)>;

struct TranResult {
    std::vector<double> values;  ///< n x T, row-major: values[i * T + k] = x_i(t_k)
    std::int64_t newton_iterations = 0;
    std::int64_t junction_switches = 0;     ///< Newton solves that switched to the junction strategy
    std::int64_t limited_steps = 0;         ///< Newton iterations changed by the junction limiter
    std::int64_t monotonicity_accepts = 0;  ///< damping trials accepted by the monotonicity test
    std::int64_t factorizations = 0;
    std::int64_t progress_calls = 0;
    std::vector<KrylovStat> krylov_stats;
};

/// x0: the n initial state values (t_grid[0]); b_dc: n values; t_grid: at
/// least one strictly increasing point (transient_time_points()).
TranResult transient(const circuit::Circuit& circuit, const double* x0, const std::vector<double>& t_grid,
                     const double* b_dc, const std::vector<TimeSource>& sources, const TranOptions& options,
                     const ProgressFn& progress = {});

/// b_dc + sum of the sources at time t (n values).
std::vector<double> source_vector(std::int32_t n, const double* b_dc, const std::vector<TimeSource>& sources,
                                  double t);

/// Python's f"{t:.2e}".
std::string format_time(double t);

}  // namespace mna::analyses
