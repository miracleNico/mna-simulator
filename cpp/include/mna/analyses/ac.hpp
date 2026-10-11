// Small-signal AC sweep on a compiled circuit, parallel over frequencies.
//
// Linearization: G_lin = G + gmin * gmin_stamps (+ J(x_op), the device
// Jacobian at the operating point, when x_op is given). For every frequency f_i
// the complex system (G_lin + j * 2 * pi * f_i * C) x = b_ac is assembled in
// the circuit's CSC pattern and solved; |x| becomes column i of the label-major
// (n, points) magnitude array.
//
// Direct path: the complex direct solver of the dense/KLU policy
// (linear::make_linear_solver<Complex>), one solver per thread, analyzed once
// per thread, a full factorization (fresh partial pivoting) per frequency.
// Krylov path (options.use_krylov): the matrix is classified, routed and solved
// with the native complex Krylov kernels exactly like solvers.solve_linear
// does through krylov_solve_linear (zero initial guess); the answer is accepted
// when it converged or its relative residual is <= max(10 * tol, 1e-8),
// otherwise the direct solver runs and the entry records
// used_direct_fallback = true.
//
// Threads: frequencies are distributed over k = min(num_threads, points / 2)
// threads when k >= 2 (i.e. only when points >= 2 * k); otherwise the sweep
// runs on the calling thread and num_threads goes to the Krylov kernels'
// matrix-vector products. Every frequency is solved from scratch by exactly one
// thread and writes only its own output column and stats entry, so the result
// is bitwise identical for every thread count.
//
// Errors (mna::Error):
//   AC_ERROR       points <= 0; a singular system: "AC solve failed at
//                  frequency 1.000e+03 Hz." for the lowest-index frequency
//                  whose system is singular (what a serial sweep reports)
//   NETLIST_FATAL  malformed input (propagated from the solvers)
// std::invalid_argument for an unknown Krylov method.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mna/circuit/circuit.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"
#include "mna/linear/linear_solver.hpp"

namespace mna::analyses {

struct AcOptions {
    bool use_krylov = false;
    /// Krylov method or route alias ("auto" classifies each frequency's matrix).
    std::string krylov_method = "auto";
    double krylov_tol = 1e-9;
    std::int64_t krylov_max_iter = 2000;
    /// Resolved Krylov rank (solvers.resolve_krylov_rank), >= 1.
    std::int64_t krylov_rank = 80;
    linear::SolverKind solver_kind = linear::SolverKind::Auto;
};

/// One Krylov attempt (one per frequency, in frequency order).
struct AcKrylovStat {
    linear::MatrixKind matrix_kind = linear::MatrixKind::General;
    linear::KrylovInfo info;
    /// Nonzero entries of G_lin + j omega C (numpy.count_nonzero of the dense system).
    std::int64_t matrix_nnz = 0;
    bool used_direct_fallback = false;
};

struct AcResult {
    std::int32_t n = 0;
    std::int64_t points = 0;
    /// |x| as an n x points row-major array (row = unknown, column = frequency).
    std::vector<double> magnitudes;
    /// One entry per frequency when options.use_krylov, empty otherwise.
    std::vector<AcKrylovStat> krylov_stats;
    /// Threads that solved frequencies (1 when the sweep ran on the caller).
    int threads_used = 1;
};

/// G + gmin * gmin_stamps, plus the device Jacobian at x_op (n values) when
/// x_op is not null, in pattern order.
std::vector<double> linearized_g(const circuit::Circuit& circuit, const double* x_op, double gmin, int num_threads);

/// "AC solve failed at frequency 1.000e+03 Hz." (printf "%.3e", like Python's format).
std::string ac_failure_message(double frequency);

/// The sweep described above. x_op: n operating-point values or nullptr (no
/// linearization); b_ac: n complex values; frequencies: `points` values in Hz.
AcResult ac_sweep(const circuit::Circuit& circuit, const double* x_op, const linear::Complex* b_ac,
                  const double* frequencies, std::int64_t points, double gmin, const AcOptions& options,
                  int num_threads);

}  // namespace mna::analyses
