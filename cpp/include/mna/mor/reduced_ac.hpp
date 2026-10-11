// Frequency loop of mor/linear.py::solve_ac_mor on the reduced system.
//
// For every frequency f_i (omega = 2 * pi * f_i, evaluated left to right):
//     z = _safe_solve(Gr + j omega Cr, Br),  y = Sr z,  magnitudes[:, i] = |y|
// with Gr, Cr real k x k, Br complex (k), Sr real p x k (all column-major).
// The complex system is assembled as Complex(gr, omega * cr) (what
// `Gr + 1j * omega * Cr` gives) and solved with a dense partial-pivoting LU;
// an exactly singular one falls back to the least squares of _safe_solve
// (mna/mor/safe_solve.hpp), so the sweep itself never fails.
//
// Threads: like the full AC sweep (mna/analyses/ac.hpp), frequencies are
// split over t = min(num_threads, points / 2) threads when t >= 2 and the
// sweep is big enough to pay for the thread starts (points * k^3 >=
// kReducedAcParallelMinWork; a 12-dimensional model over 100 points is about
// half a millisecond of work), each thread with its own workspace. Every
// frequency is solved from scratch by one thread and writes only its own
// column, so the result is bitwise identical for every thread count.
//
// Errors: NETLIST_FATAL for malformed input; JACOBIAN_SINGULAR from the
// least-squares fallback on a non-finite system.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace mna::mor {

/// Minimum points * k^3 for a frequency-parallel reduced sweep.
inline constexpr double kReducedAcParallelMinWork = 2e6;

struct ReducedAcResult {
    std::int32_t outputs = 0;
    std::int64_t points = 0;
    std::vector<double> magnitudes;  ///< row-major outputs x points
    std::int64_t lstsq_points = 0;   ///< frequencies solved by the least-squares fallback
    int threads_used = 1;
};

ReducedAcResult reduced_ac(std::int32_t k, const double* gr, const double* cr, const std::complex<double>* br,
                           std::int32_t outputs, const double* sr, const double* frequencies, std::int64_t points,
                           int num_threads);

}  // namespace mna::mor
