// Fixed-step transient time grid: a bitwise port of the Python core's
// solvers._transient_time_points.
//
// The grid starts at 0 and is built by repeated addition (current + t_step,
// never i * t_step), so (1.0, 0.1) contains 0.30000000000000004 exactly like
// the reference. A point that lands within
//     tolerance = max(|t_stop|, |t_step|, DBL_EPSILON) * 1e-9
// of t_stop (or beyond it) is replaced by t_stop, which shortens the last step
// (or appends a tiny one when the overshoot is outside the tolerance), and the
// last point is always exactly t_stop.
//
// Errors (mna::Error, code TRAN_ERROR), checked in this order:
//   t_stop < 0                      "t_stop must be non-negative."
//   t_step <= 0                     "t_step must be positive."
//   t_stop not finite (inf / NaN)   "t_stop must be finite."
//   t_step not finite (inf / NaN)   "t_step must be finite."
//   a point that does not advance   "transient time grid failed to advance."
// The first two are the reference messages; NaN passed both reference checks
// and produced the one-point grid [nan], an infinite t_stop looped forever.
#pragma once

#include <vector>

namespace mna::analyses {

/// Snapping tolerance factor relative to max(|t_stop|, |t_step|, DBL_EPSILON).
inline constexpr double kTimeGridRelTol = 1e-9;

/// Time points 0 = t_0 < t_1 < ... < t_last = t_stop; t_stop == 0 gives {0.0}.
std::vector<double> transient_time_points(double t_stop, double t_step);

}  // namespace mna::analyses
