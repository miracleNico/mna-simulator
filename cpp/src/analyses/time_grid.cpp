#include "mna/analyses/time_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mna/error.hpp"

namespace mna::analyses {

std::vector<double> transient_time_points(double t_stop, double t_step) {
    if (t_stop < 0.0) {
        throw Error(ErrorCode::TranError, "t_stop must be non-negative.");
    }
    if (t_step <= 0.0) {
        throw Error(ErrorCode::TranError, "t_step must be positive.");
    }
    if (!std::isfinite(t_stop)) {
        throw Error(ErrorCode::TranError, "t_stop must be finite.");
    }
    if (!std::isfinite(t_step)) {
        throw Error(ErrorCode::TranError, "t_step must be finite.");
    }
    if (t_stop == 0.0) {
        return {0.0};
    }

    // Same operations in the same order as the reference (Python's max keeps
    // the first of equal values, which is the same double anyway).
    const double tolerance =
        (std::max)({std::fabs(t_stop), std::fabs(t_step), std::numeric_limits<double>::epsilon()}) * kTimeGridRelTol;
    std::vector<double> points{0.0};
    double current = 0.0;
    while (current < t_stop - tolerance) {
        const double next_time = current + t_step;
        current = next_time >= t_stop - tolerance ? t_stop : next_time;
        if (current <= points.back()) {
            throw Error(ErrorCode::TranError, "transient time grid failed to advance.");
        }
        points.push_back(current);
    }
    if (points.back() != t_stop) {
        points.back() = t_stop;
    }
    return points;
}

}  // namespace mna::analyses
