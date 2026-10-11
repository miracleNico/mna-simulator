#include "mna/mor/reduced_transient.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "mna/error.hpp"
#include "mna/mor/projection.hpp"
#include "mna/mor/safe_solve.hpp"

namespace mna::mor {

void check_reduced_inputs(const char* fn, std::int32_t n, std::int32_t k, const std::vector<double>& t_grid,
                          const std::vector<analyses::TimeSource>& sources) {
    auto bad = [fn](const std::string& message) {
        throw Error(ErrorCode::NetlistFatal, std::string(fn) + ": " + message);
    };
    if (n < 0 || k < 0) {
        bad("negative dimension");
    }
    if (t_grid.empty()) {
        bad("the time grid is empty");
    }
    for (std::size_t s = 1; s < t_grid.size(); ++s) {
        if (!(t_grid[s] > t_grid[s - 1])) {
            bad("the time grid must be strictly increasing (point " + std::to_string(s) + ")");
        }
    }
    for (const analyses::TimeSource& source : sources) {
        if (source.row < 0 || source.row >= n) {
            bad("time source row " + std::to_string(source.row) + " outside [0, " + std::to_string(n) + ")");
        }
        if (!source.program) {
            bad("null time source program");
        }
        if (!source.program->unknowns().empty()) {
            bad("a time source program reads circuit unknowns");
        }
    }
}

ReducedTransientResult reduced_transient(std::int32_t n, std::int32_t k, const double* gr, const double* cr,
                                         const double* V, std::int32_t outputs, const double* sr, const double* x0,
                                         const std::vector<double>& t_grid, const double* b_dc,
                                         const std::vector<analyses::TimeSource>& sources) {
    check_reduced_inputs("mor_reduced_transient", n, k, t_grid, sources);
    if (outputs < 0) {
        throw Error(ErrorCode::NetlistFatal, "mor_reduced_transient: negative output count");
    }
    const auto uk = static_cast<std::size_t>(k);
    const auto up = static_cast<std::size_t>(outputs);
    const std::size_t points = t_grid.size();

    ReducedTransientResult result;
    result.outputs = outputs;
    result.points = static_cast<std::int64_t>(points);
    result.values.assign(up * points, 0.0);

    auto record = [&](const std::vector<double>& z, std::size_t step) {
        for (std::size_t r = 0; r < up; ++r) {
            double y = 0.0;
            for (std::size_t c = 0; c < uk; ++c) {
                y += sr[r + c * up] * z[c];
            }
            result.values[r * points + step] = y;
        }
    };

    std::vector<double> z(uk);
    project_vector(n, k, V, x0, z.data());
    record(z, 0);

    DenseSafeSolver<double> solver;
    std::vector<double> crh(uk * uk);
    std::vector<double> a(uk * uk);
    std::vector<double> vb(uk);
    std::vector<double> rhs(uk);
    double cached_h = 0.0;
    bool have_factor = false;
    for (std::size_t step = 1; step < points; ++step) {
        const double h = t_grid[step] - t_grid[step - 1];
        if (!have_factor || h != cached_h) {
            for (std::size_t e = 0; e < uk * uk; ++e) {
                crh[e] = cr[e] / h;
                a[e] = gr[e] + crh[e];
            }
            solver.factor(k, a.data());
            cached_h = h;
            have_factor = true;
            ++result.factorizations;
        }
        // rhs = Vr @ (b_dc + b_time(t)) + Cr_h @ z_prev
        const std::vector<double> b = analyses::source_vector(n, b_dc, sources, t_grid[step]);
        project_vector(n, k, V, b.data(), vb.data());
        for (std::size_t r = 0; r < uk; ++r) {
            double s = 0.0;
            for (std::size_t c = 0; c < uk; ++c) {
                s += crh[r + c * uk] * z[c];
            }
            rhs[r] = vb[r] + s;
        }
        solver.solve(rhs.data(), 1);
        z = rhs;
        result.lstsq_steps += solver.path() == SolvePath::LU ? 0 : 1;
        bool finite = true;
        double squares = 0.0;
        for (double value : z) {
            finite = finite && std::isfinite(value);
            squares += value * value;
        }
        if (!finite || std::sqrt(squares) > 1e12) {
            result.failed = true;
            result.failed_step = static_cast<std::int64_t>(step);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (std::size_t r = 0; r < up; ++r) {
                for (std::size_t s = step; s < points; ++s) {
                    result.values[r * points + s] = nan;
                }
            }
            break;
        }
        record(z, step);
    }
    return result;
}

}  // namespace mna::mor
