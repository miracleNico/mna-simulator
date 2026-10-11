#include "mna/mor/tpwl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/devices/fp_contract.hpp"
#include "mna/error.hpp"
#include "mna/linear/linear_solver.hpp"
#include "mna/mor/krylov_basis.hpp"
#include "mna/mor/projection.hpp"
#include "mna/mor/reduced_transient.hpp"
#include "mna/mor/safe_solve.hpp"
#include "mna/threads.hpp"

namespace mna::mor {
namespace {

[[noreturn]] void bad_input(const char* fn, const std::string& message) {
    throw Error(ErrorCode::NetlistFatal, std::string(fn) + ": " + message);
}

}  // namespace

std::vector<std::int64_t> training_indexes(std::int32_t n, std::int64_t T, const double* snapshots,
                                           std::int64_t basis_columns, std::int64_t output_count) {
    MNA_NO_FP_CONTRACT
    if (n < 0 || T < 0) {
        bad_input("mor_training_indexes", "negative dimension");
    }
    if (n > 0 && T > 0 && snapshots == nullptr) {
        bad_input("mor_training_indexes", "null snapshot array");
    }
    const std::int64_t steps = T;
    const std::int64_t count =
        (std::min)(steps, (std::max)(std::int64_t{4}, (std::min)(std::int64_t{16}, basis_columns + output_count)));
    std::set<std::int64_t> indexes;
    // np.linspace(0, steps - 1, count, dtype=int): float64 points j * step, the last one set to
    // steps - 1 exactly, floored, cast to int.
    if (count == 1) {
        indexes.insert(0);
    } else if (count > 1) {
        const double step = static_cast<double>(steps - 1) / static_cast<double>(count - 1);
        for (std::int64_t j = 0; j < count - 1; ++j) {
            indexes.insert(static_cast<std::int64_t>(std::floor(static_cast<double>(j) * step)));
        }
        indexes.insert(steps - 1);
    }
    if (steps > 2) {
        // np.linalg.norm(np.diff(snapshots, axis=1), axis=0): squares summed row by row.
        const auto un = static_cast<std::size_t>(n);
        const auto ndeltas = static_cast<std::size_t>(steps - 1);
        std::vector<double> deltas(ndeltas, 0.0);
        for (std::size_t j = 0; j < ndeltas; ++j) {
            const double* a = snapshots + j * un;
            const double* b = snapshots + (j + 1) * un;
            double sum = 0.0;
            for (std::size_t i = 0; i < un; ++i) {
                const double d = b[i] - a[i];
                sum += d * d;
            }
            deltas[j] = std::sqrt(sum);
        }
        std::vector<double> positive;
        positive.reserve(ndeltas);
        for (double d : deltas) {
            if (d > 0.0) {
                positive.push_back(d);
            }
        }
        if (!positive.empty()) {
            std::sort(positive.begin(), positive.end());
            const std::size_t m = positive.size();
            const double trigger =
                (m % 2 == 1) ? positive[m / 2] : (positive[m / 2 - 1] + positive[m / 2]) / 2.0;  // np.median
            for (std::size_t j = 0; j < ndeltas; ++j) {
                if (deltas[j] >= trigger) {
                    indexes.insert(static_cast<std::int64_t>(j) + 1);
                }
            }
        }
    }
    return std::vector<std::int64_t>(indexes.begin(), indexes.end());
}

TpwlModels tpwl_local_models(const circuit::Circuit& circuit, std::int64_t T, const double* snapshots,
                             const std::vector<std::int64_t>& indexes, std::int32_t k, const double* V,
                             int num_threads) {
    const char* fn = "mor_tpwl_models";
    const std::int32_t n = circuit.n();
    if (T < 0 || k < 0) {
        bad_input(fn, "negative dimension");
    }
    if ((n > 0 && T > 0 && snapshots == nullptr) || (n > 0 && k > 0 && V == nullptr)) {
        bad_input(fn, "null input array");
    }
    for (std::int64_t index : indexes) {
        if (index < 0 || index >= T) {
            bad_input(fn, "training index " + std::to_string(index) + " outside [0, " + std::to_string(T) + ")");
        }
    }
    linear::CscPattern pattern;
    pattern.n = n;
    pattern.col_ptr = circuit.col_ptr();
    pattern.row_idx = circuit.row_idx();

    TpwlModels models;
    models.n = n;
    models.k = k;
    models.nnz = circuit.nnz();
    models.count = static_cast<std::int64_t>(indexes.size());
    const auto un = static_cast<std::size_t>(n);
    const auto uk = static_cast<std::size_t>(k);
    const auto unnz = static_cast<std::size_t>(models.nnz);
    const auto count = static_cast<std::size_t>(models.count);
    models.z.assign(count * uk, 0.0);
    models.f.assign(count * un, 0.0);
    models.jvals.assign(count * unnz, 0.0);
    models.jr.assign(count * uk * uk, 0.0);
    models.q.assign(count * uk, 0.0);

    auto build = [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> jx(un);
        for (std::int64_t m = begin; m < end; ++m) {
            const auto um = static_cast<std::size_t>(m);
            const double* x = snapshots + static_cast<std::size_t>(indexes[um]) * un;
            double* f = models.f.data() + um * un;
            double* jvals = models.jvals.data() + um * unnz;
            assembly::evaluate(circuit, x, 0.0, 1, f, jvals);
            project_vector(n, k, V, x, models.z.data() + um * uk);
            const std::vector<double> jr = project_matrix(pattern, jvals, V, k);
            std::copy(jr.begin(), jr.end(), models.jr.begin() + static_cast<std::ptrdiff_t>(um * uk * uk));
            sparse_matvec(pattern, jvals, x, jx.data());
            for (std::size_t i = 0; i < un; ++i) {
                jx[i] -= f[i];
            }
            project_vector(n, k, V, jx.data(), models.q.data() + um * uk);
        }
    };
    const int threads = n >= kMorParallelMinUnknowns ? effective_threads(num_threads) : 1;
    if (threads > 1 && models.count > 1) {
        models.threads_used = parallel_for(models.count, threads, build);
    } else {
        build(0, models.count);
    }
    return models;
}

TpwlSimulation tpwl_simulate(std::int32_t n, std::int32_t k, const double* gr, const double* cr, const double* V,
                             std::int32_t outputs, const double* sr, std::int64_t count, const double* model_z,
                             const double* model_jr, const double* model_q, const double* x0,
                             const std::vector<double>& t_grid, const double* b_dc,
                             const std::vector<analyses::TimeSource>& sources) {
    const char* fn = "mor_tpwl_simulate";
    check_reduced_inputs(fn, n, k, t_grid, sources);
    if (outputs < 0) {
        bad_input(fn, "negative output count");
    }
    if (count <= 0) {
        bad_input(fn, "no local models");
    }
    const auto uk = static_cast<std::size_t>(k);
    const auto up = static_cast<std::size_t>(outputs);
    const std::size_t points = t_grid.size();

    TpwlSimulation result;
    result.outputs = outputs;
    result.points = static_cast<std::int64_t>(points);
    result.values.assign(up * points, 0.0);
    result.model.assign(points, -1);

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
    std::int64_t cached_model = -1;
    for (std::size_t step = 1; step < points; ++step) {
        const double h = t_grid[step] - t_grid[step - 1];
        // _nearest_model: argmin of the Euclidean distances (first minimum; NaN wins like numpy.argmin).
        std::int64_t nearest = 0;
        double best = 0.0;
        for (std::int64_t m = 0; m < count; ++m) {
            const double* zm = model_z + static_cast<std::size_t>(m) * uk;
            double squares = 0.0;
            for (std::size_t c = 0; c < uk; ++c) {
                const double d = z[c] - zm[c];
                squares += d * d;
            }
            const double distance = std::sqrt(squares);
            if (std::isnan(distance)) {
                nearest = m;
                break;
            }
            if (m == 0 || distance < best) {
                best = distance;
                nearest = m;
            }
        }
        result.model[step] = nearest;
        const auto um = static_cast<std::size_t>(nearest);
        if (nearest != cached_model || h != cached_h) {
            const double* jr = model_jr + um * uk * uk;
            for (std::size_t e = 0; e < uk * uk; ++e) {
                crh[e] = cr[e] / h;
                a[e] = (gr[e] + crh[e]) + jr[e];
            }
            solver.factor(k, a.data());
            cached_h = h;
            cached_model = nearest;
            ++result.factorizations;
        }
        const std::vector<double> b = analyses::source_vector(n, b_dc, sources, t_grid[step]);
        project_vector(n, k, V, b.data(), vb.data());
        const double* q = model_q + um * uk;
        for (std::size_t r = 0; r < uk; ++r) {
            double s = 0.0;
            for (std::size_t c = 0; c < uk; ++c) {
                s += crh[r + c * uk] * z[c];
            }
            rhs[r] = (vb[r] + s) + q[r];
        }
        solver.solve(rhs.data(), 1);
        z = rhs;
        result.lstsq_steps += solver.path() == SolvePath::LU ? 0 : 1;
        record(z, step);
    }
    return result;
}

}  // namespace mna::mor
