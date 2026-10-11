#include "mna/analyses/ac.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/error.hpp"
#include "mna/linear/sparse.hpp"
#include "mna/threads.hpp"

namespace mna::analyses {

namespace {

using linear::Complex;

// numpy.pi; 2 * pi * f is evaluated left to right as in `2 * np.pi * frequency`.
constexpr double kPi = 3.14159265358979323846;

linear::CscPattern circuit_pattern(const circuit::Circuit& circuit) {
    linear::CscPattern pattern;
    pattern.n = circuit.n();
    pattern.col_ptr = circuit.col_ptr();
    pattern.row_idx = circuit.row_idx();
    return pattern;
}

/// csr.values[p] = csc_values[map[p]]: the CSC slot behind every entry of the
/// row view. Built once by transposing the slot numbers themselves (exact in
/// double precision for any 32-bit slot).
std::vector<std::int32_t> csr_slot_map(const linear::CscPattern& pattern, linear::CsrMatrix<Complex>& csr_out) {
    const auto nnz = static_cast<std::size_t>(pattern.nnz());
    std::vector<double> slots(nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
        slots[k] = static_cast<double>(k);
    }
    const auto ids = linear::CsrMatrix<double>::from_csc(pattern, slots.data(), slots.size());
    std::vector<std::int32_t> map(nnz);
    for (std::size_t p = 0; p < nnz; ++p) {
        map[p] = static_cast<std::int32_t>(ids.values[p]);
    }
    csr_out.n = ids.n;
    csr_out.row_ptr = ids.row_ptr;
    csr_out.col_idx = ids.col_idx;
    csr_out.values.assign(nnz, Complex{});
    return map;
}

/// Read-only data shared by every thread of one sweep.
struct Shared {
    const linear::CscPattern* pattern = nullptr;
    const double* glin = nullptr;  // G_lin in pattern order
    const double* cvals = nullptr;  // C in pattern order
    const Complex* b_ac = nullptr;
    const double* frequencies = nullptr;
    const std::vector<std::int32_t>* csr_map = nullptr;  // Krylov only
    const linear::CsrMatrix<Complex>* csr_template = nullptr;  // Krylov only
    const AcOptions* options = nullptr;
    linear::KrylovSettings krylov;
    std::size_t n = 0;
    std::size_t nnz = 0;
};

/// Per-thread solver and buffers.
struct Workspace {
    std::unique_ptr<linear::LinearSolver<Complex>> solver;
    std::vector<Complex> values;
    std::vector<Complex> rhs;
    linear::CsrMatrix<Complex> csr;
    std::vector<Complex> x;

    explicit Workspace(const Shared& s)
        : solver(linear::make_linear_solver<Complex>(*s.pattern, s.options->solver_kind)),
          values(s.nnz),
          rhs(s.n) {
        if (s.options->use_krylov) {
            csr = *s.csr_template;
            x.assign(s.n, Complex{});
        }
    }
};

/// Solves frequency i into `column` (n magnitudes) and, with Krylov, `stat`.
/// Returns false when the direct solver finds the system singular.
bool solve_frequency(const Shared& s, Workspace& ws, std::int64_t i, double* column, AcKrylovStat* stat) {
    const double omega = 2.0 * kPi * s.frequencies[i];
    for (std::size_t k = 0; k < s.nnz; ++k) {
        ws.values[k] = Complex(s.glin[k], omega * s.cvals[k]);
    }

    if (stat != nullptr) {
        // solvers.solve_linear(use_krylov=True) on this frequency's system.
        const std::vector<std::int32_t>& map = *s.csr_map;
        for (std::size_t p = 0; p < s.nnz; ++p) {
            ws.csr.values[p] = ws.values[static_cast<std::size_t>(map[p])];
        }
        stat->matrix_kind = linear::classify_matrix(ws.csr, *s.pattern, ws.values.data());
        const linear::KrylovRoute route = linear::krylov_route(stat->matrix_kind, s.options->krylov_method);
        std::fill(ws.x.begin(), ws.x.end(), Complex{});
        stat->info = linear::krylov_solve(ws.csr, s.b_ac, ws.x.data(), route, s.krylov);
        std::int64_t nonzero = 0;
        for (std::size_t k = 0; k < s.nnz; ++k) {
            nonzero += ws.values[k] != Complex{} ? 1 : 0;
        }
        stat->matrix_nnz = nonzero;
        const double accept = (std::max)(s.options->krylov_tol * 10.0, 1e-8);
        if (stat->info.converged || stat->info.relative_residual <= accept) {
            stat->used_direct_fallback = false;
            for (std::size_t r = 0; r < s.n; ++r) {
                column[r] = std::abs(ws.x[r]);
            }
            return true;
        }
        stat->used_direct_fallback = true;
    }

    try {
        ws.solver->factor(ws.values.data(), ws.values.size());
    } catch (const Error& e) {
        if (e.code() == ErrorCode::JacobianSingular) {
            return false;
        }
        throw;
    }
    std::copy(s.b_ac, s.b_ac + s.n, ws.rhs.begin());
    ws.solver->solve(ws.rhs.data(), ws.rhs.size(), 1);
    for (std::size_t r = 0; r < s.n; ++r) {
        column[r] = std::abs(ws.rhs[r]);
    }
    return true;
}

void atomic_min(std::atomic<std::int64_t>& target, std::int64_t value) noexcept {
    std::int64_t current = target.load();
    while (value < current && !target.compare_exchange_weak(current, value)) {
    }
}

}  // namespace

std::vector<double> linearized_g(const circuit::Circuit& circuit, const double* x_op, double gmin, int num_threads) {
    std::vector<double> g = circuit.g_effective(gmin);
    if (x_op != nullptr && circuit.has_nonlinear()) {
        std::vector<double> jvals(g.size());
        assembly::evaluate(circuit, x_op, 0.0, num_threads, nullptr, jvals.data());
        for (std::size_t k = 0; k < g.size(); ++k) {
            g[k] += jvals[k];
        }
    }
    return g;
}

std::string ac_failure_message(double frequency) {
    char buffer[64];
    if (std::isnan(frequency)) {
        std::snprintf(buffer, sizeof buffer, "nan");  // Python prints "nan" whatever the sign bit
    } else {
        std::snprintf(buffer, sizeof buffer, "%.3e", frequency);
    }
    return std::string("AC solve failed at frequency ") + buffer + " Hz.";
}

AcResult ac_sweep(const circuit::Circuit& circuit, const double* x_op, const Complex* b_ac, const double* frequencies,
                  std::int64_t points, double gmin, const AcOptions& options, int num_threads) {
    if (points <= 0) {
        throw Error(ErrorCode::AcError, "AC sweep needs at least one frequency point (points = " +
                                            std::to_string(points) + ").");
    }
    const int threads = num_threads < 1 ? 1 : num_threads;
    if (options.use_krylov) {
        if (options.krylov_rank < 1) {
            throw Error(ErrorCode::NetlistFatal, "ac_sweep: krylov_rank must be a positive integer");
        }
        // Fails here, before any work, for an unknown method (std::invalid_argument).
        (void)linear::krylov_route(linear::MatrixKind::General, options.krylov_method);
    }

    const linear::CscPattern pattern = circuit_pattern(circuit);
    const std::vector<double> glin = linearized_g(circuit, x_op, gmin, threads);
    const std::vector<double>& cvals = circuit.values(circuit::MatrixKind::C);

    const std::int64_t frequency_threads = (std::min)(static_cast<std::int64_t>(threads), points / 2);
    const bool parallel = frequency_threads >= 2;

    Shared s;
    s.pattern = &pattern;
    s.glin = glin.data();
    s.cvals = cvals.data();
    s.b_ac = b_ac;
    s.frequencies = frequencies;
    s.options = &options;
    s.n = static_cast<std::size_t>(circuit.n());
    s.nnz = glin.size();
    s.krylov.tol = options.krylov_tol;
    s.krylov.max_iter = options.krylov_max_iter;
    s.krylov.rank = options.krylov_rank;
    s.krylov.num_threads = parallel ? 1 : threads;
    std::vector<std::int32_t> csr_map;
    linear::CsrMatrix<Complex> csr_template;
    if (options.use_krylov) {
        csr_map = csr_slot_map(pattern, csr_template);
        s.csr_map = &csr_map;
        s.csr_template = &csr_template;
    }

    AcResult result;
    result.n = circuit.n();
    result.points = points;
    const auto npoints = static_cast<std::size_t>(points);
    std::vector<double> columns(s.n * npoints, 0.0);  // frequency-major: column i at i * n
    if (options.use_krylov) {
        result.krylov_stats.resize(npoints);
    }

    // Frequencies are handed out in increasing order; a thread stops once its
    // next index lies beyond a recorded failure. Every index below the lowest
    // failing one is therefore solved, and the lowest failure is found for any
    // thread count.
    // Any other exception stops every thread at its next frequency and is
    // rethrown by parallel_for.
    std::atomic<std::int64_t> next{0};
    std::atomic<std::int64_t> first_failure{points};
    std::atomic<bool> aborted{false};
    auto sweep = [&](std::int64_t, std::int64_t) {
        std::unique_ptr<Workspace> ws;
        try {
            for (;;) {
                const std::int64_t i = next.fetch_add(1);
                if (i >= points || i > first_failure.load() || aborted.load()) {
                    return;
                }
                if (!ws) {
                    ws = std::make_unique<Workspace>(s);  // symbolic analysis once per thread
                }
                const auto u = static_cast<std::size_t>(i);
                AcKrylovStat* stat = options.use_krylov ? &result.krylov_stats[u] : nullptr;
                if (!solve_frequency(s, *ws, i, columns.data() + u * s.n, stat)) {
                    atomic_min(first_failure, i);
                }
            }
        } catch (...) {
            aborted.store(true);
            throw;
        }
    };
    if (parallel) {
        // One item per thread: each owns one workspace (solver analyzed once).
        result.threads_used = parallel_for(frequency_threads, static_cast<int>(frequency_threads), sweep);
    } else {
        sweep(0, 1);
        result.threads_used = 1;
    }

    const std::int64_t failed = first_failure.load();
    if (failed < points) {
        throw Error(ErrorCode::AcError, ac_failure_message(frequencies[failed]));
    }

    result.magnitudes.assign(s.n * npoints, 0.0);
    for (std::size_t i = 0; i < npoints; ++i) {
        const double* column = columns.data() + i * s.n;
        for (std::size_t r = 0; r < s.n; ++r) {
            result.magnitudes[r * npoints + i] = column[r];
        }
    }
    return result;
}

}  // namespace mna::analyses
