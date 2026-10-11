// Bindings of the basic analyses: transient time grid, DC operating point
// (damped Newton) and the Backward-Euler transient (mna/analyses/*.hpp).
//
// Every solve releases the GIL for the whole native region; the transient's
// optional progress callback re-acquires it only for the call. Krylov
// statistics come back as the dicts the Python solve_linear facade appends
// to krylov_stats (same keys, same order).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mna/analyses/newton.hpp"
#include "mna/analyses/op.hpp"
#include "mna/analyses/time_grid.hpp"
#include "mna/analyses/transient.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::analyses::KrylovStat;
using mna::circuit::Circuit;

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(holder->data(), shape, owner);
}

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

void check_size(const char* fn, const char* what, std::size_t got, std::size_t expected) {
    if (got != expected) {
        bad_input(std::string(fn) + ": " + what + " has " + std::to_string(got) + " entries, expected " +
                  std::to_string(expected));
    }
}

/// Krylov settings shared by solve_op and solve_transient, plus what the
/// stats dicts need (the request echo and the resolved rank's mode).
struct KrylovRequest {
    mna::analyses::KrylovOptions options;
    std::string rank_mode;
};

KrylovRequest krylov_request(bool use_krylov, const std::string& method, double tol, std::int64_t max_iter,
                             std::int64_t rank, const std::string& rank_mode, const char* fn) {
    if (use_krylov && rank < 1) {
        throw nb::value_error((std::string(fn) + ": krylov_rank must be a resolved positive integer").c_str());
    }
    KrylovRequest request;
    request.options.enabled = use_krylov;
    request.options.method = method;
    request.options.tol = tol;
    request.options.max_iter = max_iter;
    request.options.rank = rank;
    request.rank_mode = rank_mode;
    return request;
}

/// The stats entry of solvers.solve_linear: krylov_solve_linear's info plus
/// used_direct_fallback, keys in the facade's insertion order.
nb::dict krylov_stat_dict(const KrylovStat& stat, const KrylovRequest& request) {
    using namespace mna::linear;
    const KrylovInfo& info = stat.info;
    const bool general = info.route == KrylovRoute::General;
    const double density =
        stat.dimension > 0
            ? static_cast<double>(stat.nnz) / (static_cast<double>(stat.dimension) * static_cast<double>(stat.dimension))
            : 0.0;
    nb::dict d;
    d["method"] = info.method;
    d["engine"] = "mna_core";
    if (info.scipy_semantics) {
        d["operator_storage"] = "csr";
        d["matrix_nnz"] = stat.nnz;
        d["matrix_density"] = density;
        d["converged"] = info.converged;
        d["iterations"] = info.iterations;
        d["relative_residual"] = info.relative_residual;
        d["iteration_budget"] = info.iteration_budget;
        if (info.effective_restart >= 0) {
            d["effective_restart"] = info.effective_restart;
        } else {
            d["effective_restart"] = nb::none();
        }
        d["exit_code"] = info.exit_code;
    } else {
        d["converged"] = info.converged;
        d["iterations"] = info.iterations;
        d["relative_residual"] = info.relative_residual;
    }
    d["requested_method"] = request.options.method;
    d["matrix_kind"] = matrix_kind_name(stat.matrix_kind);
    d["route"] = krylov_route_name(info.route);
    d["matrix_dimension"] = stat.dimension;
    if (!d.contains("matrix_nnz")) {
        d["matrix_nnz"] = stat.nnz;
    }
    if (!d.contains("matrix_density")) {
        d["matrix_density"] = density;
    }
    if (!d.contains("operator_storage")) {
        d["operator_storage"] = stat.csr_operator ? "csr" : "dense";
    }
    d["resolved_rank"] = request.options.rank;
    d["rank_mode"] = request.rank_mode;
    if (!d.contains("iteration_budget")) {
        d["iteration_budget"] = general ? request.options.max_iter : request.options.rank;
    }
    if (general && !d.contains("effective_restart")) {
        d["effective_restart"] = (std::min)(request.options.rank, stat.dimension);
    }
    d["threads_used"] = info.threads_used;
    d["used_direct_fallback"] = stat.used_direct_fallback;
    return d;
}

nb::list krylov_stat_list(const std::vector<KrylovStat>& stats, const KrylovRequest& request) {
    nb::list out;
    for (const KrylovStat& stat : stats) {
        out.append(krylov_stat_dict(stat, request));
    }
    return out;
}

std::vector<mna::analyses::TimeSource> time_sources(const Circuit& circuit, In1D<std::int64_t> rows,
                                                    const nb::handle& programs, const char* fn) {
    std::vector<mna::analyses::TimeSource> sources;
    const std::size_t count = rows.shape(0);
    std::size_t index = 0;
    for (nb::handle item : programs) {
        if (index >= count) {
            bad_input(std::string(fn) + ": more source programs than source rows");
        }
        std::shared_ptr<mna::expr::Program> program;
        if (!nb::try_cast<std::shared_ptr<mna::expr::Program>>(item, program) || !program) {
            bad_input(std::string(fn) + ": source_programs must contain _core.Program objects");
        }
        const std::int64_t row = rows.data()[index];
        if (row < 0 || row >= circuit.n()) {
            bad_input(std::string(fn) + ": source row " + std::to_string(row) + " outside [0, " +
                      std::to_string(circuit.n()) + ")");
        }
        sources.push_back({static_cast<std::int32_t>(row), std::move(program)});
        ++index;
    }
    if (index != count) {
        bad_input(std::string(fn) + ": " + std::to_string(count) + " source rows but " + std::to_string(index) +
                  " programs");
    }
    return sources;
}

nb::ndarray<nb::numpy, double> transient_time_points(double t_stop, double t_step) {
    std::vector<double> points = mna::analyses::transient_time_points(t_stop, t_step);
    const std::size_t size = points.size();
    return to_numpy(std::move(points), {size});
}

nb::ndarray<nb::numpy, double> time_source_vector(const Circuit& circuit, In1D<double> b_dc,
                                                  In1D<std::int64_t> source_rows, nb::handle source_programs,
                                                  double t) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("time_source_vector", "b_dc", b_dc.shape(0), n);
    const auto sources = time_sources(circuit, source_rows, source_programs, "time_source_vector");
    std::vector<double> b = mna::analyses::source_vector(circuit.n(), b_dc.data(), sources, t);
    return to_numpy(std::move(b), {n});
}

nb::tuple solve_op(const Circuit& circuit, In1D<double> b, nb::object x0, double gmin, std::int64_t max_iter,
                   double v_tol, double f_tol, int num_threads, bool use_krylov, const std::string& krylov_method,
                   double krylov_tol, std::int64_t krylov_max_iter, std::int64_t krylov_rank,
                   const std::string& krylov_rank_mode, bool junction_limiting, bool gmin_continuation) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("solve_op", "b", b.shape(0), n);
    std::vector<double> start;
    if (!x0.is_none()) {
        In1D<double> array;
        if (!nb::try_cast<In1D<double>>(x0, array)) {
            bad_input("solve_op: x0 must be a contiguous 1-D float64 array");
        }
        check_size("solve_op", "x0", array.shape(0), n);
        start.assign(array.data(), array.data() + n);
    }
    const KrylovRequest krylov =
        krylov_request(use_krylov, krylov_method, krylov_tol, krylov_max_iter, krylov_rank, krylov_rank_mode, "solve_op");
    mna::analyses::NewtonOptions options;
    options.max_iter = max_iter;
    options.v_tol = v_tol;
    options.f_tol = f_tol;
    options.num_threads = num_threads;
    options.krylov = krylov.options;
    options.junction_limiting = junction_limiting;
    options.gmin_continuation = gmin_continuation;

    mna::analyses::OpResult result;
    {
        nb::gil_scoped_release release;
        result = mna::analyses::op(circuit, b.data(), x0.is_none() ? nullptr : start.data(), gmin, options);
    }
    nb::dict info;
    info["iterations"] = result.iterations;
    info["factorizations"] = result.factorizations;
    info["junction_switches"] = result.junction_switches;
    info["limited_steps"] = result.limited_steps;
    info["monotonicity_accepts"] = result.monotonicity_accepts;
    info["continuation_levels"] = result.continuation_levels;
    info["continuation_used"] = result.continuation_used;
    info["krylov_stats"] = krylov_stat_list(result.krylov_stats, krylov);
    return nb::make_tuple(to_numpy(std::move(result.x), {n}), info);
}

nb::tuple solve_transient(const Circuit& circuit, In1D<double> x0, In1D<double> t_grid, In1D<double> b_dc,
                          In1D<std::int64_t> source_rows, nb::handle source_programs, double gmin,
                          std::int64_t max_iter, double v_tol, double f_tol, int num_threads, bool use_krylov,
                          const std::string& krylov_method, double krylov_tol, std::int64_t krylov_max_iter,
                          std::int64_t krylov_rank, const std::string& krylov_rank_mode, bool krylov_csr_operator,
                          nb::object progress, std::int64_t progress_every, bool junction_limiting) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("solve_transient", "x0", x0.shape(0), n);
    check_size("solve_transient", "b_dc", b_dc.shape(0), n);
    const auto sources = time_sources(circuit, source_rows, source_programs, "solve_transient");
    const std::vector<double> grid(t_grid.data(), t_grid.data() + t_grid.shape(0));
    if (!progress.is_none() && !PyCallable_Check(progress.ptr())) {
        throw nb::type_error("solve_transient: progress must be callable or None");
    }
    const KrylovRequest krylov = krylov_request(use_krylov, krylov_method, krylov_tol, krylov_max_iter, krylov_rank,
                                                krylov_rank_mode, "solve_transient");
    mna::analyses::TranOptions options;
    options.newton.max_iter = max_iter;
    options.newton.v_tol = v_tol;
    options.newton.f_tol = f_tol;
    options.newton.num_threads = num_threads;
    options.newton.krylov = krylov.options;
    options.newton.junction_limiting = junction_limiting;
    options.gmin = gmin;
    options.krylov_csr_operator = krylov_csr_operator;
    options.progress_every = progress_every;

    mna::analyses::ProgressFn callback;
    if (!progress.is_none()) {
        callback = [&progress, n](std::int64_t step, double t, const double* x) {
            nb::gil_scoped_acquire acquire;
            progress(step, t, to_numpy(std::vector<double>(x, x + n), {n}));
        };
    }
    mna::analyses::TranResult result;
    {
        nb::gil_scoped_release release;
        result = mna::analyses::transient(circuit, x0.data(), grid, b_dc.data(), sources, options, callback);
    }
    nb::dict info;
    info["newton_iterations"] = result.newton_iterations;
    info["junction_switches"] = result.junction_switches;
    info["limited_steps"] = result.limited_steps;
    info["monotonicity_accepts"] = result.monotonicity_accepts;
    info["factorizations"] = result.factorizations;
    info["progress_calls"] = result.progress_calls;
    info["krylov_stats"] = krylov_stat_list(result.krylov_stats, krylov);
    const std::size_t points = grid.size();
    return nb::make_tuple(to_numpy(std::move(result.values), {n, points}), info);
}

void bind_analyses_basic(nb::module_& m) {
    m.attr("PARALLEL_DEVICE_MIN_COUNT") = mna::analyses::kParallelDeviceMinCount;

    m.def("transient_time_points", &transient_time_points, "t_stop"_a, "t_step"_a,
          "Fixed-step transient grid (bitwise port of solvers._transient_time_points). Raises\n"
          "CoreError(TRAN_ERROR) for t_stop < 0, t_step <= 0 or a non-finite argument.");
    m.def("time_source_vector", &time_source_vector, "circuit"_a, "b_dc"_a, "source_rows"_a, "source_programs"_a,
          "t"_a, "b_dc with every time source program added to its row, evaluated at t.");
    m.def("solve_op", &solve_op, "circuit"_a, "b"_a, "x0"_a = nb::none(), "gmin"_a = 0.0, "max_iter"_a = 10000,
          "v_tol"_a = 1e-9, "f_tol"_a = 1e-9, "num_threads"_a = 1, "use_krylov"_a = false, "krylov_method"_a = "auto",
          "krylov_tol"_a = 1e-9, "krylov_max_iter"_a = 2000, "krylov_rank"_a = 80, "krylov_rank_mode"_a = "manual",
          "junction_limiting"_a = true, "gmin_continuation"_a = true,
          "DC operating point on G + gmin * gmin_stamps: one linear solve for a linear circuit, the damped\n"
          "Newton driver (from x0, zeros when None) otherwise, switching to the P3b junction strategy (pnjlim\n"
          "and the natural monotonicity test) when it fails or stalls (junction_limiting), and retrying with\n"
          "gmin continuation when it still fails (gmin_continuation). Returns (x, info) with info keys\n"
          "iterations, factorizations, junction_switches, limited_steps, monotonicity_accepts,\n"
          "continuation_levels, continuation_used and krylov_stats (solve_linear stats dicts). GIL released.");
    m.def("solve_transient", &solve_transient, "circuit"_a, "x0"_a, "t_grid"_a, "b_dc"_a, "source_rows"_a,
          "source_programs"_a, "gmin"_a = 0.0, "max_iter"_a = 10000, "v_tol"_a = 1e-9, "f_tol"_a = 1e-9,
          "num_threads"_a = 1, "use_krylov"_a = false, "krylov_method"_a = "auto", "krylov_tol"_a = 1e-9,
          "krylov_max_iter"_a = 2000, "krylov_rank"_a = 80, "krylov_rank_mode"_a = "manual",
          "krylov_csr_operator"_a = true, "progress"_a = nb::none(), "progress_every"_a = 0,
          "junction_limiting"_a = true,
          "Backward-Euler transient from the state x0 over t_grid (junction_limiting: the Newton junction\n"
          "strategy of solve_op on every step). Returns (values (n, T), info) with info keys newton_iterations,\n"
          "junction_switches, limited_steps, monotonicity_accepts, factorizations, progress_calls and\n"
          "krylov_stats. progress(step, t, x) runs after every progress_every-th step and after the last one,\n"
          "with the GIL held; the rest of the call runs without the GIL.");
}

const mna::bindings::Registrar reg_analyses_basic{"analyses_basic", &bind_analyses_basic};

}  // namespace
