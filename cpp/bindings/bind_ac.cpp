// Bindings of the AC sweep (mna/analyses/ac.hpp).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mna/analyses/ac.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"
#include "mna/threads.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::circuit::Circuit;
using Complex = std::complex<double>;

using RealVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ComplexVector = nb::ndarray<const Complex, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

nb::ndarray<nb::numpy, double> to_numpy(std::vector<double>&& values, std::initializer_list<std::size_t> shape) {
    auto* holder = new std::vector<double>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::ndarray<nb::numpy, double>(holder->data(), shape, owner);
}

mna::analyses::AcOptions parse_options(const nb::handle& options) {
    mna::analyses::AcOptions out;
    if (options.is_none()) {
        return out;
    }
    if (!nb::isinstance<nb::dict>(options)) {
        bad_input("ac_sweep: options must be a dict or None");
    }
    for (auto [key, value] : nb::borrow<nb::dict>(options)) {
        const std::string name = nb::cast<std::string>(key);
        if (name == "use_krylov") {
            out.use_krylov = nb::cast<bool>(value);
        } else if (name == "krylov_method") {
            out.krylov_method = nb::cast<std::string>(value);
        } else if (name == "krylov_tol") {
            out.krylov_tol = nb::cast<double>(value);
        } else if (name == "krylov_max_iter") {
            out.krylov_max_iter = nb::cast<std::int64_t>(value);
        } else if (name == "krylov_rank") {
            out.krylov_rank = nb::cast<std::int64_t>(value);
        } else {
            bad_input("ac_sweep: unknown option '" + name + "'");
        }
    }
    return out;
}

nb::dict krylov_entry(const mna::analyses::AcKrylovStat& stat, std::int32_t n) {
    using namespace mna::linear;
    const KrylovInfo& info = stat.info;
    nb::dict d;
    d["method"] = info.method;
    d["kernel"] = krylov_algorithm_name(info.algorithm);
    d["route"] = krylov_route_name(info.route);
    d["matrix_kind"] = matrix_kind_name(stat.matrix_kind);
    d["converged"] = info.converged;
    d["iterations"] = info.iterations;
    d["relative_residual"] = info.relative_residual;
    d["iteration_budget"] = info.iteration_budget;
    if (info.effective_restart >= 0) {
        d["effective_restart"] = info.effective_restart;
    } else if (info.scipy_semantics) {
        d["effective_restart"] = nb::none();
    }
    if (info.scipy_semantics) {
        d["exit_code"] = info.exit_code;
    }
    d["scipy_semantics"] = info.scipy_semantics;
    d["matrix_dimension"] = static_cast<std::int64_t>(n);
    d["matrix_nnz"] = stat.matrix_nnz;
    d["threads_used"] = info.threads_used;
    d["used_direct_fallback"] = stat.used_direct_fallback;
    return d;
}

nb::tuple ac_sweep(const Circuit& circuit, std::optional<RealVector> x_op, ComplexVector b_ac, RealVector frequencies,
                   double gmin, nb::handle options, int num_threads) {
    const auto n = static_cast<std::size_t>(circuit.n());
    if (x_op.has_value() && x_op->shape(0) != n) {
        bad_input("ac_sweep: x_op has " + std::to_string(x_op->shape(0)) + " entries, expected n = " +
                  std::to_string(n));
    }
    if (b_ac.shape(0) != n) {
        bad_input("ac_sweep: b_ac has " + std::to_string(b_ac.shape(0)) + " entries, expected n = " +
                  std::to_string(n));
    }
    const mna::analyses::AcOptions opts = parse_options(options);
    const double* x = x_op.has_value() ? x_op->data() : nullptr;
    const auto points = static_cast<std::int64_t>(frequencies.shape(0));
    const int threads = mna::effective_threads(num_threads);

    mna::analyses::AcResult result;
    {
        nb::gil_scoped_release release;
        result = mna::analyses::ac_sweep(circuit, x, b_ac.data(), frequencies.data(), points, gmin, opts, threads);
    }

    nb::dict info;
    info["threads_used"] = result.threads_used;
    if (opts.use_krylov) {
        nb::list entries;
        for (const mna::analyses::AcKrylovStat& stat : result.krylov_stats) {
            entries.append(krylov_entry(stat, result.n));
        }
        info["krylov"] = entries;
    } else {
        info["krylov"] = nb::none();
    }
    const auto npoints = static_cast<std::size_t>(points);
    return nb::make_tuple(to_numpy(std::move(result.magnitudes), {n, npoints}), info);
}

nb::ndarray<nb::numpy, double> linearized_g(const Circuit& circuit, std::optional<RealVector> x_op, double gmin,
                                            int num_threads) {
    const auto n = static_cast<std::size_t>(circuit.n());
    if (x_op.has_value() && x_op->shape(0) != n) {
        bad_input("linearized_g: x_op has " + std::to_string(x_op->shape(0)) + " entries, expected n = " +
                  std::to_string(n));
    }
    const double* x = x_op.has_value() ? x_op->data() : nullptr;
    std::vector<double> values;
    {
        nb::gil_scoped_release release;
        values = mna::analyses::linearized_g(circuit, x, gmin, mna::effective_threads(num_threads));
    }
    const std::size_t size = values.size();
    return to_numpy(std::move(values), {size});
}

void bind_ac(nb::module_& m) {
    m.def("ac_sweep", &ac_sweep, "circuit"_a, "x_op"_a.none(), "b_ac"_a, "frequencies"_a, "gmin"_a = 0.0,
          "options"_a = nb::none(), "num_threads"_a = 1,
          "Small-signal AC sweep: |x| of (G + gmin*gmin_stamps [+ J(x_op)] + j*2*pi*f*C) x = b_ac for every\n"
          "frequency f. x_op: operating point (n,) or None (no linearization); b_ac: complex128 (n,);\n"
          "options: None or a dict with use_krylov, krylov_method, krylov_tol, krylov_max_iter and\n"
          "krylov_rank (resolved). Parallel over frequencies when points >= 2 * threads; bitwise identical\n"
          "for any thread count; GIL released. Returns (magnitudes (n, points), info) where info holds\n"
          "threads_used and krylov (one dict per frequency with use_krylov, else None). Raises CoreError\n"
          "AC_ERROR ('AC solve failed at frequency 1.000e+03 Hz.') at the first singular frequency and for\n"
          "an empty sweep.");
    m.def("linearized_g", &linearized_g, "circuit"_a, "x_op"_a.none(), "gmin"_a = 0.0, "num_threads"_a = 1,
          "G + gmin * gmin_stamps (+ the device Jacobian at x_op) in pattern order.");
}

const mna::bindings::Registrar reg_ac{"ac", &bind_ac};

}  // namespace
