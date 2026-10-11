// mna_simulation._core: Python bindings of the native numerical core.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/threads.hpp"
#include "mna/version.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <class T>
using InVector = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InMatrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutVector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Hands a std::vector to NumPy without copying. Needs the GIL.
OutVector to_numpy(std::vector<double>&& values) {
    auto* holder = new std::vector<double>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);  // keep data() non-null for zero-length arrays
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return OutVector(holder->data(), {holder->size()}, owner);
}

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

OutVector klu_solve(InVector<std::int32_t> Ap, InVector<std::int32_t> Ai, InVector<double> Ax, InVector<double> b) {
    if (Ap.shape(0) == 0) {
        bad_input("klu_solve: Ap must have n + 1 entries");
    }
    const std::size_t n = Ap.shape(0) - 1;
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        bad_input("klu_solve: matrix too large for 32-bit indices");
    }
    if (b.shape(0) != n) {
        bad_input("klu_solve: b has " + std::to_string(b.shape(0)) + " entries, expected n = " + std::to_string(n));
    }
    std::vector<double> x(b.data(), b.data() + n);
    {
        nb::gil_scoped_release release;
        mna::linear::KluSolver lu;
        lu.analyze(static_cast<std::int32_t>(n), Ap.data(), Ap.shape(0), Ai.data(), Ai.shape(0));
        lu.factor(Ax.data(), Ax.shape(0));
        lu.solve(x);
    }
    return to_numpy(std::move(x));
}

std::pair<OutVector, int> parallel_row_sums(InMatrix a, int num_threads) {
    const std::size_t rows = a.shape(0);
    const std::size_t cols = a.shape(1);
    std::vector<double> sums(rows);
    int used = 1;
    {
        nb::gil_scoped_release release;
        const double* data = a.data();
        double* out = sums.data();
        used = mna::parallel_for(static_cast<std::int64_t>(rows), mna::effective_threads(num_threads),
                                 [data, out, cols](std::int64_t begin, std::int64_t end) {
                                     for (std::int64_t r = begin; r < end; ++r) {
                                         const double* row = data + static_cast<std::size_t>(r) * cols;
                                         double s = 0.0;
                                         for (std::size_t c = 0; c < cols; ++c) {
                                             s += row[c];
                                         }
                                         out[r] = s;
                                     }
                                 });
    }
    return {to_numpy(std::move(sums)), used};
}

nb::dict build_info_dict() {
    const mna::BuildInfo info = mna::build_info();
    nb::dict d;
    d["version"] = info.version;
    d["compiler"] = info.compiler;
    d["compiler_version"] = info.compiler_version;
    d["cxx_standard"] = info.cxx_standard;
    d["build_type"] = info.build_type;
    d["threads_impl"] = info.threads_impl;
    d["hardware_concurrency"] = info.hardware_concurrency;
    d["eigen_version"] = info.eigen_version;
    d["suitesparse_version"] = info.suitesparse_version;
    d["klu_version"] = info.klu_version;
    d["platform"] = info.platform;
    d["sanitizers"] = info.sanitizers;
    d["coverage"] = info.coverage;
    d["native_arch"] = info.native_arch;
    d["debug_asserts"] = info.debug_asserts;
    return d;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Native numerical core of mna_simulation (C++17, KLU, Eigen).";
    m.attr("__version__") = mna::version();

    // CoreError(RuntimeError) carries .code (the Python core's error code
    // string, e.g. "JACOBIAN_SINGULAR") and .message. The type object lives as
    // long as the process; this module keeps one strong reference to it.
    PyObject* core_error = PyErr_NewException("mna_simulation._core.CoreError", PyExc_RuntimeError, nullptr);
    if (core_error == nullptr) {
        throw nb::python_error();
    }
    m.attr("CoreError") = nb::handle(core_error);
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* payload) {
            try {
                std::rethrow_exception(p);
            } catch (const mna::Error& e) {
                nb::handle type(static_cast<PyObject*>(payload));
                nb::object exc = type(e.what());
                exc.attr("code") = e.code_name();
                exc.attr("message") = e.what();
                PyErr_SetObject(type.ptr(), exc.ptr());
            }
        },
        core_error);

    m.def("version", &mna::version, "Version string of the native core.");
    m.def("build_info", &build_info_dict, "How the native core was built (compiler, versions, flags).");
    m.def("effective_threads", &mna::effective_threads, "requested"_a,
          "Thread count used for a request: <= 0 -> 1, capped at the hardware concurrency.");
    m.def("klu_solve", &klu_solve, "Ap"_a, "Ai"_a, "Ax"_a, "b"_a,
          "Solve A x = b for a square CSC matrix (int32 indices) with KLU (analyze + factor + solve).\n"
          "Raises CoreError with code JACOBIAN_SINGULAR for a singular matrix and NETLIST_FATAL for\n"
          "malformed input.");
    m.def("parallel_row_sums", &parallel_row_sums, "a"_a, "num_threads"_a = 1,
          "Row sums of a 2-D float64 array computed on up to num_threads threads with the GIL released.\n"
          "Returns (sums, threads_used); the result is bitwise identical for every thread count.");

    // Units registered from bindings/bind_*.cpp (see registry.hpp).
    for (const mna::bindings::Entry& entry : mna::bindings::registry()) {
        entry.fn(m);
    }
}
