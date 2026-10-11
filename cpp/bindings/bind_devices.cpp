// Bindings of the device kernels and the deterministic assembly
// (mna/devices/*.hpp, mna/assembly.hpp).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mna/assembly.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/devices/bjt_level1.hpp"
#include "mna/devices/mos_level1.hpp"
#include "mna/error.hpp"
#include "mna/threads.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::circuit::Circuit;

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(holder->data(), shape, owner);
}

/// A new NumPy float64 array with uninitialized contents (created with the GIL
/// held; the caller fills `data` afterwards, possibly without the GIL).
struct NewArray {
    double* data;
    nb::ndarray<nb::numpy, double> array;
};

NewArray new_array(std::initializer_list<std::size_t> shape) {
    std::size_t count = 1;
    for (std::size_t extent : shape) {
        count *= extent;
    }
    auto* data = new double[count == 0 ? 1 : count];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return {data, nb::ndarray<nb::numpy, double>(data, shape, owner)};
}

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

void check_size(const char* fn, const char* what, std::size_t got, std::size_t expected) {
    if (got != expected) {
        bad_input(std::string(fn) + ": " + what + " has " + std::to_string(got) + " entries, expected " +
                  std::to_string(expected));
    }
}

void evaluate_devices(const Circuit& circuit, In1D<double> x, double t, int num_threads, nb::object out_f,
                      nb::object out_jvals) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("evaluate_devices", "x", x.shape(0), n);
    double* f = nullptr;
    double* jv = nullptr;
    Out1D f_array;
    Out1D j_array;
    if (!out_f.is_none()) {
        if (!nb::try_cast<Out1D>(out_f, f_array, false)) {
            bad_input("evaluate_devices: out_f must be a writable contiguous float64 vector");
        }
        check_size("evaluate_devices", "out_f", f_array.shape(0), n);
        f = f_array.data();
    }
    if (!out_jvals.is_none()) {
        if (!nb::try_cast<Out1D>(out_jvals, j_array, false)) {
            bad_input("evaluate_devices: out_jvals must be a writable contiguous float64 vector");
        }
        check_size("evaluate_devices", "out_jvals", j_array.shape(0), static_cast<std::size_t>(circuit.nnz()));
        jv = j_array.data();
    }
    const int threads = mna::effective_threads(num_threads);
    nb::gil_scoped_release release;
    mna::assembly::evaluate(circuit, x.data(), t, threads, f, jv);
}

nb::ndarray<nb::numpy, double> device_f(const Circuit& circuit, In1D<double> x, double t) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("device_f", "x", x.shape(0), n);
    NewArray f = new_array({n, 1});
    {
        nb::gil_scoped_release release;
        mna::assembly::evaluate(circuit, x.data(), t, 1, f.data, nullptr);
    }
    return f.array;
}

nb::ndarray<nb::numpy, double> device_jacobian_dense(const Circuit& circuit, In1D<double> x, double t) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("device_jacobian_dense", "x", x.shape(0), n);
    NewArray j = new_array({n, n});
    {
        nb::gil_scoped_release release;
        mna::assembly::evaluate_dense_jacobian(circuit, x.data(), t, 1, nullptr, j.data);
    }
    return j.array;
}

std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> device_f_jacobian_dense(
    const Circuit& circuit, In1D<double> x, double t) {
    const auto n = static_cast<std::size_t>(circuit.n());
    check_size("device_f_jacobian_dense", "x", x.shape(0), n);
    NewArray f = new_array({n, 1});
    NewArray j = new_array({n, n});
    {
        nb::gil_scoped_release release;
        mna::assembly::evaluate_dense_jacobian(circuit, x.data(), t, 1, f.data, j.data);
    }
    return {f.array, j.array};
}

std::tuple<double, double, double, double> eval_mos_level1(int sign, double beta, double vth, double lam, double vd,
                                                           double vg, double vs) {
    const mna::devices::MosEval r = mna::devices::mos_level1(sign, beta, vth, lam, vd, vg, vs);
    return {r.ids, r.d_drain, r.d_gate, r.d_source};
}

using Triple = std::tuple<double, double, double>;

std::tuple<Triple, std::tuple<Triple, Triple, Triple>> eval_bjt_level1(int sign, double is, double bf, double br,
                                                                       double vaf, double var, double vc, double vb,
                                                                       double ve) {
    const mna::devices::BjtEval r = mna::devices::bjt_level1(sign, is, bf, br, vaf, var, vc, vb, ve);
    auto row = [&r](int i) { return Triple{r.d[i][0], r.d[i][1], r.d[i][2]}; };
    return {Triple{r.i[0], r.i[1], r.i[2]}, {row(0), row(1), row(2)}};
}

nb::dict operating_points(const Circuit& circuit, In1D<double> x) {
    check_size("operating_points", "x", x.shape(0), static_cast<std::size_t>(circuit.n()));
    mna::assembly::OperatingPoints op = mna::assembly::operating_points(circuit, x.data());
    const std::size_t nm = op.mos_ids.size();
    const std::size_t nb_ = op.bjt_ic.size();
    nb::dict mos;
    mos["ids"] = to_numpy(std::move(op.mos_ids), {nm});
    mos["gds"] = to_numpy(std::move(op.mos_gds), {nm});
    mos["gm"] = to_numpy(std::move(op.mos_gm), {nm});
    mos["d_source"] = to_numpy(std::move(op.mos_d_source), {nm});
    mos["region"] = to_numpy(std::move(op.mos_region), {nm});
    nb::dict bjt;
    bjt["ic"] = to_numpy(std::move(op.bjt_ic), {nb_});
    bjt["ib"] = to_numpy(std::move(op.bjt_ib), {nb_});
    bjt["ie"] = to_numpy(std::move(op.bjt_ie), {nb_});
    bjt["gm"] = to_numpy(std::move(op.bjt_gm), {nb_});
    bjt["go"] = to_numpy(std::move(op.bjt_go), {nb_});
    bjt["rpi"] = to_numpy(std::move(op.bjt_rpi), {nb_});
    bjt["region"] = to_numpy(std::move(op.bjt_region), {nb_});
    nb::dict out;
    out["mos"] = mos;
    out["bjt"] = bjt;
    return out;
}

void bind_devices(nb::module_& m) {
    m.def("evaluate_devices", &evaluate_devices, "circuit"_a, "x"_a, "t"_a, "num_threads"_a, "out_f"_a.none(),
          "out_jvals"_a.none(),
          "Device currents into out_f (n) and Jacobian values in pattern order into out_jvals (nnz); either\n"
          "may be None. Two-phase deterministic assembly, GIL released; bitwise identical for any thread count.");
    m.def("device_f", &device_f, "circuit"_a, "x"_a, "t"_a = 0.0, "Device currents f(x, t) as an (n, 1) array.");
    m.def("device_jacobian_dense", &device_jacobian_dense, "circuit"_a, "x"_a, "t"_a = 0.0,
          "Device Jacobian df/dx as a dense (n, n) array.");
    m.def("device_f_jacobian_dense", &device_f_jacobian_dense, "circuit"_a, "x"_a, "t"_a = 0.0,
          "(f (n, 1), dense Jacobian (n, n)) from one evaluation.");
    m.def("eval_mos_level1", &eval_mos_level1, "sign"_a, "beta"_a, "vth"_a, "lam"_a, "vd"_a, "vg"_a, "vs"_a,
          "Level-1 MOS (sign +1 NMOS, -1 PMOS): (ids, dIds/dVd, dIds/dVg, dIds/dVs).");
    m.def("eval_bjt_level1", &eval_bjt_level1, "sign"_a, "is_"_a, "bf"_a, "br"_a, "vaf"_a, "var"_a, "vc"_a, "vb"_a,
          "ve"_a,
          "Level-1 BJT (sign +1 NPN, -1 PNP): ((ic, ib, ie), ((dic/dvc, dic/dvb, dic/dve), (dib/...), (die/...))).");
    m.def("operating_points", &operating_points, "circuit"_a, "x"_a,
          "{'mos': {ids, gds, gm, d_source, region}, 'bjt': {ic, ib, ie, gm, go, rpi, region}} at x; region\n"
          "codes index mos_region_names / bjt_region_names.");
    m.attr("mos_region_names") = nb::make_tuple("off", "triode", "saturation", "reverse");
    m.attr("bjt_region_names") = nb::make_tuple("cutoff", "forward_active", "reverse_active", "saturation");
}

const mna::bindings::Registrar reg_devices{"devices", &bind_devices};

}  // namespace
