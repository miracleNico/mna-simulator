// Bindings of the compiled circuit (mna/circuit/circuit.hpp).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;
using mna::circuit::MatrixKind;

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

[[noreturn]] void bad_spec(const std::string& message) {
    throw mna::Error(mna::ErrorCode::NetlistFatal, "build_circuit: " + message);
}

/// spec[name] as a dict, or an empty dict when absent / None.
nb::dict sub_table(const nb::dict& spec, const char* name) {
    if (!spec.contains(name)) {
        return nb::dict();
    }
    nb::object value = spec[name];
    if (value.is_none()) {
        return nb::dict();
    }
    if (!nb::isinstance<nb::dict>(value)) {
        bad_spec(std::string("'") + name + "' must be a dict of arrays");
    }
    return nb::borrow<nb::dict>(value);
}

template <class T>
std::vector<T> column(const nb::dict& table, const char* table_name, const char* name) {
    if (!table.contains(name)) {
        return {};
    }
    In1D<T> array;
    if (!nb::try_cast<In1D<T>>(table[name], array)) {
        bad_spec(std::string(table_name) + "." + name + " must be a contiguous 1-D " +
                 (sizeof(T) == 8 ? (std::is_floating_point_v<T> ? "float64" : "int64") : "int32") + " array");
    }
    return std::vector<T>(array.data(), array.data() + array.shape(0));
}

void read_triplets(const nb::dict& spec, const char* name, mna::circuit::TripletList& out) {
    const nb::dict t = sub_table(spec, name);
    out.rows = column<std::int64_t>(t, name, "rows");
    out.cols = column<std::int64_t>(t, name, "cols");
    out.values = column<double>(t, name, "values");
}

std::shared_ptr<Circuit> build_circuit(const nb::dict& spec) {
    CircuitSpec s;
    if (!spec.contains("n")) {
        bad_spec("missing 'n'");
    }
    s.n = nb::cast<std::int64_t>(spec["n"]);
    if (spec.contains("gmin")) {
        s.gmin = nb::cast<double>(spec["gmin"]);
    }
    if (spec.contains("node_count") && !spec["node_count"].is_none()) {
        s.node_count = nb::cast<std::int64_t>(spec["node_count"]);
    }
    read_triplets(spec, "G", s.g);
    read_triplets(spec, "C", s.c);
    read_triplets(spec, "gmin_stamps", s.gmin_stamps);

    const nb::dict d = sub_table(spec, "diodes");
    s.diodes.anode = column<std::int64_t>(d, "diodes", "anode");
    s.diodes.cathode = column<std::int64_t>(d, "diodes", "cathode");
    s.diodes.is = column<double>(d, "diodes", "is");

    const nb::dict mos = sub_table(spec, "mos");
    s.mos.drain = column<std::int64_t>(mos, "mos", "drain");
    s.mos.gate = column<std::int64_t>(mos, "mos", "gate");
    s.mos.source = column<std::int64_t>(mos, "mos", "source");
    s.mos.beta = column<double>(mos, "mos", "beta");
    s.mos.vth = column<double>(mos, "mos", "vth");
    s.mos.lambda = column<double>(mos, "mos", "lambda");
    s.mos.sign = column<std::int32_t>(mos, "mos", "sign");

    const nb::dict bjt = sub_table(spec, "bjt");
    s.bjt.collector = column<std::int64_t>(bjt, "bjt", "collector");
    s.bjt.base = column<std::int64_t>(bjt, "bjt", "base");
    s.bjt.emitter = column<std::int64_t>(bjt, "bjt", "emitter");
    s.bjt.is = column<double>(bjt, "bjt", "is");
    s.bjt.bf = column<double>(bjt, "bjt", "bf");
    s.bjt.br = column<double>(bjt, "bjt", "br");
    s.bjt.vaf = column<double>(bjt, "bjt", "vaf");
    s.bjt.var = column<double>(bjt, "bjt", "var");
    s.bjt.sign = column<std::int32_t>(bjt, "bjt", "sign");

    const nb::dict sd = sub_table(spec, "spice_diodes");
    s.spice_diodes.anode = column<std::int64_t>(sd, "spice_diodes", "anode");
    s.spice_diodes.cathode = column<std::int64_t>(sd, "spice_diodes", "cathode");
    s.spice_diodes.is = column<double>(sd, "spice_diodes", "is");
    s.spice_diodes.n = column<double>(sd, "spice_diodes", "n");
    s.spice_diodes.bv = column<double>(sd, "spice_diodes", "bv");
    s.spice_diodes.ibv = column<double>(sd, "spice_diodes", "ibv");

    const nb::dict gp = sub_table(spec, "gummel_poon");
    s.gummel_poon.collector = column<std::int64_t>(gp, "gummel_poon", "collector");
    s.gummel_poon.base = column<std::int64_t>(gp, "gummel_poon", "base");
    s.gummel_poon.emitter = column<std::int64_t>(gp, "gummel_poon", "emitter");
    s.gummel_poon.is = column<double>(gp, "gummel_poon", "is");
    s.gummel_poon.bf = column<double>(gp, "gummel_poon", "bf");
    s.gummel_poon.nf = column<double>(gp, "gummel_poon", "nf");
    s.gummel_poon.br = column<double>(gp, "gummel_poon", "br");
    s.gummel_poon.nr = column<double>(gp, "gummel_poon", "nr");
    s.gummel_poon.vaf = column<double>(gp, "gummel_poon", "vaf");
    s.gummel_poon.var = column<double>(gp, "gummel_poon", "var");
    s.gummel_poon.ikf = column<double>(gp, "gummel_poon", "ikf");
    s.gummel_poon.ikr = column<double>(gp, "gummel_poon", "ikr");
    s.gummel_poon.ise = column<double>(gp, "gummel_poon", "ise");
    s.gummel_poon.ne = column<double>(gp, "gummel_poon", "ne");
    s.gummel_poon.isc = column<double>(gp, "gummel_poon", "isc");
    s.gummel_poon.nc = column<double>(gp, "gummel_poon", "nc");
    s.gummel_poon.sign = column<std::int32_t>(gp, "gummel_poon", "sign");

    const nb::dict m4 = sub_table(spec, "mos4");
    s.mos4.drain = column<std::int64_t>(m4, "mos4", "drain");
    s.mos4.gate = column<std::int64_t>(m4, "mos4", "gate");
    s.mos4.source = column<std::int64_t>(m4, "mos4", "source");
    s.mos4.bulk = column<std::int64_t>(m4, "mos4", "bulk");
    s.mos4.beta = column<double>(m4, "mos4", "beta");
    s.mos4.vto = column<double>(m4, "mos4", "vto");
    s.mos4.lambda = column<double>(m4, "mos4", "lambda");
    s.mos4.gamma = column<double>(m4, "mos4", "gamma");
    s.mos4.phi = column<double>(m4, "mos4", "phi");
    s.mos4.sign = column<std::int32_t>(m4, "mos4", "sign");

    const nb::dict beh = sub_table(spec, "behavioral");
    s.behavioral.node_p = column<std::int64_t>(beh, "behavioral", "node_p");
    s.behavioral.node_n = column<std::int64_t>(beh, "behavioral", "node_n");
    if (beh.contains("programs")) {
        for (nb::handle item : beh["programs"]) {
            std::shared_ptr<mna::expr::Program> program;
            if (!nb::try_cast<std::shared_ptr<mna::expr::Program>>(item, program) || !program) {
                bad_spec("behavioral.programs must contain _core.Program objects");
            }
            s.behavioral.programs.push_back(std::move(program));
        }
    }

    std::shared_ptr<Circuit> circuit;
    {
        nb::gil_scoped_release release;
        circuit = std::make_shared<Circuit>(s);
    }
    return circuit;
}

MatrixKind parse_kind(const std::string& name) {
    if (name == "G" || name == "g") {
        return MatrixKind::G;
    }
    if (name == "C" || name == "c") {
        return MatrixKind::C;
    }
    if (name == "gmin") {
        return MatrixKind::Gmin;
    }
    throw mna::Error(mna::ErrorCode::NetlistFatal, "unknown matrix '" + name + "' (expected 'G', 'C' or 'gmin')");
}

void bind_circuit(nb::module_& m) {
    nb::class_<Circuit>(m, "Circuit",
                        "Compiled circuit: union CSC pattern of G, C, gmin stamps and device Jacobians, static values\n"
                        "in pattern order, device tables with pattern slots and the assembly map.")
        .def_prop_ro("n", &Circuit::n)
        .def_prop_ro("nnz", &Circuit::nnz)
        .def_prop_ro("gmin", &Circuit::gmin)
        .def_prop_ro("node_count", &Circuit::node_count,
                     "Node-voltage unknowns [0, node_count) carrying a node-to-ground gmin stamp.")
        .def_prop_ro("has_nonlinear", &Circuit::has_nonlinear)
        .def_prop_ro("device_counts",
                     [](const Circuit& c) {
                         nb::dict d;
                         d["diode"] = c.device_counts().diodes;
                         d["mos"] = c.device_counts().mos;
                         d["bjt"] = c.device_counts().bjt;
                         d["behavioral"] = c.device_counts().behavioral;
                         return d;
                     },
                     "Devices per Level-1 / ideal class: diode, mos, bjt, behavioral (model-card devices are in\n"
                     "card_device_counts).")
        .def_prop_ro("card_device_counts",
                     [](const Circuit& c) {
                         nb::dict d;
                         d["spice_diode"] = c.device_counts().spice_diodes;
                         d["gummel_poon"] = c.device_counts().gummel_poon;
                         d["mos4"] = c.device_counts().mos4;
                         return d;
                     },
                     "Model-card devices (phase M2): spice_diode, gummel_poon, mos4.")
        .def("pattern",
             [](const Circuit& c) {
                 std::vector<std::int32_t> col_ptr = c.col_ptr();
                 std::vector<std::int32_t> row_idx = c.row_idx();
                 const std::size_t np = col_ptr.size();
                 const std::size_t nz = row_idx.size();
                 return std::make_tuple(to_numpy(std::move(col_ptr), {np}), to_numpy(std::move(row_idx), {nz}));
             },
             "(col_ptr, row_idx) of the CSC pattern, int32.")
        .def("slot", &Circuit::slot, "row"_a, "col"_a, "Pattern slot of (row, col), -1 when absent.")
        .def("values",
             [](const Circuit& c, const std::string& name) {
                 std::vector<double> v = c.values(parse_kind(name));
                 const std::size_t size = v.size();
                 return to_numpy(std::move(v), {size});
             },
             "name"_a,
             "Values of 'G', 'C' or 'gmin' in pattern order ('gmin': the unit gmin stamps of the builder plus one\n"
             "unit node-to-ground stamp per node unknown).")
        .def("g_effective",
             [](const Circuit& c, double gmin) {
                 std::vector<double> v = c.g_effective(gmin);
                 const std::size_t size = v.size();
                 return to_numpy(std::move(v), {size});
             },
             "gmin"_a, "Values of G + gmin * gmin_stamps in pattern order.")
        .def("to_dense",
             [](const Circuit& c, const std::string& name) {
                 const auto n = static_cast<std::size_t>(c.n());
                 return to_numpy(c.to_dense(parse_kind(name)), {n, n});
             },
             "name"_a, "Dense n x n copy of 'G', 'C' or 'gmin'.")
        .def("device_slots",
             [](const Circuit& c, const std::string& kind) {
                 std::vector<std::int32_t> slots;
                 if (kind == "diode") {
                     slots = c.diodes().slots;
                 } else if (kind == "mos") {
                     slots = c.mos().slots;
                 } else if (kind == "bjt") {
                     slots = c.bjt().slots;
                 } else if (kind == "behavioral") {
                     slots = c.behavioral().slots;
                 } else if (kind == "spice_diode") {
                     slots = c.spice_diodes().slots;
                 } else if (kind == "gummel_poon") {
                     slots = c.gummel_poon().slots;
                 } else if (kind == "mos4") {
                     slots = c.mos4().slots;
                 } else {
                     throw mna::Error(mna::ErrorCode::NetlistFatal, "unknown device kind '" + kind + "'");
                 }
                 const std::size_t size = slots.size();
                 return to_numpy(std::move(slots), {size});
             },
             "kind"_a, "Pattern slots of every device Jacobian entry (-1 for ground), in device order.")
        .def("junctions",
             [](const Circuit& c) {
                 const mna::circuit::Junctions& j = c.junctions();
                 const std::size_t size = j.size();
                 nb::dict d;
                 d["anode"] = to_numpy(std::vector<std::int32_t>(j.anode), {size});
                 d["cathode"] = to_numpy(std::vector<std::int32_t>(j.cathode), {size});
                 d["vt"] = to_numpy(std::vector<double>(j.vt), {size});
                 d["is"] = to_numpy(std::vector<double>(j.is), {size});
                 d["vcrit"] = to_numpy(std::vector<double>(j.vcrit), {size});
                 return d;
             },
             "Exponential junctions used by the junction voltage-step limiter: dict of arrays anode, cathode\n"
             "(int32, -1 = ground), vt, is, vcrit (diodes first, then each Level-1 BJT's base-emitter and\n"
             "base-collector junction, then the SPICE diodes, then each Gummel-Poon BJT's junctions).")
        .def("spice_diode_breakdown_voltages",
             [](const Circuit& c) {
                 std::vector<double> v = c.spice_diodes().bv_eff;
                 const std::size_t size = v.size();
                 return to_numpy(std::move(v), {size});
             },
             "Effective (continuity-shifted) breakdown voltage of every SPICE diode, +inf without breakdown.");

    m.def("build_circuit", &build_circuit, "spec"_a,
          "Compile a circuit from plain tables (see mna_simulation/native/circuit.py for the layout).\n"
          "Raises CoreError(NETLIST_FATAL) on malformed input.");
}

const mna::bindings::Registrar reg_circuit{"circuit", &bind_circuit};

}  // namespace
