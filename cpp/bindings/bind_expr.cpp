// Bindings of the expression bytecode VM (mna/expr/program.hpp).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::expr::Program;

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);  // keep data() non-null for zero-length arrays
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(holder->data(), shape, owner);
}

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

void check_x(const Program& program, const In1D<double>& x, const char* fn) {
    if (x.shape(0) < static_cast<std::size_t>(program.n_unknowns())) {
        bad_input(std::string(fn) + ": x has " + std::to_string(x.shape(0)) + " entries, the program needs " +
                  std::to_string(program.n_unknowns()));
    }
}

/// Tables of compile_program: a sequence of (times, values, period, origin) tuples.
std::vector<mna::expr::Table> tables_from(const nb::handle& tables) {
    std::vector<mna::expr::Table> out;
    if (tables.is_none()) {
        return out;
    }
    std::size_t k = 0;
    for (nb::handle item : tables) {
        const std::string where = "compile_program: table " + std::to_string(k);
        nb::tuple entry;
        if (!nb::try_cast<nb::tuple>(item, entry) || entry.size() != 4) {
            bad_input(where + " must be a (times, values, period, origin) tuple");
        }
        In1D<double> times;
        In1D<double> values;
        double period = 0.0;
        double origin = 0.0;
        if (!nb::try_cast<In1D<double>>(entry[0], times) || !nb::try_cast<In1D<double>>(entry[1], values)) {
            bad_input(where + ": times and values must be contiguous 1-D float64 arrays");
        }
        if (!nb::try_cast<double>(entry[2], period) || !nb::try_cast<double>(entry[3], origin)) {
            bad_input(where + ": period and origin must be numbers");
        }
        mna::expr::Table table;
        table.times.assign(times.data(), times.data() + times.shape(0));
        table.values.assign(values.data(), values.data() + values.shape(0));
        table.period = period;
        table.origin = origin;
        out.push_back(std::move(table));
        ++k;
    }
    return out;
}

std::shared_ptr<Program> compile_program(In1D<std::int32_t> opcodes, In1D<std::int32_t> args, In1D<double> consts,
                                         int n_unknowns, bool time_dependent, nb::handle tables) {
    if (opcodes.shape(0) != args.shape(0)) {
        bad_input("compile_program: opcodes and args differ in length (" + std::to_string(opcodes.shape(0)) + " vs " +
                  std::to_string(args.shape(0)) + ")");
    }
    std::vector<double> c(consts.data(), consts.data() + consts.shape(0));
    std::vector<mna::expr::Table> t = tables_from(tables);
    return std::make_shared<Program>(opcodes.data(), args.data(), opcodes.shape(0), std::move(c), n_unknowns,
                                     time_dependent, std::move(t));
}

double evaluate_program(const Program& program, In1D<double> x, double t) {
    check_x(program, x, "evaluate_program");
    return mna::expr::evaluate(program, x.data(), t);
}

nb::ndarray<nb::numpy, double> evaluate_program_batch(const Program& program, In1D<double> t_array) {
    const std::size_t count = t_array.shape(0);
    std::vector<double> out(count);
    {
        nb::gil_scoped_release release;
        mna::expr::evaluate_many(program, t_array.data(), count, out.data());
    }
    return to_numpy(std::move(out), {count});
}

std::tuple<double, nb::ndarray<nb::numpy, double>> evaluate_program_gradient(const Program& program, In1D<double> x,
                                                                             double t, In1D<std::int32_t> wrt) {
    check_x(program, x, "evaluate_program_gradient");
    const std::size_t count = wrt.shape(0);
    std::vector<double> grad(count);
    double value = 0.0;
    mna::expr::evaluate_with_gradient(program, x.data(), t, wrt.data(), static_cast<int>(count), &value, grad.data());
    return {value, to_numpy(std::move(grad), {count})};
}

nb::dict opcode_table() {
    nb::dict d;
    for (std::int32_t op = 0; op < mna::expr::kOpCount; ++op) {
        d[mna::expr::op_name(static_cast<mna::expr::Op>(op))] = op;
    }
    return d;
}

void bind_expr(nb::module_& m) {
    nb::class_<Program>(m, "Program",
                        "Validated postfix expression program (see mna_simulation/native/expressions.py).")
        .def_prop_ro("n_unknowns", &Program::n_unknowns)
        .def_prop_ro("time_dependent", &Program::time_dependent)
        .def_prop_ro("max_depth", [](const Program& p) { return p.max_depth(); })
        .def_prop_ro("n_tables", [](const Program& p) { return p.tables().size(); },
                     "Number of piecewise-linear tables read by TABLE instructions.")
        .def_prop_ro("unknowns", [](const Program& p) { return p.unknowns(); },
                     "Sorted unknown indices read by the program.")
        .def("__len__", [](const Program& p) { return p.code().size(); })
        .def("opcodes",
             [](const Program& p) {
                 std::vector<std::int32_t> ops;
                 ops.reserve(p.code().size());
                 for (const auto& in : p.code()) {
                     ops.push_back(static_cast<std::int32_t>(in.op));
                 }
                 return to_numpy(std::move(ops), {p.code().size()});
             })
        .def("args",
             [](const Program& p) {
                 std::vector<std::int32_t> args;
                 args.reserve(p.code().size());
                 for (const auto& in : p.code()) {
                     args.push_back(in.arg);
                 }
                 return to_numpy(std::move(args), {p.code().size()});
             })
        .def("consts", [](const Program& p) {
            std::vector<double> c = p.consts();
            const std::size_t size = c.size();
            return to_numpy(std::move(c), {size});
        });

    m.def("expr_opcodes", &opcode_table, "Opcode numbers of the expression VM, keyed by mnemonic.");
    m.def("compile_program", &compile_program, "opcodes"_a, "args"_a, "consts"_a, "n_unknowns"_a,
          "time_dependent"_a, "tables"_a = nb::none(),
          "Validate an expression program (int32 opcodes and operands, float64 constants).\n"
          "tables: None or a sequence of (times, values, period, origin) tuples read by TABLE\n"
          "instructions (float64 arrays, strictly increasing times; period 0 means no repetition).\n"
          "Raises CoreError(NETLIST_FATAL) for a malformed program or table.");
    m.def("evaluate_program", &evaluate_program, "program"_a, "x"_a, "t"_a = 0.0,
          "Value of a program at unknowns x (at least n_unknowns entries) and time t.");
    m.def("evaluate_program_batch", &evaluate_program_batch, "program"_a, "t_array"_a,
          "Values of a program that reads no unknown at every time in t_array (GIL released).");
    m.def("evaluate_program_gradient", &evaluate_program_gradient, "program"_a, "x"_a, "t"_a, "wrt"_a,
          "(value, gradient) with respect to the unknowns listed in wrt (forward-mode dual numbers).");
}

const mna::bindings::Registrar reg_expr{"expr", &bind_expr};

}  // namespace
