// Expression VM: every opcode, NumPy scalar semantics, validation, gradients.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/expr/program.hpp"

using mna::expr::Op;
using mna::expr::Program;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Program make(std::initializer_list<std::pair<Op, std::int32_t>> code, std::vector<double> consts = {},
             std::int32_t n_unknowns = 0, bool time_dependent = true) {
    std::vector<std::int32_t> ops;
    std::vector<std::int32_t> args;
    for (const auto& [op, arg] : code) {
        ops.push_back(static_cast<std::int32_t>(op));
        args.push_back(arg);
    }
    return Program(ops.data(), args.data(), ops.size(), std::move(consts), n_unknowns, time_dependent);
}

/// Program computing op(c0, c1) (binary) or op(c0) (unary) on constants.
double binary(Op op, double a, double b) {
    return mna::expr::evaluate(make({{Op::Const, 0}, {Op::Const, 1}, {op, 0}}, {a, b}), nullptr, 0.0);
}
double unary(Op op, double a) { return mna::expr::evaluate(make({{Op::Const, 0}, {op, 0}}, {a}), nullptr, 0.0); }

bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || (a == b && std::signbit(a) == std::signbit(b)); }

}  // namespace

TEST_CASE("leaf opcodes: const, load_x, load_t") {
    const double x[3] = {1.5, -2.0, 7.0};
    CHECK(mna::expr::evaluate(make({{Op::Const, 1}}, {3.0, 4.0}), nullptr, 0.0) == 4.0);
    CHECK(mna::expr::evaluate(make({{Op::LoadX, 2}}, {}, 3), x, 0.0) == 7.0);
    CHECK(mna::expr::evaluate(make({{Op::LoadT, 0}}), nullptr, 0.25) == 0.25);
}

TEST_CASE("arithmetic opcodes follow IEEE float64") {
    CHECK(binary(Op::Add, 0.1, 0.2) == 0.1 + 0.2);
    CHECK(binary(Op::Sub, 0.3, 0.1) == 0.3 - 0.1);
    CHECK(binary(Op::Mul, 1e308, 10.0) == kInf);
    CHECK(binary(Op::Div, 1.0, 3.0) == 1.0 / 3.0);
    CHECK(binary(Op::Div, 1.0, 0.0) == kInf);
    CHECK(binary(Op::Div, -1.0, 0.0) == -kInf);
    CHECK(std::isnan(binary(Op::Div, 0.0, 0.0)));
    CHECK(unary(Op::Neg, 2.5) == -2.5);
    CHECK(same(unary(Op::Neg, 0.0), -0.0));
    CHECK(binary(Op::Pow, 2.0, 10.0) == 1024.0);
    CHECK(binary(Op::Pow, 1.7, 2.0) == std::pow(1.7, 2.0));
    CHECK(binary(Op::Pow, 0.0, -1.0) == kInf);
    CHECK(std::isnan(binary(Op::Pow, -8.0, 1.0 / 3.0)));
    CHECK(binary(Op::Pow, 10.0, 400.0) == kInf);
}

TEST_CASE("elementary functions") {
    CHECK(unary(Op::Sqrt, 2.0) == std::sqrt(2.0));
    CHECK(std::isnan(unary(Op::Sqrt, -1.0)));
    CHECK(unary(Op::Exp, 1.0) == std::exp(1.0));
    CHECK(unary(Op::Exp, 800.0) == kInf);  // no clipping, overflow to inf
    CHECK(unary(Op::Exp, -800.0) == 0.0);
    CHECK(unary(Op::Log, 10.0) == std::log(10.0));
    CHECK(unary(Op::Log, 0.0) == -kInf);
    CHECK(unary(Op::Sin, 0.5) == std::sin(0.5));
    CHECK(unary(Op::Cos, 0.5) == std::cos(0.5));
    CHECK(unary(Op::Tan, 0.5) == std::tan(0.5));
    CHECK(unary(Op::Abs, -3.0) == 3.0);
    CHECK(same(unary(Op::Abs, -0.0), 0.0));
}

TEST_CASE("heaviside is numpy.heaviside with h0 as second operand") {
    CHECK(binary(Op::Heaviside, -1e-300, 0.5) == 0.0);
    CHECK(binary(Op::Heaviside, 0.0, 0.5) == 0.5);
    CHECK(binary(Op::Heaviside, -0.0, 1.0) == 1.0);
    CHECK(binary(Op::Heaviside, 0.0, 1.0) == 1.0);
    CHECK(binary(Op::Heaviside, 1e-300, 0.5) == 1.0);
    CHECK(binary(Op::Heaviside, kInf, 0.0) == 1.0);
    CHECK(std::isnan(binary(Op::Heaviside, kNaN, 1.0)));
}

TEST_CASE("pymod is numpy.mod: sign of the divisor, exact remainder") {
    CHECK(binary(Op::PyMod, 7.0, 3.0) == 1.0);
    CHECK(binary(Op::PyMod, -7.0, 3.0) == 2.0);
    CHECK(binary(Op::PyMod, 7.0, -3.0) == -2.0);
    CHECK(binary(Op::PyMod, -7.0, -3.0) == -1.0);
    CHECK(binary(Op::PyMod, 5.5, 2.0) == 1.5);
    CHECK(same(binary(Op::PyMod, 6.0, 3.0), 0.0));
    CHECK(same(binary(Op::PyMod, 6.0, -3.0), -0.0));
    CHECK(same(binary(Op::PyMod, -0.0, 1.0), 0.0));
    CHECK(std::isnan(binary(Op::PyMod, 1.0, 0.0)));
    CHECK(std::isnan(binary(Op::PyMod, kInf, 1.0)));
    CHECK(binary(Op::PyMod, 1.0, kInf) == 1.0);
    // fmod is exact, unlike a - b*floor(a/b) (which gives 0.0 and 1.1102230246251565e-16 here):
    // values from numpy.mod.
    CHECK(binary(Op::PyMod, 1.0, 0.1) == 0.09999999999999995);
    CHECK(binary(Op::PyMod, -0.7, 0.1) == 8.326672684688674e-17);
    CHECK(binary(Op::PyMod, 5e-9, 1e-9) == 9.999999999999999e-10);
    CHECK(binary(Op::PyMod, 0.7, 0.1) == 0.09999999999999992);
    // Tiny negative dividend wraps to just below the divisor.
    CHECK(binary(Op::PyMod, -1e-20, 1.0) == 1.0);
}

TEST_CASE("min and max are numpy.minimum / numpy.maximum") {
    CHECK(binary(Op::Max, 1.0, 2.0) == 2.0);
    CHECK(binary(Op::Min, 1.0, 2.0) == 1.0);
    CHECK(std::isnan(binary(Op::Max, kNaN, 1.0)));
    CHECK(std::isnan(binary(Op::Max, 1.0, kNaN)));
    CHECK(std::isnan(binary(Op::Min, kNaN, 1.0)));
    CHECK(std::isnan(binary(Op::Min, 1.0, kNaN)));
    CHECK(same(binary(Op::Max, -0.0, 0.0), 0.0));  // ties return the second operand, like NumPy
    CHECK(same(binary(Op::Max, 0.0, -0.0), -0.0));
    CHECK(same(binary(Op::Min, 0.0, -0.0), -0.0));
    CHECK(same(binary(Op::Min, -0.0, 0.0), 0.0));
}

TEST_CASE("comparisons and logic produce 1.0 / 0.0") {
    CHECK(binary(Op::Lt, 1.0, 2.0) == 1.0);
    CHECK(binary(Op::Lt, 2.0, 2.0) == 0.0);
    CHECK(binary(Op::Le, 2.0, 2.0) == 1.0);
    CHECK(binary(Op::Gt, 3.0, 2.0) == 1.0);
    CHECK(binary(Op::Ge, 2.0, 2.0) == 1.0);
    CHECK(binary(Op::Eq, 2.0, 2.0) == 1.0);
    CHECK(binary(Op::Ne, 2.0, 2.0) == 0.0);
    CHECK(binary(Op::Lt, kNaN, 1.0) == 0.0);
    CHECK(binary(Op::Ne, kNaN, kNaN) == 1.0);
    CHECK(binary(Op::And, 1.0, 0.0) == 0.0);
    CHECK(binary(Op::And, 1.0, kNaN) == 1.0);  // NaN is truthy, as numpy.logical_and
    CHECK(binary(Op::Or, 0.0, 0.0) == 0.0);
    CHECK(binary(Op::Or, 0.0, -2.0) == 1.0);
    CHECK(unary(Op::Not, 0.0) == 1.0);
    CHECK(unary(Op::Not, 3.0) == 0.0);
    CHECK(unary(Op::Not, kNaN) == 0.0);
}

TEST_CASE("select takes the first true branch and yields NaN when none is true") {
    // select([t < 1, t < 2, True], [10, 20, 30])
    const Program p = make({{Op::Const, 0}, {Op::LoadT, 0}, {Op::Const, 1}, {Op::Lt, 0},
                            {Op::Const, 2}, {Op::LoadT, 0}, {Op::Const, 3}, {Op::Lt, 0},
                            {Op::Const, 4}, {Op::Const, 5}, {Op::Select, 3}},
                           {10.0, 1.0, 20.0, 2.0, 30.0, 1.0});
    CHECK(mna::expr::evaluate(p, nullptr, 0.5) == 10.0);
    CHECK(mna::expr::evaluate(p, nullptr, 1.0) == 20.0);
    CHECK(mna::expr::evaluate(p, nullptr, 2.0) == 30.0);
    CHECK(mna::expr::evaluate(p, nullptr, kNaN) == 30.0);
    const Program none = make({{Op::Const, 0}, {Op::LoadT, 0}, {Op::Const, 1}, {Op::Lt, 0}, {Op::Select, 1}},
                              {10.0, 1.0});
    CHECK(mna::expr::evaluate(none, nullptr, 0.0) == 10.0);
    CHECK(std::isnan(mna::expr::evaluate(none, nullptr, 5.0)));
    CHECK(p.max_depth() == 6);
}

TEST_CASE("programs are validated at construction") {
    auto rejects = [](auto&& build) {
        try {
            build();
        } catch (const mna::Error& e) {
            CHECK(e.code() == mna::ErrorCode::NetlistFatal);
            return true;
        }
        return false;
    };
    CHECK(rejects([] { make({}); }));
    CHECK(rejects([] { make({{static_cast<Op>(99), 0}}); }));
    CHECK(rejects([] { make({{static_cast<Op>(-1), 0}}); }));
    CHECK(rejects([] { make({{Op::Const, 1}}, {1.0}); }));
    CHECK(rejects([] { make({{Op::Const, -1}}, {1.0}); }));
    CHECK(rejects([] { make({{Op::LoadX, 2}}, {}, 2); }));
    CHECK(rejects([] { make({{Op::LoadT, 0}}, {}, 0, false); }));
    CHECK(rejects([] { make({{Op::Const, 0}, {Op::Add, 0}}, {1.0}); }));
    CHECK(rejects([] { make({{Op::Const, 0}, {Op::Const, 0}}, {1.0}); }));
    CHECK(rejects([] { make({{Op::Const, 0}, {Op::Select, 0}}, {1.0}); }));
    CHECK(rejects([] { make({{Op::Const, 0}, {Op::Select, 1}}, {1.0}); }));
    CHECK(rejects([] {
        std::vector<std::int32_t> ops = {0};
        std::vector<std::int32_t> args = {0};
        Program(ops.data(), args.data(), 1, {1.0}, -1, false);
    }));
    CHECK_FALSE(rejects([] { make({{Op::LoadX, 1}, {Op::LoadX, 0}, {Op::Add, 0}}, {}, 2, false); }));
}

TEST_CASE("unknowns and load positions") {
    const Program p = make({{Op::LoadX, 4}, {Op::LoadX, 1}, {Op::Mul, 0}, {Op::LoadX, 4}, {Op::Add, 0}}, {}, 6);
    CHECK(p.unknowns() == std::vector<std::int32_t>{1, 4});
    CHECK(p.load_positions() == std::vector<std::int32_t>{1, 0, -1, 1, -1});
    CHECK(p.max_depth() == 2);
}

TEST_CASE("evaluate_many evaluates a time program on a grid and rejects unknowns") {
    const Program p = make({{Op::LoadT, 0}, {Op::Const, 0}, {Op::Mul, 0}, {Op::Sin, 0}}, {2.0});
    const double ts[4] = {0.0, 0.1, 0.2, 0.3};
    double out[4] = {};
    mna::expr::evaluate_many(p, ts, 4, out);
    for (int i = 0; i < 4; ++i) {
        CHECK(out[i] == std::sin(ts[i] * 2.0));
    }
    const Program px = make({{Op::LoadX, 0}}, {}, 1);
    CHECK_THROWS_AS(mna::expr::evaluate_many(px, ts, 4, out), mna::Error);
}

TEST_CASE("deep programs use the heap stack") {
    // 1 + 1 + ... built as c c c ... c add add ... (depth 100).
    std::vector<std::int32_t> ops;
    std::vector<std::int32_t> args;
    for (int i = 0; i < 100; ++i) {
        ops.push_back(static_cast<std::int32_t>(Op::Const));
        args.push_back(0);
    }
    for (int i = 0; i < 99; ++i) {
        ops.push_back(static_cast<std::int32_t>(Op::Add));
        args.push_back(0);
    }
    const Program p(ops.data(), args.data(), ops.size(), {1.0}, 0, false);
    CHECK(p.max_depth() == 100);
    CHECK(mna::expr::evaluate(p, nullptr, 0.0) == 100.0);
    double value = 0.0;
    double grad = 0.0;
    const std::int32_t wrt = 0;
    mna::expr::evaluate_with_gradient(p, nullptr, 0.0, &wrt, 1, &value, &grad);
    CHECK(value == 100.0);
    CHECK(grad == 0.0);
}

namespace {

/// value and gradient over wrt = {0, 1, ...}.
std::vector<double> grad_of(const Program& p, std::vector<double> x, double t = 0.0, double* value = nullptr) {
    std::vector<std::int32_t> wrt(x.size());
    for (std::size_t i = 0; i < wrt.size(); ++i) {
        wrt[i] = static_cast<std::int32_t>(i);
    }
    std::vector<double> g(x.size());
    double v = 0.0;
    mna::expr::evaluate_with_gradient(p, x.data(), t, wrt.data(), static_cast<int>(wrt.size()), &v, g.data());
    if (value != nullptr) {
        *value = v;
    }
    return g;
}

}  // namespace

TEST_CASE("dual-number gradients of arithmetic and functions") {
    // x0 * x1 / x2 - x0 ** 2.0
    const Program p = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::Mul, 0}, {Op::LoadX, 2}, {Op::Div, 0},
                            {Op::LoadX, 0}, {Op::Const, 0}, {Op::Pow, 0}, {Op::Sub, 0}},
                           {2.0}, 3);
    double value = 0.0;
    const std::vector<double> g = grad_of(p, {1.5, -2.0, 4.0}, 0.0, &value);
    CHECK(value == doctest::Approx(1.5 * -2.0 / 4.0 - 2.25));
    CHECK(g[0] == doctest::Approx(-2.0 / 4.0 - 3.0));
    CHECK(g[1] == doctest::Approx(1.5 / 4.0));
    CHECK(g[2] == doctest::Approx(-(1.5 * -2.0) / 16.0));

    // exp(sin(x0)) + sqrt(x1) + log(x1) + tan(x0) + cos(x1)
    const Program f = make({{Op::LoadX, 0}, {Op::Sin, 0}, {Op::Exp, 0}, {Op::LoadX, 1}, {Op::Sqrt, 0}, {Op::Add, 0},
                            {Op::LoadX, 1}, {Op::Log, 0}, {Op::Add, 0}, {Op::LoadX, 0}, {Op::Tan, 0}, {Op::Add, 0},
                            {Op::LoadX, 1}, {Op::Cos, 0}, {Op::Add, 0}},
                           {}, 2);
    const double a = 0.3;
    const double b = 2.0;
    const std::vector<double> gf = grad_of(f, {a, b});
    CHECK(gf[0] == doctest::Approx(std::cos(a) * std::exp(std::sin(a)) + 1.0 + std::tan(a) * std::tan(a)));
    CHECK(gf[1] == doctest::Approx(0.5 / std::sqrt(b) + 1.0 / b - std::sin(b)));

    // x0 ** x1: both partials
    const Program pw = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::Pow, 0}}, {}, 2);
    const std::vector<double> gp = grad_of(pw, {2.0, 3.0});
    CHECK(gp[0] == doctest::Approx(12.0));
    CHECK(gp[1] == doctest::Approx(8.0 * std::log(2.0)));
    // pymod(x0, x1)
    const Program pm = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::PyMod, 0}}, {}, 2);
    const std::vector<double> gm = grad_of(pm, {7.0, 3.0});
    CHECK(gm[0] == 1.0);
    CHECK(gm[1] == -2.0);
    // -x0
    CHECK(grad_of(make({{Op::LoadX, 0}, {Op::Neg, 0}}, {}, 1), {5.0})[0] == -1.0);
}

TEST_CASE("gradient conventions at non-differentiable points") {
    // Heaviside(x0, 0.5): derivative 0 everywhere (also at 0).
    const Program h = make({{Op::LoadX, 0}, {Op::Const, 0}, {Op::Heaviside, 0}}, {0.5}, 1);
    CHECK(grad_of(h, {0.0})[0] == 0.0);
    CHECK(grad_of(h, {1.0})[0] == 0.0);
    // abs: sign(x) with 0 at 0.
    const Program ab = make({{Op::LoadX, 0}, {Op::Abs, 0}}, {}, 1);
    CHECK(grad_of(ab, {-2.0})[0] == -1.0);
    CHECK(grad_of(ab, {2.0})[0] == 1.0);
    CHECK(grad_of(ab, {0.0})[0] == 0.0);
    // max(x0, x1): derivative of the selected operand, the second at ties.
    const Program mx = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::Max, 0}}, {}, 2);
    CHECK(grad_of(mx, {1.0, 2.0}) == std::vector<double>{0.0, 1.0});
    CHECK(grad_of(mx, {3.0, 2.0}) == std::vector<double>{1.0, 0.0});
    CHECK(grad_of(mx, {2.0, 2.0}) == std::vector<double>{0.0, 1.0});
    const Program mn = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::Min, 0}}, {}, 2);
    CHECK(grad_of(mn, {1.0, 2.0}) == std::vector<double>{1.0, 0.0});
    CHECK(grad_of(mn, {2.0, 2.0}) == std::vector<double>{0.0, 1.0});
    // select([x0 < 0, True], [-x0, 3*x0]): selected branch only; comparisons give 0.
    const Program sel = make({{Op::LoadX, 0}, {Op::Neg, 0}, {Op::LoadX, 0}, {Op::Const, 0}, {Op::Lt, 0},
                              {Op::Const, 1}, {Op::LoadX, 0}, {Op::Mul, 0}, {Op::Const, 2}, {Op::Select, 2}},
                             {0.0, 3.0, 1.0}, 1);
    CHECK(grad_of(sel, {-1.0})[0] == -1.0);
    CHECK(grad_of(sel, {1.0})[0] == 3.0);
    CHECK(grad_of(sel, {0.0})[0] == 3.0);
    // Comparisons and logic carry no derivative.
    const Program cmp = make({{Op::LoadX, 0}, {Op::LoadX, 1}, {Op::Lt, 0}, {Op::LoadX, 0}, {Op::And, 0},
                              {Op::Not, 0}},
                             {}, 2);
    CHECK(grad_of(cmp, {1.0, 2.0}) == std::vector<double>{0.0, 0.0});
}

TEST_CASE("structurally zero derivative terms are skipped") {
    // 2.0 * x0 at x0 = inf: d/dx0 = 2 (not 0 * inf); d/dx1 = 0 for an unread unknown.
    const Program p = make({{Op::Const, 0}, {Op::LoadX, 0}, {Op::Mul, 0}}, {2.0}, 2);
    const std::vector<double> g = grad_of(p, {kInf, 1.0});
    CHECK(g[0] == 2.0);
    CHECK(g[1] == 0.0);
    // x0 ** 2.0 at x0 < 0 has no log(x0) term.
    const Program sq = make({{Op::LoadX, 0}, {Op::Const, 0}, {Op::Pow, 0}}, {2.0}, 1);
    CHECK(grad_of(sq, {-3.0})[0] == -6.0);
    // Diode-shaped 1e-15 * exp(40 x0 - 40 x1) - 1e-15 overflows to inf like the value.
    const Program d = make({{Op::Const, 0}, {Op::Const, 1}, {Op::LoadX, 0}, {Op::Mul, 0}, {Op::Const, 1},
                            {Op::LoadX, 1}, {Op::Mul, 0}, {Op::Sub, 0}, {Op::Exp, 0}, {Op::Mul, 0},
                            {Op::Const, 0}, {Op::Sub, 0}},
                           {1e-15, 40.0}, 2);
    double value = 0.0;
    const std::vector<double> gd = grad_of(d, {20.0, 0.0}, 0.0, &value);
    CHECK(value == kInf);
    CHECK(gd[0] == kInf);
    CHECK(gd[1] == -kInf);
    // exp(x0) * x1 + x2 with exp overflowing: d/dx2 stays 1 (no 0 * inf from the first term),
    // d/dx1 = exp(x0) = inf.
    const Program row = make({{Op::LoadX, 0}, {Op::Exp, 0}, {Op::LoadX, 1}, {Op::Mul, 0}, {Op::LoadX, 2}, {Op::Add, 0}},
                             {}, 3);
    const std::vector<double> gr = grad_of(row, {800.0, 2.0, 5.0});
    CHECK(gr[0] == kInf);
    CHECK(gr[1] == kInf);
    CHECK(gr[2] == 1.0);
}

TEST_CASE("evaluate_with_gradient maps arbitrary wrt lists") {
    const Program p = make({{Op::LoadX, 2}, {Op::LoadX, 0}, {Op::Mul, 0}}, {}, 3);
    const double x[3] = {3.0, 100.0, 5.0};
    const std::int32_t wrt[4] = {2, 1, 0, 2};
    double g[4] = {};
    double value = 0.0;
    mna::expr::evaluate_with_gradient(p, x, 0.0, wrt, 4, &value, g);
    CHECK(value == 15.0);
    CHECK(g[0] == 3.0);
    CHECK(g[1] == 0.0);
    CHECK(g[2] == 5.0);
    CHECK(g[3] == 3.0);
    double gu[2] = {};
    mna::expr::evaluate_gradient_unknowns(p, x, 0.0, &value, gu);
    CHECK(gu[0] == 5.0);
    CHECK(gu[1] == 3.0);
}

namespace {

Program table_program(std::vector<mna::expr::Table> tables, std::int32_t index = 0, bool time_dependent = true) {
    const std::int32_t ops[1] = {static_cast<std::int32_t>(Op::Table)};
    const std::int32_t args[1] = {index};
    return Program(ops, args, 1, {}, 0, time_dependent, std::move(tables));
}

mna::expr::Table make_table(std::vector<double> times, std::vector<double> values, double period = 0.0,
                            double origin = 0.0) {
    mna::expr::Table table;
    table.times = std::move(times);
    table.values = std::move(values);
    table.period = period;
    table.origin = origin;
    return table;
}

}  // namespace

TEST_CASE("table: linear interpolation with the first and last values held") {
    const Program p = table_program({make_table({1.0, 2.0, 4.0}, {0.0, 3.0, -1.0})});
    CHECK(p.tables().size() == 1);
    CHECK(mna::expr::evaluate(p, nullptr, -5.0) == 0.0);
    CHECK(mna::expr::evaluate(p, nullptr, 1.0) == 0.0);
    CHECK(mna::expr::evaluate(p, nullptr, 1.5) == 0.0 + 3.0 * 0.5);
    CHECK(mna::expr::evaluate(p, nullptr, 2.0) == 3.0);  // a corner gives its value exactly
    CHECK(mna::expr::evaluate(p, nullptr, 3.0) == 3.0 + (-4.0 / 2.0) * 1.0);
    CHECK(mna::expr::evaluate(p, nullptr, 4.0) == -1.0);
    CHECK(mna::expr::evaluate(p, nullptr, 1e9) == -1.0);
    CHECK(mna::expr::evaluate(p, nullptr, kInf) == -1.0);
    CHECK(mna::expr::evaluate(p, nullptr, -kInf) == 0.0);
    CHECK(std::isnan(mna::expr::evaluate(p, nullptr, kNaN)));
    // One point: a constant.
    const Program one = table_program({make_table({0.5}, {2.5})});
    CHECK(mna::expr::evaluate(one, nullptr, 0.0) == 2.5);
    CHECK(mna::expr::evaluate(one, nullptr, 7.0) == 2.5);
}

TEST_CASE("table: periodic repetition folds t >= origin into [origin, origin + period)") {
    // Trapezoid 1 -> 5 starting at td = 1, period 10 (a PULSE(1 5 1 1 1 2 10)).
    const Program p = table_program({make_table({1.0, 2.0, 4.0, 5.0}, {1.0, 5.0, 5.0, 1.0}, 10.0, 1.0)});
    CHECK(mna::expr::evaluate(p, nullptr, 0.0) == 1.0);
    CHECK(mna::expr::evaluate(p, nullptr, 1.5) == 3.0);
    CHECK(mna::expr::evaluate(p, nullptr, 11.5) == 3.0);
    CHECK(mna::expr::evaluate(p, nullptr, 13.0) == 5.0);
    CHECK(mna::expr::evaluate(p, nullptr, 8.0) == 1.0);    // after the last corner of the period
    CHECK(mna::expr::evaluate(p, nullptr, 11.0) == 1.0);   // the fold point is the origin
    CHECK(mna::expr::evaluate(p, nullptr, 1001.5) == 3.0);
    CHECK(std::isnan(mna::expr::evaluate(p, nullptr, kInf)));
    // Repetition from an interior origin (SPICE PWL R=): [2, 4) repeats after t = 4.
    const Program r = table_program({make_table({0.0, 2.0, 3.0, 4.0}, {7.0, 0.0, 1.0, 0.0}, 2.0, 2.0)});
    CHECK(mna::expr::evaluate(r, nullptr, 1.0) == 3.5);
    CHECK(mna::expr::evaluate(r, nullptr, 3.0) == 1.0);
    CHECK(mna::expr::evaluate(r, nullptr, 5.0) == 1.0);
    CHECK(mna::expr::evaluate(r, nullptr, 4.5) == 0.5);
}

TEST_CASE("table: mixed with other instructions, several tables, zero derivative") {
    // table0(t) - 2 * table1(t) + x0
    const std::int32_t ops[] = {static_cast<std::int32_t>(Op::Table), static_cast<std::int32_t>(Op::Const),
                                static_cast<std::int32_t>(Op::Table), static_cast<std::int32_t>(Op::Mul),
                                static_cast<std::int32_t>(Op::Sub),   static_cast<std::int32_t>(Op::LoadX),
                                static_cast<std::int32_t>(Op::Add)};
    const std::int32_t args[] = {0, 0, 1, 0, 0, 0, 0};
    const Program p(ops, args, 7, {2.0}, 1, true,
                    {make_table({0.0, 1.0}, {0.0, 1.0}), make_table({0.0, 2.0}, {4.0, 0.0})});
    const double x[1] = {10.0};
    CHECK(mna::expr::evaluate(p, x, 0.5) == 0.5 - 2.0 * 3.0 + 10.0);
    double value = 0.0;
    double grad = 0.0;
    const std::int32_t wrt = 0;
    mna::expr::evaluate_with_gradient(p, x, 0.5, &wrt, 1, &value, &grad);
    CHECK(value == 0.5 - 2.0 * 3.0 + 10.0);
    CHECK(grad == 1.0);
    const double ts[3] = {0.0, 1.0, 2.0};
    double out[3] = {};
    const Program q = table_program({make_table({0.0, 1.0}, {0.0, 1.0}), make_table({0.0, 2.0}, {4.0, 0.0})}, 1);
    mna::expr::evaluate_many(q, ts, 3, out);
    CHECK(out[0] == 4.0);
    CHECK(out[1] == 2.0);
    CHECK(out[2] == 0.0);
}

TEST_CASE("table: a 1e5-point table") {
    std::vector<double> times(100000);
    std::vector<double> values(100000);
    for (std::size_t i = 0; i < times.size(); ++i) {
        times[i] = static_cast<double>(i) * 1e-9;
        values[i] = (i % 2 == 0) ? 0.0 : 3.3;
    }
    const Program p = table_program({make_table(times, values)});
    CHECK(mna::expr::evaluate(p, nullptr, times[12345]) == 3.3);
    CHECK(mna::expr::evaluate(p, nullptr, times[12346]) == 0.0);
    CHECK(mna::expr::evaluate(p, nullptr, 0.5 * (times[12345] + times[12346])) == doctest::Approx(1.65));
}

TEST_CASE("table: validation") {
    auto rejects = [](auto&& build) {
        try {
            build();
        } catch (const mna::Error& e) {
            CHECK(e.code() == mna::ErrorCode::NetlistFatal);
            return true;
        }
        return false;
    };
    CHECK(rejects([] { table_program({}); }));                                       // index out of range
    CHECK(rejects([] { table_program({make_table({0.0}, {1.0})}, 1); }));            // index out of range
    CHECK(rejects([] { table_program({make_table({0.0}, {1.0})}, -1); }));
    CHECK(rejects([] { table_program({make_table({0.0}, {1.0})}, 0, false); }));     // not time dependent
    CHECK(rejects([] { table_program({make_table({}, {})}); }));                     // empty
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0})}); }));          // lengths differ
    CHECK(rejects([] { table_program({make_table({0.0, 0.0}, {1.0, 2.0})}); }));     // not increasing
    CHECK(rejects([] { table_program({make_table({1.0, 0.0}, {1.0, 2.0})}); }));
    CHECK(rejects([] { table_program({make_table({0.0, kNaN}, {1.0, 2.0})}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, kNaN})}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, kInf})}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, 2.0}, -1.0)}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, 2.0}, kInf)}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, 2.0}, 1.0, kNaN)}); }));
    CHECK(rejects([] { table_program({make_table({0.0, 1e-300}, {-1e300, 1e300})}); }));  // slope overflows
    CHECK_FALSE(rejects([] { table_program({make_table({0.0, 1.0}, {1.0, 2.0}, 0.5, 0.0)}); }));
    CHECK(std::string(mna::expr::op_name(Op::Table)) == "table");
}
