#include "mna/expr/program.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "mna/error.hpp"

// No fused multiply-add anywhere in this file (see mna/devices/fp_contract.hpp).
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::expr {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void invalid(const std::string& message) {
    throw Error(ErrorCode::NetlistFatal, "invalid expression program: " + message);
}

inline bool truthy(double v) noexcept { return v != 0.0; }  // NaN != 0 is true, as numpy.logical_*

// Chain-rule products that skip zero partials: a partial that is exactly zero
// means the term is absent from the symbolic derivative, so 0 * inf (or 0 / 0)
// must not turn into NaN.
inline double times(double partial, double factor) noexcept { return partial == 0.0 ? 0.0 : partial * factor; }
inline double over(double partial, double divisor) noexcept { return partial == 0.0 ? 0.0 : partial / divisor; }
inline double boolean(bool b) noexcept { return b ? 1.0 : 0.0; }

/// Stack effect of one instruction: operands consumed (results pushed is 1).
std::size_t operand_count(Op op, std::int32_t arg) noexcept {
    switch (op) {
        case Op::Const:
        case Op::LoadX:
        case Op::LoadT:
            return 0;
        case Op::Neg:
        case Op::Sqrt:
        case Op::Exp:
        case Op::Log:
        case Op::Sin:
        case Op::Cos:
        case Op::Tan:
        case Op::Abs:
        case Op::Not:
            return 1;
        case Op::Select:
            return 2 * static_cast<std::size_t>(arg);
        default:
            return 2;
    }
}

/// Scratch buffers reused by the evaluators on each thread (no allocation once warm).
struct Scratch {
    std::vector<double> values;
    std::vector<unsigned char> has;
    std::vector<double> grads;
    std::vector<double> unknown_grad;
};

Scratch& scratch() {
    thread_local Scratch s;
    return s;
}

}  // namespace

const char* op_name(Op op) noexcept {
    switch (op) {
        case Op::Const: return "const";
        case Op::LoadX: return "load_x";
        case Op::LoadT: return "load_t";
        case Op::Add: return "add";
        case Op::Sub: return "sub";
        case Op::Mul: return "mul";
        case Op::Div: return "div";
        case Op::Neg: return "neg";
        case Op::Pow: return "pow";
        case Op::Sqrt: return "sqrt";
        case Op::Exp: return "exp";
        case Op::Log: return "log";
        case Op::Sin: return "sin";
        case Op::Cos: return "cos";
        case Op::Tan: return "tan";
        case Op::Abs: return "abs";
        case Op::Heaviside: return "heaviside";
        case Op::PyMod: return "pymod";
        case Op::Min: return "min";
        case Op::Max: return "max";
        case Op::Lt: return "lt";
        case Op::Le: return "le";
        case Op::Gt: return "gt";
        case Op::Ge: return "ge";
        case Op::Eq: return "eq";
        case Op::Ne: return "ne";
        case Op::And: return "and";
        case Op::Or: return "or";
        case Op::Not: return "not";
        case Op::Select: return "select";
    }
    return "?";
}

double py_mod(double a, double b) noexcept {
    // numpy npy_remainder / npy_divmod (float64), modulus part only.
    if (b == 0.0) {
        return std::fmod(a, b);  // NaN
    }
    double mod = std::fmod(a, b);
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) {
            mod += b;
        }
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

double np_maximum(double a, double b) noexcept { return (a > b || std::isnan(a)) ? a : b; }

double np_minimum(double a, double b) noexcept { return (a < b || std::isnan(a)) ? a : b; }

double np_heaviside(double x, double h0) noexcept {
    if (x < 0.0) {
        return 0.0;
    }
    if (x == 0.0) {
        return h0;
    }
    if (x > 0.0) {
        return 1.0;
    }
    return x;  // NaN
}

Program::Program(const std::int32_t* opcodes, const std::int32_t* args, std::size_t size, std::vector<double> consts,
                 std::int32_t n_unknowns, bool time_dependent)
    : consts_(std::move(consts)), n_unknowns_(n_unknowns), time_dependent_(time_dependent) {
    if (n_unknowns < 0) {
        invalid("n_unknowns must be non-negative");
    }
    if (size == 0) {
        invalid("empty program");
    }
    if (opcodes == nullptr || args == nullptr) {
        invalid("null instruction arrays");
    }
    code_.reserve(size);
    load_pos_.assign(size, -1);
    std::size_t depth = 0;
    for (std::size_t i = 0; i < size; ++i) {
        const std::int32_t raw = opcodes[i];
        const std::int32_t arg = args[i];
        const std::string where = "instruction " + std::to_string(i);
        if (raw < 0 || raw >= kOpCount) {
            invalid(where + ": unknown opcode " + std::to_string(raw));
        }
        const Op op = static_cast<Op>(raw);
        switch (op) {
            case Op::Const:
                if (arg < 0 || static_cast<std::size_t>(arg) >= consts_.size()) {
                    invalid(where + ": constant index " + std::to_string(arg) + " out of range");
                }
                break;
            case Op::LoadX:
                if (arg < 0 || arg >= n_unknowns_) {
                    invalid(where + ": unknown index " + std::to_string(arg) + " out of range for " +
                            std::to_string(n_unknowns_) + " unknowns");
                }
                unknowns_.push_back(arg);
                break;
            case Op::LoadT:
                if (!time_dependent_) {
                    invalid(where + ": load_t in a program that is not time dependent");
                }
                break;
            case Op::Select:
                if (arg < 1 || static_cast<std::size_t>(arg) > kMaxStackDepth) {
                    invalid(where + ": select needs at least one (value, condition) pair");
                }
                break;
            default:
                break;
        }
        const std::size_t pops = operand_count(op, arg);
        if (depth < pops) {
            invalid(where + " (" + op_name(op) + "): stack underflow");
        }
        depth = depth - pops + 1;
        if (depth > kMaxStackDepth) {
            invalid("stack depth exceeds " + std::to_string(kMaxStackDepth));
        }
        max_depth_ = (std::max)(max_depth_, depth);
        code_.push_back({op, arg});
    }
    if (depth != 1) {
        invalid("program leaves " + std::to_string(depth) + " values on the stack (expected 1)");
    }
    std::sort(unknowns_.begin(), unknowns_.end());
    unknowns_.erase(std::unique(unknowns_.begin(), unknowns_.end()), unknowns_.end());
    for (std::size_t i = 0; i < size; ++i) {
        if (code_[i].op == Op::LoadX) {
            const auto it = std::lower_bound(unknowns_.begin(), unknowns_.end(), code_[i].arg);
            load_pos_[i] = static_cast<std::int32_t>(it - unknowns_.begin());
        }
    }
}

double evaluate(const Program& program, const double* x, double t) {
    constexpr std::size_t kLocal = 64;
    double local[kLocal];
    double* stack = local;
    if (program.max_depth() > kLocal) {
        std::vector<double>& big = scratch().values;
        if (big.size() < program.max_depth()) {
            big.resize(program.max_depth());
        }
        stack = big.data();
    }
    const double* consts = program.consts().data();
    std::size_t sp = 0;
    for (const Instruction& in : program.code()) {
        switch (in.op) {
            case Op::Const: stack[sp++] = consts[in.arg]; break;
            case Op::LoadX: stack[sp++] = x[in.arg]; break;
            case Op::LoadT: stack[sp++] = t; break;
            case Op::Add: --sp; stack[sp - 1] = stack[sp - 1] + stack[sp]; break;
            case Op::Sub: --sp; stack[sp - 1] = stack[sp - 1] - stack[sp]; break;
            case Op::Mul: --sp; stack[sp - 1] = stack[sp - 1] * stack[sp]; break;
            case Op::Div: --sp; stack[sp - 1] = stack[sp - 1] / stack[sp]; break;
            case Op::Pow: --sp; stack[sp - 1] = std::pow(stack[sp - 1], stack[sp]); break;
            case Op::Heaviside: --sp; stack[sp - 1] = np_heaviside(stack[sp - 1], stack[sp]); break;
            case Op::PyMod: --sp; stack[sp - 1] = py_mod(stack[sp - 1], stack[sp]); break;
            case Op::Min: --sp; stack[sp - 1] = np_minimum(stack[sp - 1], stack[sp]); break;
            case Op::Max: --sp; stack[sp - 1] = np_maximum(stack[sp - 1], stack[sp]); break;
            case Op::Lt: --sp; stack[sp - 1] = boolean(stack[sp - 1] < stack[sp]); break;
            case Op::Le: --sp; stack[sp - 1] = boolean(stack[sp - 1] <= stack[sp]); break;
            case Op::Gt: --sp; stack[sp - 1] = boolean(stack[sp - 1] > stack[sp]); break;
            case Op::Ge: --sp; stack[sp - 1] = boolean(stack[sp - 1] >= stack[sp]); break;
            case Op::Eq: --sp; stack[sp - 1] = boolean(stack[sp - 1] == stack[sp]); break;
            case Op::Ne: --sp; stack[sp - 1] = boolean(stack[sp - 1] != stack[sp]); break;
            case Op::And: --sp; stack[sp - 1] = boolean(truthy(stack[sp - 1]) && truthy(stack[sp])); break;
            case Op::Or: --sp; stack[sp - 1] = boolean(truthy(stack[sp - 1]) || truthy(stack[sp])); break;
            case Op::Neg: stack[sp - 1] = -stack[sp - 1]; break;
            case Op::Sqrt: stack[sp - 1] = std::sqrt(stack[sp - 1]); break;
            case Op::Exp: stack[sp - 1] = std::exp(stack[sp - 1]); break;
            case Op::Log: stack[sp - 1] = std::log(stack[sp - 1]); break;
            case Op::Sin: stack[sp - 1] = std::sin(stack[sp - 1]); break;
            case Op::Cos: stack[sp - 1] = std::cos(stack[sp - 1]); break;
            case Op::Tan: stack[sp - 1] = std::tan(stack[sp - 1]); break;
            case Op::Abs: stack[sp - 1] = std::fabs(stack[sp - 1]); break;
            case Op::Not: stack[sp - 1] = boolean(!truthy(stack[sp - 1])); break;
            case Op::Select: {
                const std::size_t pairs = static_cast<std::size_t>(in.arg);
                const std::size_t base = sp - 2 * pairs;
                double result = kNaN;
                for (std::size_t i = 0; i < pairs; ++i) {
                    if (truthy(stack[base + 2 * i + 1])) {
                        result = stack[base + 2 * i];
                        break;
                    }
                }
                sp = base;
                stack[sp++] = result;
                break;
            }
        }
    }
    return stack[0];
}

void evaluate_many(const Program& program, const double* t_array, std::size_t count, double* out) {
    if (!program.unknowns().empty()) {
        throw Error(ErrorCode::NetlistFatal, "evaluate_many: the program reads unknowns; pass x to evaluate()");
    }
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = evaluate(program, nullptr, t_array[i]);
    }
}

namespace {

/// Forward-mode evaluation over k directions. load_pos maps each instruction
/// to the direction of its LOAD_X operand (-1: treated as a constant).
void evaluate_dual(const Program& program, const double* x, double t, const std::int32_t* load_pos, std::size_t k,
                   double* value, double* grad) {
    Scratch& s = scratch();
    const std::size_t depth = program.max_depth();
    if (s.values.size() < depth) {
        s.values.resize(depth);
    }
    if (s.has.size() < depth) {
        s.has.resize(depth);
    }
    if (s.grads.size() < depth * k) {
        s.grads.resize(depth * k);
    }
    double* v = s.values.data();
    unsigned char* h = s.has.data();
    double* g = s.grads.data();
    auto G = [g, k](std::size_t i) { return g + i * k; };
    const double* consts = program.consts().data();
    const std::vector<Instruction>& code = program.code();

    std::size_t sp = 0;
    for (std::size_t pc = 0; pc < code.size(); ++pc) {
        const Instruction in = code[pc];
        switch (in.op) {
            case Op::Const:
                v[sp] = consts[in.arg];
                h[sp] = 0;
                ++sp;
                break;
            case Op::LoadT:
                v[sp] = t;
                h[sp] = 0;
                ++sp;
                break;
            case Op::LoadX: {
                v[sp] = x[in.arg];
                const std::int32_t pos = load_pos[pc];
                if (pos >= 0 && static_cast<std::size_t>(pos) < k) {
                    double* gi = G(sp);
                    std::fill(gi, gi + k, 0.0);
                    gi[pos] = 1.0;
                    h[sp] = 1;
                } else {
                    h[sp] = 0;
                }
                ++sp;
                break;
            }
            case Op::Add:
            case Op::Sub:
            case Op::Mul:
            case Op::Div:
            case Op::Pow:
            case Op::PyMod: {
                --sp;
                const std::size_t a = sp - 1;
                const std::size_t b = sp;
                const double va = v[a];
                const double vb = v[b];
                const bool ha = h[a] != 0;
                const bool hb = h[b] != 0;
                double* ga = G(a);
                const double* gb = G(b);
                double result = 0.0;
                switch (in.op) {
                    case Op::Add:
                        result = va + vb;
                        if (ha && hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = ga[j] + gb[j];
                        } else if (hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = gb[j];
                        }
                        break;
                    case Op::Sub:
                        result = va - vb;
                        if (ha && hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = ga[j] - gb[j];
                        } else if (hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = -gb[j];
                        }
                        break;
                    case Op::Mul:
                        result = va * vb;
                        if (ha && hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(ga[j], vb) + times(gb[j], va);
                        } else if (ha) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(ga[j], vb);
                        } else if (hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(gb[j], va);
                        }
                        break;
                    case Op::Div: {
                        result = va / vb;
                        const double b2 = vb * vb;
                        if (ha && hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = over(ga[j], vb) - over(times(gb[j], va), b2);
                        } else if (ha) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = over(ga[j], vb);
                        } else if (hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = -over(times(gb[j], va), b2);
                        }
                        break;
                    }
                    case Op::Pow: {
                        result = std::pow(va, vb);
                        const double ca = ha ? vb * std::pow(va, vb - 1.0) : 0.0;
                        const double cb = hb ? result * std::log(va) : 0.0;
                        if (ha && hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(ga[j], ca) + times(gb[j], cb);
                        } else if (ha) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(ga[j], ca);
                        } else if (hb) {
                            for (std::size_t j = 0; j < k; ++j) ga[j] = times(gb[j], cb);
                        }
                        break;
                    }
                    default: {  // PyMod
                        result = py_mod(va, vb);
                        if (hb) {
                            const double q = -std::floor(va / vb);
                            if (ha) {
                                for (std::size_t j = 0; j < k; ++j) ga[j] = ga[j] + times(gb[j], q);
                            } else {
                                for (std::size_t j = 0; j < k; ++j) ga[j] = times(gb[j], q);
                            }
                        }
                        break;
                    }
                }
                v[a] = result;
                h[a] = static_cast<unsigned char>(ha || hb);
                break;
            }
            case Op::Min:
            case Op::Max: {
                --sp;
                const std::size_t a = sp - 1;
                const std::size_t b = sp;
                const bool take_a = in.op == Op::Max ? (v[a] > v[b] || std::isnan(v[a]))
                                                     : (v[a] < v[b] || std::isnan(v[a]));
                if (!take_a) {
                    v[a] = v[b];
                    h[a] = h[b];
                    if (h[b] != 0) {
                        std::memcpy(G(a), G(b), k * sizeof(double));
                    }
                }
                break;
            }
            case Op::Heaviside:
            case Op::Lt:
            case Op::Le:
            case Op::Gt:
            case Op::Ge:
            case Op::Eq:
            case Op::Ne:
            case Op::And:
            case Op::Or: {
                --sp;
                const double va = v[sp - 1];
                const double vb = v[sp];
                double result = 0.0;
                switch (in.op) {
                    case Op::Heaviside: result = np_heaviside(va, vb); break;
                    case Op::Lt: result = boolean(va < vb); break;
                    case Op::Le: result = boolean(va <= vb); break;
                    case Op::Gt: result = boolean(va > vb); break;
                    case Op::Ge: result = boolean(va >= vb); break;
                    case Op::Eq: result = boolean(va == vb); break;
                    case Op::Ne: result = boolean(va != vb); break;
                    case Op::And: result = boolean(truthy(va) && truthy(vb)); break;
                    default: result = boolean(truthy(va) || truthy(vb)); break;
                }
                v[sp - 1] = result;
                h[sp - 1] = 0;
                break;
            }
            case Op::Not:
                v[sp - 1] = boolean(!truthy(v[sp - 1]));
                h[sp - 1] = 0;
                break;
            case Op::Neg:
            case Op::Sqrt:
            case Op::Exp:
            case Op::Log:
            case Op::Sin:
            case Op::Cos:
            case Op::Tan:
            case Op::Abs: {
                const std::size_t a = sp - 1;
                const double va = v[a];
                double result = 0.0;
                double c = 0.0;  // d result / d a
                switch (in.op) {
                    case Op::Neg: result = -va; c = -1.0; break;
                    case Op::Sqrt: result = std::sqrt(va); c = 0.5 / result; break;
                    case Op::Exp: result = std::exp(va); c = result; break;
                    case Op::Log: result = std::log(va); c = 1.0 / va; break;
                    case Op::Sin: result = std::sin(va); c = std::cos(va); break;
                    case Op::Cos: result = std::cos(va); c = -std::sin(va); break;
                    case Op::Tan: result = std::tan(va); c = 1.0 + result * result; break;
                    default:  // Abs
                        result = std::fabs(va);
                        c = va > 0.0 ? 1.0 : (va < 0.0 ? -1.0 : (va == 0.0 ? 0.0 : va));
                        break;
                }
                if (h[a] != 0) {
                    double* ga = G(a);
                    if (in.op == Op::Neg) {
                        for (std::size_t j = 0; j < k; ++j) ga[j] = -ga[j];
                    } else if (in.op == Op::Log) {
                        for (std::size_t j = 0; j < k; ++j) ga[j] = over(ga[j], va);
                    } else {
                        for (std::size_t j = 0; j < k; ++j) ga[j] = times(ga[j], c);
                    }
                }
                v[a] = result;
                break;
            }
            case Op::Select: {
                const std::size_t pairs = static_cast<std::size_t>(in.arg);
                const std::size_t base = sp - 2 * pairs;
                std::size_t chosen = pairs;
                for (std::size_t i = 0; i < pairs; ++i) {
                    if (truthy(v[base + 2 * i + 1])) {
                        chosen = i;
                        break;
                    }
                }
                if (chosen == pairs) {
                    v[base] = kNaN;
                    h[base] = 1;
                    double* gb = G(base);
                    std::fill(gb, gb + k, kNaN);
                } else if (chosen != 0) {
                    const std::size_t src = base + 2 * chosen;
                    v[base] = v[src];
                    h[base] = h[src];
                    if (h[src] != 0) {
                        std::memcpy(G(base), G(src), k * sizeof(double));
                    }
                }
                sp = base + 1;
                break;
            }
        }
    }
    *value = v[0];
    if (h[0] != 0) {
        std::memcpy(grad, G(0), k * sizeof(double));
    } else {
        std::fill(grad, grad + k, 0.0);
    }
}

}  // namespace

void evaluate_gradient_unknowns(const Program& program, const double* x, double t, double* value, double* grad) {
    evaluate_dual(program, x, t, program.load_positions().data(), program.unknowns().size(), value, grad);
}

void evaluate_with_gradient(const Program& program, const double* x, double t, const std::int32_t* wrt, int count,
                            double* value, double* grad) {
    const std::vector<std::int32_t>& unknowns = program.unknowns();
    std::vector<double>& g = scratch().unknown_grad;
    // evaluate_dual uses the other scratch buffers only.
    g.resize((std::max)(unknowns.size(), std::size_t{1}));
    evaluate_dual(program, x, t, program.load_positions().data(), unknowns.size(), value, g.data());
    for (int j = 0; j < count; ++j) {
        const auto it = std::lower_bound(unknowns.begin(), unknowns.end(), wrt[j]);
        grad[j] = (it != unknowns.end() && *it == wrt[j]) ? g[static_cast<std::size_t>(it - unknowns.begin())] : 0.0;
    }
}

}  // namespace mna::expr
