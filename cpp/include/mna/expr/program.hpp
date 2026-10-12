// Postfix bytecode for circuit expressions (FUNC time sources, B behavioral
// sources, nonlinear current rows) and its stack evaluators.
//
// Programs are produced by mna_simulation/native/expressions.py, which lowers
// the exact NumPy code SymPy's lambdify would execute into this instruction
// set, so evaluating a program reproduces the reference core operation by
// operation. Scalar semantics follow NumPy float64 (IEEE 754, no fast-math):
//
//   * DIV by zero gives +-inf or NaN, POW is std::pow, overflow gives inf.
//   * PYMOD(a, b) is NumPy's `mod` (npy_remainder): fmod-based, result takes
//     the sign of b; mathematically a - b*floor(a/b), computed exactly; b == 0
//     gives NaN.
//   * MIN/MAX are numpy.minimum/maximum: NaN propagates; with equal operands
//     (including 0.0 and -0.0) the second one is returned.
//   * HEAVISIDE(x, h0) is numpy.heaviside: 0 for x < 0, h0 for x == 0, 1 for
//     x > 0, NaN for NaN.
//   * Booleans are 1.0/0.0; AND/OR/NOT/SELECT treat any value != 0 (NaN
//     included) as true, like numpy.logical_*.
//   * SELECT k pops k (value, condition) pairs pushed in order and yields the
//     value of the first true condition, NaN if none is true
//     (numpy.select(..., default=nan)).
//   * TABLE k pushes table k of the program evaluated at t (time-dependent
//     programs only): a piecewise-linear lookup (binary search, one linear
//     interpolation v_i + slope_i * (t - t_i) with slope_i precomputed as
//     (v_{i+1} - v_i) / (t_{i+1} - t_i)); the first value is held before (and
//     at) the first point, the last value at and after the last point. With
//     period > 0 a time t >= origin is first folded to
//     origin + PYMOD(t - origin, period), so the waveform repeats from origin
//     (MIXED_SIGNAL.md phase D1). A NaN time gives NaN.
//
// Derivatives (evaluate_with_gradient) use forward-mode dual numbers. Choices
// at non-differentiable points: HEAVISIDE and comparisons/logic have derivative
// 0, SELECT differentiates the selected branch only, MIN/MAX the selected
// operand (the second one at ties, the operand the value comes from), ABS has
// derivative sign(x) with 0 at x == 0,
// PYMOD has d/da = 1 and d/db = -floor(a/b), TABLE has no dependence on the
// unknowns (derivative 0 in every direction). Chain-rule terms whose partial is
// zero (constants, t, unknowns outside `wrt`, or a component the operand does
// not depend on) are skipped, as they are absent from SymPy's symbolic
// derivative, so 0 * inf never turns an unrelated entry into NaN.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mna::expr {

enum class Op : std::int32_t {
    Const = 0,  ///< push consts[arg]
    LoadX = 1,  ///< push x[arg]
    LoadT = 2,  ///< push t (time-dependent programs only)
    Add = 3,
    Sub = 4,
    Mul = 5,
    Div = 6,
    Neg = 7,
    Pow = 8,
    Sqrt = 9,
    Exp = 10,
    Log = 11,
    Sin = 12,
    Cos = 13,
    Tan = 14,
    Abs = 15,
    Heaviside = 16,  ///< (x, h0)
    PyMod = 17,      ///< (a, b)
    Min = 18,
    Max = 19,
    Lt = 20,
    Le = 21,
    Gt = 22,
    Ge = 23,
    Eq = 24,
    Ne = 25,
    And = 26,
    Or = 27,
    Not = 28,
    Select = 29,  ///< arg = number of (value, condition) pairs
    Table = 30,   ///< push tables()[arg] evaluated at t (time-dependent programs only)
};

inline constexpr std::int32_t kOpCount = 31;

/// Upper bound on the evaluation stack depth of a valid program.
inline constexpr std::size_t kMaxStackDepth = std::size_t{1} << 16;

/// Lower-case mnemonic, e.g. "load_x"; "?" for an invalid value.
const char* op_name(Op op) noexcept;

struct Instruction {
    Op op;
    std::int32_t arg;
};

/// Piecewise-linear table read by Op::Table: strictly increasing finite times,
/// one finite value per time (at least one point), and an optional repetition
/// (period > 0: a time t >= origin is folded into [origin, origin + period)).
struct Table {
    std::vector<double> times;
    std::vector<double> values;
    double period = 0.0;
    double origin = 0.0;
};

/// Value of a validated table at time t (see Op::Table). `slopes` holds the
/// size() - 1 segment slopes (v_{i+1} - v_i) / (t_{i+1} - t_i).
double table_value(const Table& table, const std::vector<double>& slopes, double t) noexcept;

/// Programs are immutable after construction and safe to evaluate from several
/// threads at once. enable_shared_from_this lets the Python bindings hand the
/// same shared_ptr back to C++ (no Python reference is held by C++ owners).
class Program : public std::enable_shared_from_this<Program> {
public:
    /// Validates and stores a program. Throws mna::Error(NetlistFatal) for an
    /// unknown opcode, an out-of-range CONST/LOAD_X operand, LOAD_T in a
    /// program that is not time dependent, SELECT with k < 1, a stack
    /// underflow, a final stack depth other than 1, or a depth above
    /// kMaxStackDepth.
    Program(const std::int32_t* opcodes, const std::int32_t* args, std::size_t size, std::vector<double> consts,
            std::int32_t n_unknowns, bool time_dependent);

    /// As above, with the tables read by TABLE instructions. Also throws
    /// mna::Error(NetlistFatal) for a TABLE operand outside [0, tables.size()),
    /// TABLE in a program that is not time dependent, and an invalid table
    /// (no point, times and values of different lengths, a non-finite time or
    /// value, times not strictly increasing, a negative or non-finite period,
    /// a non-finite origin).
    Program(const std::int32_t* opcodes, const std::int32_t* args, std::size_t size, std::vector<double> consts,
            std::int32_t n_unknowns, bool time_dependent, std::vector<Table> tables);

    const std::vector<Instruction>& code() const noexcept { return code_; }
    const std::vector<double>& consts() const noexcept { return consts_; }
    const std::vector<Table>& tables() const noexcept { return tables_; }
    /// Segment slopes of tables()[k] (size() - 1 entries).
    const std::vector<double>& table_slopes(std::size_t k) const noexcept { return slopes_[k]; }
    std::int32_t n_unknowns() const noexcept { return n_unknowns_; }
    bool time_dependent() const noexcept { return time_dependent_; }
    /// Largest stack depth reached while evaluating.
    std::size_t max_depth() const noexcept { return max_depth_; }
    /// Sorted, unique LOAD_X operands: the unknowns the program reads.
    const std::vector<std::int32_t>& unknowns() const noexcept { return unknowns_; }
    /// Per instruction: position of the LOAD_X operand in unknowns(), else -1.
    const std::vector<std::int32_t>& load_positions() const noexcept { return load_pos_; }

private:
    std::vector<Instruction> code_;
    std::vector<double> consts_;
    std::vector<Table> tables_;
    std::vector<std::vector<double>> slopes_;
    std::vector<std::int32_t> unknowns_;
    std::vector<std::int32_t> load_pos_;
    std::int32_t n_unknowns_;
    bool time_dependent_;
    std::size_t max_depth_ = 0;
};

/// Value of the program at unknowns `x` (at least n_unknowns() entries; may be
/// null when the program reads no unknown) and time `t`.
double evaluate(const Program& program, const double* x, double t);

/// out[i] = evaluate(program, nullptr, t_array[i]). Throws mna::Error if the
/// program reads unknowns.
void evaluate_many(const Program& program, const double* t_array, std::size_t count, double* out);

/// Value and gradient with respect to the unknowns wrt[0..count): grad[j] is
/// d value / d x[wrt[j]] (0 for an unknown the program does not read).
void evaluate_with_gradient(const Program& program, const double* x, double t, const std::int32_t* wrt, int count,
                            double* value, double* grad);

/// Value and gradient with respect to program.unknowns(), in that order
/// (grad has program.unknowns().size() entries). Used by the device kernels.
void evaluate_gradient_unknowns(const Program& program, const double* x, double t, double* value, double* grad);

// Scalar semantics shared with the evaluators (exposed for tests).
double py_mod(double a, double b) noexcept;
double np_maximum(double a, double b) noexcept;
double np_minimum(double a, double b) noexcept;
double np_heaviside(double x, double h0) noexcept;

}  // namespace mna::expr
