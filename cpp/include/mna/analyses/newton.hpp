// Damped Newton driver and linear-system front end shared by .op and .tran.
//
// A port of the Python core's solve_dc_nr / _damped_newton_step on the
// compiled circuit (CPP_MIGRATION_PLAN.md 2.4 and 6.1). For a static matrix A
// given as values in the circuit's CSC pattern (G_eff = G + gmin * gmin_stamps
// for .op, G_eff + C / h for a Backward-Euler step) it solves
//
//     F(x) = A x + f(x) - b = 0
//
// where f is the device current vector of mna::assembly::evaluate. Per
// iteration i (0 <= i < max_iter):
//   1. F at x (the residual of the accepted damping trial is reused: it is
//      the same computation at the same point), ||F|| as finite_norm();
//      ||F|| < f_tol -> converged (also before the first solve).
//   2. J dx = -F with J = A + device Jacobian values in the same pattern: the
//      direct solver refactors (pivot reuse, falling back to a factorization
//      with fresh pivoting inside the solver); with Krylov enabled the Krylov
//      kernel runs first with the solve_linear acceptance rule. A singular
//      direct solve -> NewtonStatus::Singular at iteration i.
//   3. Damping: trials x + dx * s for s = 1, 1/2, ... (28 trials); the first
//      trial with a finite norm <= ||F|| is accepted, the best strictly
//      smaller finite trial is tracked as the fallback, otherwise the step
//      scale is 0.
//   4. ||F(x_next)|| < f_tol -> converged; scale 0 -> NoReduction at
//      iteration i; ||x_next - x|| < v_tol -> converged.
// After max_iter iterations -> MaxIterations. Callers turn the status into
// the reference error text of their analysis.
//
// Junction strategy (CPP_MIGRATION_PLAN.md 2.4.1, P3b: "improve, not
// parity"; circuits with exponential junctions and
// NewtonOptions::junction_limiting set). Every solve starts with the
// reference iteration above, so every run it solves without stalling comes
// out bit for bit as before. It switches to the junction strategy for the rest of the solve
// (same iterate, same iteration budget) when the reference iteration fails
// (no damping trial accepted: the same iteration is retried with the
// strategy) or stalls (kStallIterations consecutive iterations accepted only
// at a damping scale <= kStallScale). In junction mode, per iteration:
//   * the update is junction-limited (SPICE pnjlim,
//     mna/analyses/limiting.hpp) before the damping loop; when the limiter
//     changed it, the damping loop runs on the limited step first and falls
//     back to the plain Newton step when no trial of the limited one is
//     accepted;
//   * a damping trial whose residual norm grew is still accepted when it
//     passes the restricted natural monotonicity test (Deuflhard):
//     ||J(x)^-1 F(x + s dx)|| <= (1 - s / 4) ||dx||, with the factorization
//     of this iteration (direct solves only, not with Krylov). The residual
//     norm mixes volts (KVL rows) and amperes (KCL rows); near a junction
//     carrying a large current the exponential's linearization error swamps
//     the KVL residual, so the residual test alone accepts only tiny steps
//     (a diode across an ideal source crept ~vt per iteration and never
//     converged).
// Why not limit every iteration from the start: on converging circuits it
// moves the point where the tolerances stop the iteration, by up to the
// Newton tolerance; the differential corpus compares at 1e-6 and some
// reference solutions are themselves only that close to the exact one
// (rand_bjt_ce_s1's initial operating point steers a transient that is
// 7e-7 V off a tightly converged one).
//
// Device evaluation runs on mna::assembly::evaluate with
// device_eval_threads(circuit, num_threads) threads: one thread below
// kParallelDeviceMinCount devices (thread start-up costs more than the work),
// the requested count above. The assembly is bitwise identical for every
// thread count, so results never depend on num_threads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mna/circuit/circuit.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"
#include "mna/linear/linear_solver.hpp"

namespace mna::analyses {

/// Device evaluation uses more than one thread only from this many devices on
/// (CPP_MIGRATION_PLAN.md 2.5).
inline constexpr std::int64_t kParallelDeviceMinCount = 2048;

/// Halvings tried by the damped Newton step.
inline constexpr int kMaxDampingTrials = 28;
/// Junction strategy trigger: this many consecutive reference iterations
/// accepted at a damping scale <= kStallScale (on the corpus no operating
/// point the reference solves takes more than 2 consecutive iterations at a
/// scale <= 1/16, or any at <= 1/64; the clamped diode creeps at ~1/1000).
inline constexpr int kStallIterations = 4;
inline constexpr double kStallScale = 1.0 / 64.0;

/// Thread count for device evaluation: 1 below kParallelDeviceMinCount
/// devices, effective_threads(requested) otherwise.
int device_eval_threads(const circuit::Circuit& circuit, int requested) noexcept;

/// The solve_linear Krylov options (rank already resolved by the caller with
/// resolve_krylov_rank for the circuit dimension).
struct KrylovOptions {
    bool enabled = false;
    std::string method = "auto";  ///< requested method: "auto", a route name or any alias
    double tol = 1e-9;
    std::int64_t max_iter = 2000;
    std::int64_t rank = 80;
};

/// One Krylov attempt, with the fields the Python facade puts in its
/// krylov_stats entries.
struct KrylovStat {
    linear::KrylovInfo info;
    linear::MatrixKind matrix_kind = linear::MatrixKind::General;
    std::int64_t dimension = 0;
    std::int64_t nnz = 0;           ///< nonzero values of the operator
    bool csr_operator = false;      ///< the facade passed a sparse matrix (else a dense one)
    bool used_direct_fallback = false;
};

struct NewtonOptions {
    std::int64_t max_iter = 10000;
    double v_tol = 1e-9;
    double f_tol = 1e-9;
    int num_threads = 1;
    KrylovOptions krylov;
    /// Junction strategy after a failed or stalled reference iteration
    /// (pnjlim + natural monotonicity test, see above).
    bool junction_limiting = true;
    /// .op only: gmin continuation after a failed Newton solve (op.hpp).
    bool gmin_continuation = true;
};

/// +inf when any entry is not finite, otherwise sqrt(sum v_i^2) (NumPy's
/// norm: no scaling, so a huge finite vector may also give +inf).
double finite_norm(const double* v, std::size_t n) noexcept;

/// y = A x for values in the circuit pattern (y is overwritten).
void pattern_matvec(const circuit::Circuit& circuit, const double* values, const double* x, double* y) noexcept;

/// How SystemSolver::solve prepares the direct solver.
enum class FactorMode {
    Factor,    ///< numeric factorization with fresh partial pivoting
    Refactor,  ///< reuse the previous pivot order (the solver falls back to Factor)
    Cached,    ///< reuse the current factorization if it is valid, else Factor
};

/// Linear systems A x = rhs on the circuit's CSC pattern with the semantics
/// of the Python solve_linear facade: Krylov first when enabled (accepted if
/// converged or relative residual <= max(10 tol, 1e-8), else recorded with
/// used_direct_fallback and solved directly), then the direct solver (KLU).
class SystemSolver {
public:
    /// `stats` (may be null) receives one entry per Krylov attempt.
    SystemSolver(const circuit::Circuit& circuit, const KrylovOptions& krylov, int num_threads,
                 std::vector<KrylovStat>* stats);
    ~SystemSolver();
    SystemSolver(const SystemSolver&) = delete;
    SystemSolver& operator=(const SystemSolver&) = delete;

    /// x <- A^-1 rhs (x and rhs may not alias). `x0` is the Krylov initial
    /// guess (nullptr: zeros). `csr_operator` only labels the Krylov stats.
    /// Returns false when the direct solver finds A singular.
    bool solve(const double* values, const double* rhs, double* x, const double* x0, bool csr_operator,
               FactorMode mode);

    /// x <- A^-1 rhs with the numeric factorization of the last successful
    /// direct solve (no refactorization). Returns false when there is none:
    /// the last solve was answered by Krylov or found A singular.
    bool resolve(const double* rhs, double* x);

    /// Numeric factorization of `values` with the direct solver (Krylov is
    /// not consulted); false when the matrix is singular.
    bool factor_check(const double* values);

    /// The values changed: the next Cached solve factors again and the
    /// cached Krylov operator is rebuilt.
    void invalidate() noexcept;

    /// Numeric factorizations (factor or refactor calls) so far.
    std::int64_t factorizations() const noexcept { return factorizations_; }

private:
    bool direct(const double* values, const double* rhs, double* x, FactorMode mode);
    void build_krylov_operator(const double* values);

    const circuit::Circuit& circuit_;
    KrylovOptions krylov_;
    int num_threads_;
    std::vector<KrylovStat>* stats_;
    linear::CscPattern pattern_;
    std::unique_ptr<linear::LinearSolver<double>> direct_;
    bool factor_valid_ = false;
    bool numeric_valid_ = false;  ///< direct_ holds the factorization of the last solve
    std::int64_t factorizations_ = 0;
    // Krylov operator: the nonzero values only (the facade passed dense or
    // SciPy matrices, whose conversion drops zeros), its row view and kind.
    bool krylov_valid_ = false;
    linear::CscPattern krylov_pattern_;
    std::vector<double> krylov_values_;
    linear::CsrMatrix<double> krylov_matrix_;
    linear::MatrixKind krylov_kind_ = linear::MatrixKind::General;
};

enum class NewtonStatus { Converged, Singular, NoReduction, MaxIterations };

struct NewtonOutcome {
    NewtonStatus status = NewtonStatus::Converged;
    std::int64_t iteration = 0;   ///< iteration index of a Singular / NoReduction failure
    std::int64_t iterations = 0;  ///< linear (Jacobian) solves performed
    std::int64_t limited_steps = 0;     ///< iterations whose update the junction limiter changed
    std::int64_t fallback_steps = 0;    ///< of those, iterations that fell back to the plain step
    std::int64_t monotonicity_accepts = 0;  ///< trials accepted by the natural monotonicity test
    std::int64_t junction_switch = -1;  ///< iteration at which the junction strategy took over (-1: never)
};

/// Reusable damped Newton solver for one circuit: the direct solver keeps its
/// symbolic analysis and pivot order across solve() calls (transient steps).
class NewtonDriver {
public:
    NewtonDriver(const circuit::Circuit& circuit, const NewtonOptions& options, std::vector<KrylovStat>* stats);
    ~NewtonDriver();
    NewtonDriver(const NewtonDriver&) = delete;
    NewtonDriver& operator=(const NewtonDriver&) = delete;

    /// Solves A x + f(x, t) = b starting from x (n entries, overwritten with
    /// the last iterate). `a_static` holds nnz values in pattern order.
    /// `max_iter` < 0 uses NewtonOptions::max_iter.
    NewtonOutcome solve(const double* a_static, const double* b, double t, std::vector<double>& x,
                        std::int64_t max_iter = -1);

    std::int64_t factorizations() const noexcept { return system_.factorizations(); }

    /// True when the Newton matrix A + J(x, t) is singular for the direct solver.
    bool jacobian_singular(const double* a_static, double t, const std::vector<double>& x);

private:
    void evaluate_residual(const double* a_static, const double* b, double t, const double* x, double* f,
                           double* jvals, double* residual);

    /// Result of the damping loop on one candidate step.
    struct Damping {
        bool accepted = false;
        double next_norm = 0.0;
        double step_scale = 0.0;
        double best_norm = 0.0;
        double best_scale = 0.0;
        bool by_monotonicity = false;
    };
    /// The damping loop (step 3) on `step` from x; the accepted trial is left
    /// in trial_ / trial_f_ / trial_jdev_ / trial_residual_, the best strictly
    /// smaller one in best_.
    Damping damp(const double* a_static, const double* b, double t, const std::vector<double>& x,
                 const std::vector<double>& step, double norm_now, bool monotonicity);

    const circuit::Circuit& circuit_;
    NewtonOptions options_;
    int device_threads_;
    SystemSolver system_;
    std::vector<double> f_, jdev_, residual_;
    std::vector<double> trial_, trial_f_, trial_jdev_, trial_residual_;
    std::vector<double> best_, jacobian_, rhs_, delta_, ax_;
    std::vector<double> limited_, correction_;
};

}  // namespace mna::analyses
