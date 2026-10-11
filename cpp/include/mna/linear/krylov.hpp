// Krylov subspace solvers for A x = b over the CSR row view, real and complex,
// with the routing, kernel selection and metadata of the Python core
// (solvers.krylov_solve_linear and its helpers).
//
// Routing (solvers._krylov_route): "auto" follows the matrix kind; aliases
//   cg, conjugate_gradient, positive_definite, spd          -> positive_definite
//   cr, minres, conjugate_residual, symmetric, hermitian    -> symmetric
//   arnoldi, gmres, arnoldi_gmres, general                  -> general
// case-insensitively; anything else is std::invalid_argument
// ("Unknown Krylov method '<method>'.", a ValueError in Python).
//
// Kernel selection. The reference hands systems with n >= 64 to SciPy
// (scipy.sparse.linalg cg / minres / gmres; MINRES for real matrices only) and
// solves the others with its own NumPy loops. Each regime is reproduced with
// the semantics that decide `converged` there:
//
//   route              n < 64, or symmetric+complex     n >= 64 (kScipyEngineMinN)
//   positive_definite  LegacyCg   (_conjugate_gradient)  Cg     (scipy cg)
//   symmetric          LegacyCr   (_conjugate_residual)  Minres (scipy minres, real)
//   general            LegacyGmres (_arnoldi_gmres)      RestartedGmres (scipy gmres)
//
// Legacy kernels: faithful ports of the reference loops. The relative residual
// ||b - A x|| / ||b|| is recomputed explicitly after every update, the
// breakdown tolerance is eps * 100 (absolute), the best iterate is tracked and
// returned on failure, a zero right-hand side returns zeros after 0
// iterations. LegacyGmres solves the small least-squares problem with Givens
// rotations (the reference calls numpy.linalg.lstsq) and treats R pivots below
// eps * max(m, n) * max|R_kk| as zero, like lstsq's default rcond. Iteration
// budgets: LegacyGmres max_iter inner steps with restart = rank; LegacyCg and
// LegacyCr `rank` steps and they report `rank` iterations when they fail.
//
// SciPy-equivalent kernels (ports of scipy/sparse/linalg/_isolve): Cg checks
// ||r|| < tol ||b|| at the start of each of `rank` iterations; Minres is the
// Paige-Saunders algorithm with SciPy's stopping tests (converged unless the
// iteration limit `rank` is reached); RestartedGmres runs up to max_iter
// restart cycles of min(rank, n) MGS-Arnoldi steps with Givens rotations,
// stops a cycle on the rotated residual estimate and accepts on the true
// residual at the end of a cycle. They report exit_code (0 or the SciPy
// iteration limit) and their relative residual is recomputed from the
// returned x, as the reference does after calling SciPy.
#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

#include "mna/linear/classify.hpp"
#include "mna/linear/sparse.hpp"

namespace mna::linear {

enum class KrylovRoute { PositiveDefinite, Symmetric, General };

/// "positive_definite", "symmetric", "general".
const char* krylov_route_name(KrylovRoute route) noexcept;
/// The route "auto" selects for a matrix kind.
KrylovRoute route_for_kind(MatrixKind kind) noexcept;
/// True when `method` is "auto" (case-insensitive), i.e. the route needs the matrix kind.
bool method_is_auto(std::string_view method) noexcept;
/// solvers._krylov_route; throws std::invalid_argument for an unknown method.
KrylovRoute krylov_route(MatrixKind kind, std::string_view method);

enum class KrylovAlgorithm {
    LegacyGmres,     // port of solvers._arnoldi_gmres
    LegacyCg,        // port of solvers._conjugate_gradient
    LegacyCr,        // port of solvers._conjugate_residual
    RestartedGmres,  // scipy.sparse.linalg.gmres semantics
    Cg,              // scipy.sparse.linalg.cg semantics
    Minres,          // scipy.sparse.linalg.minres (Paige-Saunders), real matrices only
};

/// Systems at least this large use the SciPy-equivalent kernels (see above).
inline constexpr std::int32_t kScipyEngineMinN = 64;
/// Breakdown tolerance of the legacy loops: np.finfo(float).eps * 100.
inline constexpr double kLegacyBreakdownTol = std::numeric_limits<double>::epsilon() * 100.0;

/// "legacy_gmres", "legacy_cg", "legacy_cr", "restarted_gmres", "cg", "minres".
const char* krylov_algorithm_name(KrylovAlgorithm algorithm) noexcept;
/// The metadata `method`: "arnoldi_gmres", "conjugate_gradient",
/// "conjugate_residual" (LegacyCr) or "minres".
const char* krylov_method_name(KrylovAlgorithm algorithm) noexcept;
KrylovRoute route_of_algorithm(KrylovAlgorithm algorithm) noexcept;
/// The kernel the reference runs for a route, dimension and scalar type.
KrylovAlgorithm select_krylov_algorithm(KrylovRoute route, std::int32_t n, bool is_complex) noexcept;

struct KrylovSettings {
    double tol = 1e-9;
    /// Budget of the general route: inner steps (LegacyGmres) or restart cycles (RestartedGmres).
    std::int64_t max_iter = 2000;
    /// Resolved Krylov rank: GMRES restart length, iteration budget of CG / CR / MINRES.
    std::int64_t rank = 80;
    int num_threads = 1;
};

struct KrylovInfo {
    KrylovRoute route = KrylovRoute::General;
    KrylovAlgorithm algorithm = KrylovAlgorithm::LegacyGmres;
    const char* method = "arnoldi_gmres";
    bool converged = false;
    std::int64_t iterations = 0;
    double relative_residual = 0.0;
    /// True for the SciPy-equivalent kernels, whose reference metadata carries
    /// exit_code (and effective_restart = None on the CG / MINRES routes).
    bool scipy_semantics = false;
    std::int64_t exit_code = 0;
    /// max_iter on the general route, rank otherwise (the reference's iteration_budget).
    std::int64_t iteration_budget = 0;
    /// min(rank, n) on the general route, -1 otherwise.
    std::int64_t effective_restart = -1;
    int threads_used = 1;
};

/// Solves A x = b with the kernel select_krylov_algorithm() picks for `route`.
/// `x` holds the initial guess on entry (n values) and the solution on exit.
template <class Scalar>
KrylovInfo krylov_solve(const CsrMatrix<Scalar>& A, const Scalar* b, Scalar* x, KrylovRoute route,
                        const KrylovSettings& settings);

/// Runs one specific kernel (any dimension). Minres with complex values throws
/// std::invalid_argument.
template <class Scalar>
KrylovInfo run_krylov_algorithm(KrylovAlgorithm algorithm, const CsrMatrix<Scalar>& A, const Scalar* b, Scalar* x,
                                const KrylovSettings& settings);

extern template KrylovInfo krylov_solve<double>(const CsrMatrix<double>&, const double*, double*, KrylovRoute,
                                                const KrylovSettings&);
extern template KrylovInfo krylov_solve<Complex>(const CsrMatrix<Complex>&, const Complex*, Complex*, KrylovRoute,
                                                 const KrylovSettings&);
extern template KrylovInfo run_krylov_algorithm<double>(KrylovAlgorithm, const CsrMatrix<double>&, const double*,
                                                        double*, const KrylovSettings&);
extern template KrylovInfo run_krylov_algorithm<Complex>(KrylovAlgorithm, const CsrMatrix<Complex>&, const Complex*,
                                                         Complex*, const KrylovSettings&);

}  // namespace mna::linear
