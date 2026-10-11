// Krylov solvers: routing, kernel selection, legacy ports and SciPy-equivalent kernels.
#include <doctest/doctest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "mna/linear/klu_solver.hpp"
#include "mna/linear/krylov.hpp"

using mna::linear::Complex;
using mna::linear::CsrMatrix;
using mna::linear::KrylovAlgorithm;
using mna::linear::KrylovInfo;
using mna::linear::KrylovRoute;
using mna::linear::KrylovSettings;
using mna::linear::MatrixKind;

namespace {

using IVec = std::vector<std::int32_t>;

template <class T>
struct Csc {
    std::int32_t n = 0;
    IVec Ap{0};
    IVec Ai;
    std::vector<T> Ax;

    CsrMatrix<T> csr() const {
        return CsrMatrix<T>::from_csc(n, Ap.data(), Ap.size(), Ai.data(), Ai.size(), Ax.data(), Ax.size());
    }
};

template <class T>
Csc<T> from_dense(std::int32_t n, const std::vector<T>& colmajor) {
    Csc<T> m;
    m.n = n;
    for (std::int32_t j = 0; j < n; ++j) {
        for (std::int32_t i = 0; i < n; ++i) {
            const T v = colmajor[static_cast<std::size_t>(i + j * n)];
            if (v != T{}) {
                m.Ai.push_back(i);
                m.Ax.push_back(v);
            }
        }
        m.Ap.push_back(static_cast<std::int32_t>(m.Ai.size()));
    }
    return m;
}

// 5-point Laplacian on side x side with diagonal 4 - shift (+ i * imag_shift).
template <class T>
Csc<T> laplacian(std::int32_t side, double shift = 0.0, double imag_shift = 0.0) {
    Csc<T> m;
    m.n = side * side;
    for (std::int32_t col = 0; col < m.n; ++col) {
        const std::int32_t r = col / side;
        const std::int32_t c = col % side;
        auto add = [&](std::int32_t row, T v) {
            m.Ai.push_back(row);
            m.Ax.push_back(v);
        };
        if (r > 0) add(col - side, T(-1.0));
        if (c > 0) add(col - 1, T(-1.0));
        if constexpr (std::is_same_v<T, double>) {
            add(col, 4.0 - shift);
        } else {
            add(col, T(4.0 - shift, imag_shift));
        }
        if (c + 1 < side) add(col + 1, T(-1.0));
        if (r + 1 < side) add(col + side, T(-1.0));
        m.Ap.push_back(static_cast<std::int32_t>(m.Ai.size()));
    }
    return m;
}

template <class T>
double residual_norm(const CsrMatrix<T>& A, const std::vector<T>& x, const std::vector<T>& b) {
    std::vector<T> ax(b.size());
    mna::linear::spmv(A, x.data(), ax.data(), 1);
    double r = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        r += std::norm(b[i] - ax[i]);
        nb += std::norm(b[i]);
    }
    return std::sqrt(r) / (nb > 0 ? std::sqrt(nb) : 1.0);
}

template <class T>
std::vector<T> ramp(std::int32_t n) {
    std::vector<T> b(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(i)] = T(1.0 + 0.01 * i);
    }
    return b;
}

KrylovSettings settings(double tol, std::int64_t max_iter, std::int64_t rank, int threads = 1) {
    KrylovSettings s;
    s.tol = tol;
    s.max_iter = max_iter;
    s.rank = rank;
    s.num_threads = threads;
    return s;
}

template <class T>
KrylovInfo run(KrylovAlgorithm algorithm, const Csc<T>& m, const std::vector<T>& b, std::vector<T>& x,
               const KrylovSettings& s) {
    return mna::linear::run_krylov_algorithm(algorithm, m.csr(), b.data(), x.data(), s);
}

// The 3x3 matrices of tests/test_krylov_solvers.py (column-major) and b = (1, 2, 3).
const std::vector<double> kGeneral = {4, 2, 0, 1, 3, 1, 0, 1, 2};
const std::vector<double> kSymmetric = {4, 1, 0, 1, -3, 1, 0, 1, 2};
const std::vector<double> kSpd = {4, 1, 0, 1, 3, 1, 0, 1, 2};
const std::vector<double> kB = {1, 2, 3};

}  // namespace

TEST_CASE("method aliases route like solvers._krylov_route") {
    using mna::linear::krylov_route;
    for (const char* m : {"cg", "conjugate_gradient", "positive_definite", "spd", "CG", "Spd"}) {
        CHECK(krylov_route(MatrixKind::General, m) == KrylovRoute::PositiveDefinite);
    }
    for (const char* m : {"cr", "minres", "conjugate_residual", "symmetric", "hermitian", "MINRES"}) {
        CHECK(krylov_route(MatrixKind::PositiveDefinite, m) == KrylovRoute::Symmetric);
    }
    for (const char* m : {"arnoldi", "gmres", "arnoldi_gmres", "general", "GMRES"}) {
        CHECK(krylov_route(MatrixKind::PositiveDefinite, m) == KrylovRoute::General);
    }
    CHECK(krylov_route(MatrixKind::General, "auto") == KrylovRoute::General);
    CHECK(krylov_route(MatrixKind::Symmetric, "Auto") == KrylovRoute::Symmetric);
    CHECK(krylov_route(MatrixKind::PositiveDefinite, "AUTO") == KrylovRoute::PositiveDefinite);
    CHECK(mna::linear::method_is_auto("aUtO"));
    CHECK_FALSE(mna::linear::method_is_auto("auto "));
    CHECK_FALSE(mna::linear::method_is_auto("gmres"));
    try {
        krylov_route(MatrixKind::General, "foo");
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()) == "Unknown Krylov method 'foo'.");
    }
    CHECK_THROWS_AS(krylov_route(MatrixKind::General, " gmres"), std::invalid_argument);
    CHECK(std::string(mna::linear::krylov_route_name(KrylovRoute::PositiveDefinite)) == "positive_definite");
}

TEST_CASE("kernel selection reproduces the reference's engine split at n = 64") {
    using mna::linear::select_krylov_algorithm;
    CHECK(select_krylov_algorithm(KrylovRoute::General, 63, false) == KrylovAlgorithm::LegacyGmres);
    CHECK(select_krylov_algorithm(KrylovRoute::General, 64, false) == KrylovAlgorithm::RestartedGmres);
    CHECK(select_krylov_algorithm(KrylovRoute::General, 64, true) == KrylovAlgorithm::RestartedGmres);
    CHECK(select_krylov_algorithm(KrylovRoute::PositiveDefinite, 63, false) == KrylovAlgorithm::LegacyCg);
    CHECK(select_krylov_algorithm(KrylovRoute::PositiveDefinite, 64, true) == KrylovAlgorithm::Cg);
    CHECK(select_krylov_algorithm(KrylovRoute::Symmetric, 63, false) == KrylovAlgorithm::LegacyCr);
    CHECK(select_krylov_algorithm(KrylovRoute::Symmetric, 64, false) == KrylovAlgorithm::Minres);
    CHECK(select_krylov_algorithm(KrylovRoute::Symmetric, 1035, true) == KrylovAlgorithm::LegacyCr);
    // Metadata names: the symmetric route says "minres" only for the SciPy-equivalent kernel.
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::Minres)) == "minres");
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::LegacyCr)) == "conjugate_residual");
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::Cg)) == "conjugate_gradient");
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::LegacyCg)) == "conjugate_gradient");
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::RestartedGmres)) == "arnoldi_gmres");
    CHECK(std::string(mna::linear::krylov_method_name(KrylovAlgorithm::LegacyGmres)) == "arnoldi_gmres");
}

TEST_CASE("legacy kernels on the test_krylov_solvers matrices") {
    struct Case {
        const std::vector<double>* A;
        KrylovRoute route;
        const char* method;
    };
    for (const Case& c : {Case{&kGeneral, KrylovRoute::General, "arnoldi_gmres"},
                          Case{&kSymmetric, KrylovRoute::Symmetric, "conjugate_residual"},
                          Case{&kSpd, KrylovRoute::PositiveDefinite, "conjugate_gradient"}}) {
        const Csc<double> m = from_dense(3, *c.A);
        std::vector<double> x(3, 0.0);
        const KrylovInfo info = mna::linear::krylov_solve(m.csr(), kB.data(), x.data(), c.route, settings(1e-12, 30, 3));
        CHECK(std::string(info.method) == c.method);
        CHECK(info.converged);
        CHECK(info.iterations >= 1);
        CHECK(info.iterations <= 3);
        CHECK_FALSE(info.scipy_semantics);
        CHECK(info.relative_residual <= 1e-12);
        CHECK(residual_norm(m.csr(), x, kB) < 1e-10);
        CHECK(info.iteration_budget == (c.route == KrylovRoute::General ? 30 : 3));
        CHECK(info.effective_restart == (c.route == KrylovRoute::General ? 3 : -1));
    }
    // Explicit Arnoldi with rank 2 on the positive-definite matrix.
    const Csc<double> spd = from_dense(3, kSpd);
    std::vector<double> x(3, 0.0);
    const KrylovInfo info =
        mna::linear::krylov_solve(spd.csr(), kB.data(), x.data(), KrylovRoute::General, settings(1e-12, 30, 2));
    CHECK(info.converged);
    CHECK(info.effective_restart == 2);
    CHECK(residual_norm(spd.csr(), x, kB) < 1e-10);
}

TEST_CASE("legacy GMRES failure keeps the best iterate and counts inner steps") {
    // krylov_max_iter = 1, rank 1 on the general 3x3: one step, no convergence.
    const Csc<double> m = from_dense(3, kGeneral);
    std::vector<double> x(3, 0.0);
    const KrylovInfo info = run(KrylovAlgorithm::LegacyGmres, m, kB, x, settings(1e-12, 1, 1));
    CHECK_FALSE(info.converged);
    CHECK(info.iterations == 1);
    CHECK(info.relative_residual > 1e-3);
    CHECK(info.relative_residual < 1.0);  // the one step improved on x0 = 0
    CHECK(residual_norm(m.csr(), x, kB) == doctest::Approx(info.relative_residual).epsilon(1e-12));

    // Legacy CG / CR report the full budget when they fail.
    const Csc<double> lap = laplacian<double>(5);  // n = 25
    const std::vector<double> b = ramp<double>(25);
    for (KrylovAlgorithm alg : {KrylovAlgorithm::LegacyCg, KrylovAlgorithm::LegacyCr}) {
        std::vector<double> y(25, 0.0);
        const KrylovInfo f = run(alg, lap, b, y, settings(1e-14, 2000, 2));
        CHECK_FALSE(f.converged);
        CHECK(f.iterations == 2);
        CHECK(f.relative_residual < 1.0);
    }
}

TEST_CASE("zero right-hand side returns zeros after 0 iterations for every kernel") {
    const Csc<double> lap = laplacian<double>(9);  // n = 81
    const std::vector<double> zero(81, 0.0);
    for (KrylovAlgorithm alg : {KrylovAlgorithm::LegacyGmres, KrylovAlgorithm::LegacyCg, KrylovAlgorithm::LegacyCr,
                                KrylovAlgorithm::RestartedGmres, KrylovAlgorithm::Cg, KrylovAlgorithm::Minres}) {
        std::vector<double> x(81, 0.5);  // a nonzero initial guess is discarded
        const KrylovInfo info = run(alg, lap, zero, x, settings(1e-9, 100, 41));
        CHECK(info.converged);
        CHECK(info.iterations == 0);
        CHECK(info.relative_residual == 0.0);
        CHECK(info.exit_code == 0);
        for (double v : x) {
            CHECK(v == 0.0);
        }
    }
}

TEST_CASE("exact initial guess: SciPy kernels stop at once, legacy CG / CR keep their quirk") {
    // x = (1, 2, 3) solves A x = (6, 10, 8) exactly in floating point.
    const Csc<double> m = from_dense(3, kSpd);
    const std::vector<double> b = {6.0, 10.0, 8.0};
    const std::vector<double> exact = {1.0, 2.0, 3.0};
    // Reference _conjugate_gradient / _conjugate_residual: r = 0 -> denominator
    // 0 -> break -> "not converged" with the budget as iteration count and the
    // residual of the (exact) best iterate, which solve_linear then accepts.
    for (KrylovAlgorithm alg : {KrylovAlgorithm::LegacyCg, KrylovAlgorithm::LegacyCr}) {
        std::vector<double> x = exact;
        const KrylovInfo legacy = run(alg, m, b, x, settings(1e-9, 30, 3));
        CHECK_FALSE(legacy.converged);
        CHECK(legacy.iterations == 3);
        CHECK(legacy.relative_residual == 0.0);
        CHECK(x == exact);
    }
    // _arnoldi_gmres and SciPy cg / minres / gmres accept x0 before iterating.
    for (KrylovAlgorithm alg : {KrylovAlgorithm::LegacyGmres, KrylovAlgorithm::Cg, KrylovAlgorithm::Minres,
                                KrylovAlgorithm::RestartedGmres}) {
        std::vector<double> y = exact;
        const KrylovInfo info = run(alg, m, b, y, settings(1e-9, 30, 3));
        CAPTURE(mna::linear::krylov_algorithm_name(alg));
        CHECK(info.converged);
        CHECK(info.iterations == 0);
        CHECK(y == exact);
    }
}

TEST_CASE("SciPy-equivalent kernels converge on a 2-D Laplacian") {
    const Csc<double> lap = laplacian<double>(12);  // n = 144, SPD
    const std::vector<double> b = ramp<double>(lap.n);
    std::vector<double> direct = b;
    mna::linear::KluSolver lu;
    lu.analyze(lap.n, lap.Ap, lap.Ai);
    lu.factor(lap.Ax);
    lu.solve(direct);

    for (KrylovAlgorithm alg : {KrylovAlgorithm::Cg, KrylovAlgorithm::Minres, KrylovAlgorithm::RestartedGmres}) {
        std::vector<double> x(static_cast<std::size_t>(lap.n), 0.0);
        const KrylovInfo info = run(alg, lap, b, x, settings(1e-10, 2000, 72));
        CAPTURE(mna::linear::krylov_algorithm_name(alg));
        CHECK(info.converged);
        CHECK(info.scipy_semantics);
        CHECK(info.exit_code == 0);
        CHECK(info.iterations >= 1);
        CHECK(info.iterations <= 144);
        // MINRES stops on SciPy's backward-error test ||r|| / (||A|| ||x||) <= tol,
        // which is looser than ||r|| / ||b|| <= tol.
        const bool minres = alg == KrylovAlgorithm::Minres;
        CHECK(info.relative_residual < (minres ? 1e-7 : 1e-10));
        double err = 0.0;
        double ref = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            err += (x[i] - direct[i]) * (x[i] - direct[i]);
            ref += direct[i] * direct[i];
        }
        CHECK(std::sqrt(err / ref) < (minres ? 1e-6 : 1e-8));
    }
}

TEST_CASE("MINRES handles symmetric indefinite systems") {
    const Csc<double> m = laplacian<double>(10, /*shift=*/2.0);  // eigenvalues in (-2, 6)
    const std::vector<double> b = ramp<double>(m.n);
    std::vector<double> x(static_cast<std::size_t>(m.n), 0.0);
    const KrylovInfo info = run(KrylovAlgorithm::Minres, m, b, x, settings(1e-10, 2000, 100));
    CHECK(info.converged);
    CHECK(std::string(info.method) == "minres");
    CHECK(residual_norm(m.csr(), x, b) < 1e-6);
    // Conjugate Residual on the same system.
    std::vector<double> y(static_cast<std::size_t>(m.n), 0.0);
    const KrylovInfo cr = run(KrylovAlgorithm::LegacyCr, m, b, y, settings(1e-8, 2000, 400));
    CHECK(cr.converged);
    CHECK(residual_norm(m.csr(), y, b) <= 1e-8);
}

TEST_CASE("SciPy-equivalent kernels report the iteration limit as exit_code") {
    const Csc<double> lap = laplacian<double>(10);  // n = 100
    const std::vector<double> b = ramp<double>(lap.n);
    {
        std::vector<double> x(100, 0.0);
        const KrylovInfo info = run(KrylovAlgorithm::Cg, lap, b, x, settings(1e-12, 2000, 2));
        CHECK_FALSE(info.converged);
        CHECK(info.iterations == 2);
        CHECK(info.exit_code == 2);
        CHECK(info.iteration_budget == 2);
    }
    {
        std::vector<double> x(100, 0.0);
        const KrylovInfo info = run(KrylovAlgorithm::Minres, lap, b, x, settings(1e-12, 2000, 3));
        CHECK_FALSE(info.converged);
        CHECK(info.iterations == 3);
        CHECK(info.exit_code == 3);
    }
    {
        std::vector<double> x(100, 0.0);
        const KrylovInfo info = run(KrylovAlgorithm::RestartedGmres, lap, b, x, settings(1e-12, 3, 2));
        CHECK_FALSE(info.converged);
        CHECK(info.iterations == 6);  // 3 restart cycles of 2 steps
        CHECK(info.exit_code == 3);
        CHECK(info.iteration_budget == 3);
        CHECK(info.effective_restart == 2);
    }
}

TEST_CASE("complex kernels: GMRES on non-Hermitian, CR on Hermitian") {
    // Complex non-Hermitian 3x3 (legacy GMRES).
    const std::vector<Complex> A = {{4, 1}, {0, 2}, {0, 0}, {1, 0}, {3, -1}, {1, 0}, {0, 0}, {1, 1}, {2, 2}};
    const Csc<Complex> m = from_dense(3, A);
    const std::vector<Complex> b = {{1, 0}, {0, 2}, {3, -1}};
    std::vector<Complex> x(3);
    KrylovInfo info = mna::linear::krylov_solve(m.csr(), b.data(), x.data(), KrylovRoute::General, settings(1e-12, 30, 3));
    CHECK(info.algorithm == KrylovAlgorithm::LegacyGmres);
    CHECK(info.converged);
    CHECK(residual_norm(m.csr(), x, b) < 1e-11);

    // Complex shifted Laplacian, n = 100: restarted GMRES.
    const Csc<Complex> lap = laplacian<Complex>(10, 0.5, 0.75);
    const std::vector<Complex> rhs = ramp<Complex>(lap.n);
    std::vector<Complex> y(static_cast<std::size_t>(lap.n));
    info = mna::linear::krylov_solve(lap.csr(), rhs.data(), y.data(), KrylovRoute::General, settings(1e-10, 2000, 50));
    CHECK(info.algorithm == KrylovAlgorithm::RestartedGmres);
    CHECK(info.converged);
    CHECK(info.iterations >= 1);
    CHECK(residual_norm(lap.csr(), y, rhs) <= 1e-10);

    // Hermitian (complex) symmetric route stays on Conjugate Residual even for n >= 64.
    Csc<Complex> herm = laplacian<Complex>(9);  // real-valued Hermitian, n = 81
    for (std::size_t p = 0; p < herm.Ax.size(); ++p) {
        if (herm.Ax[p] == Complex(-1.0)) {
            herm.Ax[p] = Complex(-0.5, 0.0);
        }
    }
    const std::vector<Complex> hb = ramp<Complex>(herm.n);
    std::vector<Complex> hx(static_cast<std::size_t>(herm.n));
    // (The reference's Conjugate Residual recurrence stalls near 1e-9 here; 1e-8 is reachable.)
    info = mna::linear::krylov_solve(herm.csr(), hb.data(), hx.data(), KrylovRoute::Symmetric, settings(1e-8, 2000, 400));
    CHECK(info.algorithm == KrylovAlgorithm::LegacyCr);
    CHECK(std::string(info.method) == "conjugate_residual");
    CHECK(info.converged);
    CHECK_FALSE(info.scipy_semantics);

    std::vector<Complex> z(static_cast<std::size_t>(herm.n));
    CHECK_THROWS_AS(run(KrylovAlgorithm::Minres, herm, hb, z, settings(1e-10, 10, 10)), std::invalid_argument);
}

TEST_CASE("Krylov solves above the parallel SpMV threshold are bitwise identical for any thread count") {
    const Csc<double> lap = laplacian<double>(66, 0.0);  // n = 4356 >= kParallelSpmvMinN
    REQUIRE(lap.n >= mna::linear::kParallelSpmvMinN);
    std::vector<double> b(static_cast<std::size_t>(lap.n));
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = std::sin(0.37 * static_cast<double>(i)) + 0.1;
    }
    const CsrMatrix<double> A = lap.csr();
    for (KrylovAlgorithm alg : {KrylovAlgorithm::Cg, KrylovAlgorithm::RestartedGmres}) {
        std::vector<double> x1(b.size(), 0.0);
        const KrylovInfo i1 = mna::linear::run_krylov_algorithm(alg, A, b.data(), x1.data(), settings(1e-8, 4, 30, 1));
        CHECK(i1.threads_used == 1);
        for (int threads : {2, 4}) {
            std::vector<double> xk(b.size(), 0.0);
            const KrylovInfo ik =
                mna::linear::run_krylov_algorithm(alg, A, b.data(), xk.data(), settings(1e-8, 4, 30, threads));
            CHECK(ik.threads_used == threads);
            CHECK(ik.iterations == i1.iterations);
            CHECK(ik.converged == i1.converged);
            CHECK(ik.relative_residual == i1.relative_residual);
            CHECK(std::memcmp(xk.data(), x1.data(), x1.size() * sizeof(double)) == 0);
        }
    }
}
