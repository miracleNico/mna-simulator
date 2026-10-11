// Harmonic balance: Fourier helpers, the block-triangular direct solver and hb_solve.
#include <doctest/doctest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mna/analyses/hb.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"

namespace hb = mna::analyses;
using mna::circuit::Circuit;
using mna::circuit::CircuitSpec;

namespace {

constexpr double kPi = 3.141592653589793;

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

/// amplitude * sin(omega * t) as a time-dependent program.
std::shared_ptr<const mna::expr::Program> sine_program(double amplitude, double omega) {
    const std::int32_t ops[6] = {0, 0, 2, 5, 12, 5};  // const, const, t, mul, sin, mul
    const std::int32_t args[6] = {0, 1, 0, 0, 0, 0};
    return std::make_shared<mna::expr::Program>(ops, args, 6, std::vector<double>{amplitude, omega}, 0, true);
}

/// The HB_ERROR message thrown by fn, or "" when it does not throw one.
template <class Fn>
std::string hb_error_message(Fn&& fn) {
    try {
        fn();
    } catch (const mna::Error& e) {
        if (e.code() == mna::ErrorCode::HbError) {
            return e.what();
        }
        return std::string("other code: ") + e.code_name();
    }
    return "";
}

struct Csc {
    std::int32_t n = 0;
    std::vector<std::int32_t> col_ptr;
    std::vector<std::int32_t> row_idx;
    std::vector<double> values;
};

Csc to_csc(const Eigen::MatrixXd& a) {
    Csc m;
    m.n = static_cast<std::int32_t>(a.rows());
    m.col_ptr.push_back(0);
    for (Eigen::Index j = 0; j < a.cols(); ++j) {
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            if (a(i, j) != 0.0) {
                m.row_idx.push_back(static_cast<std::int32_t>(i));
                m.values.push_back(a(i, j));
            }
        }
        m.col_ptr.push_back(static_cast<std::int32_t>(m.row_idx.size()));
    }
    return m;
}

std::vector<double> btf_solution(const Csc& m, const std::vector<double>& b, int threads,
                                 hb::BtfSolver* keep = nullptr) {
    hb::BtfSolver local;
    hb::BtfSolver& s = keep != nullptr ? *keep : local;
    s.analyze(m.n, m.col_ptr.data(), m.row_idx.data());
    s.factor(m.values.data(), threads);
    std::vector<double> x = b;
    s.solve(x.data());
    return x;
}

double rel_error(const std::vector<double>& x, const Eigen::VectorXd& ref) {
    double num = 0.0;
    double den = 0.0;
    for (Eigen::Index i = 0; i < ref.size(); ++i) {
        num = std::fmax(num, std::fabs(x[static_cast<std::size_t>(i)] - ref[i]));
        den = std::fmax(den, std::fabs(ref[i]));
    }
    return num / den;
}

/// A matrix with a known block upper triangular structure: `blocks` diagonal
/// blocks of the given sizes (dense or cyclic tridiagonal), random entries above,
/// then a random symmetric permutation of rows and columns.
Eigen::MatrixXd structured_matrix(const std::vector<int>& sizes, bool sparse_blocks, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    int n = 0;
    for (int s : sizes) {
        n += s;
    }
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n, n);
    int begin = 0;
    for (int s : sizes) {
        for (int i = 0; i < s; ++i) {
            if (sparse_blocks) {
                a(begin + i, begin + i) = 4.0 + u(rng);
                a(begin + i, begin + (i + 1) % s) = u(rng);
                a(begin + (i + 1) % s, begin + i) = u(rng);
            } else {
                for (int j = 0; j < s; ++j) {
                    a(begin + i, begin + j) = u(rng) + (i == j ? s : 0.0);
                }
            }
        }
        // couplings to later blocks only (upper block triangular)
        for (int i = 0; i < s; ++i) {
            for (int j = begin + s; j < n; ++j) {
                if ((rng() % 7) == 0) {
                    a(begin + i, j) = u(rng);
                }
            }
        }
        begin += s;
    }
    std::vector<int> perm(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        perm[static_cast<std::size_t>(i)] = i;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> qperm = perm;
    std::shuffle(qperm.begin(), qperm.end(), rng);
    Eigen::MatrixXd p(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            p(i, j) = a(perm[static_cast<std::size_t>(i)], qperm[static_cast<std::size_t>(j)]);
        }
    }
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Fourier helpers
// ---------------------------------------------------------------------------

TEST_CASE("hb_fourier_matrices follows create_fourier_matrices") {
    const hb::FourierMatrices m = hb::hb_fourier_matrices(3, 7);
    CHECK(m.rows == 7);
    CHECK(m.points == 7);
    for (int k = 0; k < 7; ++k) {
        CHECK(m.gamma[static_cast<std::size_t>(k)] == 1.0 / 7.0);
        CHECK(m.gamma_inv[static_cast<std::size_t>(k * 7)] == 1.0);
        for (int h = 1; h <= 3; ++h) {
            const double angle = 2.0 * kPi * h * k / 7.0;
            CHECK(m.gamma_inv[static_cast<std::size_t>(k * 7 + 2 * h - 1)] == std::cos(angle));
            CHECK(m.gamma_inv[static_cast<std::size_t>(k * 7 + 2 * h)] == std::sin(angle));
            CHECK(m.gamma[static_cast<std::size_t>((2 * h - 1) * 7 + k)] == (2.0 / 7.0) * std::cos(angle));
            CHECK(m.gamma[static_cast<std::size_t>(2 * h * 7 + k)] == (2.0 / 7.0) * std::sin(angle));
        }
    }
    // Gamma Gamma_inv = I for K = 2H + 1.
    const Eigen::Map<const Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> g(m.gamma.data());
    const Eigen::Map<const Eigen::Matrix<double, 7, 7, Eigen::RowMajor>> gi(m.gamma_inv.data());
    CHECK((g * gi - Eigen::Matrix<double, 7, 7>::Identity()).cwiseAbs().maxCoeff() < 1e-14);

    const hb::FourierMatrices rect = hb::hb_fourier_matrices(2, 4);  // K != S is allowed
    CHECK(rect.gamma.size() == 5 * 4);
    CHECK(rect.gamma_inv.size() == 4 * 5);
    CHECK_THROWS_AS(hb::hb_fourier_matrices(-1, 3), std::invalid_argument);
    CHECK_THROWS_AS(hb::hb_fourier_matrices(1, 0), std::domain_error);
}

TEST_CASE("hb_time_points is linspace(0, period, K, endpoint=False)") {
    const double omega0 = 2.0 * kPi * 5e6;
    const std::vector<double> t = hb::hb_time_points(omega0, 411);
    REQUIRE(t.size() == 411);
    const double step = (2.0 * kPi / omega0) / 411.0;
    for (int k = 0; k < 411; ++k) {
        CHECK(t[static_cast<std::size_t>(k)] == static_cast<double>(k) * step);
    }
    CHECK(hb::hb_time_points(1.0, 0).empty());
}

TEST_CASE("hb_reconstruct sums the harmonics and is thread independent") {
    const std::int64_t n = 2;
    const std::int64_t H = 3;
    std::vector<double> x(static_cast<std::size_t>(n * 7), 0.0);
    x[0] = 0.5;      // node 0 DC
    x[1] = 1.0;      // node 0 cos 1
    x[7 + 4] = 2.0;  // node 1 sin 2
    std::vector<double> t(1000);
    for (std::size_t j = 0; j < t.size(); ++j) {
        t[j] = 1e-6 * static_cast<double>(j);
    }
    const double omega0 = 2.0 * kPi * 1e3;
    std::vector<double> out1(static_cast<std::size_t>(n) * t.size());
    std::vector<double> out4(out1.size());
    hb::hb_reconstruct(x.data(), n, H, omega0, t.data(), static_cast<std::int64_t>(t.size()), 1, out1.data());
    hb::hb_reconstruct(x.data(), n, H, omega0, t.data(), static_cast<std::int64_t>(t.size()), 4, out4.data());
    CHECK(bitwise_equal(out1, out4));
    for (std::size_t j = 0; j < t.size(); j += 97) {
        CHECK(out1[j] == doctest::Approx(0.5 + std::cos(omega0 * t[j])).epsilon(1e-13));
        CHECK(out1[t.size() + j] == doctest::Approx(2.0 * std::sin(2.0 * omega0 * t[j])).epsilon(1e-12));
    }
}

TEST_CASE("hb_gram_columns gives Gamma diag(g) Gamma_inv") {
    for (std::int64_t H : {1, 4, 40}) {
        const std::int64_t K = 2 * H + 1;
        const hb::FourierMatrices m = hb::hb_fourier_matrices(H, K);
        std::mt19937 rng(static_cast<unsigned>(H));
        std::uniform_real_distribution<double> u(-2.0, 3.0);
        std::vector<double> g(static_cast<std::size_t>(K));
        for (double& v : g) {
            v = u(rng);
        }
        std::vector<double> gram(static_cast<std::size_t>(K * K), 0.0);
        std::vector<double> scratch(static_cast<std::size_t>(K * hb::kHbBlockChunk));
        for (std::int64_t c0 = 0; c0 < K; c0 += hb::kHbBlockChunk) {
            hb::hb_gram_columns(m, g.data(), c0, std::min(K, c0 + hb::kHbBlockChunk), scratch.data(), gram.data());
        }
        double worst = 0.0;
        double scale = 0.0;
        for (std::int64_t a = 0; a < K; ++a) {
            for (std::int64_t b = 0; b < K; ++b) {
                double ref = 0.0;
                for (std::int64_t k = 0; k < K; ++k) {
                    ref += m.gamma[static_cast<std::size_t>(a * K + k)] * g[static_cast<std::size_t>(k)] *
                           m.gamma_inv[static_cast<std::size_t>(k * K + b)];
                }
                const double upper = a <= b ? gram[static_cast<std::size_t>(a + b * K)]
                                            : gram[static_cast<std::size_t>(b + a * K)];
                const double value = (a == 0 ? 1.0 / static_cast<double>(K) : 2.0 / static_cast<double>(K)) * upper;
                worst = std::fmax(worst, std::fabs(value - ref));
                scale = std::fmax(scale, std::fabs(ref));
            }
        }
        CHECK(worst <= 1e-13 * scale);
    }
}

// ---------------------------------------------------------------------------
// BtfSolver
// ---------------------------------------------------------------------------

TEST_CASE("BtfSolver solves block triangular systems with dense, KLU and 1 x 1 blocks") {
    SUBCASE("dense blocks") {
        const Eigen::MatrixXd a = structured_matrix({1, 3, 40, 1, 7}, false, 11);
        const Csc m = to_csc(a);
        std::vector<double> b(static_cast<std::size_t>(m.n));
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = std::sin(static_cast<double>(i) + 1.0);
        }
        hb::BtfSolver s;
        const std::vector<double> x = btf_solution(m, b, 1, &s);
        const Eigen::VectorXd ref = a.partialPivLu().solve(Eigen::Map<const Eigen::VectorXd>(b.data(), m.n));
        CHECK(rel_error(x, ref) < 1e-12);
        CHECK(s.blocks() == 5);
        CHECK(s.singleton_blocks() == 2);
        CHECK(s.dense_blocks() == 3);
        CHECK(s.klu_blocks() == 0);
        CHECK(s.largest_block() == 40);
        CHECK(s.structural_rank() == m.n);
    }
    SUBCASE("sparse blocks go to KLU, refactor reuses the analysis") {
        const Eigen::MatrixXd a = structured_matrix({120, 2, 90}, true, 5);
        Csc m = to_csc(a);
        std::vector<double> b(static_cast<std::size_t>(m.n), 1.0);
        hb::BtfSolver s;
        std::vector<double> x = btf_solution(m, b, 2, &s);
        CHECK(rel_error(x, a.partialPivLu().solve(Eigen::VectorXd::Ones(m.n))) < 1e-12);
        CHECK(s.klu_blocks() == 2);
        CHECK(s.dense_blocks() == 1);
        // New values on the same pattern.
        Eigen::MatrixXd a2 = a;
        for (double& v : m.values) {
            v *= 1.5;
        }
        a2 *= 1.5;
        s.factor(m.values.data(), 2);
        x = b;
        s.solve(x.data());
        CHECK(rel_error(x, a2.partialPivLu().solve(Eigen::VectorXd::Ones(m.n))) < 1e-12);
    }
    SUBCASE("factorization is identical for any thread count") {
        const Eigen::MatrixXd a = structured_matrix({150, 150, 140, 3, 160}, false, 7);
        const Csc m = to_csc(a);
        std::vector<double> b(static_cast<std::size_t>(m.n));
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = std::cos(0.3 * static_cast<double>(i));
        }
        hb::BtfSolver s4;
        const std::vector<double> x1 = btf_solution(m, b, 1);
        const std::vector<double> x4 = btf_solution(m, b, 4, &s4);
        CHECK(bitwise_equal(x1, x4));
        CHECK(s4.threads_used() > 1);
    }
    SUBCASE("empty system") {
        hb::BtfSolver s;
        const std::int32_t col_ptr[1] = {0};
        s.analyze(0, col_ptr, nullptr);
        s.factor(nullptr, 1);
        s.solve(nullptr);
        CHECK(s.blocks() == 0);
    }
}

TEST_CASE("BtfSolver reports singular matrices") {
    auto code_of = [](const Eigen::MatrixXd& a) {
        const Csc m = to_csc(a);
        hb::BtfSolver s;
        s.analyze(m.n, m.col_ptr.data(), m.row_idx.data());
        try {
            s.factor(m.values.data(), 1);
        } catch (const mna::Error& e) {
            return e.code();
        }
        return mna::ErrorCode::Unknown;
    };
    Eigen::MatrixXd structural(3, 3);
    structural << 1, 2, 0, 3, 4, 0, 5, 6, 0;  // empty column
    CHECK(code_of(structural) == mna::ErrorCode::JacobianSingular);
    Eigen::MatrixXd dense(3, 3);
    dense << 1, 2, 3, 1, 2, 3, 0, 1, 1;  // exact zero pivot after elimination
    CHECK(code_of(dense) == mna::ErrorCode::JacobianSingular);

    // An explicit zero on the diagonal of a 1 x 1 block.
    const std::int32_t col_ptr[3] = {0, 1, 2};
    const std::int32_t row_idx[2] = {0, 1};
    const double values[2] = {2.0, 0.0};
    hb::BtfSolver s;
    s.analyze(2, col_ptr, row_idx);
    CHECK_THROWS_AS(s.factor(values, 1), mna::Error);
    CHECK_FALSE(s.factored());
}

// ---------------------------------------------------------------------------
// hb_solve
// ---------------------------------------------------------------------------

namespace {

/// One node: conductance g and capacitance c to ground.
CircuitSpec rc_node(double g, double c) {
    CircuitSpec s;
    s.n = 1;
    s.g = {{0}, {0}, {g}};
    if (c != 0.0) {
        s.c = {{0}, {0}, {c}};
    }
    return s;
}

}  // namespace

TEST_CASE("linear RC harmonic balance equals the phasor solution") {
    const double g = 1e-3;
    const double cap = 1e-6;
    const double omega0 = 2.0 * kPi * 1e3;
    const double amp = 2e-3;
    const Circuit circuit(rc_node(g, cap));
    const double b_dc[1] = {1e-3};
    std::vector<hb::HbTimeSource> sources{{0, sine_program(amp, omega0)}};
    hb::HbOptions options;
    options.omega0 = omega0;
    options.harmonics = 3;
    const hb::HbResult r = hb::hb_solve(circuit, b_dc, sources, options);
    REQUIRE(r.x_bar.size() == 7);
    CHECK(r.stats.linear);
    CHECK(r.stats.linear_solves == 1);
    // x = a cos + b sin with G a + C w b = 0, G b - C w a = amp.
    const double cw = cap * omega0;
    const double den = g * g + cw * cw;
    CHECK(r.x_bar[0] == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(r.x_bar[1] == doctest::Approx(-amp * cw / den).epsilon(1e-12));
    CHECK(r.x_bar[2] == doctest::Approx(amp * g / den).epsilon(1e-12));
    for (int s = 3; s < 7; ++s) {
        CHECK(std::fabs(r.x_bar[static_cast<std::size_t>(s)]) < 1e-15);
    }

    SUBCASE("Krylov option keeps the answer and records the attempt") {
        hb::HbOptions k = options;
        k.use_krylov = true;
        k.krylov_rank = 4;
        const hb::HbResult rk = hb::hb_solve(circuit, b_dc, sources, k);
        REQUIRE(rk.stats.krylov.size() == 1);
        CHECK_FALSE(rk.stats.krylov[0].used_direct_fallback);
        CHECK(rk.stats.factorizations == 0);
        double worst = 0.0;
        for (std::size_t i = 0; i < 7; ++i) {
            worst = std::fmax(worst, std::fabs(rk.x_bar[i] - r.x_bar[i]));
        }
        CHECK(worst <= 1e-8);  // |x| ~ 1, Krylov tolerance 1e-9
    }
}

TEST_CASE("nonlinear harmonic balance satisfies KCL at the samples and is thread independent") {
    CircuitSpec s = rc_node(1e-3, 0.0);
    s.diodes = {{0}, {-1}, {1e-14}};
    const Circuit circuit(s);
    const double omega0 = 2.0 * kPi * 1e6;
    const double b_dc[1] = {1e-3};
    std::vector<hb::HbTimeSource> sources{{0, sine_program(2e-4, omega0)}};
    hb::HbOptions options;
    options.omega0 = omega0;
    options.harmonics = 150;  // K = 301: the Jacobian products are split across threads
    options.tol = 1e-10;
    const hb::HbResult r1 = hb::hb_solve(circuit, b_dc, sources, options);
    options.num_threads = 4;
    const hb::HbResult r4 = hb::hb_solve(circuit, b_dc, sources, options);
    CHECK(bitwise_equal(r1.x_bar, r4.x_bar));
    CHECK(r4.stats.threads_used > 1);
    CHECK_FALSE(r1.stats.linear);
    CHECK(r1.stats.device_slots == 1);
    CHECK(r1.stats.dense_blocks >= 1);
    CHECK(r1.stats.residual_norm < 1e-10);
    // With no capacitance HB is collocation: KCL holds at every sample.
    const std::int64_t K = 301;
    const hb::FourierMatrices m = hb::hb_fourier_matrices(150, K);
    const std::vector<double> t = hb::hb_time_points(omega0, K);
    double worst = 0.0;
    for (std::int64_t k = 0; k < K; ++k) {
        double v = 0.0;
        for (std::int64_t q = 0; q < K; ++q) {
            v += m.gamma_inv[static_cast<std::size_t>(k * K + q)] * r1.x_bar[static_cast<std::size_t>(q)];
        }
        const double kcl = 1e-3 * v + 1e-14 * (std::exp(v / 0.025) - 1.0) - (1e-3 + 2e-4 * std::sin(omega0 * t[k]));
        worst = std::fmax(worst, std::fabs(kcl));
    }
    CHECK(worst < 1e-9);
}

TEST_CASE("hb_solve errors keep the reference texts") {
    const Circuit circuit(rc_node(1e-3, 0.0));
    const double b_dc[1] = {1e-3};
    hb::HbOptions options;
    options.omega0 = 2.0 * kPi * 1e3;
    options.harmonics = 0;
    CHECK(hb_error_message([&] { hb::hb_solve(circuit, b_dc, {}, options); }) ==
          "harmonics must be a positive integer, got 0.");

    options.harmonics = 50;
    options.max_nnz = 1000;
    const std::string too_large = hb_error_message([&] { hb::hb_solve(circuit, b_dc, {}, options); });
    CHECK(too_large.rfind("HB system too large (N=101, nnz=", 0) == 0);
    CHECK(too_large.find("); reduce harmonics") != std::string::npos);

    // A capacitor-only node has no DC path: Y_bar is singular.
    const Circuit floating(rc_node(0.0, 1e-6));
    options.max_nnz = hb::kHbDefaultMaxNnz;
    CHECK(hb_error_message([&] { hb::hb_solve(floating, b_dc, {}, options); }) == "Linear harmonic solve failed.");

    CircuitSpec s = rc_node(1e-3, 0.0);
    s.diodes = {{0}, {-1}, {1e-14}};
    const Circuit diode(s);
    options.max_iter = 0;
    CHECK(hb_error_message([&] { hb::hb_solve(diode, b_dc, {}, options); }) ==
          "Harmonic balance solver did not converge within limit.");

    std::vector<hb::HbTimeSource> bad{{3, sine_program(1.0, 1.0)}};
    CHECK_THROWS_AS(hb::hb_solve(diode, b_dc, bad, options), mna::Error);
}
