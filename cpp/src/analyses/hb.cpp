#include "mna/analyses/hb.hpp"

#include <btf.h>

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "mna/assembly.hpp"
#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/linear/linear_solver.hpp"
#include "mna/linear/sparse.hpp"
#include "mna/threads.hpp"

// Every product and sum is rounded on its own (the Fourier matrices and the
// reconstruction reproduce NumPy bit for bit); see mna/devices/fp_contract.hpp.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mna::analyses {

namespace {

constexpr double kPi = 3.141592653589793;  // numpy.pi

constexpr const char* kNotConverged = "Harmonic balance solver did not converge within limit.";
constexpr const char* kSingularJacobian = "Singular HB Jacobian.";
constexpr const char* kLinearFailed = "Linear harmonic solve failed.";

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void hb_error(const std::string& message) { throw Error(ErrorCode::HbError, message); }

[[noreturn]] void bad_input(const std::string& message) { throw Error(ErrorCode::NetlistFatal, message); }

/// The reference's `2 * np.pi * h * k / K`, evaluated left to right.
inline double fourier_angle(std::int64_t h, std::int64_t k, std::int64_t points) {
    return 2.0 * kPi * static_cast<double>(h) * static_cast<double>(k) / static_cast<double>(points);
}

inline std::size_t idx(std::int64_t v) { return static_cast<std::size_t>(v); }

constexpr std::int64_t kInt64Max = std::numeric_limits<std::int64_t>::max();

/// Smallest amount of work (multiply-adds) a parallel loop is split for (a few ms of
/// work: every parallel region starts its threads afresh).
constexpr std::int64_t kHbParallelMinWork = std::int64_t{1} << 23;

std::int64_t sat_mul(std::int64_t a, std::int64_t b) {
    if (a <= 0 || b <= 0) {
        return 0;
    }
    return a > kInt64Max / b ? kInt64Max : a * b;
}

std::int64_t sat_add(std::int64_t a, std::int64_t b) { return a > kInt64Max - b ? kInt64Max : a + b; }

}  // namespace

// ---------------------------------------------------------------------------
// Fourier helpers
// ---------------------------------------------------------------------------

FourierMatrices hb_fourier_matrices(std::int64_t harmonics, std::int64_t num_points) {
    if (harmonics < 0) {
        throw std::invalid_argument("hb_fourier_matrices: n_harmonics must be non-negative");
    }
    if (num_points < 0) {
        throw std::invalid_argument("hb_fourier_matrices: the number of time points must be non-negative");
    }
    if (num_points == 0) {
        throw std::domain_error("hb_fourier_matrices: float division by zero (no time points)");
    }
    const std::int64_t S = 2 * harmonics + 1;
    const std::int64_t K = num_points;
    FourierMatrices m;
    m.harmonics = harmonics;
    m.rows = S;
    m.points = K;
    m.gamma.assign(idx(S) * idx(K), 0.0);
    m.gamma_inv.assign(idx(K) * idx(S), 0.0);

    for (std::int64_t k = 0; k < K; ++k) {
        double* row = m.gamma_inv.data() + idx(k * S);
        row[0] = 1.0;
        for (std::int64_t h = 1; h <= harmonics; ++h) {
            const double angle = fourier_angle(h, k, K);
            row[2 * h - 1] = std::cos(angle);
            row[2 * h] = std::sin(angle);
        }
    }
    // Gamma[2h-1, k] = (2.0 / K) * np.cos(angle) with the same angle, so the
    // cos / sin values are taken from Gamma_inv instead of being recomputed.
    const double dc = 1.0 / static_cast<double>(K);
    const double scale = 2.0 / static_cast<double>(K);
    for (std::int64_t k = 0; k < K; ++k) {
        m.gamma[idx(k)] = dc;
    }
    for (std::int64_t h = 1; h <= harmonics; ++h) {
        double* cos_row = m.gamma.data() + idx((2 * h - 1) * K);
        double* sin_row = m.gamma.data() + idx(2 * h * K);
        for (std::int64_t k = 0; k < K; ++k) {
            const double* inv_row = m.gamma_inv.data() + idx(k * S);
            cos_row[k] = scale * inv_row[2 * h - 1];
            sin_row[k] = scale * inv_row[2 * h];
        }
    }
    return m;
}

std::vector<double> hb_time_points(double omega0, std::int64_t num_points) {
    if (num_points < 0) {
        throw std::invalid_argument("hb_time_points: the number of points must be non-negative");
    }
    std::vector<double> t(idx(num_points), 0.0);
    if (num_points == 0) {
        return t;
    }
    // numpy.linspace(0, period, K, endpoint=False): step = delta / K,
    // y = arange(K) * step (or arange(K) / K * delta when step == 0), y += start.
    const double period = 2.0 * kPi / omega0;
    const double div = static_cast<double>(num_points);
    const double step = period / div;
    for (std::int64_t k = 0; k < num_points; ++k) {
        const double y = step == 0.0 ? static_cast<double>(k) / div * period : static_cast<double>(k) * step;
        t[idx(k)] = y + 0.0;
    }
    return t;
}

void hb_reconstruct(const double* x_bar, std::int64_t n, std::int64_t harmonics, double omega0, const double* t,
                    std::int64_t num_t, int num_threads, double* out) {
    if (n < 0 || harmonics < 0 || num_t < 0) {
        throw std::invalid_argument("hb_reconstruct: negative size");
    }
    if (n == 0 || num_t == 0) {
        return;
    }
    const std::int64_t K = 2 * harmonics + 1;
    parallel_for(num_t, num_threads, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> values(idx(n));
        for (std::int64_t j = begin; j < end; ++j) {
            for (std::int64_t i = 0; i < n; ++i) {
                values[idx(i)] = x_bar[i * K];
            }
            for (std::int64_t h = 1; h <= harmonics; ++h) {
                // h * omega_0 * t_array: (h * omega_0) first, then times t.
                const double arg = static_cast<double>(h) * omega0 * t[j];
                const double c = std::cos(arg);
                const double s = std::sin(arg);
                for (std::int64_t i = 0; i < n; ++i) {
                    const double a_h = x_bar[i * K + 2 * h - 1];
                    const double b_h = x_bar[i * K + 2 * h];
                    values[idx(i)] += a_h * c + b_h * s;
                }
            }
            for (std::int64_t i = 0; i < n; ++i) {
                out[i * num_t + j] = values[idx(i)];
            }
        }
    });
}

void hb_gram_columns(const FourierMatrices& fm, const double* g, std::int64_t col_begin, std::int64_t col_end,
                     double* scratch, double* gram) {
    using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const std::int64_t S = fm.rows;
    const std::int64_t K = fm.points;
    const std::int64_t width = col_end - col_begin;
    if (width <= 0) {
        return;
    }
    // weighted[:, col_begin:col_end] = g_waveform[:, None] * Gamma_inv[:, col_begin:col_end]
    for (std::int64_t c = 0; c < width; ++c) {
        double* column = scratch + c * K;
        const double* source = fm.gamma_inv.data() + idx(col_begin + c);
        for (std::int64_t k = 0; k < K; ++k) {
            column[k] = g[k] * source[k * S];
        }
    }
    const auto eS = static_cast<Eigen::Index>(S);
    const auto eK = static_cast<Eigen::Index>(K);
    const Eigen::Map<const RowMajor> gamma_inv(fm.gamma_inv.data(), eK, eS);
    const Eigen::Map<const Eigen::MatrixXd> weighted(scratch, eK, static_cast<Eigen::Index>(width));
    Eigen::Map<Eigen::MatrixXd> m(gram, eS, eS);
    // Rows [0, col_end): the upper triangle of these columns (plus the lower part of the chunk's diagonal block).
    const auto rows = static_cast<Eigen::Index>(col_end);
    m.block(0, static_cast<Eigen::Index>(col_begin), rows, static_cast<Eigen::Index>(width)).noalias() =
        gamma_inv.leftCols(rows).transpose() * weighted;
}

// ---------------------------------------------------------------------------
// BtfSolver
// ---------------------------------------------------------------------------

struct BtfSolver::Impl {
    enum class Kind : std::uint8_t { Singleton, Dense, Klu };
    using DenseLu = Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>>;

    struct Block {
        std::int32_t begin = 0;
        std::int32_t size = 0;
        Kind kind = Kind::Singleton;
        std::int64_t value_begin = 0;  // first value in diag_values
        std::int64_t nnz = 0;
        std::vector<std::int32_t> col_ptr;  // local CSC (Dense and Klu blocks)
        std::vector<std::int32_t> row_idx;
        double pivot = 0.0;  // Singleton
        Eigen::MatrixXd dense;  // Dense: factored in place by `lu`
        std::optional<DenseLu> lu;
        std::unique_ptr<linear::KluSolver> klu;
        bool factored = false;
    };

    std::int32_t n = 0;
    std::int64_t nnz = 0;
    std::int32_t structural_rank = 0;
    bool analyzed = false;
    bool factored = false;
    int threads_used = 1;
    std::vector<std::int32_t> P;  // row i = P[k] of A is row k of P A Q
    std::vector<std::int32_t> Q;  // column j = Q[k] of A is column k of P A Q
    std::vector<std::int64_t> dest;  // per slot: >= 0 index into diag_values, < 0: -(off index) - 1
    std::vector<double> diag_values;
    std::vector<std::int32_t> off_ptr;  // per permuted column, size n + 1
    std::vector<std::int32_t> off_row;  // permuted row of each off-diagonal entry
    std::vector<double> off_values;
    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<double> work;
    // Rough factorization cost of all blocks and of the largest one (dense: size^3,
    // KLU: nnz^2 / size, 1 x 1: 1); factor() goes parallel only when the blocks
    // other than the largest are worth it.
    double total_cost = 0.0;
    double max_cost = 0.0;

    void factor_block(Block& b) const {
        const double* v = diag_values.data() + b.value_begin;
        b.factored = false;
        switch (b.kind) {
            case Kind::Singleton: {
                const double pivot = b.nnz > 0 ? v[0] : 0.0;
                if (pivot == 0.0) {
                    throw Error(ErrorCode::JacobianSingular, "BTF solver: zero pivot in a 1 x 1 block");
                }
                b.pivot = pivot;
                break;
            }
            case Kind::Dense: {
                b.lu.reset();
                b.dense.setZero();
                for (std::int32_t c = 0; c < b.size; ++c) {
                    for (std::int32_t e = b.col_ptr[idx(c)]; e < b.col_ptr[idx(c) + 1]; ++e) {
                        b.dense(b.row_idx[idx(e)], c) = v[e];
                    }
                }
                b.lu.emplace(b.dense);
                const auto diag = b.lu->matrixLU().diagonal();
                for (Eigen::Index k = 0; k < diag.size(); ++k) {
                    if (diag[k] == 0.0) {
                        throw Error(ErrorCode::JacobianSingular,
                                    "BTF solver: zero pivot in a dense block of size " + std::to_string(b.size));
                    }
                }
                break;
            }
            case Kind::Klu:
                if (b.klu->factored()) {
                    b.klu->refactor(v, idx(b.nnz));
                } else {
                    b.klu->factor(v, idx(b.nnz));
                }
                break;
        }
        b.factored = true;
    }

    std::int32_t count(Kind kind) const noexcept {
        std::int32_t total = 0;
        for (const auto& b : blocks) {
            total += b->kind == kind ? 1 : 0;
        }
        return total;
    }

    void solve_block(Block& b, double* y) const {
        switch (b.kind) {
            case Kind::Singleton:
                y[0] = y[0] / b.pivot;
                break;
            case Kind::Dense: {
                Eigen::Map<Eigen::VectorXd> v(y, b.size);
                const Eigen::VectorXd x = b.lu->solve(v);
                v = x;
                break;
            }
            case Kind::Klu:
                b.klu->solve(y, idx(b.size), 1);
                break;
        }
    }
};

BtfSolver::BtfSolver() : impl_(std::make_unique<Impl>()) {}
BtfSolver::~BtfSolver() = default;
BtfSolver::BtfSolver(BtfSolver&&) noexcept = default;
BtfSolver& BtfSolver::operator=(BtfSolver&&) noexcept = default;

void BtfSolver::analyze(std::int32_t n, const std::int32_t* col_ptr, const std::int32_t* row_idx) {
    if (n < 0) {
        bad_input("BTF solver: negative dimension " + std::to_string(n));
    }
    if (col_ptr == nullptr) {
        bad_input("BTF solver: null column pointer array");
    }
    const linear::CscPattern pattern =
        linear::CscPattern::from_arrays(n, col_ptr, idx(n) + 1, row_idx, col_ptr[n] < 0 ? 0 : idx(col_ptr[n]));

    auto impl = std::make_unique<Impl>();
    impl->n = n;
    impl->nnz = pattern.nnz();
    impl->dest.assign(idx(impl->nnz), 0);
    impl->off_ptr.assign(idx(n) + 1, 0);
    impl->work.assign(idx(n), 0.0);
    if (n == 0) {
        impl->analyzed = true;
        impl_ = std::move(impl);
        return;
    }

    impl->P.assign(idx(n), 0);
    impl->Q.assign(idx(n), 0);
    std::vector<std::int32_t> R(idx(n) + 1, 0);
    std::vector<std::int32_t> btf_work(5 * idx(n), 0);
    std::int32_t dummy_row = 0;
    double work = 0.0;
    std::int32_t nmatch = 0;
    // BTF never writes through Ap / Ai; its C API just lacks const.
    std::int32_t* Ap = const_cast<std::int32_t*>(pattern.col_ptr.data());
    std::int32_t* Ai = pattern.row_idx.empty() ? &dummy_row : const_cast<std::int32_t*>(pattern.row_idx.data());
    const std::int32_t nblocks =
        btf_order(n, Ap, Ai, 0.0, &work, impl->P.data(), impl->Q.data(), R.data(), &nmatch, btf_work.data());
    impl->structural_rank = nmatch;
    for (std::int32_t& q : impl->Q) {
        q = BTF_UNFLIP(q);
    }

    std::vector<std::int32_t> pinv(idx(n), 0);
    for (std::int32_t k = 0; k < n; ++k) {
        pinv[idx(impl->P[idx(k)])] = k;
    }
    std::vector<std::int32_t> block_of(idx(n), 0);
    for (std::int32_t b = 0; b < nblocks; ++b) {
        for (std::int32_t k = R[idx(b)]; k < R[idx(b) + 1]; ++k) {
            block_of[idx(k)] = b;
        }
    }

    std::int64_t diag_count = 0;
    std::int64_t off_count = 0;
    std::vector<std::pair<std::int32_t, std::int32_t>> local;  // (local row, slot)
    impl->blocks.reserve(idx(nblocks));
    for (std::int32_t b = 0; b < nblocks; ++b) {
        auto block = std::make_unique<Impl::Block>();
        block->begin = R[idx(b)];
        block->size = R[idx(b) + 1] - R[idx(b)];
        block->value_begin = diag_count;
        block->col_ptr.assign(idx(block->size) + 1, 0);
        for (std::int32_t lc = 0; lc < block->size; ++lc) {
            const std::int32_t kc = block->begin + lc;
            const std::int32_t j = impl->Q[idx(kc)];
            local.clear();
            for (std::int32_t p = pattern.col_ptr[idx(j)]; p < pattern.col_ptr[idx(j) + 1]; ++p) {
                const std::int32_t kr = pinv[idx(pattern.row_idx[idx(p)])];
                const std::int32_t br = block_of[idx(kr)];
                if (br == b) {
                    local.emplace_back(kr - block->begin, p);
                } else if (br < b) {
                    impl->dest[idx(p)] = -(off_count + 1);
                    impl->off_row.push_back(kr);
                    ++off_count;
                } else {
                    throw Error(ErrorCode::Unknown, "BTF solver: permuted matrix is not block upper triangular");
                }
            }
            impl->off_ptr[idx(kc) + 1] = static_cast<std::int32_t>(off_count);
            std::sort(local.begin(), local.end());
            for (const auto& entry : local) {
                impl->dest[idx(entry.second)] = diag_count;
                block->row_idx.push_back(entry.first);
                ++diag_count;
            }
            block->col_ptr[idx(lc) + 1] = static_cast<std::int32_t>(diag_count - block->value_begin);
        }
        block->nnz = diag_count - block->value_begin;
        const double size = static_cast<double>(block->size);
        if (block->size == 1) {
            block->kind = Impl::Kind::Singleton;
            block->col_ptr.clear();
            block->row_idx.clear();
        } else if (block->size <= kBtfDenseMaxSmall ||
                   static_cast<double>(block->nnz) >= kBtfDenseFraction * size * size) {
            block->kind = Impl::Kind::Dense;
            block->dense.resize(block->size, block->size);
        } else {
            block->kind = Impl::Kind::Klu;
            block->klu = std::make_unique<linear::KluSolver>();
            block->klu->analyze(block->size, block->col_ptr, block->row_idx);
        }
        const double nnz = static_cast<double>(block->nnz);
        const double cost = block->kind == Impl::Kind::Singleton ? 1.0
                            : block->kind == Impl::Kind::Dense ? size * size * size
                                                                : (std::max)(size, nnz * nnz / size);
        impl->total_cost += cost;
        impl->max_cost = (std::max)(impl->max_cost, cost);
        impl->blocks.push_back(std::move(block));
    }
    impl->diag_values.assign(idx(diag_count), 0.0);
    impl->off_values.assign(idx(off_count), 0.0);
    impl->analyzed = true;
    impl_ = std::move(impl);
}

void BtfSolver::factor(const double* values, int num_threads) {
    Impl& s = *impl_;
    if (!s.analyzed) {
        bad_input("BTF solver: factor() called before analyze()");
    }
    s.factored = false;
    if (s.structural_rank < s.n) {
        throw Error(ErrorCode::JacobianSingular, "BTF solver: matrix is structurally singular (structural rank " +
                                                     std::to_string(s.structural_rank) + " of " +
                                                     std::to_string(s.n) + ")");
    }
    if (s.nnz > 0 && values == nullptr) {
        bad_input("BTF solver: null value array");
    }
    for (std::int64_t p = 0; p < s.nnz; ++p) {
        const std::int64_t d = s.dest[idx(p)];
        if (d >= 0) {
            s.diag_values[idx(d)] = values[p];
        } else {
            s.off_values[idx(-d - 1)] = values[p];
        }
    }
    const auto count = static_cast<std::int64_t>(s.blocks.size());
    const int threads = s.total_cost - s.max_cost >= kHbParallelMinWork ? num_threads : 1;
    s.threads_used = parallel_for(count, threads, [&s](std::int64_t begin, std::int64_t end) {
        for (std::int64_t b = begin; b < end; ++b) {
            s.factor_block(*s.blocks[idx(b)]);
        }
    });
    s.factored = true;
}

void BtfSolver::solve(double* rhs) {
    Impl& s = *impl_;
    if (!s.factored) {
        bad_input("BTF solver: solve() called before a successful factor()");
    }
    if (s.n == 0) {
        return;
    }
    double* y = s.work.data();
    for (std::int32_t k = 0; k < s.n; ++k) {
        y[k] = rhs[s.P[idx(k)]];
    }
    for (auto it = s.blocks.rbegin(); it != s.blocks.rend(); ++it) {
        Impl::Block& b = **it;
        s.solve_block(b, y + b.begin);
        for (std::int32_t kc = b.begin; kc < b.begin + b.size; ++kc) {
            const double xk = y[kc];
            for (std::int32_t e = s.off_ptr[idx(kc)]; e < s.off_ptr[idx(kc) + 1]; ++e) {
                y[s.off_row[idx(e)]] -= s.off_values[idx(e)] * xk;
            }
        }
    }
    for (std::int32_t k = 0; k < s.n; ++k) {
        rhs[s.Q[idx(k)]] = y[k];
    }
}

bool BtfSolver::analyzed() const noexcept { return impl_->analyzed; }
bool BtfSolver::factored() const noexcept { return impl_->factored; }
std::int32_t BtfSolver::size() const noexcept { return impl_->n; }
std::int32_t BtfSolver::structural_rank() const noexcept { return impl_->structural_rank; }
std::int32_t BtfSolver::blocks() const noexcept { return static_cast<std::int32_t>(impl_->blocks.size()); }

std::int32_t BtfSolver::dense_blocks() const noexcept { return impl_->count(Impl::Kind::Dense); }
std::int32_t BtfSolver::klu_blocks() const noexcept { return impl_->count(Impl::Kind::Klu); }
std::int32_t BtfSolver::singleton_blocks() const noexcept { return impl_->count(Impl::Kind::Singleton); }
std::int32_t BtfSolver::largest_block() const noexcept {
    std::int32_t largest = 0;
    for (const auto& b : impl_->blocks) {
        largest = (std::max)(largest, b->size);
    }
    return largest;
}

std::int64_t BtfSolver::nnz_factors() const noexcept {
    std::int64_t total = static_cast<std::int64_t>(impl_->off_values.size());
    for (const auto& b : impl_->blocks) {
        switch (b->kind) {
            case Impl::Kind::Singleton: total += 1; break;
            case Impl::Kind::Dense: total += static_cast<std::int64_t>(b->size) * b->size; break;
            case Impl::Kind::Klu: total += b->klu->nnz_lu(); break;
        }
    }
    return total;
}

std::int64_t BtfSolver::memory_bytes() const noexcept {
    const Impl& s = *impl_;
    std::int64_t bytes = static_cast<std::int64_t>(s.dest.size() * sizeof(std::int64_t) +
                                                   s.diag_values.size() * sizeof(double) +
                                                   s.off_values.size() * (sizeof(double) + sizeof(std::int32_t)) +
                                                   (s.P.size() + s.Q.size() + s.off_ptr.size()) * sizeof(std::int32_t));
    for (const auto& b : s.blocks) {
        bytes += static_cast<std::int64_t>((b->col_ptr.size() + b->row_idx.size()) * sizeof(std::int32_t));
        if (b->kind == Impl::Kind::Dense) {
            bytes += static_cast<std::int64_t>(b->size) * b->size * static_cast<std::int64_t>(sizeof(double));
        } else if (b->kind == Impl::Kind::Klu) {
            // L and U values and row indices.
            bytes += b->klu->nnz_lu() * static_cast<std::int64_t>(sizeof(double) + sizeof(std::int32_t));
        }
    }
    return bytes;
}

int BtfSolver::threads_used() const noexcept { return impl_->threads_used; }

// ---------------------------------------------------------------------------
// hb_solve
// ---------------------------------------------------------------------------

namespace {

/// The HB system of one run: union CSC pattern of Y_bar and the dense device
/// blocks, the constant Y_bar values in that pattern, and where each device
/// block column starts.
struct HbSystem {
    std::int64_t n = 0;
    std::int64_t H = 0;
    std::int64_t K = 0;
    std::int64_t N = 0;
    linear::CscPattern pattern;
    std::vector<double> y_values;
    std::vector<std::int32_t> device_slots;  // circuit pattern slot of each device block
    std::vector<std::int64_t> block_col_start;  // device d, block column b -> value index of row 0
    std::vector<std::int32_t> read_nodes;  // unknowns the devices read
    std::vector<std::int32_t> f_rows;      // rows with device current contributions
};

/// Partner index of harmonic column b > 0 (cos h <-> sin h).
inline std::int64_t partner(std::int64_t b) { return (b % 2 == 1) ? b + 1 : b - 1; }

/// Omega[a, b] for a == partner(b): +h omega0 at (cos h, sin h), -h omega0 at (sin h, cos h).
inline double omega_entry(std::int64_t a, double omega0) {
    const std::int64_t h = (a + 1) / 2;
    const double w = static_cast<double>(h) * omega0;  // w_h = h * omega_0
    return (a % 2 == 1) ? w : -w;
}

}  // namespace

HbResult hb_solve(const circuit::Circuit& circuit, const double* b_dc, const std::vector<HbTimeSource>& sources,
                  const HbOptions& options) {
    const auto t_setup = Clock::now();
    if (options.harmonics < 1) {
        hb_error("harmonics must be a positive integer, got " + std::to_string(options.harmonics) + ".");
    }
    if (!(options.omega0 > 0.0) || !std::isfinite(options.omega0)) {
        hb_error("the base angular frequency must be positive and finite.");
    }
    const int threads = options.num_threads < 1 ? 1 : options.num_threads;
    const std::int64_t n = circuit.n();
    const std::int64_t H = options.harmonics;
    const std::int64_t K = H > kHbMaxDimension ? kInt64Max : 2 * H + 1;
    const std::int64_t N = sat_mul(n, K);
    const bool nonlinear = options.nonlinear && circuit.has_nonlinear();

    for (const HbTimeSource& source : sources) {
        if (source.row < 0 || source.row >= n) {
            bad_input("hb_solve: time source row " + std::to_string(source.row) + " outside [0, " +
                      std::to_string(n) + ")");
        }
        if (!source.program) {
            bad_input("hb_solve: null time source program");
        }
    }
    if (n > 0 && b_dc == nullptr) {
        bad_input("hb_solve: null b_dc");
    }

    HbResult result;
    HbStats& stats = result.stats;
    stats.linear = !nonlinear;
    stats.harmonics = H;
    stats.samples = K;
    stats.dimension = N;
    if (n == 0) {
        stats.setup_seconds = seconds_since(t_setup);
        return result;
    }

    // ---- Memory policy (before any allocation proportional to K) ----------
    const std::int32_t nnz_c = circuit.nnz();
    const std::vector<std::int32_t>& cp = circuit.col_ptr();
    const std::vector<std::int32_t>& ri = circuit.row_idx();
    const std::vector<double> g_eff = circuit.g_effective(options.gmin);
    const std::vector<double>& c_vals = circuit.values(circuit::MatrixKind::C);
    std::vector<std::int32_t> device_index(idx(nnz_c), -1);
    std::int64_t device_slots = 0;
    if (nonlinear) {
        const auto& jp = circuit.assembly().j.ptr;
        for (std::int32_t s = 0; s < nnz_c; ++s) {
            if (jp[idx(s) + 1] > jp[idx(s)]) {
                device_index[idx(s)] = static_cast<std::int32_t>(device_slots++);
            }
        }
    }
    std::int64_t y_nnz = 0;
    std::int64_t assembled = 0;
    {
        const std::int64_t k_eff = (std::min)(K, kHbMaxDimension + 1);
        for (std::int32_t s = 0; s < nnz_c; ++s) {
            const std::int64_t ys = (g_eff[idx(s)] != 0.0 ? k_eff : 0) + (c_vals[idx(s)] != 0.0 ? k_eff - 1 : 0);
            y_nnz = sat_add(y_nnz, ys);
            assembled = sat_add(assembled, device_index[idx(s)] >= 0 ? sat_mul(k_eff, k_eff) : ys);
        }
    }
    const std::int64_t k2 = sat_mul(K, K);
    const std::int64_t stored = sat_add(sat_add(y_nnz, sat_mul(device_slots, k2)), sat_mul(2, k2));
    stats.y_nnz = y_nnz;
    stats.device_slots = device_slots;
    stats.stored_nnz = stored;
    if (N > kHbMaxDimension || stored > options.max_nnz ||
        assembled > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max() - 1)) {
        hb_error("HB system too large (N=" + std::to_string(N) + ", nnz=" + std::to_string(stored) +
                 "); reduce harmonics");
    }
    stats.matrix_nnz = assembled;

    // ---- Fourier matrices, samples, b_bar ---------------------------------
    const FourierMatrices fm = hb_fourier_matrices(H, K);
    const std::int64_t S = fm.rows;  // == K
    const std::vector<double> t_points = hb_time_points(options.omega0, K);

    std::vector<double> b_bar(idx(N), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        b_bar[idx(i * K)] = b_dc[i];
    }
    if (!sources.empty()) {
        // b_time rows (duplicate rows add up), then b_bar[i*K + s] += sum_k Gamma[s, k] b_time_i(t_k).
        std::vector<std::int32_t> rows;
        std::vector<double> b_time;
        std::vector<double> buffer(idx(K));
        for (const HbTimeSource& source : sources) {
            expr::evaluate_many(*source.program, t_points.data(), idx(K), buffer.data());
            auto it = std::find(rows.begin(), rows.end(), source.row);
            std::size_t slot = 0;
            if (it == rows.end()) {
                slot = rows.size();
                rows.push_back(source.row);
                b_time.resize(b_time.size() + idx(K), 0.0);
            } else {
                slot = static_cast<std::size_t>(it - rows.begin());
            }
            double* dst = b_time.data() + slot * idx(K);
            for (std::int64_t k = 0; k < K; ++k) {
                dst[k] += buffer[idx(k)];
            }
        }
        for (std::size_t q = 0; q < rows.size(); ++q) {
            const double* bt = b_time.data() + q * idx(K);
            double* dst = b_bar.data() + idx(rows[q]) * idx(K);
            for (std::int64_t s = 0; s < S; ++s) {
                const double* gamma_row = fm.gamma.data() + idx(s * K);
                double acc = 0.0;
                for (std::int64_t k = 0; k < K; ++k) {
                    acc += gamma_row[k] * bt[k];
                }
                dst[s] += acc;
            }
        }
    }

    // ---- Union pattern and constant Y_bar values ---------------------------
    HbSystem sys;
    sys.n = n;
    sys.H = H;
    sys.K = K;
    sys.N = N;
    sys.pattern.n = static_cast<std::int32_t>(N);
    sys.pattern.col_ptr.assign(idx(N) + 1, 0);
    sys.pattern.row_idx.resize(idx(assembled));
    sys.y_values.resize(idx(assembled));
    sys.device_slots.resize(idx(device_slots));
    sys.block_col_start.resize(idx(device_slots) * idx(K));
    {
        std::int64_t pos = 0;
        std::int32_t* rows_out = sys.pattern.row_idx.data();
        double* vals_out = sys.y_values.data();
        for (std::int64_t j = 0; j < n; ++j) {
            for (std::int64_t b = 0; b < K; ++b) {
                sys.pattern.col_ptr[idx(j * K + b)] = static_cast<std::int32_t>(pos);
                const std::int64_t pb = b == 0 ? -1 : partner(b);
                for (std::int32_t s = cp[idx(j)]; s < cp[idx(j) + 1]; ++s) {
                    const std::int64_t base = static_cast<std::int64_t>(ri[idx(s)]) * K;
                    const double g = g_eff[idx(s)];
                    const double c = c_vals[idx(s)];
                    const std::int32_t d = device_index[idx(s)];
                    if (d >= 0) {
                        sys.block_col_start[idx(d) * idx(K) + idx(b)] = pos;
                        for (std::int64_t a = 0; a < K; ++a) {
                            rows_out[pos] = static_cast<std::int32_t>(base + a);
                            vals_out[pos] = a == b ? g : (a == pb ? c * omega_entry(a, options.omega0) : 0.0);
                            ++pos;
                        }
                        continue;
                    }
                    const bool has_g = g != 0.0;
                    const bool has_c = c != 0.0 && pb >= 0;
                    // Rows b (G) and partner(b) (C Omega), in increasing order.
                    if (has_c && pb < b) {
                        rows_out[pos] = static_cast<std::int32_t>(base + pb);
                        vals_out[pos] = c * omega_entry(pb, options.omega0);
                        ++pos;
                    }
                    if (has_g) {
                        rows_out[pos] = static_cast<std::int32_t>(base + b);
                        vals_out[pos] = g;
                        ++pos;
                    }
                    if (has_c && pb > b) {
                        rows_out[pos] = static_cast<std::int32_t>(base + pb);
                        vals_out[pos] = c * omega_entry(pb, options.omega0);
                        ++pos;
                    }
                }
            }
        }
        sys.pattern.col_ptr[idx(N)] = static_cast<std::int32_t>(pos);
        if (pos != assembled) {
            throw Error(ErrorCode::Unknown, "hb_solve: pattern size mismatch");
        }
    }
    if (nonlinear) {
        std::vector<char> reads(idx(n), 0);
        for (std::int32_t col = 0; col < static_cast<std::int32_t>(n); ++col) {
            for (std::int32_t s = cp[idx(col)]; s < cp[idx(col) + 1]; ++s) {
                const std::int32_t d = device_index[idx(s)];
                if (d >= 0) {
                    sys.device_slots[idx(d)] = s;
                    reads[idx(col)] = 1;
                }
            }
        }
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(n); ++i) {
            if (reads[idx(i)] != 0) {
                sys.read_nodes.push_back(i);
            }
        }
        const auto& fp = circuit.assembly().f.ptr;
        for (std::int32_t r = 0; r < static_cast<std::int32_t>(n); ++r) {
            if (fp[idx(r) + 1] > fp[idx(r)]) {
                sys.f_rows.push_back(r);
            }
        }
    }

    BtfSolver direct;
    direct.analyze(sys.pattern.n, sys.pattern.col_ptr.data(), sys.pattern.row_idx.data());
    stats.btf_blocks = direct.blocks();
    stats.dense_blocks = direct.dense_blocks();
    stats.klu_blocks = direct.klu_blocks();
    stats.singleton_blocks = direct.singleton_blocks();
    stats.largest_block = direct.largest_block();

    const std::int64_t n_read = static_cast<std::int64_t>(sys.read_nodes.size());
    const std::int64_t n_frows = static_cast<std::int64_t>(sys.f_rows.size());
    // Fourier matrices, Y_bar and Newton values, N-vectors, samples, the Gram buffer and
    // per-thread scratch, plus the pattern indices; the factorization is added below.
    const std::int64_t per_thread = K * kHbBlockChunk + 2 * n + nnz_c;
    const std::int64_t base_bytes =
        static_cast<std::int64_t>(sizeof(double)) *
            (3 * K * K + 2 * assembled + 6 * N + (n_read + n_frows + device_slots) * K + threads * per_thread) +
        static_cast<std::int64_t>(sizeof(std::int32_t)) * (N + 1 + assembled);
    stats.memory_bytes = base_bytes + direct.memory_bytes();
    stats.setup_seconds = seconds_since(t_setup);

    int threads_used = 1;
    auto note_threads = [&threads_used](int used) { threads_used = (std::max)(threads_used, used); };
    // Loops below kHbParallelMinWork multiply-adds (or device evaluations x 64) run inline:
    // spawning threads costs more than they save. Results do not depend on the choice.
    auto threads_for = [threads](std::int64_t work) { return work >= kHbParallelMinWork ? threads : 1; };
    const std::int64_t device_work = 64 * circuit.device_counts().total();

    // ---- Linear solve: Krylov first when requested, then the direct solver --
    auto solve_system = [&](const std::vector<double>& values, std::vector<double>& rhs, const char* singular_text) {
        if (options.use_krylov) {
            const auto t0 = Clock::now();
            const linear::CsrMatrix<double> A =
                linear::CsrMatrix<double>::from_csc(sys.pattern, values.data(), values.size());
            const linear::MatrixKind kind = linear::classify_matrix(A, sys.pattern, values.data());
            const linear::KrylovRoute route = linear::krylov_route(kind, options.krylov_method);
            linear::KrylovSettings settings;
            settings.tol = options.krylov_tol;
            settings.max_iter = options.krylov_max_iter;
            settings.rank = options.krylov_rank;
            settings.num_threads = threads;
            std::vector<double> x(idx(N), 0.0);
            HbKrylovRecord record;
            record.info = linear::krylov_solve(A, rhs.data(), x.data(), route, settings);
            record.matrix_kind = kind;
            record.matrix_nnz = static_cast<std::int64_t>(
                std::count_if(values.begin(), values.end(), [](double v) { return v != 0.0; }));
            const double accept = (std::max)(options.krylov_tol * 10.0, 1e-8);
            const bool accepted = record.info.converged || record.info.relative_residual <= accept;
            record.used_direct_fallback = !accepted;
            note_threads(record.info.threads_used);
            stats.krylov.push_back(record);
            stats.solve_seconds += seconds_since(t0);
            ++stats.linear_solves;
            if (accepted) {
                rhs.swap(x);
                return;
            }
        } else {
            ++stats.linear_solves;
        }
        const auto t1 = Clock::now();
        try {
            direct.factor(values.data(), threads);
        } catch (const Error& e) {
            if (e.code() == ErrorCode::JacobianSingular) {
                hb_error(singular_text);
            }
            throw;
        }
        ++stats.factorizations;
        note_threads(direct.threads_used());
        const auto t2 = Clock::now();
        stats.factor_seconds += std::chrono::duration<double>(t2 - t1).count();
        direct.solve(rhs.data());
        stats.solve_seconds += seconds_since(t2);
    };

    auto finish = [&]() {
        stats.factor_nnz = direct.factored() ? direct.nnz_factors() : 0;
        stats.memory_bytes = base_bytes + direct.memory_bytes();
        stats.threads_used = threads_used;
    };

    if (!nonlinear) {
        std::vector<double> rhs = b_bar;
        solve_system(sys.y_values, rhs, kLinearFailed);
        result.x_bar = std::move(rhs);
        finish();
        return result;
    }

    // ---- Newton --------------------------------------------------------------
    std::vector<double> x(idx(N), 0.0);
    std::vector<double> residual(idx(N), 0.0);
    std::vector<double> a_values(idx(assembled), 0.0);
    std::vector<double> x_time(idx(n_read * K), 0.0);
    std::vector<double> f_time(idx(n_frows * K), 0.0);
    std::vector<double> g_time(idx(device_slots * K), 0.0);
    std::vector<double> f_bar(idx(n_frows * K), 0.0);
    const std::int64_t chunks = (K + kHbBlockChunk - 1) / kHbBlockChunk;
    std::vector<double> gram(device_slots > 0 ? idx(S) * idx(S) : 0, 0.0);
    std::vector<double> gram_scale(idx(S), 2.0 / static_cast<double>(K));
    gram_scale[0] = 1.0 / static_cast<double>(K);
    const double* gamma = fm.gamma.data();
    const double* gamma_inv = fm.gamma_inv.data();

    for (std::int64_t iteration = 0; iteration < options.max_iter; ++iteration) {
        const auto t_eval = Clock::now();
        // x(t_k) of the unknowns the devices read: x_reshaped @ Gamma_inv.T.
        note_threads(parallel_for(n_read * K, threads_for(n_read * K * S), [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t e = begin; e < end; ++e) {
                const std::int64_t q = e / K;
                const std::int64_t k = e % K;
                const double* xs = x.data() + idx(sys.read_nodes[idx(q)]) * idx(K);
                const double* row = gamma_inv + idx(k * S);
                double acc = 0.0;
                for (std::int64_t s = 0; s < S; ++s) {
                    acc += xs[s] * row[s];
                }
                x_time[idx(e)] = acc;
            }
        }));
        // Device currents and Jacobian values at every sample (one slot per sample).
        note_threads(parallel_for(K, threads_for(K * device_work), [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> xs(idx(n), 0.0);
            std::vector<double> fs(idx(n), 0.0);
            std::vector<double> js(idx(nnz_c) + 1, 0.0);
            for (std::int64_t k = begin; k < end; ++k) {
                for (std::int64_t q = 0; q < n_read; ++q) {
                    xs[idx(sys.read_nodes[idx(q)])] = x_time[idx(q * K + k)];
                }
                assembly::evaluate(circuit, xs.data(), 0.0, 1, fs.data(), js.data());
                for (std::int64_t q = 0; q < n_frows; ++q) {
                    f_time[idx(q * K + k)] = fs[idx(sys.f_rows[idx(q)])];
                }
                for (std::int64_t d = 0; d < device_slots; ++d) {
                    g_time[idx(d * K + k)] = js[idx(sys.device_slots[idx(d)])];
                }
            }
        }));
        // f_bar = i_nl_time @ Gamma.T on the device rows.
        note_threads(parallel_for(n_frows * K, threads_for(n_frows * K * K), [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t e = begin; e < end; ++e) {
                const std::int64_t q = e / K;
                const std::int64_t s = e % K;
                const double* ft = f_time.data() + idx(q * K);
                const double* row = gamma + idx(s * K);
                double acc = 0.0;
                for (std::int64_t k = 0; k < K; ++k) {
                    acc += ft[k] * row[k];
                }
                f_bar[idx(e)] = acc;
            }
        }));
        // residual = (Y_bar x_bar + f_bar) - b_bar
        std::fill(residual.begin(), residual.end(), 0.0);
        for (std::int64_t j = 0; j < n; ++j) {
            const double* xj = x.data() + idx(j * K);
            for (std::int32_t s = cp[idx(j)]; s < cp[idx(j) + 1]; ++s) {
                double* out = residual.data() + idx(ri[idx(s)]) * idx(K);
                const double g = g_eff[idx(s)];
                if (g != 0.0) {
                    for (std::int64_t a = 0; a < K; ++a) {
                        out[a] += g * xj[a];
                    }
                }
                const double c = c_vals[idx(s)];
                if (c != 0.0) {
                    for (std::int64_t h = 1; h <= H; ++h) {
                        const double w = static_cast<double>(h) * options.omega0;
                        out[2 * h - 1] += (c * w) * xj[2 * h];
                        out[2 * h] += (c * -w) * xj[2 * h - 1];
                    }
                }
            }
        }
        for (std::int64_t q = 0; q < n_frows; ++q) {
            double* out = residual.data() + idx(sys.f_rows[idx(q)]) * idx(K);
            const double* fb = f_bar.data() + idx(q * K);
            for (std::int64_t s = 0; s < K; ++s) {
                out[s] += fb[s];
            }
        }
        double sum_sq = 0.0;
        for (std::int64_t i = 0; i < N; ++i) {
            residual[idx(i)] -= b_bar[idx(i)];
            sum_sq += residual[idx(i)] * residual[idx(i)];
        }
        const double norm = std::sqrt(sum_sq);
        stats.evaluate_seconds += seconds_since(t_eval);
        if (!std::isfinite(norm)) {
            // The reference keeps iterating on non-finite values until max_iter.
            hb_error(kNotConverged);
        }
        if (norm < options.tol) {
            stats.newton_iterations = iteration;
            stats.residual_norm = norm;
            result.x_bar = std::move(x);
            finish();
            return result;
        }

        // Newton matrix: Y_bar + Gamma diag(g_rc(t)) Gamma_inv on the device slots.
        const auto t_jac = Clock::now();
        std::copy(sys.y_values.begin(), sys.y_values.end(), a_values.begin());
        // block = Gamma diag(g) Gamma_inv = D M with the symmetric M = Gamma_inv^T diag(g) Gamma_inv
        // (Gamma = D Gamma_inv^T, D = diag(1/K, 2/K, ..., 2/K)): the upper triangle of M in fixed
        // column chunks (the unit of parallel work, independent of the thread count), then
        // block[a, b] = D[a] * M[min(a, b), max(a, b)].
        for (std::int64_t d = 0; d < device_slots; ++d) {
            const double* g = g_time.data() + idx(d * K);
            note_threads(parallel_for(chunks, threads_for(K * K * S / 2), [&](std::int64_t begin, std::int64_t end) {
                std::vector<double> scratch(idx(K) * idx(kHbBlockChunk));
                for (std::int64_t item = begin; item < end; ++item) {
                    // Chunk c costs ~ c + 1 (rows [0, c1)); interleave cheap and costly chunks so the
                    // contiguous item ranges of parallel_for carry similar work.
                    const std::int64_t c = item % 2 == 0 ? item / 2 : chunks - 1 - item / 2;
                    const std::int64_t c0 = c * kHbBlockChunk;
                    const std::int64_t c1 = (std::min)(K, c0 + kHbBlockChunk);
                    hb_gram_columns(fm, g, c0, c1, scratch.data(), gram.data());
                }
            }));
            note_threads(parallel_for(K, threads_for(K * K * 8), [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t b = begin; b < end; ++b) {
                    const std::int64_t start = sys.block_col_start[idx(d * K + b)];
                    const double* upper = gram.data() + idx(b * S);  // M[0:b+1, b]
                    for (std::int64_t a = 0; a <= b; ++a) {
                        a_values[idx(start + a)] = sys.y_values[idx(start + a)] + gram_scale[idx(a)] * upper[a];
                    }
                    for (std::int64_t a = b + 1; a < K; ++a) {  // M[a, b] = M[b, a]
                        a_values[idx(start + a)] =
                            sys.y_values[idx(start + a)] + gram_scale[idx(a)] * gram[idx(a * S + b)];
                    }
                }
            }));
        }
        stats.jacobian_seconds += seconds_since(t_jac);

        // delta = solve(J, -residual); x_bar = x_bar + delta
        for (double& v : residual) {
            v = -v;
        }
        solve_system(a_values, residual, kSingularJacobian);
        for (std::int64_t i = 0; i < N; ++i) {
            x[idx(i)] = x[idx(i)] + residual[idx(i)];
        }
        stats.newton_iterations = iteration + 1;
    }
    finish();
    hb_error(kNotConverged);
}

}  // namespace mna::analyses
