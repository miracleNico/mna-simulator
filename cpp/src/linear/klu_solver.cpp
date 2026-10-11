#include "mna/linear/klu_solver.hpp"

#include <klu.h>

#include <new>
#include <string>
#include <utility>

#include "mna/error.hpp"

namespace mna::linear {
namespace {

[[noreturn]] void fatal(const std::string& message) { throw Error(ErrorCode::NetlistFatal, message); }

// Maps a failed KLU call to an exception.
[[noreturn]] void throw_status(const klu_common& common, const char* operation) {
    const std::string op = operation;
    switch (common.status) {
        case KLU_SINGULAR:
            throw Error(ErrorCode::JacobianSingular,
                        op + ": matrix is singular (zero pivot in column " + std::to_string(common.singular_col) +
                            ", structural rank " + std::to_string(common.structural_rank) + ")");
        case KLU_OUT_OF_MEMORY:
            throw std::bad_alloc();
        case KLU_TOO_LARGE:
            fatal(op + ": problem too large for 32-bit KLU indices");
        case KLU_INVALID:
            fatal(op + ": KLU rejected the input as invalid");
        default:
            throw Error(ErrorCode::Unknown, op + ": KLU failed with status " + std::to_string(common.status));
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// CscPattern
// ----------------------------------------------------------------------------

CscPattern CscPattern::from_arrays(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size,
                                   const std::int32_t* Ai, std::size_t ai_size) {
    if (n < 0) {
        fatal("CSC pattern: negative dimension " + std::to_string(n));
    }
    const auto un = static_cast<std::size_t>(n);
    if (ap_size != un + 1) {
        fatal("CSC pattern: column pointer array has " + std::to_string(ap_size) + " entries, expected n + 1 = " +
              std::to_string(un + 1));
    }
    if (Ap == nullptr) {
        fatal("CSC pattern: null column pointer array");
    }
    if (Ap[0] != 0) {
        fatal("CSC pattern: column pointers must start at 0");
    }
    for (std::size_t j = 0; j < un; ++j) {
        if (Ap[j + 1] < Ap[j]) {
            fatal("CSC pattern: column pointers decrease at column " + std::to_string(j));
        }
    }
    const auto nnz = static_cast<std::size_t>(Ap[un]);
    if (ai_size != nnz) {
        fatal("CSC pattern: row index array has " + std::to_string(ai_size) + " entries, column pointers say " +
              std::to_string(nnz));
    }
    if (nnz > 0 && Ai == nullptr) {
        fatal("CSC pattern: null row index array");
    }

    // Row range and per-column duplicate check (marker holds the last column
    // that touched each row).
    std::vector<std::int32_t> marker(un, -1);
    for (std::size_t j = 0; j < un; ++j) {
        for (auto p = static_cast<std::size_t>(Ap[j]); p < static_cast<std::size_t>(Ap[j + 1]); ++p) {
            const std::int32_t i = Ai[p];
            if (i < 0 || i >= n) {
                fatal("CSC pattern: row index " + std::to_string(i) + " out of range in column " + std::to_string(j));
            }
            if (marker[static_cast<std::size_t>(i)] == static_cast<std::int32_t>(j)) {
                fatal("CSC pattern: duplicate entry (" + std::to_string(i) + ", " + std::to_string(j) + ")");
            }
            marker[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(j);
        }
    }

    CscPattern pattern;
    pattern.n = n;
    pattern.col_ptr.assign(Ap, Ap + ap_size);
    pattern.row_idx.assign(Ai, Ai + nnz);
    return pattern;
}

// ----------------------------------------------------------------------------
// KluSolver
// ----------------------------------------------------------------------------

struct KluSolver::Impl {
    klu_common common{};
    klu_symbolic* symbolic = nullptr;
    klu_numeric* numeric = nullptr;
    CscPattern pattern;
    bool analyzed = false;
    bool factored = false;

    Impl() {
        klu_defaults(&common);  // halt_if_singular = TRUE, AMD ordering, BTF on
    }
    ~Impl() {
        free_numeric();
        free_symbolic();
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void free_numeric() noexcept {
        if (numeric != nullptr) {
            klu_free_numeric(&numeric, &common);
        }
        numeric = nullptr;
        factored = false;
    }
    void free_symbolic() noexcept {
        if (symbolic != nullptr) {
            klu_free_symbolic(&symbolic, &common);
        }
        symbolic = nullptr;
        analyzed = false;
    }

    // KLU never writes through these pointers; its C API just lacks const.
    std::int32_t* ap() noexcept { return pattern.col_ptr.data(); }
    std::int32_t* ai() noexcept { return pattern.row_idx.empty() ? &empty_slot : pattern.row_idx.data(); }
    static double* ax(const double* values) noexcept { return const_cast<double*>(values); }

    void require_analyzed(const char* operation) const {
        if (!analyzed) {
            fatal(std::string(operation) + " called before analyze()");
        }
    }
    void require_values(const double* values, std::size_t size, const char* operation) const {
        if (size != static_cast<std::size_t>(pattern.nnz())) {
            fatal(std::string(operation) + ": got " + std::to_string(size) + " values, the pattern has " +
                  std::to_string(pattern.nnz()) + " entries");
        }
        if (size > 0 && values == nullptr) {
            fatal(std::string(operation) + ": null value array");
        }
    }

    std::int32_t empty_slot = 0;
};

KluSolver::KluSolver() : impl_(std::make_unique<Impl>()) {}
KluSolver::~KluSolver() = default;
KluSolver::KluSolver(KluSolver&&) noexcept = default;
KluSolver& KluSolver::operator=(KluSolver&&) noexcept = default;

void KluSolver::analyze(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                        std::size_t ai_size) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();  // moved-from solver becomes usable again
    }
    Impl& s = *impl_;
    CscPattern pattern = CscPattern::from_arrays(n, Ap, ap_size, Ai, ai_size);

    s.free_numeric();
    s.free_symbolic();
    s.pattern = std::move(pattern);
    if (s.pattern.n > 0) {
        s.symbolic = klu_analyze(s.pattern.n, s.ap(), s.ai(), &s.common);
        if (s.symbolic == nullptr) {
            throw_status(s.common, "KLU analyze");
        }
    }
    s.analyzed = true;
}

void KluSolver::analyze(std::int32_t n, const std::vector<std::int32_t>& Ap, const std::vector<std::int32_t>& Ai) {
    analyze(n, Ap.data(), Ap.size(), Ai.data(), Ai.size());
}

void KluSolver::factor(const double* Ax, std::size_t ax_size) {
    if (!impl_) {
        fatal("factor called before analyze()");
    }
    Impl& s = *impl_;
    s.require_analyzed("factor");
    s.require_values(Ax, ax_size, "factor");

    s.free_numeric();
    if (s.pattern.n > 0) {
        klu_numeric* numeric = klu_factor(s.ap(), s.ai(), Impl::ax(Ax), s.symbolic, &s.common);
        if (numeric == nullptr || s.common.status != KLU_OK) {
            if (numeric != nullptr) {
                klu_free_numeric(&numeric, &s.common);
            }
            throw_status(s.common, "KLU factor");
        }
        s.numeric = numeric;
    }
    s.factored = true;
}

void KluSolver::factor(const std::vector<double>& Ax) { factor(Ax.data(), Ax.size()); }

FactorPath KluSolver::refactor(const double* Ax, std::size_t ax_size) {
    if (!impl_) {
        fatal("refactor called before analyze()");
    }
    Impl& s = *impl_;
    s.require_analyzed("refactor");
    s.require_values(Ax, ax_size, "refactor");

    if (!s.factored) {
        factor(Ax, ax_size);
        return FactorPath::FullFactor;
    }
    if (s.pattern.n == 0) {
        return FactorPath::Refactor;
    }
    const int ok = klu_refactor(s.ap(), s.ai(), Impl::ax(Ax), s.symbolic, s.numeric, &s.common);
    if (ok && s.common.status == KLU_OK) {
        return FactorPath::Refactor;
    }
    // The old pivot order no longer works for these values (or KLU hit another
    // problem): the numeric object is unusable, redo the factorization with pivoting.
    factor(Ax, ax_size);
    return FactorPath::FullFactor;
}

FactorPath KluSolver::refactor(const std::vector<double>& Ax) { return refactor(Ax.data(), Ax.size()); }

void KluSolver::solve(double* b, std::size_t b_size, std::int32_t nrhs) {
    if (!impl_ || !impl_->factored) {
        fatal("solve called before factor()");
    }
    Impl& s = *impl_;
    if (nrhs < 0) {
        fatal("solve: negative number of right-hand sides");
    }
    const std::size_t expected = static_cast<std::size_t>(s.pattern.n) * static_cast<std::size_t>(nrhs);
    if (b_size != expected) {
        fatal("solve: right-hand side has " + std::to_string(b_size) + " values, expected n * nrhs = " +
              std::to_string(expected));
    }
    if (expected == 0) {
        return;
    }
    if (b == nullptr) {
        fatal("solve: null right-hand side");
    }
    if (!klu_solve(s.symbolic, s.numeric, s.pattern.n, nrhs, b, &s.common)) {
        throw_status(s.common, "KLU solve");
    }
}

void KluSolver::solve(std::vector<double>& b, std::int32_t nrhs) { solve(b.data(), b.size(), nrhs); }

double KluSolver::rcond() {
    if (!impl_ || !impl_->factored) {
        fatal("rcond called before factor()");
    }
    Impl& s = *impl_;
    if (s.pattern.n == 0) {
        return 1.0;
    }
    if (!klu_rcond(s.symbolic, s.numeric, &s.common)) {
        throw_status(s.common, "KLU rcond");
    }
    return s.common.rcond;
}

std::int64_t KluSolver::nnz_lu() const noexcept {
    if (!impl_ || impl_->numeric == nullptr) {
        return 0;
    }
    return static_cast<std::int64_t>(impl_->numeric->lnz) + static_cast<std::int64_t>(impl_->numeric->unz);
}

std::int64_t KluSolver::nnz_offdiag() const noexcept {
    if (!impl_ || impl_->numeric == nullptr) {
        return 0;
    }
    return static_cast<std::int64_t>(impl_->numeric->nzoff);
}

std::int32_t KluSolver::size() const noexcept { return impl_ ? impl_->pattern.n : 0; }
std::int64_t KluSolver::nnz() const noexcept { return impl_ ? impl_->pattern.nnz() : 0; }
bool KluSolver::analyzed() const noexcept { return impl_ && impl_->analyzed; }
bool KluSolver::factored() const noexcept { return impl_ && impl_->factored; }

}  // namespace mna::linear
