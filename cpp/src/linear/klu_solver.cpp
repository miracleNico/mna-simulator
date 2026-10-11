#include "mna/linear/klu_solver.hpp"

#include <klu.h>

#include <new>
#include <string>
#include <utility>

#include "mna/error.hpp"

namespace mna::linear {
namespace {

[[noreturn]] void fatal(const std::string& message) { throw Error(ErrorCode::NetlistFatal, message); }

// Maps a failed KLU call to an exception (shared by the real and complex solvers).
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

// The numeric half of the KLU API, per scalar type. Symbolic analysis
// (klu_analyze / klu_free_symbolic) and klu_free_numeric are shared.
template <class Scalar>
struct KluApi;

template <>
struct KluApi<double> {
    static constexpr const char* kName = "KLU";
    // KLU never writes through value pointers; its C API just lacks const.
    static double* values(const double* v) noexcept { return const_cast<double*>(v); }
    static klu_numeric* factor(int32_t* Ap, int32_t* Ai, double* Ax, klu_symbolic* S, klu_common* c) {
        return klu_factor(Ap, Ai, Ax, S, c);
    }
    static int refactor(int32_t* Ap, int32_t* Ai, double* Ax, klu_symbolic* S, klu_numeric* N, klu_common* c) {
        return klu_refactor(Ap, Ai, Ax, S, N, c);
    }
    static int solve(klu_symbolic* S, klu_numeric* N, int32_t ld, int32_t nrhs, double* B, klu_common* c) {
        return klu_solve(S, N, ld, nrhs, B, c);
    }
    static int tsolve(klu_symbolic* S, klu_numeric* N, int32_t ld, int32_t nrhs, double* B, bool /*conjugate*/,
                      klu_common* c) {
        return klu_tsolve(S, N, ld, nrhs, B, c);
    }
    static int rcond(klu_symbolic* S, klu_numeric* N, klu_common* c) { return klu_rcond(S, N, c); }
};

template <>
struct KluApi<Complex> {
    static constexpr const char* kName = "KLU (complex)";
    // std::complex<double> arrays may be accessed as interleaved (re, im)
    // double pairs ([complex.numbers]), which is KLU's complex layout.
    static double* values(const Complex* v) noexcept {
        return reinterpret_cast<double*>(const_cast<Complex*>(v));
    }
    static klu_numeric* factor(int32_t* Ap, int32_t* Ai, double* Ax, klu_symbolic* S, klu_common* c) {
        return klu_z_factor(Ap, Ai, Ax, S, c);
    }
    static int refactor(int32_t* Ap, int32_t* Ai, double* Ax, klu_symbolic* S, klu_numeric* N, klu_common* c) {
        return klu_z_refactor(Ap, Ai, Ax, S, N, c);
    }
    static int solve(klu_symbolic* S, klu_numeric* N, int32_t ld, int32_t nrhs, double* B, klu_common* c) {
        return klu_z_solve(S, N, ld, nrhs, B, c);
    }
    static int tsolve(klu_symbolic* S, klu_numeric* N, int32_t ld, int32_t nrhs, double* B, bool conjugate,
                      klu_common* c) {
        return klu_z_tsolve(S, N, ld, nrhs, B, conjugate ? 1 : 0, c);
    }
    static int rcond(klu_symbolic* S, klu_numeric* N, klu_common* c) { return klu_z_rcond(S, N, c); }
};

}  // namespace

// ----------------------------------------------------------------------------
// CscPattern, policy helpers
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

const char* factor_path_name(FactorPath path) noexcept {
    return path == FactorPath::Refactor ? "refactor" : "full_factor";
}

// ----------------------------------------------------------------------------
// KluSolverT
// ----------------------------------------------------------------------------

template <class Scalar>
struct KluSolverT<Scalar>::Impl {
    using Api = KluApi<Scalar>;

    klu_common common{};
    klu_symbolic* symbolic = nullptr;
    klu_numeric* numeric = nullptr;
    CscPattern pattern;
    bool analyzed = false;
    bool factored = false;
    std::int32_t empty_slot = 0;
    double empty_values[2] = {0.0, 0.0};

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
            klu_free_numeric(&numeric, &common);  // frees real and complex Numeric objects alike
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
    // KLU rejects null arrays, so empty ones point at dummy slots (a pattern
    // without entries is then reported as structurally singular).
    std::int32_t* ap() noexcept { return pattern.col_ptr.data(); }
    std::int32_t* ai() noexcept { return pattern.row_idx.empty() ? &empty_slot : pattern.row_idx.data(); }
    double* ax(const Scalar* values) noexcept {
        return values == nullptr ? empty_values : Api::values(values);
    }

    void require_analyzed(const char* operation) const {
        if (!analyzed) {
            fatal(std::string(operation) + " called before analyze()");
        }
    }
    void require_values(const Scalar* values, std::size_t size, const char* operation) const {
        if (size != static_cast<std::size_t>(pattern.nnz())) {
            fatal(std::string(operation) + ": got " + std::to_string(size) + " values, the pattern has " +
                  std::to_string(pattern.nnz()) + " entries");
        }
        if (size > 0 && values == nullptr) {
            fatal(std::string(operation) + ": null value array");
        }
    }
    // Checks a column-major n x nrhs right-hand side; false when there is nothing to do.
    bool check_rhs(const Scalar* b, std::size_t b_size, std::int32_t nrhs, const char* operation) const {
        if (nrhs < 0) {
            fatal(std::string(operation) + ": negative number of right-hand sides");
        }
        const std::size_t expected = static_cast<std::size_t>(pattern.n) * static_cast<std::size_t>(nrhs);
        if (b_size != expected) {
            fatal(std::string(operation) + ": right-hand side has " + std::to_string(b_size) +
                  " values, expected n * nrhs = " + std::to_string(expected));
        }
        if (expected == 0) {
            return false;
        }
        if (b == nullptr) {
            fatal(std::string(operation) + ": null right-hand side");
        }
        return true;
    }

    void analyze(CscPattern&& validated) {
        free_numeric();
        free_symbolic();
        pattern = std::move(validated);
        if (pattern.n > 0) {
            symbolic = klu_analyze(pattern.n, ap(), ai(), &common);
            if (symbolic == nullptr) {
                throw_status(common, "KLU analyze");
            }
        }
        analyzed = true;
    }

    void factor(const Scalar* Ax) {
        free_numeric();
        if (pattern.n > 0) {
            klu_numeric* fresh = Api::factor(ap(), ai(), ax(Ax), symbolic, &common);
            if (fresh == nullptr || common.status != KLU_OK) {
                if (fresh != nullptr) {
                    klu_free_numeric(&fresh, &common);
                }
                throw_status(common, (std::string(Api::kName) + " factor").c_str());
            }
            numeric = fresh;
        }
        factored = true;
    }

    double rcond() {
        if (pattern.n == 0) {
            return 1.0;
        }
        if (!Api::rcond(symbolic, numeric, &common)) {
            throw_status(common, (std::string(Api::kName) + " rcond").c_str());
        }
        return common.rcond;
    }
};

template <class Scalar>
KluSolverT<Scalar>::KluSolverT() : impl_(std::make_unique<Impl>()) {}
template <class Scalar>
KluSolverT<Scalar>::~KluSolverT() = default;
template <class Scalar>
KluSolverT<Scalar>::KluSolverT(KluSolverT&&) noexcept = default;
template <class Scalar>
KluSolverT<Scalar>& KluSolverT<Scalar>::operator=(KluSolverT&&) noexcept = default;

template <class Scalar>
void KluSolverT<Scalar>::analyze(std::int32_t n, const std::int32_t* Ap, std::size_t ap_size, const std::int32_t* Ai,
                                 std::size_t ai_size) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();  // moved-from solver becomes usable again
    }
    impl_->analyze(CscPattern::from_arrays(n, Ap, ap_size, Ai, ai_size));
}

template <class Scalar>
void KluSolverT<Scalar>::analyze(std::int32_t n, const std::vector<std::int32_t>& Ap,
                                 const std::vector<std::int32_t>& Ai) {
    analyze(n, Ap.data(), Ap.size(), Ai.data(), Ai.size());
}

template <class Scalar>
void KluSolverT<Scalar>::analyze(const CscPattern& pattern) {
    analyze(pattern.n, pattern.col_ptr.data(), pattern.col_ptr.size(), pattern.row_idx.data(),
            pattern.row_idx.size());
}

template <class Scalar>
void KluSolverT<Scalar>::factor(const Scalar* Ax, std::size_t ax_size) {
    if (!impl_) {
        fatal("factor called before analyze()");
    }
    Impl& s = *impl_;
    s.require_analyzed("factor");
    s.require_values(Ax, ax_size, "factor");
    s.factor(Ax);
}

template <class Scalar>
void KluSolverT<Scalar>::factor(const std::vector<Scalar>& Ax) {
    factor(Ax.data(), Ax.size());
}

template <class Scalar>
FactorPath KluSolverT<Scalar>::refactor(const Scalar* Ax, std::size_t ax_size) {
    if (!impl_) {
        fatal("refactor called before analyze()");
    }
    Impl& s = *impl_;
    s.require_analyzed("refactor");
    s.require_values(Ax, ax_size, "refactor");

    if (!s.factored) {
        s.factor(Ax);
        return FactorPath::FullFactor;
    }
    if (s.pattern.n == 0) {
        return FactorPath::Refactor;
    }
    using Api = typename Impl::Api;
    const int ok = Api::refactor(s.ap(), s.ai(), s.ax(Ax), s.symbolic, s.numeric, &s.common);
    if (ok && s.common.status == KLU_OK && s.rcond() >= kRefactorRcondFloor) {
        return FactorPath::Refactor;
    }
    // The old pivot order no longer works for these values (zero pivot, a
    // collapsed pivot ratio, or another KLU problem): redo the factorization
    // with partial pivoting. A NaN rcond also lands here.
    s.factor(Ax);
    return FactorPath::FullFactor;
}

template <class Scalar>
FactorPath KluSolverT<Scalar>::refactor(const std::vector<Scalar>& Ax) {
    return refactor(Ax.data(), Ax.size());
}

template <class Scalar>
void KluSolverT<Scalar>::solve(Scalar* b, std::size_t b_size, std::int32_t nrhs) {
    if (!impl_ || !impl_->factored) {
        fatal("solve called before factor()");
    }
    Impl& s = *impl_;
    if (!s.check_rhs(b, b_size, nrhs, "solve")) {
        return;
    }
    using Api = typename Impl::Api;
    if (!Api::solve(s.symbolic, s.numeric, s.pattern.n, nrhs, Api::values(b), &s.common)) {
        throw_status(s.common, (std::string(Api::kName) + " solve").c_str());
    }
}

template <class Scalar>
void KluSolverT<Scalar>::solve(std::vector<Scalar>& b, std::int32_t nrhs) {
    solve(b.data(), b.size(), nrhs);
}

template <class Scalar>
void KluSolverT<Scalar>::tsolve(Scalar* b, std::size_t b_size, std::int32_t nrhs, bool conjugate) {
    if (!impl_ || !impl_->factored) {
        fatal("tsolve called before factor()");
    }
    Impl& s = *impl_;
    if (!s.check_rhs(b, b_size, nrhs, "tsolve")) {
        return;
    }
    using Api = typename Impl::Api;
    if (!Api::tsolve(s.symbolic, s.numeric, s.pattern.n, nrhs, Api::values(b), conjugate, &s.common)) {
        throw_status(s.common, (std::string(Api::kName) + " tsolve").c_str());
    }
}

template <class Scalar>
void KluSolverT<Scalar>::tsolve(std::vector<Scalar>& b, std::int32_t nrhs, bool conjugate) {
    tsolve(b.data(), b.size(), nrhs, conjugate);
}

template <class Scalar>
double KluSolverT<Scalar>::rcond() {
    if (!impl_ || !impl_->factored) {
        fatal("rcond called before factor()");
    }
    return impl_->rcond();
}

template <class Scalar>
std::int64_t KluSolverT<Scalar>::nnz_lu() const noexcept {
    if (!impl_ || impl_->numeric == nullptr) {
        return 0;
    }
    return static_cast<std::int64_t>(impl_->numeric->lnz) + static_cast<std::int64_t>(impl_->numeric->unz);
}

template <class Scalar>
std::int64_t KluSolverT<Scalar>::nnz_offdiag() const noexcept {
    if (!impl_ || impl_->numeric == nullptr) {
        return 0;
    }
    return static_cast<std::int64_t>(impl_->numeric->nzoff);
}

template <class Scalar>
std::int32_t KluSolverT<Scalar>::size() const noexcept {
    return impl_ ? impl_->pattern.n : 0;
}
template <class Scalar>
std::int64_t KluSolverT<Scalar>::nnz() const noexcept {
    return impl_ ? impl_->pattern.nnz() : 0;
}
template <class Scalar>
bool KluSolverT<Scalar>::analyzed() const noexcept {
    return impl_ && impl_->analyzed;
}
template <class Scalar>
bool KluSolverT<Scalar>::factored() const noexcept {
    return impl_ && impl_->factored;
}

template class KluSolverT<double>;
template class KluSolverT<Complex>;

}  // namespace mna::linear
