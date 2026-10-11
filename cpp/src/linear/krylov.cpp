#include "mna/linear/krylov.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "mna/error.hpp"

namespace mna::linear {
namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

inline double conj_of(double v) noexcept { return v; }
inline Complex conj_of(const Complex& v) noexcept { return std::conj(v); }
inline double abs2(double v) noexcept { return v * v; }
inline double abs2(const Complex& v) noexcept { return v.real() * v.real() + v.imag() * v.imag(); }

// sum conj(a_i) b_i (np.vdot; np.dot / np.inner for real vectors).
template <class S>
S vdot(std::size_t n, const S* a, const S* b) noexcept {
    S sum{};
    for (std::size_t i = 0; i < n; ++i) {
        sum += conj_of(a[i]) * b[i];
    }
    return sum;
}

template <class S>
double norm2(std::size_t n, const S* a) noexcept {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += abs2(a[i]);
    }
    return std::sqrt(sum);
}

// The operator x -> A x, remembering how many threads its products used.
template <class S>
class Operator {
public:
    Operator(const CsrMatrix<S>& A, int num_threads) : A_(A), num_threads_(num_threads) {}
    void apply(const S* x, S* y) { threads_used_ = std::max(threads_used_, spmv(A_, x, y, num_threads_)); }
    std::size_t n() const noexcept { return static_cast<std::size_t>(A_.n); }
    int threads_used() const noexcept { return threads_used_; }

private:
    const CsrMatrix<S>& A_;
    int num_threads_;
    int threads_used_ = 1;
};

// solvers._relative_residual: ||b - A x|| / (||b|| if ||b|| > 0 else 1).
template <class S>
double relative_residual(Operator<S>& A, const S* b, const S* x, double b_norm, std::vector<S>& work) {
    const std::size_t n = A.n();
    A.apply(x, work.data());
    for (std::size_t i = 0; i < n; ++i) {
        work[i] = b[i] - work[i];
    }
    const double scale = b_norm > 0.0 ? b_norm : 1.0;
    return norm2(n, work.data()) / scale;
}

// r = b - A x.
template <class S>
void residual_vector(Operator<S>& A, const S* b, const S* x, S* r) {
    A.apply(x, r);
    for (std::size_t i = 0; i < A.n(); ++i) {
        r[i] = b[i] - r[i];
    }
}

// Plane rotation [c s; -conj(s) c] [f; g] = [r; 0] with real c (LAPACK
// dlartg / zlartg conventions in their unscaled range).
template <class S>
void make_givens(const S& f, const S& g, double& c, S& s, S& r) {
    if (g == S{}) {
        c = 1.0;
        s = S{};
        r = f;
        return;
    }
    if (f == S{}) {
        const double ag = std::abs(g);
        c = 0.0;
        s = conj_of(g) / ag;
        r = S(ag);
        return;
    }
    const double af = std::abs(f);
    const double d = std::sqrt(abs2(f) + abs2(g));
    const S phase = f / af;
    c = af / d;
    s = phase * conj_of(g) / d;
    r = phase * d;
}

template <class S>
void apply_givens(double c, const S& s, S& a, S& b) noexcept {
    const S a0 = a;
    const S b0 = b;
    a = c * a0 + s * b0;
    b = -conj_of(s) * a0 + c * b0;
}

void set_converged(KrylovInfo& info, std::int64_t iterations, double residual) {
    info.converged = true;
    info.iterations = iterations;
    info.relative_residual = residual;
    info.exit_code = 0;
}

// ---------------------------------------------------------------------------
// Legacy kernels: ports of solvers._arnoldi_gmres / _conjugate_gradient /
// _conjugate_residual.
// ---------------------------------------------------------------------------

template <class S>
void legacy_gmres(Operator<S>& A, const S* b, S* x, double tol, std::int64_t max_iter_in, std::int64_t restart_in,
                  KrylovInfo& info) {
    const std::size_t n = A.n();
    const double rhs_norm = norm2(n, b);
    if (rhs_norm == 0.0) {
        std::fill(x, x + n, S{});
        set_converged(info, 0, 0.0);
        return;
    }
    const std::int64_t max_iter = std::max<std::int64_t>(1, max_iter_in);
    const std::int64_t requested = restart_in != 0 ? restart_in : static_cast<std::int64_t>(n);
    const std::int64_t restart = std::max<std::int64_t>(1, std::min<std::int64_t>(requested, static_cast<std::int64_t>(n)));

    std::vector<S> work(n);
    std::vector<S> best_x(x, x + n);
    double best_residual = relative_residual(A, b, x, rhs_norm, work);
    if (best_residual <= tol) {
        set_converged(info, 0, best_residual);
        return;
    }

    const auto m_max = static_cast<std::size_t>(restart);
    const std::size_t ld = m_max + 1;  // leading dimension of H and R (column-major (m+1) x m)
    std::vector<S> V(n * (m_max + 1));
    std::vector<S> H(ld * m_max);
    std::vector<S> R(ld * m_max);
    std::vector<S> g(m_max + 1);
    std::vector<S> y(m_max);
    std::vector<double> cs(m_max);
    std::vector<S> sn(m_max);
    std::vector<S> r(n);
    std::vector<S> w(n);
    std::vector<S> step(n);
    std::vector<S> candidate(n);

    std::int64_t iterations = 0;
    while (iterations < max_iter) {
        residual_vector(A, b, x, r.data());
        const double beta = norm2(n, r.data());
        if (beta / rhs_norm <= tol) {
            set_converged(info, iterations, beta / rhs_norm);
            return;
        }

        const auto m = static_cast<std::size_t>(std::min<std::int64_t>(restart, max_iter - iterations));
        std::fill(V.begin(), V.end(), S{});
        std::fill(H.begin(), H.end(), S{});
        for (std::size_t i = 0; i < n; ++i) {
            V[i] = r[i] / beta;
        }
        std::fill(g.begin(), g.end(), S{});
        g[0] = S(beta);
        std::copy(x, x + n, candidate.begin());

        bool broke_down = false;
        for (std::size_t j = 0; j < m; ++j) {
            // Arnoldi step with modified Gram-Schmidt.
            A.apply(&V[j * n], w.data());
            for (std::size_t i = 0; i <= j; ++i) {
                const S* vi = &V[i * n];
                const S h = vdot(n, vi, w.data());
                H[i + j * ld] = h;
                for (std::size_t k = 0; k < n; ++k) {
                    w[k] -= h * vi[k];
                }
            }
            const double h_next = norm2(n, w.data());
            H[(j + 1) + j * ld] = S(h_next);
            if (std::abs(h_next) > kLegacyBreakdownTol) {
                S* v_next = &V[(j + 1) * n];
                for (std::size_t k = 0; k < n; ++k) {
                    v_next[k] = w[k] / h_next;
                }
            }

            // min || beta e1 - H[:j+2, :j+1] y || by Givens rotations.
            S* Rj = &R[j * ld];
            for (std::size_t i = 0; i <= j + 1; ++i) {
                Rj[i] = H[i + j * ld];
            }
            for (std::size_t k = 0; k < j; ++k) {
                apply_givens(cs[k], sn[k], Rj[k], Rj[k + 1]);
            }
            S rotated{};
            make_givens(Rj[j], Rj[j + 1], cs[j], sn[j], rotated);
            Rj[j] = rotated;
            Rj[j + 1] = S{};
            g[j + 1] = -conj_of(sn[j]) * g[j];
            g[j] = cs[j] * g[j];

            // Back substitution; pivots below lstsq's default cutoff count as zero.
            double max_diag = 0.0;
            for (std::size_t k = 0; k <= j; ++k) {
                max_diag = std::max(max_diag, static_cast<double>(std::abs(R[k + k * ld])));
            }
            const double cutoff = kEps * static_cast<double>(j + 2) * max_diag;
            for (std::size_t kk = j + 1; kk-- > 0;) {
                const S diag = R[kk + kk * ld];
                if (!(std::abs(diag) > cutoff)) {
                    y[kk] = S{};
                    continue;
                }
                S sum = g[kk];
                for (std::size_t i = kk + 1; i <= j; ++i) {
                    sum -= R[kk + i * ld] * y[i];
                }
                y[kk] = sum / diag;
            }

            // candidate = x + V[:, :j+1] y
            std::fill(step.begin(), step.end(), S{});
            for (std::size_t i = 0; i <= j; ++i) {
                const S* vi = &V[i * n];
                const S yi = y[i];
                for (std::size_t k = 0; k < n; ++k) {
                    step[k] += vi[k] * yi;
                }
            }
            for (std::size_t k = 0; k < n; ++k) {
                candidate[k] = x[k] + step[k];
            }
            const double residual = relative_residual(A, b, candidate.data(), rhs_norm, work);
            ++iterations;

            if (residual < best_residual) {
                best_residual = residual;
                best_x = candidate;
            }
            if (residual <= tol) {
                std::copy(candidate.begin(), candidate.end(), x);
                set_converged(info, iterations, residual);
                return;
            }
            if (std::abs(h_next) <= kLegacyBreakdownTol) {
                std::copy(candidate.begin(), candidate.end(), x);
                broke_down = true;
                break;
            }
        }
        if (!broke_down) {
            std::copy(candidate.begin(), candidate.end(), x);
        }
    }

    std::copy(best_x.begin(), best_x.end(), x);
    info.converged = false;
    info.iterations = iterations;
    info.relative_residual = best_residual;
}

template <class S>
void legacy_cg(Operator<S>& A, const S* b, S* x, double tol, std::int64_t max_iter, KrylovInfo& info) {
    const std::size_t n = A.n();
    const double rhs_norm = norm2(n, b);
    if (rhs_norm == 0.0) {
        std::fill(x, x + n, S{});
        set_converged(info, 0, 0.0);
        return;
    }
    std::vector<S> work(n);
    std::vector<S> r(n);
    residual_vector(A, b, x, r.data());
    std::vector<S> p(r);
    std::vector<S> Ap(n);
    S rr = vdot(n, r.data(), r.data());
    std::vector<S> best_x(x, x + n);
    double best_residual = relative_residual(A, b, x, rhs_norm, work);
    const double denom_tol = kLegacyBreakdownTol;
    const std::int64_t budget = std::max<std::int64_t>(1, max_iter);

    for (std::int64_t iteration = 1; iteration <= budget; ++iteration) {
        A.apply(p.data(), Ap.data());
        const S denom = vdot(n, p.data(), Ap.data());
        if (std::abs(denom) <= denom_tol) {
            break;
        }
        const S alpha = rr / denom;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
        }
        const double residual = relative_residual(A, b, x, rhs_norm, work);
        if (residual < best_residual) {
            best_residual = residual;
            best_x.assign(x, x + n);
        }
        if (residual <= tol) {
            set_converged(info, iteration, residual);
            return;
        }
        const S rr_next = vdot(n, r.data(), r.data());
        if (std::abs(rr) <= denom_tol) {
            break;
        }
        const S ratio = rr_next / rr;
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = r[i] + ratio * p[i];
        }
        rr = rr_next;
    }

    std::copy(best_x.begin(), best_x.end(), x);
    info.converged = false;
    info.iterations = budget;
    info.relative_residual = best_residual;
}

template <class S>
void legacy_cr(Operator<S>& A, const S* b, S* x, double tol, std::int64_t max_iter, KrylovInfo& info) {
    const std::size_t n = A.n();
    const double rhs_norm = norm2(n, b);
    if (rhs_norm == 0.0) {
        std::fill(x, x + n, S{});
        set_converged(info, 0, 0.0);
        return;
    }
    std::vector<S> work(n);
    std::vector<S> r(n);
    residual_vector(A, b, x, r.data());
    std::vector<S> p(r);
    std::vector<S> Ap(n);
    A.apply(p.data(), Ap.data());
    std::vector<S> r_next(n);
    std::vector<S> Ar_next(n);
    std::vector<S> best_x(x, x + n);
    double best_residual = relative_residual(A, b, x, rhs_norm, work);
    const double denom_tol = kLegacyBreakdownTol;
    const std::int64_t budget = std::max<std::int64_t>(1, max_iter);

    for (std::int64_t iteration = 1; iteration <= budget; ++iteration) {
        const S denom = vdot(n, Ap.data(), Ap.data());
        if (std::abs(denom) <= denom_tol) {
            break;
        }
        const S alpha = vdot(n, r.data(), Ap.data()) / denom;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = x[i] + alpha * p[i];
            r_next[i] = r[i] - alpha * Ap[i];
        }
        const double residual = relative_residual(A, b, x, rhs_norm, work);
        if (residual < best_residual) {
            best_residual = residual;
            best_x.assign(x, x + n);
        }
        if (residual <= tol) {
            set_converged(info, iteration, residual);
            return;
        }
        A.apply(r_next.data(), Ar_next.data());
        const S beta = vdot(n, Ar_next.data(), Ap.data()) / denom;
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = r_next[i] - beta * p[i];
            Ap[i] = Ar_next[i] - beta * Ap[i];
        }
        std::swap(r, r_next);
    }

    std::copy(best_x.begin(), best_x.end(), x);
    info.converged = false;
    info.iterations = budget;
    info.relative_residual = best_residual;
}

// ---------------------------------------------------------------------------
// SciPy-equivalent kernels: ports of scipy/sparse/linalg/_isolve
// (iterative.py cg and gmres, minres.py) without preconditioner, atol = 0.
// ---------------------------------------------------------------------------

template <class S>
void scipy_cg(Operator<S>& A, const S* b, S* x, double rtol, std::int64_t maxiter, KrylovInfo& info) {
    const std::size_t n = A.n();
    const double bnrm2 = norm2(n, b);
    const double atol = std::max(0.0, rtol * bnrm2);
    if (bnrm2 == 0.0) {
        std::copy(b, b + n, x);
        set_converged(info, 0, 0.0);
        return;
    }
    std::vector<S> r(n);
    residual_vector(A, b, x, r.data());
    std::vector<S> p(n);
    std::vector<S> q(n);
    S rho_prev{};
    for (std::int64_t iteration = 0; iteration < maxiter; ++iteration) {
        if (norm2(n, r.data()) < atol) {
            set_converged(info, iteration, 0.0);
            return;
        }
        const S rho_cur = vdot(n, r.data(), r.data());
        if (iteration > 0) {
            const S beta = rho_cur / rho_prev;
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = p[i] * beta + r[i];
            }
        } else {
            p = r;
        }
        A.apply(p.data(), q.data());
        const S alpha = rho_cur / vdot(n, p.data(), q.data());
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        rho_prev = rho_cur;
    }
    info.converged = false;
    info.iterations = maxiter;
    info.exit_code = maxiter;
}

template <class S>
void scipy_gmres(Operator<S>& A, const S* b, S* x, double rtol, std::int64_t restart_in, std::int64_t maxiter,
                 KrylovInfo& info) {
    const std::size_t n = A.n();
    const double bnrm2 = norm2(n, b);
    const double atol = std::max(0.0, rtol * bnrm2);
    if (bnrm2 == 0.0) {
        std::copy(b, b + n, x);
        set_converged(info, 0, 0.0);
        return;
    }
    const auto restart = static_cast<std::size_t>(std::min<std::int64_t>(restart_in, static_cast<std::int64_t>(n)));
    const double Mb_nrm2 = bnrm2;
    double ptol_max_factor = 1.0;
    double ptol = Mb_nrm2 * std::min(ptol_max_factor, atol / bnrm2);
    double presid = 0.0;

    const std::size_t ldh = restart + 1;          // h is restart x (restart + 1), row-major: h[col][row]
    std::vector<S> v((restart + 1) * n);          // row k = k-th Krylov vector
    std::vector<S> h(restart * ldh);
    std::vector<double> giv_c(restart);
    std::vector<S> giv_s(restart);
    std::vector<S> Sg(restart + 1);
    std::vector<S> y(restart);
    std::vector<S> r(n);
    std::vector<S> step(n);
    std::int64_t inner_iter = 0;
    double rnorm = 0.0;

    for (std::int64_t iteration = 0; iteration < maxiter; ++iteration) {
        if (iteration == 0) {
            residual_vector(A, b, x, r.data());
            if (norm2(n, r.data()) < atol) {
                set_converged(info, 0, 0.0);
                return;
            }
        }
        {
            const double tmp = norm2(n, r.data());
            const double inv = 1.0 / tmp;
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = r[i] * inv;
            }
            std::fill(Sg.begin(), Sg.end(), S{});
            Sg[0] = S(tmp);
        }
        bool breakdown = false;
        std::size_t col = 0;
        for (col = 0; col < restart; ++col) {
            S* w = &v[(col + 1) * n];
            A.apply(&v[col * n], w);
            const double h0 = norm2(n, w);
            S* hcol = &h[col * ldh];
            for (std::size_t k = 0; k <= col; ++k) {
                const S* vk = &v[k * n];
                const S t = vdot(n, vk, w);
                hcol[k] = t;
                for (std::size_t i = 0; i < n; ++i) {
                    w[i] -= t * vk[i];
                }
            }
            const double h1 = norm2(n, w);
            hcol[col + 1] = S(h1);
            if (h1 <= kEps * h0) {
                hcol[col + 1] = S{};
                breakdown = true;
            } else {
                const double inv = 1.0 / h1;
                for (std::size_t i = 0; i < n; ++i) {
                    w[i] *= inv;
                }
            }
            for (std::size_t k = 0; k < col; ++k) {
                apply_givens(giv_c[k], giv_s[k], hcol[k], hcol[k + 1]);
            }
            S mag{};
            make_givens(hcol[col], hcol[col + 1], giv_c[col], giv_s[col], mag);
            hcol[col] = mag;
            hcol[col + 1] = S{};
            const S t = -conj_of(giv_s[col]) * Sg[col];
            Sg[col] = giv_c[col] * Sg[col];
            Sg[col + 1] = t;
            presid = std::abs(t);
            ++inner_iter;
            if (presid <= ptol || breakdown) {
                break;
            }
        }
        if (col == restart) {
            col = restart - 1;  // the loop ran to completion
        }

        // Solve the upper triangular system, pseudo-solving singular cases as SciPy does.
        if (h[col * ldh + col] == S{}) {
            Sg[col] = S{};
        }
        for (std::size_t k = 0; k <= col; ++k) {
            y[k] = Sg[k];
        }
        for (std::size_t k = col; k > 0; --k) {
            if (y[k] != S{}) {
                y[k] /= h[k * ldh + k];
                const S t = y[k];
                for (std::size_t i = 0; i < k; ++i) {
                    y[i] -= t * h[k * ldh + i];
                }
            }
        }
        if (y[0] != S{}) {
            y[0] /= h[0];
        }
        std::fill(step.begin(), step.end(), S{});
        for (std::size_t k = 0; k <= col; ++k) {
            const S* vk = &v[k * n];
            const S yk = y[k];
            for (std::size_t i = 0; i < n; ++i) {
                step[i] += yk * vk[i];
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += step[i];
        }
        residual_vector(A, b, x, r.data());
        rnorm = norm2(n, r.data());

        if (rnorm <= atol) {
            break;
        }
        if (breakdown) {
            break;  // exact-solution indicator but the true residual disagrees: SciPy bails out
        }
        if (presid <= ptol) {
            ptol_max_factor = std::max(kEps, 0.25 * ptol_max_factor);
        } else {
            ptol_max_factor = std::min(1.0, 1.5 * ptol_max_factor);
        }
        ptol = presid * std::min(ptol_max_factor, atol / rnorm);
    }

    info.iterations = inner_iter;
    info.converged = rnorm <= atol;
    info.exit_code = info.converged ? 0 : maxiter;
}

void minres(Operator<double>& A, const double* b, double* x, double rtol, std::int64_t maxiter, KrylovInfo& info) {
    const std::size_t n = A.n();
    std::vector<double> r1(n);
    residual_vector(A, b, x, r1.data());
    double beta1 = vdot(n, r1.data(), r1.data());
    if (beta1 == 0.0) {
        set_converged(info, 0, 0.0);
        return;
    }
    const double bnorm = norm2(n, b);
    if (bnorm == 0.0) {
        std::copy(b, b + n, x);
        set_converged(info, 0, 0.0);
        return;
    }
    beta1 = std::sqrt(beta1);

    int istop = 0;
    std::int64_t itn = 0;
    const double eps = kEps;
    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double rhs1 = beta1;
    double rhs2 = 0.0;
    double tnorm2 = 0.0;
    double gmax = 0.0;
    double gmin = std::numeric_limits<double>::max();
    double cs = -1.0;
    double sn = 0.0;
    std::vector<double> w(n, 0.0);
    std::vector<double> w1(n, 0.0);
    std::vector<double> w2(n, 0.0);
    std::vector<double> r2(r1);  // r2 = r1; y = psolve(r1) == r2
    std::vector<double> v(n);
    std::vector<double> y(n);

    while (itn < maxiter) {
        ++itn;
        const double s = 1.0 / beta;
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = s * r2[i];
        }
        A.apply(v.data(), y.data());
        if (itn >= 2) {
            const double f = beta / oldb;
            for (std::size_t i = 0; i < n; ++i) {
                y[i] = y[i] - f * r1[i];
            }
        }
        const double alfa = vdot(n, v.data(), y.data());
        {
            const double f = alfa / beta;
            for (std::size_t i = 0; i < n; ++i) {
                y[i] = y[i] - f * r2[i];
            }
        }
        std::swap(r1, r2);  // r1 = r2
        std::swap(r2, y);   // r2 = y (y is scratch again; psolve(r2) is r2 itself)
        oldb = beta;
        beta = std::sqrt(vdot(n, r2.data(), r2.data()));
        tnorm2 += alfa * alfa + oldb * oldb + beta * beta;

        if (itn == 1 && beta / beta1 <= 10.0 * eps) {
            istop = -1;  // terminate later
        }

        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        const double root = std::sqrt(gbar * gbar + dbar * dbar);

        double gamma = std::sqrt(gbar * gbar + beta * beta);
        gamma = std::max(gamma, eps);
        cs = gbar / gamma;
        sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = sn * phibar;

        const double denom = 1.0 / gamma;
        std::swap(w1, w2);  // w1 = w2
        std::swap(w2, w);   // w2 = w
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
            x[i] = x[i] + phi * w[i];
        }

        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        const double z = rhs1 / gamma;
        rhs1 = rhs2 - delta * z;
        rhs2 = -epsln * z;

        const double Anorm = std::sqrt(tnorm2);
        const double ynorm = norm2(n, x);
        const double epsx = Anorm * ynorm * eps;
        const double rnorm = phibar;
        const double inf = std::numeric_limits<double>::infinity();
        const double test1 = (ynorm == 0.0 || Anorm == 0.0) ? inf : rnorm / (Anorm * ynorm);
        const double test2 = Anorm == 0.0 ? inf : root / Anorm;
        const double Acond = gmax / gmin;

        if (istop == 0) {
            const double t1 = 1.0 + test1;
            const double t2 = 1.0 + test2;
            if (t2 <= 1.0) istop = 2;
            if (t1 <= 1.0) istop = 1;
            if (itn >= maxiter) istop = 6;
            if (Acond >= 0.1 / eps) istop = 4;
            if (epsx >= beta1) istop = 3;
            if (test2 <= rtol) istop = 2;
            if (test1 <= rtol) istop = 1;
        }
        if (istop != 0) {
            break;
        }
    }

    info.iterations = itn;
    info.converged = istop != 6;
    info.exit_code = istop == 6 ? maxiter : 0;
}

std::string lower_ascii(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Routing and names
// ---------------------------------------------------------------------------

const char* krylov_route_name(KrylovRoute route) noexcept {
    switch (route) {
        case KrylovRoute::PositiveDefinite: return "positive_definite";
        case KrylovRoute::Symmetric: return "symmetric";
        case KrylovRoute::General: break;
    }
    return "general";
}

KrylovRoute route_for_kind(MatrixKind kind) noexcept {
    switch (kind) {
        case MatrixKind::PositiveDefinite: return KrylovRoute::PositiveDefinite;
        case MatrixKind::Symmetric: return KrylovRoute::Symmetric;
        case MatrixKind::General: break;
    }
    return KrylovRoute::General;
}

bool method_is_auto(std::string_view method) noexcept {
    constexpr std::string_view kAuto = "auto";
    if (method.size() != kAuto.size()) {
        return false;
    }
    for (std::size_t i = 0; i < kAuto.size(); ++i) {
        const char c = method[i];
        if ((c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c) != kAuto[i]) {
            return false;
        }
    }
    return true;
}

KrylovRoute krylov_route(MatrixKind kind, std::string_view method) {
    const std::string m = lower_ascii(method);
    if (m == "auto") {
        return route_for_kind(kind);
    }
    if (m == "cg" || m == "conjugate_gradient" || m == "positive_definite" || m == "spd") {
        return KrylovRoute::PositiveDefinite;
    }
    if (m == "cr" || m == "minres" || m == "conjugate_residual" || m == "symmetric" || m == "hermitian") {
        return KrylovRoute::Symmetric;
    }
    if (m == "arnoldi" || m == "gmres" || m == "arnoldi_gmres" || m == "general") {
        return KrylovRoute::General;
    }
    throw std::invalid_argument("Unknown Krylov method '" + std::string(method) + "'.");
}

const char* krylov_algorithm_name(KrylovAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case KrylovAlgorithm::LegacyGmres: return "legacy_gmres";
        case KrylovAlgorithm::LegacyCg: return "legacy_cg";
        case KrylovAlgorithm::LegacyCr: return "legacy_cr";
        case KrylovAlgorithm::RestartedGmres: return "restarted_gmres";
        case KrylovAlgorithm::Cg: return "cg";
        case KrylovAlgorithm::Minres: return "minres";
    }
    return "unknown";
}

const char* krylov_method_name(KrylovAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case KrylovAlgorithm::LegacyCg:
        case KrylovAlgorithm::Cg: return "conjugate_gradient";
        case KrylovAlgorithm::LegacyCr: return "conjugate_residual";
        case KrylovAlgorithm::Minres: return "minres";
        case KrylovAlgorithm::LegacyGmres:
        case KrylovAlgorithm::RestartedGmres: break;
    }
    return "arnoldi_gmres";
}

KrylovRoute route_of_algorithm(KrylovAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case KrylovAlgorithm::LegacyCg:
        case KrylovAlgorithm::Cg: return KrylovRoute::PositiveDefinite;
        case KrylovAlgorithm::LegacyCr:
        case KrylovAlgorithm::Minres: return KrylovRoute::Symmetric;
        case KrylovAlgorithm::LegacyGmres:
        case KrylovAlgorithm::RestartedGmres: break;
    }
    return KrylovRoute::General;
}

KrylovAlgorithm select_krylov_algorithm(KrylovRoute route, std::int32_t n, bool is_complex) noexcept {
    const bool scipy = n >= kScipyEngineMinN;
    switch (route) {
        case KrylovRoute::PositiveDefinite: return scipy ? KrylovAlgorithm::Cg : KrylovAlgorithm::LegacyCg;
        case KrylovRoute::Symmetric:
            return scipy && !is_complex ? KrylovAlgorithm::Minres : KrylovAlgorithm::LegacyCr;
        case KrylovRoute::General: break;
    }
    return scipy ? KrylovAlgorithm::RestartedGmres : KrylovAlgorithm::LegacyGmres;
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

template <class Scalar>
KrylovInfo run_krylov_algorithm(KrylovAlgorithm algorithm, const CsrMatrix<Scalar>& A, const Scalar* b, Scalar* x,
                                const KrylovSettings& settings) {
    if (A.n > 0 && (b == nullptr || x == nullptr)) {
        throw Error(ErrorCode::NetlistFatal, "krylov_solve: null right-hand side or solution array");
    }
    if (algorithm == KrylovAlgorithm::Minres && !std::is_same_v<Scalar, double>) {
        throw std::invalid_argument("MINRES is implemented for real matrices only");
    }
    KrylovInfo info;
    info.algorithm = algorithm;
    info.route = route_of_algorithm(algorithm);
    info.method = krylov_method_name(algorithm);
    const auto n = static_cast<std::int64_t>(A.n);
    if (info.route == KrylovRoute::General) {
        info.iteration_budget = settings.max_iter;
        info.effective_restart = std::min(settings.rank, n);
    } else {
        info.iteration_budget = settings.rank;
    }

    Operator<Scalar> op(A, settings.num_threads);
    switch (algorithm) {
        case KrylovAlgorithm::LegacyGmres:
            legacy_gmres(op, b, x, settings.tol, settings.max_iter, settings.rank, info);
            break;
        case KrylovAlgorithm::LegacyCg:
            legacy_cg(op, b, x, settings.tol, settings.rank, info);
            break;
        case KrylovAlgorithm::LegacyCr:
            legacy_cr(op, b, x, settings.tol, settings.rank, info);
            break;
        case KrylovAlgorithm::RestartedGmres:
            info.scipy_semantics = true;
            scipy_gmres(op, b, x, settings.tol, std::max<std::int64_t>(1, std::min(settings.rank, n)),
                        std::max<std::int64_t>(1, settings.max_iter), info);
            break;
        case KrylovAlgorithm::Cg:
            info.scipy_semantics = true;
            scipy_cg(op, b, x, settings.tol, std::max<std::int64_t>(1, settings.rank), info);
            break;
        case KrylovAlgorithm::Minres:
            info.scipy_semantics = true;
            if constexpr (std::is_same_v<Scalar, double>) {
                minres(op, b, x, settings.tol, std::max<std::int64_t>(1, settings.rank), info);
            }
            break;
    }
    if (info.scipy_semantics) {
        // The reference recomputes the residual of whatever SciPy returned.
        std::vector<Scalar> work(static_cast<std::size_t>(n));
        info.relative_residual = relative_residual(op, b, x, norm2(static_cast<std::size_t>(n), b), work);
    }
    info.threads_used = op.threads_used();
    return info;
}

template <class Scalar>
KrylovInfo krylov_solve(const CsrMatrix<Scalar>& A, const Scalar* b, Scalar* x, KrylovRoute route,
                        const KrylovSettings& settings) {
    const KrylovAlgorithm algorithm = select_krylov_algorithm(route, A.n, !std::is_same_v<Scalar, double>);
    return run_krylov_algorithm(algorithm, A, b, x, settings);
}

template KrylovInfo krylov_solve<double>(const CsrMatrix<double>&, const double*, double*, KrylovRoute,
                                         const KrylovSettings&);
template KrylovInfo krylov_solve<Complex>(const CsrMatrix<Complex>&, const Complex*, Complex*, KrylovRoute,
                                          const KrylovSettings&);
template KrylovInfo run_krylov_algorithm<double>(KrylovAlgorithm, const CsrMatrix<double>&, const double*, double*,
                                                 const KrylovSettings&);
template KrylovInfo run_krylov_algorithm<Complex>(KrylovAlgorithm, const CsrMatrix<Complex>&, const Complex*,
                                                  Complex*, const KrylovSettings&);

}  // namespace mna::linear
