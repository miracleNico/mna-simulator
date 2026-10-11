// Prototype only: see benchmarks/prototypes/README.md for the build command.
// Minimal pybind11 PoC: Level-1 MOS evaluation + dense Jacobian stamping in C++.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
namespace py = pybind11;

static inline void nmos_forward(double vgs, double vds, double beta, double vth, double lam,
                                double &I, double &dgs, double &dds) {
    if (vgs <= vth) { I = dgs = dds = 0.0; return; }
    double vov = vgs - vth;
    double mod = 1.0 + lam * vds;
    if (vds < vov) {
        double ch = vov * vds - 0.5 * vds * vds;
        I = beta * ch * mod; dgs = beta * vds * mod; dds = beta * ((vov - vds) * mod + ch * lam);
    } else {
        I = 0.5 * beta * vov * vov * mod; dgs = beta * vov * mod; dds = 0.5 * beta * vov * vov * lam;
    }
}

py::tuple eval_mos(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                   py::array_t<long long> drain, py::array_t<long long> gate, py::array_t<long long> source,
                   py::array_t<double> beta, py::array_t<double> vth, py::array_t<double> lam,
                   py::array_t<double> sign, py::ssize_t n) {
    auto X = x.unchecked<1>(); auto D = drain.unchecked<1>(); auto Gt = gate.unchecked<1>(); auto S = source.unchecked<1>();
    auto B = beta.unchecked<1>(); auto VT = vth.unchecked<1>(); auto L = lam.unchecked<1>(); auto SG = sign.unchecked<1>();
    py::array_t<double> f({n, (py::ssize_t)1}); py::array_t<double> J({n, n});
    auto F = f.mutable_unchecked<2>(); auto JJ = J.mutable_unchecked<2>();
    std::fill(f.mutable_data(), f.mutable_data() + n, 0.0); std::fill(J.mutable_data(), J.mutable_data() + n * n, 0.0);
    const py::ssize_t m = D.shape(0);
    {
        py::gil_scoped_release release;   // pure numeric loop: no Python objects touched
        for (py::ssize_t k = 0; k < m; ++k) {
            long long d = D(k), g = Gt(k), s = S(k); double sg = SG(k);
            double vd = d < 0 ? 0.0 : sg * X(d), vg = g < 0 ? 0.0 : sg * X(g), vs = s < 0 ? 0.0 : sg * X(s);
            double I, dgs, dds, cur, dd, dg, ds;
            if (vd >= vs) { nmos_forward(vg - vs, vd - vs, B(k), VT(k), L(k), I, dgs, dds); cur = I; dd = dds; dg = dgs; ds = -dgs - dds; }
            else { nmos_forward(vg - vd, vs - vd, B(k), VT(k), L(k), I, dgs, dds); cur = -I; dd = dgs + dds; dg = -dgs; ds = -dds; }
            cur *= sg;
            if (d >= 0) { F(d, 0) += cur; if (d >= 0) JJ(d, d) += dd; if (g >= 0) JJ(d, g) += dg; if (s >= 0) JJ(d, s) += ds; }
            if (s >= 0) { F(s, 0) -= cur; if (d >= 0) JJ(s, d) -= dd; if (g >= 0) JJ(s, g) -= dg; if (s >= 0) JJ(s, s) -= ds; }
        }
    }
    return py::make_tuple(f, J);
}


// COO variant: f plus 6 Jacobian values per device (rows: drain(+), source(-); cols: drain, gate, source).
// Entries that touch ground stay 0; the caller maps the fixed triplet layout onto a CSR pattern once.
py::tuple eval_mos_coo(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                       py::array_t<long long> drain, py::array_t<long long> gate, py::array_t<long long> source,
                       py::array_t<double> beta, py::array_t<double> vth, py::array_t<double> lam,
                       py::array_t<double> sign, py::ssize_t n) {
    auto X = x.unchecked<1>(); auto D = drain.unchecked<1>(); auto Gt = gate.unchecked<1>(); auto S = source.unchecked<1>();
    auto B = beta.unchecked<1>(); auto VT = vth.unchecked<1>(); auto L = lam.unchecked<1>(); auto SG = sign.unchecked<1>();
    const py::ssize_t m = D.shape(0);
    py::array_t<double> f({n, (py::ssize_t)1}); py::array_t<double> vals(6 * m);
    auto F = f.mutable_unchecked<2>(); auto V = vals.mutable_unchecked<1>();
    std::fill(f.mutable_data(), f.mutable_data() + n, 0.0);
    {
        py::gil_scoped_release release;
        for (py::ssize_t k = 0; k < m; ++k) {
            long long d = D(k), g = Gt(k), s = S(k); double sg = SG(k);
            double vd = d < 0 ? 0.0 : sg * X(d), vg = g < 0 ? 0.0 : sg * X(g), vs = s < 0 ? 0.0 : sg * X(s);
            double I, dgs, dds, cur, dd, dg, ds;
            if (vd >= vs) { nmos_forward(vg - vs, vd - vs, B(k), VT(k), L(k), I, dgs, dds); cur = I; dd = dds; dg = dgs; ds = -dgs - dds; }
            else { nmos_forward(vg - vd, vs - vd, B(k), VT(k), L(k), I, dgs, dds); cur = -I; dd = dgs + dds; dg = -dgs; ds = -dds; }
            cur *= sg;
            if (d >= 0) F(d, 0) += cur;
            if (s >= 0) F(s, 0) -= cur;
            V(6 * k + 0) = dd; V(6 * k + 1) = dg; V(6 * k + 2) = ds; V(6 * k + 3) = -dd; V(6 * k + 4) = -dg; V(6 * k + 5) = -ds;
        }
    }
    return py::make_tuple(f, vals);
}

// Coarse-grained variants: evaluate many state vectors in one call. The GIL is released once for the
// whole batch (so Python threads can overlap long native regions), and the _omp variant additionally
// parallelizes across rows with OpenMP inside a single call (no Python threads needed at all).
static void eval_rows(const double *X, py::ssize_t rows, py::ssize_t n, const long long *D, const long long *Gt,
                      const long long *S, const double *B, const double *VT, const double *L, const double *SG,
                      py::ssize_t m, double *F, double *V, bool use_omp) {
#pragma omp parallel for schedule(static) if (use_omp)
    for (py::ssize_t r = 0; r < rows; ++r) {
        const double *x = X + r * n; double *f = F + r * n; double *v = V + r * 6 * m;
        for (py::ssize_t k = 0; k < m; ++k) {
            long long d = D[k], g = Gt[k], s = S[k]; double sg = SG[k];
            double vd = d < 0 ? 0.0 : sg * x[d], vg = g < 0 ? 0.0 : sg * x[g], vs = s < 0 ? 0.0 : sg * x[s];
            double I, dgs, dds, cur, dd, dg, ds;
            if (vd >= vs) { nmos_forward(vg - vs, vd - vs, B[k], VT[k], L[k], I, dgs, dds); cur = I; dd = dds; dg = dgs; ds = -dgs - dds; }
            else { nmos_forward(vg - vd, vs - vd, B[k], VT[k], L[k], I, dgs, dds); cur = -I; dd = dgs + dds; dg = -dgs; ds = -dds; }
            cur *= sg;
            if (d >= 0) f[d] += cur;
            if (s >= 0) f[s] -= cur;
            v[6 * k] = dd; v[6 * k + 1] = dg; v[6 * k + 2] = ds; v[6 * k + 3] = -dd; v[6 * k + 4] = -dg; v[6 * k + 5] = -ds;
        }
    }
}

py::tuple eval_mos_coo_batch(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                             py::array_t<long long> drain, py::array_t<long long> gate, py::array_t<long long> source,
                             py::array_t<double> beta, py::array_t<double> vth, py::array_t<double> lam,
                             py::array_t<double> sign, bool use_omp) {
    const py::ssize_t rows = X.shape(0), n = X.shape(1), m = drain.shape(0);
    py::array_t<double> F({rows, n}); py::array_t<double> V({rows, 6 * m});
    std::fill(F.mutable_data(), F.mutable_data() + rows * n, 0.0);
    {
        py::gil_scoped_release release;
        eval_rows(X.data(), rows, n, drain.data(), gate.data(), source.data(), beta.data(), vth.data(), lam.data(),
                  sign.data(), m, F.mutable_data(), V.mutable_data(), use_omp);
    }
    return py::make_tuple(F, V);
}

// Same as eval_mos_coo_batch but writes into caller-owned buffers (no allocation inside the call), which is
// how a production kernel would behave inside a Newton loop that reuses its CSR value array.
void eval_mos_coo_batch_into(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                             py::array_t<long long> drain, py::array_t<long long> gate, py::array_t<long long> source,
                             py::array_t<double> beta, py::array_t<double> vth, py::array_t<double> lam,
                             py::array_t<double> sign, py::array_t<double, py::array::c_style> F,
                             py::array_t<double, py::array::c_style> V, bool use_omp) {
    const py::ssize_t rows = X.shape(0), n = X.shape(1), m = drain.shape(0);
    if (F.shape(0) != rows || F.shape(1) != n || V.shape(0) != rows || V.shape(1) != 6 * m)
        throw std::invalid_argument("output buffers have the wrong shape");
    py::gil_scoped_release release;
    std::fill(F.mutable_data(), F.mutable_data() + rows * n, 0.0);
    eval_rows(X.data(), rows, n, drain.data(), gate.data(), source.data(), beta.data(), vth.data(), lam.data(),
              sign.data(), m, F.mutable_data(), V.mutable_data(), use_omp);
}

PYBIND11_MODULE(mos_kernel, m) {
    m.def("eval_mos_coo_batch_into", &eval_mos_coo_batch_into, "Batch evaluation into preallocated F and V buffers");
    m.def("eval_mos_coo_batch", &eval_mos_coo_batch, "Batch of state vectors; use_omp parallelizes rows with OpenMP");
    m.def("eval_mos_coo", &eval_mos_coo, "Level-1 MOS f and COO Jacobian values"); m.def("eval_mos", &eval_mos, "Level-1 MOS f and dense Jacobian"); }
