import numpy as np
from numba import njit

@njit(cache=False, fastmath=False)
def eval_mos_numba(x, drain, gate, source, beta, vth, lam, sign, n):
    f = np.zeros((n, 1)); J = np.zeros((n, n))
    for k in range(drain.shape[0]):
        d, g, s, sg = drain[k], gate[k], source[k], sign[k]
        vd = 0.0 if d < 0 else sg * x[d]; vg = 0.0 if g < 0 else sg * x[g]; vs = 0.0 if s < 0 else sg * x[s]
        if vd >= vs:
            vgs, vds = vg - vs, vd - vs
        else:
            vgs, vds = vg - vd, vs - vd
        I = 0.0; dgs = 0.0; dds = 0.0
        if vgs > vth[k]:
            vov = vgs - vth[k]; mod = 1.0 + lam[k] * vds
            if vds < vov:
                ch = vov * vds - 0.5 * vds * vds
                I = beta[k] * ch * mod; dgs = beta[k] * vds * mod; dds = beta[k] * ((vov - vds) * mod + ch * lam[k])
            else:
                I = 0.5 * beta[k] * vov * vov * mod; dgs = beta[k] * vov * mod; dds = 0.5 * beta[k] * vov * vov * lam[k]
        if vd >= vs:
            cur = I; dd = dds; dg = dgs; ds = -dgs - dds
        else:
            cur = -I; dd = dgs + dds; dg = -dgs; ds = -dds
        cur *= sg
        if d >= 0:
            f[d, 0] += cur
            J[d, d] += dd
            if g >= 0: J[d, g] += dg
            if s >= 0: J[d, s] += ds
        if s >= 0:
            f[s, 0] -= cur
            if d >= 0: J[s, d] -= dd
            if g >= 0: J[s, g] -= dg
            J[s, s] -= ds
    return f, J

eval_mos_numba_nogil = njit(nogil=True, cache=False)(eval_mos_numba.py_func)


@njit(nogil=True, cache=False)
def eval_mos_coo_numba(x, drain, gate, source, beta, vth, lam, sign, n):
    """f plus 6 Jacobian values per device (same layout as the C++ COO kernel)."""
    m = drain.shape[0]; f = np.zeros((n, 1)); vals = np.empty(6 * m)
    for k in range(m):
        d, g, s, sg = drain[k], gate[k], source[k], sign[k]
        vd = 0.0 if d < 0 else sg * x[d]; vg = 0.0 if g < 0 else sg * x[g]; vs = 0.0 if s < 0 else sg * x[s]
        if vd >= vs:
            vgs, vds = vg - vs, vd - vs
        else:
            vgs, vds = vg - vd, vs - vd
        I = 0.0; dgs = 0.0; dds = 0.0
        if vgs > vth[k]:
            vov = vgs - vth[k]; mod = 1.0 + lam[k] * vds
            if vds < vov:
                ch = vov * vds - 0.5 * vds * vds
                I = beta[k] * ch * mod; dgs = beta[k] * vds * mod; dds = beta[k] * ((vov - vds) * mod + ch * lam[k])
            else:
                I = 0.5 * beta[k] * vov * vov * mod; dgs = beta[k] * vov * mod; dds = 0.5 * beta[k] * vov * vov * lam[k]
        if vd >= vs:
            cur = I; dd = dds; dg = dgs; ds = -dgs - dds
        else:
            cur = -I; dd = dgs + dds; dg = -dgs; ds = -dds
        cur *= sg
        if d >= 0: f[d, 0] += cur
        if s >= 0: f[s, 0] -= cur
        vals[6 * k] = dd; vals[6 * k + 1] = dg; vals[6 * k + 2] = ds; vals[6 * k + 3] = -dd; vals[6 * k + 4] = -dg; vals[6 * k + 5] = -ds
    return f, vals

eval_mos_coo_numba_gil = njit(nogil=False, cache=False)(eval_mos_coo_numba.py_func)


@njit(nogil=True, cache=False)
def eval_mos_coo_batch_numba(X, drain, gate, source, beta, vth, lam, sign, n):
    """Loop over rows of X inside one nogil call (coarse-grained GIL release)."""
    rows = X.shape[0]; m = drain.shape[0]
    F = np.zeros((rows, n)); V = np.empty((rows, 6 * m))
    for r in range(rows):
        f, vals = eval_mos_coo_numba(X[r], drain, gate, source, beta, vth, lam, sign, n)
        F[r, :] = f[:, 0]; V[r, :] = vals
    return F, V
