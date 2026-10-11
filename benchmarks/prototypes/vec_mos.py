"""Vectorized NumPy prototype of _evaluate_level1_mos_devices (structure-of-arrays)."""
import numpy as np

class MosTable:
    """Pack level1 MOS device dicts into arrays once; ground (-1) maps to a zero slot at index n."""
    def __init__(self, devices, n):
        self.n = n
        m = len(devices)
        g = lambda key: np.array([int(d[key]) for d in devices], dtype=np.int64)
        self.drain, self.gate, self.source = g("drain"), g("gate"), g("source")
        self.beta = np.array([float(d["beta"]) for d in devices]); self.vth = np.array([float(d["vth"]) for d in devices]); self.lam = np.array([float(d["lambda"]) for d in devices])
        self.sign = np.where(np.array([str(d["type"]).upper() == "PMOS" for d in devices]), -1.0, 1.0)
        z = lambda idx: np.where(idx < 0, n, idx)           # -1 -> zero slot
        self.zd, self.zg, self.zs = z(self.drain), z(self.gate), z(self.source)
        # dense Jacobian flat indices for the 6 stamps: rows (drain,+), (source,-); cols (drain,gate,source)
        rows = np.concatenate([self.zd, self.zd, self.zd, self.zs, self.zs, self.zs])
        cols = np.concatenate([self.zd, self.zg, self.zs, self.zd, self.zg, self.zs])
        self.jsign = np.concatenate([np.ones(3 * m), -np.ones(3 * m)])
        self.valid = (rows < n) & (cols < n)
        self.flat = (rows * n + cols)[self.valid]
        self.f_rows = np.concatenate([self.zd, self.zs]); self.f_sign = np.concatenate([np.ones(m), -np.ones(m)])
        self.f_valid = self.f_rows < n

    def evaluate(self, x):
        n = self.n
        xe = np.append(np.asarray(x, dtype=float).ravel(), 0.0)
        s = self.sign
        vd, vg, vs = s * xe[self.zd], s * xe[self.zg], s * xe[self.zs]
        fwd = vd >= vs
        vgs = np.where(fwd, vg - vs, vg - vd)
        vds = np.where(fwd, vd - vs, vs - vd)
        vov = vgs - self.vth
        on = vgs > self.vth
        triode = on & (vds < vov)
        sat = on & ~triode
        mod = 1.0 + self.lam * vds
        ch = vov * vds - 0.5 * vds * vds
        I = np.where(triode, self.beta * ch * mod, 0.0) + np.where(sat, 0.5 * self.beta * vov * vov * mod, 0.0)
        dgs = np.where(triode, self.beta * vds * mod, 0.0) + np.where(sat, self.beta * vov * mod, 0.0)
        dds = np.where(triode, self.beta * ((vov - vds) * mod + ch * self.lam), 0.0) + np.where(sat, 0.5 * self.beta * vov * vov * self.lam, 0.0)
        cur = np.where(fwd, I, -I)
        d_d = np.where(fwd, dds, dgs + dds)
        d_g = np.where(fwd, dgs, -dgs)
        d_s = np.where(fwd, -dgs - dds, -dds)
        cur = s * cur  # PMOS: current negated, derivatives unchanged
        f = np.bincount(self.f_rows[self.f_valid], weights=(self.f_sign * np.concatenate([cur, cur]))[self.f_valid], minlength=n + 1)[:n].reshape(n, 1)
        vals = (self.jsign * np.concatenate([d_d, d_g, d_s, d_d, d_g, d_s]))[self.valid]
        J = np.bincount(self.flat, weights=vals, minlength=n * n).reshape(n, n)
        return f, J
