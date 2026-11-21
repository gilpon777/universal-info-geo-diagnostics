#!/usr/bin/env python3
"""osc_proj への周波数別寄与率 C(f) の計算（ChatGPT 第12回相当の提案、2026-11-21 Muse）。
C(f) = w(f)|r(f)|^2 / Σ_f w(f)|r(f)|^2 。r(f) は F-1 を {1, f} へ射影した残差。
SNR 重み w(f) の集中（重心 51.25 Hz）と、osc_proj に実際に寄与する周波数を分けて比べる。
実行: wf_env/bin/python gw_osc_contribution.py（gw_osc_diagnostic.py の重み構築を再利用）
"""
import numpy as np, gw_osc_diagnostic as G
FB, W, WSUM = G.FB, G.W, G.WSUM

def pct(p, x, w):
    i = np.argsort(x); cw = np.cumsum(w[i]); cw /= cw[-1]
    return float(np.interp(p, cw, x[i]))

def summ(x, w):
    w = w / w.sum()
    mu = float(np.sum(w * x)); sd = float(np.sqrt(np.sum(w * (x - mu) ** 2)))
    return mu, sd, pct(0.05, x, w), pct(0.95, x, w)

if __name__ == "__main__":
    print("w(f): centroid %.2f sd %.2f 5-95%% %.1f-%.1f" % summ(FB, W))
    for (M, y) in [(100, 0.3), (400, 1.0), (1000, 0.3), (1000, 0.5), (2000, 0.3)]:
        F, ierr = G.lens_F(M, y, G.FREQS)
        Fb = F[G.BAND]; v = Fb - 1.0
        S1 = float(np.sum(W * FB)); S2 = float(np.sum(W * FB ** 2))
        Gm = np.array([[WSUM, S1], [S1, S2]])
        rhs = np.array([np.sum(W * v), np.sum(W * FB * v)], dtype=complex)
        a, b = np.linalg.solve(Gm, rhs)
        resid = v - (a + b * FB)
        C = W * np.abs(resid) ** 2
        mu, sd, lo, hi = summ(FB, C)
        print(f"C(f) M={M} y={y}: centroid {mu:.2f} Hz, sd {sd:.2f}, 5-95% {lo:.1f}-{hi:.1f}")
