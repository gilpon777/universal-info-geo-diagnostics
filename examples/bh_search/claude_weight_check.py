#!/usr/bin/env python3
"""Claude v12 補助: osc_proj の重み依存性の再現スクリプト（固定対象外）。
Muse 返信 v15 §2 の依頼（点の選択条件・重みの関数形を明記）への回答。
入力: Muse の osc_proj 表（gw_osc_diagnostic.md の 19 点。下に埋め込み）。 依存: numpy, mpmath。
定義:
  F(w,y): 点質量レンズの増幅因子（Takahashi & Nakamura）、w = 8π G M f / c^3、G M_sun/c^3 = 4.925491e-6 s。
  f 格子: 20..448 Hz、刻み 1 Hz（429 点）。点の選択: **19 点すべて**（有意点などの絞り込みなし）。
  osc_proj(weight): x=F-1 を {1, (f-⟨f⟩_w)} の複素張る空間へ重み付き最小二乗で射影した残差の重み付き RMS。
  ばらつき = ln(予測 osc_proj / Muse の表) の標準偏差（np.std, ddof=0）。 平均比 = exp(平均)。
重み 1（推定）: w(f) = f^-2 · exp(-(f/60)^2)、20–448 Hz
重み 2（実測の重心・幅のガウス型）: w(f) = exp(-0.5((f-51.25)/11.8)^2)、20–448 Hz
期待される出力: 重み1 → ばらつき 0.080 / 重み2 → ばらつき 0.226（v11 §2 の数値）
"""
import numpy as np, mpmath as mp
G = 4.925491e-6
mp.mp.dps = 20
OSC = {(100,0.3):0.02080,(100,0.5):0.02699,(100,1.0):0.03856,(100,2.0):0.05388,(200,0.3):0.05931,(200,0.5):0.08193,
       (200,1.0):0.12111,(200,2.0):0.12028,(400,0.3):0.16400,(400,0.5):0.23536,(400,1.0):0.27496,(400,2.0):0.16557,
       (1000,0.3):0.56326,(1000,0.5):0.59858,(1000,1.0):0.40414,(1000,2.0):0.17362,(2000,0.3):0.93357,(2000,0.5):0.75135,(500,0.5):0.31622}

def F(w, y):
    xm = (y + np.sqrt(y*y + 4)) / 2
    phi = (xm - y)**2 / 2 - np.log(xm)
    w = mp.mpf(w); y = mp.mpf(y)
    return complex(mp.e**(mp.pi*w/4 + 1j*(w/2)*(mp.log(w/2) - 2*phi)) * mp.gamma(1 - 1j*w/2) * mp.hyp1f1(1j*w/2, 1, 1j*w*y*y/2))

f = np.arange(20, 448.5, 1.0)

def osc_proj(Fv, wfun):
    x = Fv - 1
    W = wfun(f); W = W / W.sum()
    A = np.vstack([np.ones_like(f), f - (W*f).sum()]).T.astype(complex)
    sw = np.sqrt(W)[:, None]
    c = np.linalg.lstsq(A*sw, x*sw[:, 0], rcond=None)[0]
    r = x - A @ c
    return np.sqrt((W*np.abs(r)**2).sum())

if __name__ == "__main__":
    keys = list(OSC)
    Ft = {k: np.array([F(8*np.pi*G*k[0]*ff, k[1]) for ff in f]) for k in keys}
    rv = np.array([OSC[k] for k in keys])
    for name, wf in (("重み1: f^-2 exp(-(f/60)^2)", lambda ff: ff**-2.0*np.exp(-(ff/60.0)**2)),
                     ("重み2: ガウス 51.25±11.8 Hz", lambda ff: np.exp(-0.5*((ff-51.25)/11.8)**2))):
        v = np.array([osc_proj(Ft[k], wf) for k in keys]); lr = np.log(v/rv)
        print(f"{name:32s} n={len(keys)}  ばらつき(ln比の標準偏差)={np.std(lr):.3f}  平均比={np.exp(lr.mean()):.2f}  最小/最大比={np.exp(lr.min()):.2f}/{np.exp(lr.max()):.2f}")
    # 重みの重心・幅
    for name, wf in (("重み1", lambda ff: ff**-2.0*np.exp(-(ff/60.0)**2)), ("重み2", lambda ff: np.exp(-0.5*((ff-51.25)/11.8)**2))):
        W = wf(f); W = W/W.sum(); c = (W*f).sum(); s = np.sqrt((W*(f-c)**2).sum()); cdf = np.cumsum(W)
        print(f"{name}: 重心 {c:.1f} Hz, 標準偏差 {s:.1f} Hz, 5–95% {f[np.searchsorted(cdf,0.05)]:.0f}–{f[np.searchsorted(cdf,0.95)]:.0f} Hz")
