"""波長選択コード BH 版 — v3 フレームワークで作り直し。

wavelength_sweep.py はバンドごとの検出限界距離の計算だった。
本コードは wavelength_sed.py (銀河 SED 版) と同じ骨格で、
観測点 = 測光バンド、モデル = BH 連星の SED の成分分解とし、
FIM の貪欲選択を通す。どのバンドから観れば系の成分が
一番早く確定するか、を出す。

基準天体は静穏時 BH 連星 (A0620-00 型)。成分は 5 つ:

  companion  伴星 (K 型矮星、黒体、T は自由パラメータ)
  disk       降着円盤 (nu^1/3 則、内外にカットオフ)
  hot spot   輝点 (9000 K 黒体、光学の余分な光)
  jet        ジェット (平坦 -> 折れ曲がりべき)
  corona     コロナ/ADAF (X 線のみのべき)

theta = (lnA_comp, T_comp[1000K], lnA_disk, lnA_spot,
         lnA_jet, lnA_corona)。
静穏時の基準振幅は A0620-00 の実測 SED の階層に合わせた
(近赤外は伴星が約 9 割、V では円盤が約 3 割、電波は
ジェットのみ、X 線はコロナのみ)。
ノイズはバンドごとの現実的な測光精度 (相対誤差) + 絶対床。
2 つ目の基準点としてハード状態 (GX 339-4 型) でも走らせ、
貪欲の順位が状態で反転するかを見る。

検証: (V1) テンプレートのピーク位置、(V2) 振幅だけの
線形部分問題でフレームワーク FIM vs 解析 FIM。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json
import sys

import numpy as np

import os as _os
_fw = _os.environ.get("INFO_GEO_DIAG_DIR")
if not _fw:
    for _cand in (_Path(__file__).resolve().parents[3],
                  _Path(__file__).resolve().parents[3] / "info_geo_diag",
                  _Path.home() / "workspace" / "info_geo_diag"):
        if (_cand / "universal_info_geo_diagnostics_v3.py").exists():
            _fw = str(_cand)
            break
if _fw:
    sys.path.insert(0, _fw)
from universal_info_geo_diagnostics_v3 import (  # noqa: E402
    UniversalInformationGeometricDiagnosticsV3 as Diag)

OUT = str(HERE)
h, kB = 6.62607015e-34, 1.380649e-23
c = 2.99792458e8

# バンド: (名前, 周波数 Hz, 相対測光誤差)
BANDS = [
    ("radio 150MHz", 1.50e8, 0.10), ("radio 1.4GHz", 1.40e9, 0.10),
    ("radio 5GHz", 5.00e9, 0.10),
    ("FIR 250um", 1.20e12, 0.30), ("FIR 160um", 1.87e12, 0.30),
    ("FIR 100um", 3.00e12, 0.30), ("FIR 70um", 4.28e12, 0.30),
    ("MIR 24um", 1.25e13, 0.20), ("MIR 8um", 3.75e13, 0.20),
    ("NIR 3.6um", 8.33e13, 0.03), ("NIR 2.2um", 1.36e14, 0.03),
    ("NIR 1.2um", 2.50e14, 0.03),
    ("optical 900nm", 3.33e14, 0.02), ("optical 550nm", 5.45e14, 0.02),
    ("optical 440nm", 6.81e14, 0.02),
    ("NUV 230nm", 1.30e15, 0.08), ("FUV 150nm", 2.00e15, 0.15),
    ("X-ray 0.5keV", 1.21e17, 0.30), ("X-ray 1keV", 2.42e17, 0.30),
    ("X-ray 2keV", 4.84e17, 0.30), ("X-ray 5keV", 1.21e18, 0.35),
    ("X-ray 10keV", 2.42e18, 0.40),
]
NU = np.array([b[1] for b in BANDS])
NAMES = [b[0] for b in BANDS]
REL = np.array([b[2] for b in BANDS])
FLOOR = 0.001


def Bnu(nu, T):
    x = h * nu / (kB * T)
    return nu ** 3 / np.expm1(np.minimum(x, 300.0))


def bb(nu, T, ref_nu):
    return Bnu(nu, T) / float(Bnu(np.array([ref_nu]), T)[0])


def t_comp(nu, T_K):
    return bb(nu, T_K * 1000.0, 1.36e14)


def t_disk(nu):
    # nu^1/3 則。内縁 (T_in=3e4 K) で指数カット、外縁側は
    # nu_out=5e13 Hz 以下で nu^2 的に落ちる (代理テンプレート)
    Tin = 3.0e4
    raw = nu ** (1.0 / 3.0) * np.exp(-h * nu / (kB * Tin))
    raw = raw * np.minimum(1.0, (nu / 5.0e13) ** 2)
    ref = (1.30e15) ** (1.0 / 3.0) * np.exp(-h * 1.30e15 / (kB * Tin))
    return raw / ref


def t_spot(nu):
    return bb(nu, 9000.0, 6.81e14)


NU_BR = 3.0e13          # ジェットの折れ曲がり (固定した仮定)


def t_jet(nu):
    return np.where(nu <= NU_BR, 1.0, (nu / NU_BR) ** -0.7)


def t_corona(nu):
    # X 線帯だけに載る (sed 版 AGN と同じ教訓: べきを全域に
    # 伸ばすと他成分を呑むテンプレートの人工物になる)
    return np.where(nu >= 2.42e16, (nu / 2.42e17) ** -0.7, 0.0)


def templates_matrix(T_K=4.5):
    cols = [t_comp(NU, T_K), t_disk(NU), t_spot(NU), t_jet(NU),
            t_corona(NU)]
    return np.stack(cols, axis=1)


COMP = ["companion", "disk", "hot spot", "jet", "corona"]
# 静穏時 (A0620 型) とハード状態 (GX 339-4 型) の基準振幅
AMP_Q = np.array([1.0, 0.12, 0.10, 0.06, 0.80])
AMP_H = np.array([0.7, 1.20, 0.30, 1.20, 3.00])
THETA_Q = np.array([0.0, 4.5, np.log(0.12), np.log(0.10),
                    np.log(0.06), np.log(0.80)])
THETA_H = np.array([np.log(0.7), 4.5, np.log(1.20), np.log(0.30),
                    np.log(1.20), np.log(3.00)])


def make_model():
    def model(theta):
        lnAc, T_K, lnAd, lnAs, lnAj, lnAcor = theta
        return (np.exp(lnAc) * t_comp(NU, T_K)
                + np.exp(lnAd) * t_disk(NU)
                + np.exp(lnAs) * t_spot(NU)
                + np.exp(lnAj) * t_jet(NU)
                + np.exp(lnAcor) * t_corona(NU))
    return model


MODEL = make_model()


def cov_of(theta):
    F = MODEL(theta)
    sig = REL * np.abs(F) + FLOOR
    return np.diag(sig ** 2), sig


def run_state(theta0, amp0, label):
    cov, sig = cov_of(theta0)
    d = Diag(MODEL)
    res = d.evaluate(theta0, covariance=cov)
    recs = d.greedy_select(theta0, cov, max_points=10)
    greedy = [{"band": NAMES[r["Index"]], "driver": r["Driver"],
               "gain_margin": float(r["GainMargin"]),
               "logdet_gain": float(r["LogdetGain"])} for r in recs]
    # 振幅だけの解析 FIM (T 固定) で CR sigma と相関
    J = templates_matrix(theta0[1])
    Jln = J * amp0[None, :]
    W = Jln / sig[:, None]
    FIM = W.T @ W
    covm = np.linalg.pinv(FIM)
    s = np.sqrt(np.diag(covm))
    corr = covm / np.outer(s, s)
    pairs = []
    for i in range(len(COMP)):
        for jx in range(i + 1, len(COMP)):
            if abs(corr[i, jx]) > 0.3:
                pairs.append((COMP[i], COMP[jx], float(corr[i, jx])))
    print(f"\n=== {label} ===")
    print("Scaled r_eff:", round(float(res["Scaled"]["EffectiveRank"]), 3),
          " Raw r_eff:", round(float(res["Raw"]["EffectiveRank"]), 3),
          " (of 6 params)")
    print("Scaled direction SNR:",
          np.round(res["Absolute"]["Scaled"]["DirectionSNR"], 2))
    print("Greedy:", [(g["band"], g["driver"]) for g in greedy])
    print("CR sigma (ln amp):",
          {c_: round(float(v), 3) for c_, v in zip(COMP, s)})
    print("strong correlations:", pairs)
    return {"label": label,
            "scaled_r_eff": float(res["Scaled"]["EffectiveRank"]),
            "raw_r_eff": float(res["Raw"]["EffectiveRank"]),
            "direction_snr": res["Absolute"]["Scaled"]["DirectionSNR"].tolist(),
            "greedy": greedy,
            "cr_sigma_lnamp": {c_: float(v) for c_, v in zip(COMP, s)},
            "correlations": pairs}


def main():
    out = {}
    # V1: テンプレートの形
    nu_fine = np.logspace(7, 19, 8000)
    pk = nu_fine[np.argmax(bb(nu_fine, 4500.0, 1.36e14))]
    print(f"V1 companion 4500K: Bnu peak {c / pk * 1e6:.2f} um "
          f"(expect ~1.1 um)")
    pk2 = nu_fine[np.argmax(bb(nu_fine, 9000.0, 6.81e14))]
    print(f"V1 hot spot 9000K: Bnu peak {c / pk2 * 1e6:.2f} um "
          f"(expect ~0.57 um)")
    assert t_jet(np.array([1.5e8]))[0] == 1.0
    assert abs(t_corona(np.array([2.42e17]))[0] - 1.0) < 1e-12
    assert t_corona(np.array([1.0e15]))[0] == 0.0
    print("V1 jet flat below break, corona X-ray-only: OK")

    # V2: 振幅だけの線形部分問題、解析 FIM vs フレームワーク
    cov, sig = cov_of(THETA_Q)
    J = templates_matrix()
    Jln = J * AMP_Q[None, :]
    FIMa = Jln.T @ np.diag(1.0 / sig ** 2) @ Jln
    d_lin = Diag(lambda la: templates_matrix() @ np.exp(la))
    res_lin = d_lin.evaluate(np.log(AMP_Q), covariance=cov)
    ev_a = np.sort(np.linalg.eigvalsh(FIMa))[::-1]
    ev_f = np.sort(res_lin["Raw"]["Eigenvalues"])[::-1]
    rel = float(np.max(np.abs(ev_a - ev_f) / ev_a))
    print(f"V2 max rel diff (ln-amp FIM): {rel:.2e}")
    out["validation"] = {"v2_max_rel_diff": rel}

    out["quiescent"] = run_state(THETA_Q, AMP_Q, "quiescent (A0620-like)")
    out["hard"] = run_state(THETA_H, AMP_H, "hard state (GX 339-4-like)")

    # 図: 静穏時 SED 分解 + 両状態の貪欲ピック
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = c / NU * 1e6
    Fq = MODEL(THETA_Q)
    J = templates_matrix()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for k, cname in enumerate(COMP):
        ax.plot(lam, J[:, k] * [AMP_Q, AMP_H][0][k], "--", lw=1,
                label=cname)
    ax.plot(lam, Fq, "k-", lw=2, label="total (quiescent)")
    ax.scatter(lam, Fq, c="k", s=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (um) — radio at left, X-ray at right")
    ax.set_ylabel("flux (template units, companion NIR = 1)")
    ax.set_title("Quiescent BH binary SED decomposition — "
                 "what each band carries")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/wavelength_bh.png", dpi=130)
    with open(f"{OUT}/wavelength_bh_numbers.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved wavelength_bh.png, wavelength_bh_numbers.json")


if __name__ == "__main__":
    main()
