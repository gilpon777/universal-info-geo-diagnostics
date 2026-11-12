"""FIRAS 平均スペクトルで μ 歪みを自分で測る。

背景 (2026-11-12 の議論): スープ時代のイベント (例: BF の分解) が
熱を出していたなら、その痕跡は温度地図の「濃い点」ではなく
スペクトル全体の形 (μ 歪み) に積分として残る。ここでは公開 FIRAS
単極スペクトルに 黒体 + ΔT + μ + y + 銀河成分 のテンプレートを
重み付き最小二乗で当て、μ と、その 2σ 上限から「晴れ上がり前に
注入できた熱の総量の予算」Δu/u ≈ μ/1.4 を出す。

入力: data/firas_spectrum.csv (nu_GHz, I_MJy_sr, sigma_MJy_sr)。
      取得した LAMBDA の表から変換して作る (変換元はレポートに明記)。
テンプレート (占有数ベース、x = hν/kT):
  ΔT:  x e^x/(e^x-1)^2            (ΔT/T あたり)
  μ :  -e^x/(e^x-1)^2             (μ あたり、T 固定の微分)
  y :  [x e^x/(e^x-1)^2] [x(e^x+1)/(e^x-1) - 4]
銀河: 固定した修正黒体 (T_d=18 K, beta=1.6) の強度テンプレート、振幅自由。
フィット範囲は CMB が支配的な 60-630 GHz を主とし、全域でも確認する。
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
H_PLANCK = 6.62607015e-34
K_B = 1.380649e-23
C = 299792458.0
T_BASE = 2.725  # K (テンプレート構築の基準; ΔT 係数がずれを吸収)


def templates(nu_GHz, gal_tab=None, T=T_BASE):
    """強度テンプレート [MJy/sr] を返す (係数は無次元パラメータ)。"""
    nu = nu_GHz * 1e9
    x = H_PLANCK * nu / (K_B * T)
    ex = np.exp(x)
    I0 = 2 * H_PLANCK * nu**3 / C**2 * 1e20  # -> MJy/sr (1 Jy = 1e-26)
    n0 = 1.0 / (ex - 1.0)
    dT = x * ex / (ex - 1.0)**2
    dmu = -ex / (ex - 1.0)**2
    dy = dT * (x * (ex + 1.0) / (ex - 1.0) - 4.0)
    # 銀河: Fixsen 表のモデル銀河スペクトル (銀河極) をそのまま形に使う
    gal = gal_tab  # 係数は無次元 (モデル通りなら ~1)
    base = I0 * n0
    return base, {"dT": I0 * dT, "mu": I0 * dmu, "y": I0 * dy, "gal": gal}


def fit(nu, I, sig, gal, label):
    base, T = templates(nu, gal_tab=gal)
    A = np.column_stack([T["dT"], T["mu"], T["y"], T["gal"]])
    w = 1.0 / sig
    Aw = A * w[:, None]
    bw = (I - base) * w
    coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    resid = I - (base + A @ coef)
    chi2 = np.sum((resid / sig)**2)
    dof = len(nu) - 4
    err = np.sqrt(np.diag(cov))
    scale = np.sqrt(chi2 / dof) if chi2 / dof > 1 else 1.0
    names = ["dT_over_T", "mu", "y", "gal_amp"]
    res = {"label": label, "n_points": int(len(nu)),
           "chi2": float(chi2), "dof": int(dof),
           "chi2_per_dof": float(chi2 / dof)}
    for n_, c_, e_ in zip(names, coef, err):
        res[n_] = float(c_)
        res[n_ + "_err"] = float(e_ * scale)   # chi2 で膨らませた誤差
        res[n_ + "_err_raw"] = float(e_)
    res["T_CMB_K"] = float(T_BASE * (1 + coef[0]))
    res["T_CMB_err_K"] = float(T_BASE * res["dT_over_T_err"])
    # μ の 2σ 上限と熱の予算 (μ 期 5e4<z<2e6 の注入に対する換算)
    mu, mu_e = coef[1], res["mu_err"]
    res["mu_2sigma_upper"] = float(abs(mu) + 2 * mu_e)
    res["heat_budget_Du_over_u"] = float(res["mu_2sigma_upper"] / 1.4)
    print(f"\n--- {label} (n={len(nu)}, chi2/dof={chi2/dof:.2f}) ---")
    for n_ in names:
        print(f"  {n_:24s} = {res[n_]:+.4g} ± {res[n_ + '_err']:.2g}")
    print(f"  T_CMB = {res['T_CMB_K']:.4f} ± {res['T_CMB_err_K']:.4f} K")
    print(f"  |mu| 2σ 上限 = {res['mu_2sigma_upper']:.3g}  "
          f"-> 熱の予算 Δu/u ≲ {res['heat_budget_Du_over_u']:.3g}")
    return res, resid


def main():
    d = np.genfromtxt(HERE / "../data/firas_spectrum.csv", delimiter=",",
                      names=True)
    nu, I, sig = d["nu_GHz"], d["I_MJy_sr"], d["sigma_MJy_sr"]
    gal = d["gal_MJy_sr"]
    out = {}
    main_sel = (nu >= 60) & (nu <= 630)
    r_main, _ = fit(nu[main_sel], I[main_sel], sig[main_sel],
                    gal[main_sel], "main 60-630 GHz")
    out["main"] = r_main
    r_all, _ = fit(nu, I, sig, gal, "full range (check)")
    out["full"] = r_all
    out["published_reference"] = {
        "Fixsen+1996_mu_95CL": 9.0e-5, "Fixsen+1996_y_95CL": 1.5e-5,
        "T_CMB_Fixsen2009_K": 2.7255}
    (HERE / "firas_mu_numbers.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nsaved: firas_mu_numbers.json")

    # 図: スペクトルと残差 (μ テンプレートの大きさ比較入り)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base, T = templates(nu, gal_tab=gal)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.errorbar(nu, I, yerr=sig, fmt=".", ms=3, label="FIRAS monopole")
    model = base + r_main["dT_over_T"] * T["dT"] + r_main["mu"] * T["mu"] \
        + r_main["y"] * T["y"] + r_main["gal_amp"] * T["gal"]
    a1.plot(nu, model, "r-", lw=1, label="fit (BB+ΔT+μ+y+gal)")
    a1.set_ylabel("I [MJy/sr]")
    a1.legend()
    a1.set_title("COBE/FIRAS monopole spectrum fit")
    pred_mu = r_main["mu_2sigma_upper"] * T["mu"]
    a2.errorbar(nu, I - (base + r_main["dT_over_T"] * T["dT"]
                         + r_main["y"] * T["y"]
                         + r_main["gal_amp"] * T["gal"]),
                yerr=sig, fmt=".", ms=3, label="residual after BB+y+gal")
    a2.plot(nu, pred_mu, "g-", lw=1,
            label=f"μ template at 2σ upper ({r_main['mu_2sigma_upper']:.1e})")
    a2.plot(nu, -pred_mu, "g-", lw=1)
    a2.set_xlabel("frequency [GHz]")
    a2.set_ylabel("residual [MJy/sr]")
    a2.legend()
    fig.tight_layout()
    fig.savefig(str(HERE / "firas_mu.png"), dpi=130)
    print("saved: firas_mu.png")


if __name__ == "__main__":
    main()
