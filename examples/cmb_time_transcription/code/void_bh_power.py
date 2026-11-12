"""ボイド銀河 中心BH過重テストの検出力計算。

観測量: Delta = log10(M_BH) - [局所 M_BH-M* 関係の予測値] (dex)。
ボイド群の平均 Delta と、質量マッチしたフィールド群の平均 Delta の差
delta が、仮説を分ける。

仮説と予測 (design_void_bh_test.md 参照):
- H_heavy (重い種が先: PBH/直接崩壊 + ボイドで宿主が育たない):
    Delta_void - Delta_field = +0.5 .. +1.5 dex
- H_std (標準シミュレーションのベースライン, EAGLE 型):
    +0.1 .. +0.2 dex 程度 (「わずかに重い」)
- H_light (軽い種がボイドで飢える): -0.3 .. -1.0 dex
分離すべき最小の差は H_heavy 下限 (+0.5) と H_std 上限 (+0.2) の間
= 0.3 dex。これが設計の物差し。

誤差予算 (1天体あたり):
- M_BH-M* 関係の固有散乱: 0.35 dex (局所関係の実測散乱 0.3-0.4)
- 質量測定誤差: ビリアル質量 0.4 dex 系統 / 力学質量 0.2 dex。
  混在サンプルとして 0.35 dex を置く。
-> sigma_tot = sqrt(0.35^2 + 0.35^2) = 0.49 ~ 0.5 dex。

検出力: 解析式 (両側5%, 検出力80%) と、打ち切り (上限値) を入れた
モンテカルロ ( censored normal の尤度で平均を推定 ) で相互確認する。
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats, optimize

rng = np.random.default_rng(20261112)
SIGMA = 0.5
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)   # 2.80


def n_analytic(delta, sigma=SIGMA):
    """2群比較: 1群あたり必要 n (上限値なし)。"""
    return 2 * Z**2 * sigma**2 / delta**2


def censored_mean_mle(x_det, x_lim):
    """打ち切り正規分布の平均の MLE (sigma は既知として扱う簡易版)。
    x_det: 検出値, x_lim: 上限値 (真値はこれ以下)。"""
    def nll(mu):
        ll = np.sum(stats.norm.logpdf(x_det, mu, SIGMA))
        ll += np.sum(stats.norm.logcdf(x_lim, mu, SIGMA))
        return -ll
    res = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return res.x


def mc_power(delta, n_per_group, censor_frac, trials=400):
    """ボイド群 (真の平均 delta) vs フィールド群 (平均 0) の2群検定。
    各群で censor_frac の割合が上限値化される (上限 = その天体の
    観測下限に対応する Delta_lim を真値 - 0.3 dex に置く簡易モデル)。
    検定は censored MLE の平均差 / 解析的標準誤差 で行う。"""
    hits = 0
    for _ in range(trials):
        ests, ses = [], []
        for mu in (delta, 0.0):
            x = rng.normal(mu, SIGMA, n_per_group)
            k = int(round(censor_frac * n_per_group))
            idx = rng.choice(n_per_group, k, replace=False) if k else []
            det = np.delete(x, idx)
            lim = x[idx] - 0.3 if k else np.array([])
            mu_hat = censored_mean_mle(det, lim) if k else det.mean()
            # Fisher 情報から SE (検出 + 打ち切りの情報量)
            info = len(det) / SIGMA**2
            if k:
                z_ = (lim - mu_hat) / SIGMA
                lam = stats.norm.pdf(z_) / np.maximum(stats.norm.cdf(z_), 1e-12)
                info += np.sum(z_ * lam + lam**2) / SIGMA**2 * -1.0 + np.sum(lam * (lam + z_)) / SIGMA**2
                info = max(info, len(det) / SIGMA**2)  # 保守的に検出分のみ下限保証
            ests.append(mu_hat)
            ses.append(1.0 / np.sqrt(info))
        d = ests[0] - ests[1]
        se = np.sqrt(ses[0]**2 + ses[1]**2)
        if abs(d / se) > stats.norm.ppf(0.975):
            hits += 1
    return hits / trials


def main():
    out = {"sigma_tot_dex": SIGMA}
    print(f"sigma_tot = {SIGMA} dex, Z(5%,80%) = {Z:.2f}\n")
    print("=== 解析式: 必要 n (1群あたり, 上限値なし) ===")
    table = {}
    for delta in [1.0, 0.7, 0.5, 0.3, 0.2]:
        n = n_analytic(delta)
        table[f"delta={delta}"] = int(np.ceil(n))
        print(f"delta = {delta:.1f} dex : n = {n:.1f} -> {int(np.ceil(n))}")
    out["n_analytic_per_group"] = table

    print("\n=== モンテカルロ検出力 (n を振って 80% に届く最小 n) ===")
    mc = {}
    for delta, cfrac in [(0.5, 0.0), (0.5, 0.5), (0.3, 0.0), (0.3, 0.5), (1.0, 0.5)]:
        found = None
        for n in [5, 10, 15, 20, 30, 45, 65, 90, 130]:
            p = mc_power(delta, n, cfrac, trials=300)
            if p >= 0.80:
                found = (n, round(p, 2))
                break
        mc[f"delta={delta},censor={cfrac}"] = found
        print(f"delta={delta}, censor={cfrac}: -> n={found}")
    out["mc_min_n_per_group"] = {k: v for k, v in mc.items()}

    Path(__file__).with_name("void_bh_power_numbers.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nsaved: void_bh_power_numbers.json")


if __name__ == "__main__":
    main()
