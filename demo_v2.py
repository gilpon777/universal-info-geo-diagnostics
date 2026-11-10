"""v2 実地テスト: 3成分指数減衰モデル（6パラメータ）。

薬物動態・分光・緩和過程で典型的な「重なり合う指数減衰の分離」問題。
減衰率が近い成分同士は強く縮退する（古典的な sloppy 構造）。
ノイズ入りの繰り返し観測サンプルから標本共分散を推定する実務設定で診断する。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universal_info_geo_diagnostics_v2 import UniversalInformationGeometricDiagnosticsV2

rng = np.random.default_rng(7)

t = np.linspace(0.05, 4.0, 60)
names = ["a1", "b1", "a2", "b2", "a3", "b3"]
theta_true = np.array([2.0, 0.5, 1.0, 0.9, 0.5, 2.5])

def model(th):
    a1, b1, a2, b2, a3, b3 = th
    return (a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t) + a3 * np.exp(-b3 * t))

# 実務設定: ノイズ sigma=0.02 の繰り返し観測 300 サンプル
noise = 0.02
samples = np.array([model(theta_true) + rng.normal(0, noise, size=len(t))
                    for _ in range(300)])

diag = UniversalInformationGeometricDiagnosticsV2(model)
res = diag.evaluate(theta_true, observed_data=samples)

F = res["FIM"]
evals, evecs = np.linalg.eigh(0.5 * (F + F.T))
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]

print("=== 診断結果 ===")
for label in ("Raw", "Scaled"):
    m = res[label]
    print(f"[{label}] log10 CN = {m['Log10ConditionNumber']:.2f} | "
          f"r_eff = {m['EffectiveRank']:.3f} / 6 | 数値ランク = {m['NumericalRank']} / 6")
print(f"共分散の条件数 = {res['CovConditionNumber']:.2f} (経路: {res['InversePath']}, "
      f"共分散ソース: {res['CovarianceSource']})")
print("\n固有値スペクトル（降順）:")
for i, ev in enumerate(evals):
    print(f"  lambda_{i+1} = {ev:.4e}")

def show_dir(v, top=3):
    idx = np.argsort(-np.abs(v))[:top]
    return ", ".join(f"{names[k]}{'+' if v[k] >= 0 else '-'}{abs(v[k]):.2f}" for k in idx)

print("\n最も stiff な方向（最大固有値）:", show_dir(evecs[:, 0]))
print("最も sloppy な方向（最小固有値）:", show_dir(evecs[:, -1]))
print("2番目に sloppy な方向          :", show_dir(evecs[:, -2]))

# 対数パラメータ感度で見た縮退ペアの確認: b1<->b2 の相関
d = np.sqrt(np.diag(F))
corr = F / np.outer(d, d)
print(f"\n相関化FIMでの b1-b2 相関要素: {corr[1, 3]:+.3f}（-1に近いほど b1 と b2 がトレードオフ）")
print(f"相関化FIMでの a2-b2 相関要素: {corr[2, 3]:+.3f}")

# --- スペクトル図 ---
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(1, 7)
ax.bar(x - 0.2, np.log10(np.maximum(res["Raw"]["Eigenvalues"], 1e-300)), width=0.4, label="Raw FIM")
ev_s = res["Scaled"]["Eigenvalues"]
ax.bar(x + 0.2, np.log10(np.maximum(ev_s, 1e-300)), width=0.4, label="Correlation-scaled FIM")
ax.set_xlabel("Eigenvalue index (descending)")
ax.set_ylabel("log10(eigenvalue)")
ax.set_title("FIM eigenvalue spectrum: 3-exponential decay (6 params)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "spectrum_demo.png"), dpi=150)
print("\n図を保存: spectrum_demo.png")
