"""ボイド内 PBH 見積もり v2 — 感想で挙げた最弱の仮定を潰しに行く版。

v1 の最弱点: 「PBH は DM と同じ割合 (0.15 倍) でボイドに残る」と置いた。
これを2つの心配に分解して、それぞれ計算で確かめる。

心配A: 出生時のバイアス
  PBH 形成は指数関数的 (beta ~ exp(-nu^2/2))。将来ボイドになる領域
  (大規模ゆらぎが負) では形成率が指数で抑えられるのではないか?
  -> peak-background split: 閾値が delta_c -> delta_c - delta_b に
     ずれる。ただし形成時刻 (QCD 期) の delta_b は原始ゆらぎの
     大きさ ~1e-5 の桁しかない (ボイドの -0.8 は後から成長した値)。
     nu = 5..8, delta_b = ±1e-5..±3e-5 で比率を計算する。

心配B: 集団出生 + 暴走合体 (ユーザー自身の描像)
  PBH は集団で生まれ、密集した集団内では合体の暴走が起き、
  集団ごとに 1 個の重い残骸 (runaway remnant) が残る、という
  再配分を入れる。パラメータ: 集団サイズ N_cl、残骸への質量効率 eta。
  残骸質量 M_rem = eta * N_cl * M_seed、残骸が担う質量分率 phi = eta
  (集団内の残りは単体として残るとする簡易モデル)。
  -> 総質量保存のため光学深度 tau の「平均」は不変 (v1 の結論は不変)。
     変わるのは (1) 質量の配分: GW 波動光学の窓 (100-3000 Msun) に
     入る質量分率、(2) ボイド1個あたりの「窓内天体」の個数、
     (3) 検出可能光学深度 tau_det = phi_window * tau。

v1 の定数・幾何は pbh_void_estimate.py と同一 (rho_DM, ボイド体積,
視線パラメータ) を再定義して使う。
"""
import json
import numpy as np
from pathlib import Path

MSUN = 1.989e30
RHO_DM_MSUN_MPC3 = 3.37e10     # v1 の計算値 (H0=67.4, Omega_c=0.265)
VOID_DM_FACTOR = 0.15
VOIDS = {"large_R30": 4/3*np.pi*30.0**3, "local_R22.5": 4/3*np.pi*22.5**3}
TAU_F1 = {"GW231123": 1.68e-4, "GW240925_LocalVoid": 2.55e-6,
          "GW241011_LocalVoid": 2.47e-6}   # v1: tau(f=1)

# ---------------- 心配A: 形成バイアス ----------------
def formation_ratio(nu, delta_b, delta_c=0.45):
    """背景 delta_b の領域での形成率 / 平均領域の形成率。
    nu_eff = nu * (1 - delta_b/delta_c); 率 ~ exp(-nu_eff^2/2)/exp(-nu^2/2)。"""
    nu_eff = nu * (1.0 - delta_b / delta_c)
    return np.exp(-(nu_eff**2 - nu**2) / 2.0)

print("=== 心配A: 形成時のバイアス (率の比) ===")
worryA = {}
for nu in [5, 6, 7, 8]:
    for db in [-3e-5, -1e-5, 1e-5, 3e-5]:
        r = formation_ratio(nu, db)
        worryA[f"nu={nu},db={db:g}"] = r
    print(f"nu={nu}: delta_b=-3e-5 -> {formation_ratio(nu, -3e-5):.4f}, "
          f"+3e-5 -> {formation_ratio(nu, 3e-5):.4f}")
print("-> 形成時刻の delta_b は 1e-5 の桁しかなく、指数の肩は動かない。")
print("   指数関数が効くのは後から delta_b ~ O(1) に育つ銀河形成の側。")
print("   PBH は『地図が描かれる前』に配られた、が定量的にも正しい。")

# ---------------- 心配B: 暴走合体の再配分 ----------------
M_SEED = 3.0  # Msun (QCD 山)
print(f"\n=== 心配B: 集団暴走合体 (M_seed = {M_SEED} Msun) ===")
scenarios = [  # (N_cl, eta)
    (30, 0.3), (100, 0.3), (300, 0.3), (100, 0.5), (300, 0.5), (1000, 0.5),
]
worryB = {}
for f in [1e-12, 1e-9]:
    total_mass = {v: f * RHO_DM_MSUN_MPC3 * VOID_DM_FACTOR * V
                  for v, V in VOIDS.items()}     # ボイド内の PBH 総質量 [Msun]
    singles = {v: total_mass[v] / M_SEED for v in VOIDS}
    print(f"\nf = {f:g}: 単体換算 個数 = " +
          ", ".join(f"{v}: {singles[v]:,.4g}" for v in VOIDS))
    for N_cl, eta in scenarios:
        M_rem = eta * N_cl * M_SEED
        # 集団の数 = 単体数 / N_cl、集団ごとに残骸 1 個
        rem = {v: singles[v] / N_cl for v in VOIDS}
        in_window = 100 <= M_rem <= 3000
        worryB[f"f={f:g},Ncl={N_cl},eta={eta}"] = {
            "M_rem_Msun": M_rem, "in_GW_window": bool(in_window),
            "remnants_per_void": rem, "phi_remnant": eta}
        print(f"  N_cl={N_cl:>4}, eta={eta}: M_rem={M_rem:>6,.0f} Msun "
              f"({'窓内' if in_window else '窓外'}) 残骸/ボイド = " +
              ", ".join(f"{v}: {rem[v]:,.3g}" for v in VOIDS))

# 検出可能光学深度: 窓内の質量分率 phi が担う分だけが「見える」
print("\n=== 検出可能光学深度 tau_det = phi_window * tau(f) ===")
tau_det = {}
for name, phi in [("v1 (全部 3 Msun, w=0.007)", 0.0),
                  ("暴走あり phi=0.3", 0.3), ("暴走あり phi=0.5", 0.5)]:
    row = {s: phi * tf1 * 1e-9 for s, tf1 in TAU_F1.items()}  # f=1e-9 の例
    tau_det[name] = row
    print(f"{name}: " + ", ".join(f"{s}={v:.3g}" for s, v in row.items()))
print("-> tau の平均は総質量保存で不変。当たる確率の絶望も不変。")
print("   変わるのは『当たった場合に見えるか』: v1 では窓の外だった")
print("   質量が、暴走があれば窓の中に入る。")

# ---------------- 設計への波及: Δ 分布の裾 ----------------
print("\n=== Δテストへの波及: 裾の過剰 (解析式) ===")
from scipy import stats
q, mu_h, sig_h, mu_0, sig_0 = 0.30, 1.2, 0.40, 0.15, 0.50
p_void = q * (1 - stats.norm.cdf(1.0, mu_h, sig_h)) + \
         (1 - q) * (1 - stats.norm.cdf(1.0, mu_0, sig_0))
p_field = 1 - stats.norm.cdf(1.0, mu_0, sig_0)
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
pbar = (p_void + p_field) / 2
n_tail = 2 * Z**2 * pbar * (1 - pbar) / (p_void - p_field)**2
print(f"仮定: ボイド側は割合 q={q} が Δ~N({mu_h},{sig_h}) の重い集団、"
      f"残りとフィールドは N({mu_0},{sig_0})")
print(f"P(Δ>1): ボイド {p_void:.3f} vs フィールド {p_field:.3f}")
print(f"裾の検定に必要な n (1群): {n_tail:.0f}")
print("-> 平均のずれ (0.3 dex 分離, n=44-90) と同じ桁で、")
print("   『裾の太さ』も独立の検定軸になる (二峰性が暴走モデルの署名)。")

out = {"formation_bias": worryA, "remnant_scenarios": worryB,
       "tau_det_f1e-9": tau_det,
       "tail_test": {"p_void_Delta>1": float(p_void),
                     "p_field_Delta>1": float(p_field),
                     "n_per_group": float(n_tail)}}
Path(__file__).with_name("pbh_void_numbers2.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False))
print("\nsaved: pbh_void_numbers2.json")
