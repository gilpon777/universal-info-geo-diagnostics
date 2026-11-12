"""ボイド内の原始ブラックホール (PBH) 種の「当たり」見積もり。

2026-11-12 の議論 (スープで生まれる種 -> ボイドに生き残る) を受けた
桁見積もり。仮定は全て明示し、結果は上限/目安として扱う。

Part 1: 質量 M・存在量 f (暗黒物質に占める割合) ごとに、
        代表的なボイド1個の中に期待される PBH の個数。
Part 2: 既に解析済みの GW 視線 (ボイド横切り) について、
        レンズがアインシュタイン半径内を通る確率 (幾何光学深度 tau)。
        tau は固定 f では質量に依らない (総質量密度で決まる) ことを
        数値でも確認する。波動光学の可視性は w = 8 pi G M f_gw / c^3
        で別に評価する (w ~ 1 が干渉縞の出る境目)。

検証アンカー (過去の実測メモとの照合):
- GW231123 の視線 (レンズ z~0.04, D_eff ~ 160 Mpc) で
  100 Msun のアインシュタイン半径 ~ 0.055 pc。
- 同視線で w=1 になる質量は 20 Hz で ~ 388 Msun。
"""
import json
import numpy as np
from pathlib import Path

# --- 定数 ---
G = 6.674e-11          # m^3 kg^-1 s^-2
C = 2.998e8            # m/s
MSUN = 1.989e30        # kg
MPC = 3.086e22         # m
PC = MPC / 1e6
H0 = 67.4e3 / MPC      # 1/s
OMEGA_C = 0.265
RHO_C = 3 * H0**2 / (8 * np.pi * G)          # kg/m^3
RHO_DM = OMEGA_C * RHO_C                      # kg/m^3 (宇宙平均)
RHO_DM_MSUN_MPC3 = RHO_DM / MSUN * MPC**3     # Msun/Mpc^3

VOID_DM_FACTOR = 0.15   # ボイド内 DM 密度 = 平均 x 0.15 (delta ~ -0.85; 括り 0.1-0.2)

# --- Part 1: ボイド内の個数 ---
# 代表ボイド: GW231123 視線の idx7 (Rv=30.5 h^-1 Mpc -> h=0.674 で 45.2 Mpc は
# 大きすぎるので、ここでは h^-1 をそのまま使わず実測系の代表として
# Rv = 30 Mpc 級と Local Void (直径 45 Mpc) の2つを置く。
VOIDS = {
    "large_void_R30Mpc": {"R_Mpc": 30.0},
    "local_void_R22.5Mpc": {"R_Mpc": 22.5},
}

SCENARIOS = [
    # (名前, M [kg], f のリスト)
    ("QCD_seed_3Msun", 3 * MSUN, [1e-12, 1e-9, 1e-6]),
    ("QCD_seed_30Msun", 30 * MSUN, [1e-12, 1e-9, 1e-6]),
    ("grown_IMBH_300Msun", 300 * MSUN, [1e-9, 1e-6, 1e-3]),
    ("asteroid_1e15kg", 1e15, [1e-6, 1e-3, 1.0]),
    ("asteroid_1e18kg", 1e18, [1e-6, 1e-3, 1.0]),
]


def n_per_mpc3(M_kg, f):
    """宇宙平均での数密度 [1/Mpc^3]。"""
    return f * RHO_DM_MSUN_MPC3 / (M_kg / MSUN)


def count_in_void(M_kg, f, R_Mpc):
    V = 4.0 / 3.0 * np.pi * R_Mpc**3
    return n_per_mpc3(M_kg, f) * VOID_DM_FACTOR * V


# --- Part 2: GW 視線の光学深度 ---
SIGHTLINES = {
    # Ds: 源までの距離, Dl: レンズ (ボイド) までの代表距離, L: ボイド内弦長合計
    "GW231123_voids": {"Ds_Mpc": 2200.0, "Dl_Mpc": 170.0, "L_Mpc": 356.0},
    "GW240925_LocalVoid": {"Ds_Mpc": 356.0, "Dl_Mpc": 20.0, "L_Mpc": 44.8},
    "GW241011_LocalVoid": {"Ds_Mpc": 212.0, "Dl_Mpc": 20.0, "L_Mpc": 45.3},
}


def einstein_radius_pc(M_kg, Dl_Mpc, Ds_Mpc):
    Dls = Ds_Mpc - Dl_Mpc
    D_eff = Dl_Mpc * Dls / Ds_Mpc * MPC       # m
    rE2 = 4 * G * M_kg / C**2 * D_eff         # m^2
    return np.sqrt(rE2) / PC


def optical_depth(f, Dl_Mpc, Ds_Mpc, L_Mpc):
    """tau = f * rho_void * (4 pi G / c^2) * D_eff * L (質量に依らない形)。"""
    Dls = Ds_Mpc - Dl_Mpc
    D_eff = Dl_Mpc * Dls / Ds_Mpc * MPC
    rho_void = VOID_DM_FACTOR * RHO_DM
    return f * rho_void * (4 * np.pi * G / C**2) * D_eff * (L_Mpc * MPC)


def optical_depth_direct(M_kg, f, Dl_Mpc, Ds_Mpc, L_Mpc):
    """n * pi rE^2 * L で直接計算 (上の式と一致するはず)。"""
    n = f * (VOID_DM_FACTOR * RHO_DM) / M_kg          # 1/m^3 (ボイド内)
    rE = einstein_radius_pc(M_kg, Dl_Mpc, Ds_Mpc) * PC
    return n * np.pi * rE**2 * (L_Mpc * MPC)


def w_wave(M_kg, f_gw=20.0):
    return 8 * np.pi * G * M_kg * f_gw / C**3


def main():
    out = {"rho_dm_msun_mpc3": RHO_DM_MSUN_MPC3, "void_dm_factor": VOID_DM_FACTOR}

    # --- アンカー照合 ---
    rE100 = einstein_radius_pc(100 * MSUN, 170.0, 2200.0)
    # w=1 になる質量 (20 Hz)
    M_w1 = C**3 / (8 * np.pi * G * 20.0) / MSUN
    out["anchors"] = {"rE_100Msun_pc_at_GW231123": rE100, "M_w1_Msun_20Hz": M_w1}
    print(f"[anchor] rE(100 Msun, GW231123 geometry) = {rE100:.3f} pc  (past memo ~0.055)")
    print(f"[anchor] M(w=1, 20 Hz) = {M_w1:.0f} Msun  (past memo ~388)")

    # --- Part 1 ---
    print("\n=== Part 1: ボイド1個あたりの期待個数 ===")
    part1 = {}
    for name, M, fs in SCENARIOS:
        part1[name] = {}
        for f in fs:
            row = {}
            for vname, v in VOIDS.items():
                row[vname] = count_in_void(M, f, v["R_Mpc"])
            part1[name][f"{f:g}"] = row
            print(f"{name:22s} f={f:<8g} " +
                  "  ".join(f"{k}={v:,.3g}" for k, v in row.items()))
    out["counts_per_void"] = part1

    # --- Part 2 ---
    print("\n=== Part 2: GW視線の光学深度 tau (当たる確率) ===")
    part2 = {}
    for sname, s in SIGHTLINES.items():
        part2[sname] = {}
        for f in [1e-12, 1e-9, 1e-6, 1e-3, 1.0]:
            tau = optical_depth(f, s["Dl_Mpc"], s["Ds_Mpc"], s["L_Mpc"])
            part2[sname][f"{f:g}"] = tau
        print(f"{sname:22s} tau(f=1)={part2[sname]['1']:.3g}  "
              f"tau(f=1e-6)={part2[sname]['1e-06']:.3g}  "
              f"tau(f=1e-12)={part2[sname]['1e-12']:.3g}")
    out["optical_depth"] = part2

    # tau の質量非依存チェック (直接式との一致)
    tau_a = optical_depth(1e-6, 170.0, 2200.0, 356.0)
    tau_b = optical_depth_direct(3 * MSUN, 1e-6, 170.0, 2200.0, 356.0)
    tau_c = optical_depth_direct(300 * MSUN, 1e-6, 170.0, 2200.0, 356.0)
    print(f"\n[check] tau closed-form={tau_a:.6g}  direct(3Msun)={tau_b:.6g}  "
          f"direct(300Msun)={tau_c:.6g}")
    assert abs(tau_a / tau_b - 1) < 1e-9 and abs(tau_a / tau_c - 1) < 1e-9
    out["tau_mass_independence_check"] = {"closed": tau_a, "direct_3Msun": tau_b,
                                          "direct_300Msun": tau_c}

    # 波動光学の可視性
    print("\n=== w = 8 pi G M f / c^3 (20 Hz) ===")
    wtab = {}
    for M in [1, 3, 10, 30, 100, 300, 1000]:
        wtab[f"{M}Msun"] = w_wave(M * MSUN)
        print(f"M={M:>5} Msun : w={wtab[f'{M}Msun']:.3g}")
    out["w_at_20Hz"] = wtab

    # 小惑星級の w (GW では完全に見えないことの確認)
    print(f"w(1e15 kg, 20 Hz) = {w_wave(1e15):.3g}  -> GW チャンネルでは不可視")

    Path(__file__).with_name("pbh_void_numbers.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nsaved: pbh_void_numbers.json")


if __name__ == "__main__":
    main()
