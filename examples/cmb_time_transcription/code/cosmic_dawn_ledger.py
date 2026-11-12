"""晴れ上がり → 初期銀河: 時間を前に進めた2つの帳簿 (2026-11-13)。

帳簿A (光子の側): 再イオン化の光学的厚さ τ — 晴れた後に自由電子が
できた総量の積分。Planck 2018 実測 τ = 0.054 ± 0.007 を入力に、
「いつから完全電離だったら τ がいくつになるか」を逆算し、
初期銀河の時代 (z>10) が電子柱に寄与できる上限を出す。

帳簿B (構造の側): Press–Schechter 崩壊割合 f_coll(z) (原子冷却
ハロー閾値) × 星の電離光子予算 — 光源がいつ・どれだけ用意できたか。

近似の明示: 転送関数は Eisenstein & Hu 1999 フィット (ゼロバリオン
補正つき簡易版)、成長因子は Carroll+ 1992 フィット、崩壊割合は
Press–Schechter (Sheth–Tormen より高z尾は控えめ)。桁と括りの計算。
"""
import json
import numpy as np
from scipy.special import erfc
from pathlib import Path

HERE = Path(__file__).parent

# ---- Planck 2018 宇宙論 ----
H0, OM, OL, OB = 67.4, 0.315, 0.685, 0.0493
S8, NS = 0.811, 0.965
h = H0 / 100.0
C_CGS = 2.99792458e10
SIGT = 6.6524587e-25            # cm^2
MPC_CM = 3.085677581e24
GYR_S = 3.15576e16
H0_S = H0 * 1e5 / MPC_CM        # 1/s
RHO_C = 3 * H0_S**2 / (8 * np.pi * 6.674e-8)   # g/cm^3
YP = 0.245
NH0 = OB * RHO_C * (1 - YP) / 1.6735e-24       # H 核 /cm^3 (現在)
FE_FULL = 1 + 2 * (YP / (4 * (1 - YP))) / ((1 - YP) / (1 - YP))  # 後段で再定義
FE = 1.0 + (YP / (4.0 * (1 - YP))) * 2.0 / 1.0  # H核あたり電子数 (H+He 完全電離)
FE = 1.0 + 2 * 0.0811           # = 1.162 (nHe/nH = 0.081)


def E(z):
    return np.sqrt(OM * (1 + z)**3 + OL)


def age_gyr(z):
    """宇宙年齢 (放射時代の残り ~0.05 Myr は無視)。"""
    zz = np.linspace(z, 3000.0, 6001)
    return np.trapezoid(1.0 / ((1 + zz) * E(zz)), zz) / H0_S / GYR_S


def tau_between(z1, z2, fe=FE):
    """z1<z2 の区間が完全電離だった場合の τ 寄与。"""
    zz = np.linspace(z1, z2, 2001)
    integ = (1 + zz)**2 / E(zz)
    return SIGT * fe * NH0 * C_CGS / H0_S * np.trapezoid(integ, zz)


# ============ 帳簿A: τ ============
TAU_OBS, TAU_ERR = 0.054, 0.007
tableA = {}
for zre in [5.3, 6, 7, 7.7, 8, 9, 10, 11, 12, 15, 20, 30]:
    tableA[f"zre={zre}"] = float(tau_between(0.0, zre))
# 逆算: τ=0.054 になる瞬時再イオン化 z
zg = np.linspace(5, 30, 251)
taus = np.array([tau_between(0.0, z) for z in zg])
z_inst = float(np.interp(TAU_OBS, taus, zg))
z_inst_lo = float(np.interp(TAU_OBS - TAU_ERR, taus, zg))
z_inst_hi = float(np.interp(TAU_OBS + TAU_ERR, taus, zg))
# 高z区間の完全電離が足す量
seg = {f"tau({a}-{b})": float(tau_between(a, b))
       for a, b in [(10, 15), (10, 20), (15, 20), (20, 30)]}
# z>10 完全電離 + 以後は実測整合の履歴、という形の上限:
# τ_obs +2σ までに許される z>10 の電離割合 x_e の上限 (10-20 区間)
budget_2s = (TAU_OBS + 2 * TAU_ERR) - tableA["zre=7.7"]
xe_max = float(budget_2s / seg["tau(10-20)"]) if seg["tau(10-20)"] > 0 else None

print("=== Ledger A: tau ===")
for k, v in tableA.items():
    print(f"  instant reion {k}: tau = {v:.4f}")
print(f"  tau=0.054 -> instant z_re = {z_inst:.1f} "
      f"(±1σ: {z_inst_lo:.1f}–{z_inst_hi:.1f})")
print(f"  segments: {seg}")
print(f"  x_e(z=10-20) upper (2σ, on top of z_re=7.7 baseline): {xe_max:.2f}")

# ============ 帳簿B: 成長と光子予算 ============
# Eisenstein & Hu 1999 (BBKS型簡易): T(k) ゼロバリオン近似 + 音響補正なし
def transfer_eh(k):
    """k in 1/Mpc. EH99 Eq. (29)-(31) の smooth 版。"""
    Omh2, Obh2 = OM * h**2, OB * h**2
    theta = 2.728 / 2.7
    zeq = 2.50e4 * Omh2 / theta**4
    keq = 7.46e-2 * Omh2 / theta**2         # 1/Mpc
    b1 = 0.313 * Omh2**(-0.419) * (1 + 0.607 * Omh2**0.674)
    b2 = 0.238 * Omh2**0.223
    zd = 1291 * Omh2**0.251 / (1 + 0.659 * Omh2**0.828) * \
        (1 + b1 * Obh2**b2)
    rd = 147.7 * (Omh2 / 0.1432)**(-0.5)     # 概算 (Mpc) — 厳密値147.1に近い
    req = 31.5 * Obh2 / theta**4 / (zeq / 1000.0)
    s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**0.75)  # Mpc
    Gamma = OM * h * (1.0)                   # shape (バリオン抑制なし簡易)
    q = k / (13.41 * keq / 1.0) * 0.0 + k / (OM * h**2) * (theta**2)
    # EH99 Eq. (29): L0, C0 (alpha=1 の smooth 形)
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 386.0 / (1 + 69.9 * q**1.08)
    return L0 / (L0 + C0 * q**2)


def sigma_r(R):
    """半径 R (Mpc) のトップハット σ (z=0)。k 積分し σ8 で規格化。"""
    k = np.logspace(-4, 2, 1200)
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    P = k**NS * transfer_eh(k)**2
    return np.sqrt(np.trapezoid(k**2 * P * W**2, k) / (2 * np.pi**2))


R8 = 8.0 / h
s_norm = S8 / sigma_r(R8)
print(f"sigma normalization factor: {s_norm:.3f}")


def sigma_m(M, z):
    """質量 M (M_sun) の σ(z)。"""
    rho_m = OM * RHO_C / 1.989e33 * MPC_CM**3     # M_sun / Mpc^3
    R = (3 * M / (4 * np.pi * rho_m))**(1 / 3)
    g = growth(z) / growth(0.0)
    # 積分は重いので R ごとにキャッシュ
    if not hasattr(sigma_m, "_cache"):
        sigma_m._cache = {}
    key = round(float(R), 3)
    if key not in sigma_m._cache:
        sigma_m._cache[key] = sigma_r(R) * s_norm
    return sigma_m._cache[key] * g


def growth(z):
    Omz = OM * (1 + z)**3 / E(z)**2
    Olz = OL / E(z)**2
    g = 2.5 * Omz / (Omz**(4 / 7) - Olz +
                     (1 + Omz / 2) * (1 + Olz / 70))
    return g / (1 + z)


def m_atomic(z):
    """原子冷却閾値 T_vir=1e4 K のハロー質量 (M_sun)、括り中心。"""
    return 1.0e8 * ((1 + z) / 10.0)**(-1.5)


def f_coll(z, mmin=None):
    if mmin is None:
        mmin = m_atomic(z)
    s = sigma_m(mmin, z)
    return float(erfc(1.686 / (np.sqrt(2) * s)))


zlist = [30, 25, 20, 17, 15, 12, 10, 9, 8, 7, 6, 5.3]
tableB = {}
for z in zlist:
    tableB[str(z)] = {"age_Gyr": float(age_gyr(z)),
                      "M_atomic": float(m_atomic(z)),
                      "sigma": float(sigma_m(m_atomic(z), z)),
                      "f_coll": f_coll(z)}
    print(f"  z={z:>4}: age {tableB[str(z)]['age_Gyr']:.2f} Gyr, "
          f"M_at {m_atomic(z):.1e}, sigma {tableB[str(z)]['sigma']:.2f}, "
          f"f_coll {tableB[str(z)]['f_coll']:.2e}")

# 光子予算: バリオンあたり電離光子数
RHO_B = OB * RHO_C / 1.989e33 * MPC_CM**3        # M_sun/Mpc^3
budget = {}
for tag, fstar, fesc, nstar in [("central", 0.10, 0.10, 4000),
                                ("optimistic", 0.20, 0.20, 4000),
                                ("pessimistic", 0.05, 0.05, 4000),
                                ("PopIII-mix", 0.05, 0.20, 20000)]:
    rows = {}
    for z in zlist:
        rows[str(z)] = float(f_coll(z) * fstar * fesc * nstar)
    budget[tag] = rows

    def cross(thr):
        zz = sorted(zlist, reverse=True)      # 高z → 低z (時間前進)
        prev = None
        for z in zz:
            if rows[str(z)] >= thr:
                if prev is None:
                    return f">{zz[0]}"
                # prev (高z側, 未達) と z の間で線形補間 (log 値)
                f = (np.log(thr) - np.log(rows[str(prev)])) / \
                    (np.log(rows[str(z)]) - np.log(rows[str(prev)]))
                return f"{prev + f * (z - prev):.1f}"
            prev = z
        return "not by z=5.3"
    print(f"  photons/baryon [{tag}]: z=15 {rows['15']:.3f}, "
          f"z=10 {rows['10']:.3f}, z=8 {rows['8']:.2f}, z=6 {rows['6']:.2f}, "
          f"crosses 1 at z~{cross(1.0)}, crosses 3 at z~{cross(3.0)}")

# JWST 照合 (z=10): 観測 ρ* ~ 10^5.5-10^6.5 M_sun/Mpc^3 (文献範囲の桁)
fc10 = f_coll(10)
for rho_star in [3e5, 1e6, 3e6]:
    print(f"  JWST z~10: rho*={rho_star:.0e} -> required efficiency "
          f"{rho_star / (fc10 * RHO_B):.3f} of collapsed baryons")

out = {"tau_table_instant": tableA,
       "z_instant_for_tau_obs": z_inst,
       "z_instant_range_1sigma": [z_inst_lo, z_inst_hi],
       "tau_segments": seg,
       "xe_10_20_upper_2sigma": xe_max,
       "growth_table": tableB,
       "photon_budget": budget,
       "rho_b_Msun_per_Mpc3": float(RHO_B)}
(HERE / "cosmic_dawn_numbers.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False))
print("saved cosmic_dawn_numbers.json")

# ---- 図 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.plot(zg, taus, color="darkblue")
ax.axhspan(TAU_OBS - TAU_ERR, TAU_OBS + TAU_ERR, color="gray", alpha=0.3,
           label="Planck τ = 0.054 ± 0.007")
ax.axhline(TAU_OBS + 2 * TAU_ERR, color="red", ls=":", lw=1, label="+2σ")
ax.set_xlabel("z (fully ionized below this z, neutral above)")
ax.set_ylabel("τ (0 → z)")
ax.set_title("Ledger A: reionization τ bookkeeping")
ax.legend(fontsize=8)
ax = axes[1]
zz = np.array(zlist, dtype=float)
fc = np.array([tableB[str(z)]["f_coll"] for z in zlist])
ax.semilogy(zz, fc, "o-", color="darkgreen", label="f_coll (atomic-cooling halos, PS)")
for tag, col in [("central", "darkorange"), ("optimistic", "gold"),
                 ("pessimistic", "brown"), ("PopIII-mix", "purple")]:
    ph = np.array([budget[tag][str(z)] for z in zlist])
    ax.semilogy(zz, ph, "s--", color=col, ms=4, label=f"ionizing photons/baryon [{tag}]")
ax.axhline(1.0, color="k", lw=0.9)
ax.axhline(3.0, color="gray", ls=":", lw=0.9)
ax.invert_xaxis()
ax.set_xlabel("z")
ax.set_title("Ledger B: collapse & photon budget")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(str(HERE / "cosmic_dawn_ledger.png"), dpi=130)
print("saved cosmic_dawn_ledger.png")
