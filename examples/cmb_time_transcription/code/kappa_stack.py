"""Planck レンズ収束 κ × VAST/VIDE ボイドのスタッキング + 理論期待値。

重ね合わせの意味 (2026-11-12): 晴れ上がりの「後」に起きたこと —
構造が育ち、ボイドが空になり、その質量の窪みが通過する CMB 光子を
曲げた累積が κ マップ。κ は経路全体の時間積分そのもの
(= 転写) で、温度の ISW (変化率) と対になる「育った構造の総量」側。

理論期待値: 自前の銀河スタック・プロファイル (中心 0.7%、壁 1.55 倍
@1.07 Rv) をバイアス b で物質に直し、球対称ボイドの中心視線で
κ(0) = (3/2) Ωm (H0/c)^2 ∫ δ (1+z) χ (χs-χ)/χs dℓ を積分する。
χs (CMB までの共動距離) >> χ_lens なので (χs-χ)/χs ≒ 1。

使い方: kappa_stack.py [κマップのパス]
"""
import sys
import json
import numpy as np
import healpy as hp
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = Path(__file__).parent
H = 0.674
OM = 0.315
C_KMS = 299792.458
H0_C = 100.0 * H / C_KMS          # H0/c [1/Mpc]
rng = np.random.default_rng(20261112)

# ---------------- ボイド (isw_stack.py と同一選択) ----------------
vraw = np.loadtxt(str(HERE / "../data/V2_VIDE-nsa_v1_0_1_zobovoids.dat"), skiprows=1)
v_xyz, v_z = vraw[:, 0:3], vraw[:, 3]
v_ra, v_dec, v_rad = vraw[:, 4], vraw[:, 5], vraw[:, 6]
v_dc = np.linalg.norm(v_xyz, axis=1)
v_b = SkyCoord(ra=v_ra * u.deg, dec=v_dec * u.deg).galactic.b.deg
sel = (v_z > 0.02) & (v_z < 0.114) & (v_rad >= 12.0) & (np.abs(v_b) > 20.0)
ra, dec, z = v_ra[sel], v_dec[sel], v_z[sel]
Rv = v_rad[sel] / H
Dc = v_dc[sel] / H
Da = Dc / (1.0 + z)
theta = np.degrees(np.arctan2(Rv, Da))
N = int(sel.sum())
print(f"voids: {N}, Rv median {np.median(Rv):.1f} Mpc, z median {np.median(z):.3f}, "
      f"theta median {np.median(theta):.2f} deg", flush=True)

EDGES = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0])
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])

# ---------------- 理論期待値 ----------------
def kappa_center(Rv_Mpc, z_l, bias):
    """自前プロファイル形 (銀河) を物質に直した球ボイドの中心 κ。"""
    # 銀河プロファイル (自前スタックのアンカー): 中心 0.007, 壁 1.55 @1.07
    # 形は Hamaus 型: delta_g(r) = dc * (1 - (r/Rs)^a) / (1 + (r/Rv)^b) の簡易版として
    # 区分線形 (r/Rv, 密度比) を使う
    prof_r = np.array([0.0, 0.5, 0.8, 0.95, 1.07, 1.25, 1.6, 2.0])
    prof_g = np.array([0.007, 0.05, 0.25, 0.70, 1.55, 1.15, 1.02, 1.00])
    delta_m = (prof_g - 1.0) / bias
    # 中心視線: 衝突パラメータ 0、経路 s ∈ [-2Rv, 2Rv]、r=|s|
    s = np.linspace(-2 * Rv_Mpc, 2 * Rv_Mpc, 801)
    d = np.interp(np.abs(s) / Rv_Mpc, prof_r, delta_m)
    Dc_l = Rv_Mpc / np.tan(np.deg2rad(1.0)) * 0  # placeholder (未使用)
    Dc_lens = (Dc[np.argmin(np.abs(z - z_l))])
    integrand = d * (1 + z_l) * Dc_lens  # (χs-χ)/χs ≒ 1
    return 1.5 * OM * H0_C**2 * np.trapezoid(integrand, s)

def kappa_profile_theory(Rv_Mpc, z_l, bias):
    """衝突パラメータ b ごとの κ(b) (自前プロファイル形、球対称)。"""
    prof_r = np.array([0.0, 0.5, 0.8, 0.95, 1.07, 1.25, 1.6, 2.0])
    prof_g = np.array([0.007, 0.05, 0.25, 0.70, 1.55, 1.15, 1.02, 1.00])
    delta_m = (prof_g - 1.0) / bias
    Dc_lens = Dc[np.argmin(np.abs(z - z_l))]
    s = np.linspace(-2.5 * Rv_Mpc, 2.5 * Rv_Mpc, 1001)
    out = []
    for bb in CENT:
        r = np.sqrt(s**2 + (bb * Rv_Mpc)**2)
        d = np.interp(r / Rv_Mpc, prof_r, delta_m)
        out.append(1.5 * OM * H0_C**2 * (1 + z_l) * Dc_lens *
                   np.trapezoid(d, s))
    return np.array(out)


pred = {}
for b in [1.0, 1.45]:
    vals = [kappa_center(r, zz, b) for r, zz in zip(Rv, z)]
    prof_th = np.mean([kappa_profile_theory(r, zz, b)
                      for r, zz in zip(Rv[::15], z[::15])], axis=0)
    # 補償開口の理論値: disc (r<1 のビン加重) − annulus (1.4–1.8 ≒ 1.375, 1.75 ビン)
    w_disc = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    disc = float(np.sum(prof_th[:5] * w_disc))
    ann = float(0.5 * (prof_th[6] + prof_th[7]))
    pred[f"bias={b}"] = {"median": float(np.median(vals)),
                         "mean": float(np.mean(vals)),
                         "profile_binned": prof_th.tolist(),
                         "compensated": disc - ann}
    print(f"theory kappa(0): bias={b}: median {np.median(vals):.2e}, "
          f"mean {np.mean(vals):.2e}, compensated {disc - ann:.2e}",
          flush=True)

if len(sys.argv) < 2:
    print("map not given yet — theory only")
    (HERE / "kappa_theory.json").write_text(json.dumps(pred, indent=2))
    sys.exit(0)

# ---------------- κ マップのスタック ----------------
kmap = hp.read_map(sys.argv[1], field=0, verbose=False)
NSIDE = hp.get_nside(kmap)
if NSIDE > 512:
    kmap = hp.ud_grade(kmap, 512)
    NSIDE = 512
kmap = hp.smoothing(kmap, fwhm=np.deg2rad(0.5), verbose=False)
print(f"kappa map ready: nside={NSIDE}, std={np.std(kmap):.4f}", flush=True)

EDGES = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0])
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])


def profile(ra_arr, dec_arr, th_arr):
    acc = np.zeros(len(EDGES) - 1)
    cnt = np.zeros(len(EDGES) - 1)
    comp = []
    for r_, d_, t_ in zip(ra_arr, dec_arr, th_arr):
        vec = hp.ang2vec(np.deg2rad(90.0 - d_), np.deg2rad(r_))
        pix = hp.query_disc(NSIDE, vec, np.deg2rad(2.0 * t_), inclusive=True)
        vecs = hp.pix2vec(NSIDE, pix)
        cosang = vecs[0] * vec[0] + vecs[1] * vec[1] + vecs[2] * vec[2]
        rr = np.degrees(np.arccos(np.clip(cosang, -1, 1))) / t_
        vals = kmap[pix]
        for i in range(len(EDGES) - 1):
            msk = (rr >= EDGES[i]) & (rr < EDGES[i + 1])
            if msk.sum() >= 3:
                acc[i] += vals[msk].mean()
                cnt[i] += 1
        comp.append(vals[rr < 1.0].mean() - vals[(rr >= 1.4) & (rr < 1.8)].mean())
    return acc / np.maximum(cnt, 1), np.array(comp)


real_prof, real_comp = profile(ra, dec, theta)
print("real kappa profile done", flush=True)

ra_lo, ra_hi = ra.min(), ra.max()
dec_lo, dec_hi = dec.min(), dec.max()
NRAND = 200
rand_prof = np.zeros((NRAND, len(EDGES) - 1))
rand_comp = np.zeros(NRAND)
for k in range(NRAND):
    ok_ra, ok_dec = [], []
    while len(ok_ra) < N:
        cra = rng.uniform(ra_lo, ra_hi, 2 * N)
        cdec = np.degrees(np.arcsin(rng.uniform(np.sin(np.deg2rad(dec_lo)),
                                                np.sin(np.deg2rad(dec_hi)), 2 * N)))
        cb = SkyCoord(ra=cra * u.deg, dec=cdec * u.deg).galactic.b.deg
        good = np.abs(cb) > 25.0
        ok_ra.extend(cra[good].tolist())
        ok_dec.extend(cdec[good].tolist())
    rp, rc = profile(np.array(ok_ra[:N]), np.array(ok_dec[:N]),
                     rng.permutation(theta))
    rand_prof[k] = rp
    rand_comp[k] = rc.mean()
    if (k + 1) % 50 == 0:
        print(f"random {k + 1}/{NRAND}", flush=True)

out = {"n_voids": N,
       "bin_centers_r_over_Rv": CENT.tolist(),
       "kappa_profile": real_prof.tolist(),
       "random_profile_mean": rand_prof.mean(axis=0).tolist(),
       "random_profile_std": rand_prof.std(axis=0).tolist(),
       "kappa_center_bin": float(real_prof[0]),
       "kappa_center_excess": float(real_prof[0] - rand_prof.mean(axis=0)[0]),
       "kappa_center_sigma": float((real_prof[0] - rand_prof.mean(axis=0)[0])
                                   / rand_prof.std(axis=0)[0]),
       "kappa_compensated": float(real_comp.mean()),
       "kappa_compensated_std": float(rand_comp.std()),
       "kappa_compensated_significance": float(real_comp.mean() / rand_comp.std()),
       "theory_kappa_center": pred}
(HERE / "kappa_numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(CENT, real_prof, yerr=rand_prof.std(axis=0), fmt="o-",
            color="darkblue", label=f"VAST voids (N={N})")
ax.fill_between(CENT, rand_prof.mean(axis=0) - rand_prof.std(axis=0),
                rand_prof.mean(axis=0) + rand_prof.std(axis=0),
                color="gray", alpha=0.25, label="random stacks (1σ)")
for b, col in [("bias=1.0", "green"), ("bias=1.45", "orange")]:
    ax.axhline(pred[b]["median"], color=col, ls=":", lw=1.2,
               label=f"theory center ({b}): {pred[b]['median']:.1e}")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(1.0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("r / R_v")
ax.set_ylabel("stacked κ")
ax.set_title("CMB lensing κ stack: Planck × VAST/VIDE voids")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(str(HERE / "kappa_stack.png"), dpi=130)
print("saved: kappa_numbers.json, kappa_stack.png")
