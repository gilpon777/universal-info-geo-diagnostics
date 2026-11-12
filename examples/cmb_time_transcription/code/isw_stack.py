"""Planck SMICA x VAST/VIDE ボイドの ISW スタッキング。

「背景放射は写真ではなく時間で転写された像」という読みに基づく検証:
ISW 効果は、光子がボイドを横切る間に重力ポテンシャルが変化
(浅く) なることで生じる温度ずれで、時間が経過しなければ存在しない。
静止画の宇宙では原理的にゼロ。検出できれば転写の直接の痕跡、
null なら変化率への上限になる。

方法 (Granett+ 式の補償開口):
- Planck PR3 SMICA I マップを Nside 256 に落とし 1 deg FWHM で平滑。
- 各ボイド: 角度半径 theta_v = R_v / D_A(z)。径方向プロファイルを
  r/theta_v のビンで測り、全ボイドで等重み平均。
- 補償開口量: disc(r<theta_v) の平均 - annulus(1.4-1.8 theta_v) の平均。
- 誤差と有意性: ボイドと同じ theta 分布を持つランダム中心
  (カタログのフットプリント内、|b|>25 deg) 200 回のスタック分布から。
- 銀河面マスクは共通マスク未取得のため |b| カットで代替 (制限として明記)。

宇宙論・カタログの規約は local_void_analysis.py と同一:
Planck2018 (Om=0.315, h=0.674)、座標と半径は h^-1 Mpc。
"""
import json
import numpy as np
import healpy as hp
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = Path(__file__).parent
H = 0.674
rng = np.random.default_rng(20261112)

# ---------------- CMB マップ ----------------
print("loading SMICA ...", flush=True)
m = hp.read_map(str(HERE / "../data/planck_smica_2048.fits"), field=0, verbose=False)
m = hp.ud_grade(m, 256)
m = hp.smoothing(m, fwhm=np.deg2rad(1.0), verbose=False)
m_uK = m * 1e6  # K -> uK
NSIDE = 256
print(f"map ready: nside={NSIDE}, rms={np.std(m_uK):.1f} uK", flush=True)

# ---------------- ボイド ----------------
vraw = np.loadtxt(str(HERE / "../data/V2_VIDE-nsa_v1_0_1_zobovoids.dat"), skiprows=1)
v_xyz, v_z = vraw[:, 0:3], vraw[:, 3]
v_ra, v_dec, v_rad = vraw[:, 4], vraw[:, 5], vraw[:, 6]   # deg, deg, h^-1 Mpc
v_dc = np.linalg.norm(v_xyz, axis=1)                       # h^-1 Mpc
gal = SkyCoord(ra=v_ra * u.deg, dec=v_dec * u.deg).galactic
v_b = gal.b.deg

sel = (v_z > 0.02) & (v_z < 0.114) & (v_rad >= 12.0) & (np.abs(v_b) > 20.0)
ra, dec, z = v_ra[sel], v_dec[sel], v_z[sel]
Rv = v_rad[sel] / H                                        # Mpc
Dc = v_dc[sel] / H                                         # Mpc
Da = Dc / (1.0 + z)
theta = np.degrees(np.arctan2(Rv, Da))                     # deg
N = sel.sum()
print(f"voids in stack: {N} (catalog {len(v_z)}), "
      f"theta_v median {np.median(theta):.2f} deg", flush=True)

EDGES = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0])
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])


def profile(ra_arr, dec_arr, th_arr):
    """各中心について r/theta ビンの平均温度を取り、等重みで平均。"""
    acc = np.zeros(len(EDGES) - 1)
    cnt = np.zeros(len(EDGES) - 1)
    comp = []
    for r_, d_, t_ in zip(ra_arr, dec_arr, th_arr):
        vec = hp.ang2vec(np.deg2rad(90.0 - d_), np.deg2rad(r_))
        pix = hp.query_disc(NSIDE, vec, np.deg2rad(2.0 * t_), inclusive=True)
        vecs = hp.pix2vec(NSIDE, pix)
        cosang = vecs[0] * vec[0] + vecs[1] * vec[1] + vecs[2] * vec[2]
        rr = np.degrees(np.arccos(np.clip(cosang, -1, 1))) / t_
        vals = m_uK[pix]
        for i in range(len(EDGES) - 1):
            msk = (rr >= EDGES[i]) & (rr < EDGES[i + 1])
            if msk.sum() >= 3:
                acc[i] += vals[msk].mean()
                cnt[i] += 1
        disc = vals[rr < 1.0].mean()
        ann = vals[(rr >= 1.4) & (rr < 1.8)].mean()
        comp.append(disc - ann)
    return acc / np.maximum(cnt, 1), np.array(comp)


real_prof, real_comp = profile(ra, dec, theta)
print("real profile done", flush=True)

# ---------------- ランダム対照 ----------------
ra_lo, ra_hi = ra.min(), ra.max()
dec_lo, dec_hi = dec.min(), dec.max()
NRAND = 200
rand_prof = np.zeros((NRAND, len(EDGES) - 1))
rand_comp = np.zeros(NRAND)
for k in range(NRAND):
    # フットプリント内で一様 (dec は sin 一様)、|b|>25 を棄却サンプリング
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

prof_mean = rand_prof.mean(axis=0)
prof_std = rand_prof.std(axis=0)
comp_real = real_comp.mean()
comp_std = rand_comp.std()
comp_p = (np.sum(np.abs(rand_comp) >= np.abs(comp_real)) + 1) / (NRAND + 1)

out = {
    "n_voids": int(N),
    "theta_median_deg": float(np.median(theta)),
    "bin_centers_r_over_Rv": CENT.tolist(),
    "profile_uK": real_prof.tolist(),
    "random_profile_std_uK": prof_std.tolist(),
    "compensated_aperture_uK": float(comp_real),
    "compensated_random_std_uK": float(comp_std),
    "compensated_random_mean_uK": float(rand_comp.mean()),
    "compensated_p_two_sided": float(comp_p),
    "significance_sigma": float(comp_real / comp_std),
    "central_bin_uK": float(real_prof[0]),
    "central_bin_sigma": float(real_prof[0] / prof_std[0]),
    "per_void_comp_std_uK": float(real_comp.std()),
}
(HERE / "isw_numbers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2))

# ---------------- 図 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(CENT, real_prof, yerr=prof_std, fmt="o-", color="darkred",
            label=f"VAST voids (N={N})")
ax.fill_between(CENT, prof_mean - prof_std, prof_mean + prof_std,
                color="gray", alpha=0.25, label="random stacks (1σ)")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(1.0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("r / R_v")
ax.set_ylabel("stacked ΔT [µK]")
ax.set_title("ISW stack: Planck SMICA × VAST/VIDE voids")
ax.legend()
fig.tight_layout()
fig.savefig(str(HERE / "isw_stack.png"), dpi=130)
print("saved: isw_numbers.json, isw_stack.png")
