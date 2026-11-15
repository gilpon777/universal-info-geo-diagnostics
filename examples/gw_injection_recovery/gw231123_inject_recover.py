#!/usr/bin/env python3
"""GW231123_135430 実ストレインへの注入・回収試験 (計画書 gw231123_injection_recovery_plan.md).

Stage A: 同形注入 f*h(t) を「フィットの前」に実ストレインへ加え, 主解析
(waveform_analysis.py) と同一の alpha・tau フィットで回収する.
Stage B: 波動光学2像レンズ (点質量) の増幅率 F(f) による歪み (F-1)*h を
同じくフィット前に注入し, フィット込みで回収する.

!!! 注入は必ずフィットの前に行う !!!
このスクリプトの中心的な構造: inject_into_strain() が返す修正済みストレインを
fit_and_stats() に渡す. フィット後の残差へ信号を足す経路は存在しない
(それは gw231123_sensitivity.py のフィット後感度であり, 本試験の対象外).

固定条件 (計画書 §2): PE サンプル idx 19435 固定 (再選択しない),
前処理・窓・統計量 S は主解析と同一, 背景は保存済み配列を再利用 (取り直さない).
"""
import json, os, sys
import numpy as np
import h5py
from scipy import signal as sig
from scipy.interpolate import CubicSpline
import mpmath as mp
import lal
import lalsimulation as lalsim

WORK = os.path.expanduser("~/workspace/info_geo_diag/void_validation")
WD = WORK + "/wf_data"
GPS0 = 1384782888.7
FS_IN, FS = 4096, 1024
F_LO, F_HI = 20.0, 448.0
SEG_T = 64.0
WINDOWS = {"plan_[-1.5,+0.5]": (-1.5, 0.5), "matched_[-0.6,+0.3]": (-0.6, 0.3)}
LAG_MAX = 0.050
TUKEY = ("tukey", 0.25)
dt = 1.0 / FS
NSEG = int(SEG_T * FS)
DF = 1.0 / SEG_T
FREQS = np.fft.rfftfreq(NSEG, dt)
BAND = (FREQS >= F_LO) & (FREQS <= F_HI)
SAMPLE_IDX = 19435
GMSUN_C3 = 4.9254909476412675e-6  # G*M_sun/c^3 [s]

# ---------------------------------------------------------------- 前処理 (主解析と同一)
pe = h5py.File(f"{WD}/GW231123_PEDataRelease.hdf5", "r")
grp = pe["C00:IMRPhenomXPHM-SpinTaylor"]
ps = grp["posterior_samples"]
pe_psd = {}
for det in ["H1", "L1"]:
    arr = grp["psds"][det][:]
    pe_psd[det] = (arr[:, 0], arr[:, 1])

def load_series(det):
    with h5py.File(f"{WD}/{det}_4096.hdf5", "r") as fh:
        x = fh["strain/Strain"][:].astype(np.float64)
        tstart = float(fh["strain/Strain"].attrs["Xstart"])
    xd = sig.decimate(x, FS_IN // FS, ftype="fir", zero_phase=True)
    sos = sig.butter(8, 15.0, btype="highpass", fs=FS, output="sos")
    return sig.sosfiltfilt(sos, xd), tstart

data, Sn_all = {}, {}
for det in ["H1", "L1"]:
    xh, tstart = load_series(det)
    data[det] = dict(x=xh, t0=tstart)
    fr_pe, P_pe = pe_psd[det]
    lg = np.interp(np.log(np.maximum(FREQS, 0.5)), np.log(fr_pe[1:]), np.log(P_pe[1:]))
    Sn = np.exp(lg); Sn[0] = P_pe[0]
    Sn_all[det] = Sn
print("strain + PE PSD ready", flush=True)

# 採用サンプル idx 19435 を直接読む (再選択 = PE のやり直しはしない)
r = ps[SAMPLE_IDX]
ml = {k: float(r[k]) for k in ps.dtype.names if np.ndim(r[k]) == 0}
ref = json.load(open(f"{WD}/waveform_results.json"))["selected_sample"]
for k in ["mass_1", "mass_2", "luminosity_distance", "geocent_time"]:
    assert abs(ml[k] - ref[k]) < 1e-6 * max(1.0, abs(ref[k])), f"sample mismatch {k}"
TC_GEO = float(ml["geocent_time"])
SEG_START = TC_GEO - SEG_T / 2
print(f"sample idx={SAMPLE_IDX}: m1={ml['mass_1']:.1f} m2={ml['mass_2']:.1f} "
      f"DL={ml['luminosity_distance']:.0f} chi_eff={ml['chi_eff']:.2f}", flush=True)

# ---------------------------------------------------------------- テンプレート (主解析と同一構築)
hp_td, hx_td = lalsim.SimInspiralChooseTDWaveform(
    float(ml["mass_1"]) * lal.MSUN_SI, float(ml["mass_2"]) * lal.MSUN_SI,
    float(ml["spin_1x"]), float(ml["spin_1y"]), float(ml["spin_1z"]),
    float(ml["spin_2x"]), float(ml["spin_2y"]), float(ml["spin_2z"]),
    float(ml["luminosity_distance"]) * 1e6 * lal.PC_SI,
    float(ml["theta_jn"]), float(ml["phase"]), 0.0, 0.0, 0.0,
    dt, 10.0, 10.0, None, lalsim.IMRPhenomXPHM)
hp_a, hx_a = hp_td.data.data, hx_td.data.data
gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(TC_GEO))
win_arr = sig.get_window(TUKEY, NSEG)

HSEG, HFD, HH, I0SEG = {}, {}, {}, {}
for det in ["H1", "L1"]:
    d = lal.cached_detector_by_prefix[det]
    Fp, Fx = lal.ComputeDetAMResponse(d.response, float(ml["ra"]), float(ml["dec"]),
                                      float(ml["psi"]), gmst)
    delay = lal.TimeDelayFromEarthCenter(d.location, float(ml["ra"]), float(ml["dec"]),
                                         lal.LIGOTimeGPS(TC_GEO))
    hdet = Fp * hp_a + Fx * hx_a
    pk_src = int(np.argmax(np.abs(hdet)))
    target = int(round((TC_GEO + delay - SEG_START) * FS))
    hseg = np.zeros(NSEG)
    off = target - pk_src
    hseg[max(off, 0):max(off, 0) + len(hdet) - max(-off, 0)] = hdet[max(-off, 0):]
    HSEG[det] = hseg
    HFD[det] = np.fft.rfft(hseg) * dt
    I0SEG[det] = int(round((SEG_START - data[det]["t0"]) * FS))

def inner_cplx(a_f, b_f, Sn):
    return 4 * DF * np.sum(a_f[BAND] * np.conj(b_f[BAND]) / Sn[BAND])

for det in ["H1", "L1"]:
    HH[det] = inner_cplx(HFD[det], HFD[det], Sn_all[det]).real

# ---------------------------------------------------------------- 注入 (フィットの前!)
def inject_into_strain(det, inj_td_segment):
    """64 s セグメント位置に inj_td_segment (長さ NSEG) を加えた実ストレインを返す.

    これが本試験の唯一の注入経路であり, 呼び出し側はこの戻り値を
    fit_and_stats() に渡す. フィット後の残差への加算は行わない.
    """
    x = data[det]["x"].copy()
    i0 = I0SEG[det]
    x[i0:i0 + NSEG] += inj_td_segment
    return x

# ---------------------------------------------------------------- フィット + 統計量 (主解析と同一)
def whiten_td_from_fd(x_f, Sn, n=NSEG):
    W = np.zeros_like(x_f)
    W[BAND] = x_f[BAND] / np.sqrt(Sn[BAND])
    return np.fft.irfft(W / dt, n=n)

def xcorr_stat(a, b, i0, i1):
    wa, wb = a[i0:i1], b[i0:i1]
    n = len(wa); maxlag = int(LAG_MAX * FS)
    na = np.sqrt(np.sum(wa ** 2)); best, bestlag = 0.0, 0
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = wa[:n - lag], wb[lag:]
        else:
            x, y = wa[-lag:], wb[:n + lag]
        nb = np.sqrt(np.sum(y ** 2))
        if nb == 0 or na == 0:
            continue
        c = abs(np.sum(x * y) / (na * nb))
        if c > best:
            best, bestlag = c, lag
    return best, bestlag / FS

def fit_one_det(det, x_full):
    """主解析と同一の alpha・tau フィット. 戻り値: dict(alpha, tau, R_fd, resid_w, sc)."""
    Sn = Sn_all[det]
    H = HFD[det]
    hh = HH[det]
    i0 = I0SEG[det]
    seg = x_full[i0:i0 + NSEG] * win_arr
    D = np.fft.rfft(seg) * dt
    A = np.zeros(NSEG, dtype=complex)
    A[:NSEG // 2 + 1][BAND] = (D * np.conj(H) / Sn)[BAND]
    rho_series = 4 * DF * NSEG * np.fft.ifft(A)
    chk = inner_cplx(D, H, Sn) / rho_series[0]
    rho_series = rho_series * chk
    maxlag = int(0.030 * FS)
    idxs = np.concatenate([np.arange(0, maxlag + 1), np.arange(NSEG - maxlag, NSEG)])
    ipk = idxs[int(np.argmax(np.abs(rho_series[idxs])))]
    tau = (ipk / FS) if ipk <= maxlag else ((ipk - NSEG) / FS)
    rho_pk = rho_series[ipk]
    alpha = rho_pk / hh
    Hfit = H * alpha * np.exp(-2j * np.pi * FREQS * tau)
    R = D - Hfit
    dw = whiten_td_from_fd(D, Sn); rw = whiten_td_from_fd(R, Sn)
    edge = np.concatenate([dw[:int(16 * FS)], dw[int(48 * FS):]])
    sc = 1.0 / np.std(edge)
    return dict(alpha=complex(alpha), tau=float(tau), R_fd=R, resid_w=rw * sc, sc=float(sc))

tc_idx = int((TC_GEO - SEG_START) * FS)

def fit_and_stats(xH1, xL1):
    fits = {"H1": fit_one_det("H1", xH1), "L1": fit_one_det("L1", xL1)}
    S = {}
    for wname, (w0, w1) in WINDOWS.items():
        i0w, i1w = tc_idx + int(w0 * FS), tc_idx + int(w1 * FS)
        S[wname] = float(xcorr_stat(fits["H1"]["resid_w"], fits["L1"]["resid_w"], i0w, i1w)[0])
    return fits, S

def cross_frac(fs_, ss_, thr):
    for a, b, sa, sb in zip(fs_[:-1], fs_[1:], ss_[:-1], ss_[1:]):
        if sa < thr <= sb:
            return float(a + (b - a) * (thr - sa) / (sb - sa))
    return None

# 背景は保存済み配列を再利用する (取り直さない)
z = np.load(f"{WD}/waveform_series.npz")
BG = {"plan_[-1.5,+0.5]": z["bg_plan"], "matched_[-0.6,+0.3]": z["bg_matched"]}
P99 = {w: float(np.percentile(BG[w], 99)) for w in WINDOWS}
BMAX = {w: float(BG[w].max()) for w in WINDOWS}
stored = json.load(open(f"{WD}/waveform_results.json"))
sens = json.load(open(f"{WD}/gw231123_sensitivity.json"))

OUT_PATH = f"{WD}/gw231123_inject_recover.json"
out = {}
if os.path.exists(OUT_PATH):
    out = json.load(open(OUT_PATH))
out["background"] = {w: dict(n=int(len(BG[w])), p99=P99[w], max=BMAX[w]) for w in WINDOWS}
out["postfit_f99_reference"] = {w: sens[w]["frac_at_p99"] for w in WINDOWS}

# ================================================================ Stage A
def run_stage_a():
    print("=== Stage A: baseline (f=0) ===", flush=True)
    fits0, S0 = fit_and_stats(data["H1"]["x"], data["L1"]["x"])
    base = {}
    for det in ["H1", "L1"]:
        a0 = fits0[det]["alpha"]
        base[det] = dict(alpha_abs=float(abs(a0)), alpha_phase_deg=float(np.degrees(np.angle(a0))),
                         tau_ms=float(fits0[det]["tau"] * 1e3),
                         stored_alpha_abs=stored["per_detector"][det]["alpha_abs"])
        print(f"  {det}: |alpha0|={abs(a0):.4f} (stored {base[det]['stored_alpha_abs']:.4f}), "
              f"tau={fits0[det]['tau']*1e3:+.2f} ms", flush=True)
    for w in WINDOWS:
        print(f"  [{w}] S0={S0[w]:.4f} (stored S_residual {stored['windows'][w]['S_residual']:.4f})",
              flush=True)
    out["stageA_baseline"] = dict(per_detector=base, S=S0,
                                  stored_S={w: stored["windows"][w]["S_residual"] for w in WINDOWS})

    FRACS_A = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    rows = []
    for f in FRACS_A:
        # ★ 注入はフィットの前: f*h(t) を実ストレインへ加えてから fit_and_stats を呼ぶ
        xH1 = inject_into_strain("H1", f * HSEG["H1"])
        xL1 = inject_into_strain("L1", f * HSEG["L1"])
        fits, S = fit_and_stats(xH1, xL1)
        row = {"f": f, "S": S, "per_detector": {}}
        for det in ["H1", "L1"]:
            a_inj = fits[det]["alpha"]; a_0 = fits0[det]["alpha"]
            dalpha = a_inj - a_0
            # 残差に残った実効 f: (R_inj - R_0) の H 方向成分
            q = inner_cplx(fits[det]["R_fd"] - fits0[det]["R_fd"], HFD[det], Sn_all[det]) / HH[det]
            row["per_detector"][det] = dict(
                alpha_abs=float(abs(a_inj)), alpha_phase_deg=float(np.degrees(np.angle(a_inj))),
                tau_ms=float(fits[det]["tau"] * 1e3),
                dalpha_re=float(dalpha.real), dalpha_im=float(dalpha.imag),
                dalpha_abs=float(abs(dalpha)), dalpha_over_f=float(abs(dalpha) / f),
                resid_eff_f_abs=float(abs(q)))
        rows.append(row)
        print(f"  f={f}: S_plan={S['plan_[-1.5,+0.5]']:.4f} S_matched={S['matched_[-0.6,+0.3]']:.4f} "
              f"|da|/f H1={row['per_detector']['H1']['dalpha_over_f']:.3f} "
              f"L1={row['per_detector']['L1']['dalpha_over_f']:.3f} "
              f"|q| H1={row['per_detector']['H1']['resid_eff_f_abs']:.4f} "
              f"L1={row['per_detector']['L1']['resid_eff_f_abs']:.4f}", flush=True)
    f99 = {}
    for w in WINDOWS:
        xs = [0.0] + FRACS_A
        ys = [S0[w]] + [r["S"][w] for r in rows]
        f99[w] = cross_frac(xs, ys, P99[w])
    out["stageA"] = dict(fractions=FRACS_A, rows=rows, f99_fit_inclusive=f99,
                         f99_postfit_reference={w: sens[w]["frac_at_p99"] for w in WINDOWS})
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"Stage A f99 (fit-inclusive): {f99}", flush=True)
    print(f"saved {OUT_PATH}", flush=True)

# ================================================================ Stage B
mp.mp.dps = 50
_GMSUN_MP = mp.mpf("4.9254909476412675e-6")

def _F_mpmath_point(M_msun, y, f):
    """F(f) の 1 点を mpmath (50 桁) で直接評価する.

    scipy.special.hyp1f1 は複素引数を扱えず, 複素128 の級数展開は
    大きな w*y^2 で桁落ち (中間項 ~1e62) するため, 高精度評価を使う.
    """
    w = 8 * mp.pi * _GMSUN_MP * mp.mpf(M_msun) * mp.mpf(f)
    if w == 0:
        return complex(1.0)
    xm = (mp.mpf(y) + mp.sqrt(mp.mpf(y) ** 2 + 4)) / 2
    phi_m = (xm - mp.mpf(y)) ** 2 / 2 - mp.log(xm)
    v = (mp.e ** (mp.pi * w / 4 + 1j * w / 2 * (mp.log(w / 2) - 2 * phi_m))
         * mp.gamma(1 - 1j * w / 2)
         * mp.hyper([1j * w / 2], [1], 1j * w * mp.mpf(y) ** 2 / 2))
    return complex(v)

def lens_F(M_msun, y, freqs):
    """点質量レンズの波動光学増幅率 F(f) (Takahashi & Nakamura 2003 の標準形).

    M_msun は赤方偏移込みレンズ質量 M_z. w = 8 pi G M_z f / c^3.
    0.25 Hz グリッドで mpmath 直接評価し, 3 次スプラインで補間する
    (F の振動周期 = うなり周期 >= 11 Hz に対し十分細かい).
    戻り値の第 2 要素は, 帯域内ランダム点での直接評価との最大相対誤差 (検証用).
    """
    fg = np.arange(0.0, 512.0 + 0.25, 0.25)
    Fg = np.array([_F_mpmath_point(M_msun, y, float(f)) for f in fg])
    F = CubicSpline(fg, Fg.real)(freqs) + 1j * CubicSpline(fg, Fg.imag)(freqs)
    F = np.asarray(F, dtype=complex)
    F[freqs == 0] = 1.0
    rng = np.random.default_rng(20261115)
    chk_f = rng.uniform(20.0, 448.0, 16)
    chk_interp = CubicSpline(fg, Fg.real)(chk_f) + 1j * CubicSpline(fg, Fg.imag)(chk_f)
    errs = []
    for ci, f in zip(chk_interp, chk_f):
        direct = _F_mpmath_point(M_msun, y, float(f))
        errs.append(abs(ci - direct) / abs(direct))
    return F, float(max(errs))

def delta_T(y):
    s = np.sqrt(y * y + 4.0)
    return 0.5 * y * s + np.log((s + y) / (s - y))

def run_stage_b():
    print("=== Stage B: wave-optics two-image lens ===", flush=True)
    fits0, S0 = fit_and_stats(data["H1"]["x"], data["L1"]["x"])
    GRID_M = [100.0, 200.0, 400.0, 1000.0]
    GRID_Y = [0.3, 0.5, 1.0, 2.0]
    G_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    results = []
    for M in GRID_M:
        for y in GRID_Y:
            F, interp_err = lens_F(M, y, FREQS)
            dtd = 4.0 * GMSUN_C3 * M * delta_T(y)
            # 検算: 帯域平均 |F|^2 vs 幾何光学 mu_tot, SNR 重み付き |F-1| 実効値
            mu_tot = (y * y + 2.0) / (y * np.sqrt(y * y + 4.0))
            meanF2 = float(np.mean(np.abs(F[BAND]) ** 2))
            num = den = 0.0
            for det in ["H1", "L1"]:
                wt = np.abs(HFD[det][BAND]) ** 2 / Sn_all[det][BAND]
                num += float(np.sum(np.abs(F[BAND] - 1.0) ** 2 * wt)); den += float(np.sum(wt))
            f_eq = float(np.sqrt(num / den))
            # 歪み系列 (検出器ごと): dH = (F-1) H, TD へ戻す
            dTD = {}
            for det in ["H1", "L1"]:
                dH = (F - 1.0) * HFD[det]
                dTD[det] = np.fft.irfft(dH / dt, n=NSEG)
            grows = []
            for g in G_GRID:
                # ★ 注入はフィットの前: g*(F-1)h を実ストレインへ加えてからフィット
                xH1 = inject_into_strain("H1", g * dTD["H1"])
                xL1 = inject_into_strain("L1", g * dTD["L1"])
                fits, S = fit_and_stats(xH1, xL1)
                grows.append({"g": g, "S": S})
            g99 = {}
            for wname in WINDOWS:
                xs = [0.0] + G_GRID
                ys = [S0[wname]] + [gr["S"][wname] for gr in grows]
                g99[wname] = cross_frac(xs, ys, P99[wname])
            rec = dict(M_msun=M, y=y, delta_t_ms=float(dtd * 1e3),
                       beat_freq_Hz=float(1.0 / dtd), mu_total=float(mu_tot),
                       F_interp_max_relerr=interp_err,
                       band_mean_absF2=meanF2, f_eq_distortion=f_eq,
                       rows=grows, g99=g99,
                       S_at_g1={wname: grows[3]["S"][wname] for wname in WINDOWS})
            results.append(rec)
            print(f"  M={M:.0f} y={y}: dt={dtd*1e3:.1f} ms beat={1.0/dtd:.0f} Hz "
                  f"mu={mu_tot:.2f} <|F|^2>={meanF2:.2f} f_eq={f_eq:.3f} "
                  f"g99 plan={g99['plan_[-1.5,+0.5]']} matched={g99['matched_[-0.6,+0.3]']}", flush=True)
    out["stageB"] = dict(grid_M=GRID_M, grid_y=GRID_Y, grid_g=G_GRID, results=results,
                         S_baseline=S0)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"saved {OUT_PATH}", flush=True)

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    if stage == "A":
        run_stage_a()
    elif stage == "B":
        run_stage_b()
    else:
        raise SystemExit(f"unknown stage {stage}")
