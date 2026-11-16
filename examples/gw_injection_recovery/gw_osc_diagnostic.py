#!/usr/bin/env python3
"""GW231123 Stage B の osc 診断量の再現と, 射影版 osc_proj の計算.

目的 (Claude v6 §4.3 への回答材料):
  1. 保存済み診断量 osc_rms (gw231123_inject_recover.json の
     stageB_osc_diagnostic, 16 格子点) を, gw231123_inject_recover.py と
     同一の実設定で再計算し, 全点で照合する.
  2. 同じ重みで osc_proj を計算する: F-1 を {1, f} の張る複素空間へ
     加重射影し, 残差の加重 RMS を取る (Claude v6 §4.3(b) の定義).
     {1} だけへの射影が osc_rms (平均除去) に相当し, osc_proj はさらに
     周波数線形成分 (フィットの tau 自由度が吸収する方向) を除いた量.
  3. Claude の予測点 (2000,0.3), (2000,0.5), (5000,1.0), (500,0.5) の
     osc_rms / osc_proj を計算する.
  4. 保存済み g99 (Stage B の注入・回収結果) と osc_proj の積が
     ほぼ一定 (Claude の言う約 0.5) になるか検算する.

!!! Stage B 本体 (実ストレインへの注入・回収) は再実行しない !!!
本スクリプトはストレインを読み込まず, テンプレートと PSD から重みを
再構成し, レンズ増幅率 F(f) の診断量だけを計算する. g99 は保存 JSON の
値をそのまま使う.

実設定 (gw231123_inject_recover.py と同一, 照合済み):
  - 帯域 20-448 Hz (64 s セグメントの rfft グリッド, df = 1/64 Hz)
  - 重み w(f) = sum_det |H_det(f)|^2 / Sn_det(f)  (H1+L1, f_eq と同一の SNR 重み)
    H は PE サンプル idx 19435 固定のテンプレート, Sn は PE リリースの PSD
  - osc_rms = F から同重みの複素加重平均 <F>_w を引いた残りの加重 RMS
  - lens_F は元スクリプトから逐語コピー (mpmath 50 桁, 0.25 Hz グリッド
    + 3 次スプライン)
"""
import json, os
import numpy as np
import h5py
from scipy.interpolate import CubicSpline
import mpmath as mp
import lal
import lalsimulation as lalsim

WORK = os.path.expanduser("~/workspace/info_geo_diag/void_validation")
WD = WORK + "/wf_data"
FS = 1024
F_LO, F_HI = 20.0, 448.0
SEG_T = 64.0
SAMPLE_IDX = 19435
dt = 1.0 / FS
NSEG = int(SEG_T * FS)
DF = 1.0 / SEG_T
FREQS = np.fft.rfftfreq(NSEG, dt)
BAND = (FREQS >= F_LO) & (FREQS <= F_HI)

# ---------------------------------------------------------------- PSD + テンプレート
# (gw231123_inject_recover.py の構築と同一. ストレイン読み込みは重みに
#  影響しないため省略する)
pe = h5py.File(f"{WD}/GW231123_PEDataRelease.hdf5", "r")
grp = pe["C00:IMRPhenomXPHM-SpinTaylor"]
ps = grp["posterior_samples"]
pe_psd = {}
for det in ["H1", "L1"]:
    arr = grp["psds"][det][:]
    pe_psd[det] = (arr[:, 0], arr[:, 1])

Sn_all = {}
for det in ["H1", "L1"]:
    fr_pe, P_pe = pe_psd[det]
    lg = np.interp(np.log(np.maximum(FREQS, 0.5)), np.log(fr_pe[1:]), np.log(P_pe[1:]))
    Sn = np.exp(lg); Sn[0] = P_pe[0]
    Sn_all[det] = Sn

r = ps[SAMPLE_IDX]
ml = {k: float(r[k]) for k in ps.dtype.names if np.ndim(r[k]) == 0}
ref = json.load(open(f"{WD}/waveform_results.json"))["selected_sample"]
for k in ["mass_1", "mass_2", "luminosity_distance", "geocent_time"]:
    assert abs(ml[k] - ref[k]) < 1e-6 * max(1.0, abs(ref[k])), f"sample mismatch {k}"
TC_GEO = float(ml["geocent_time"])
SEG_START = TC_GEO - SEG_T / 2
print(f"sample idx={SAMPLE_IDX}: m1={ml['mass_1']:.1f} m2={ml['mass_2']:.1f} "
      f"DL={ml['luminosity_distance']:.0f}", flush=True)

hp_td, hx_td = lalsim.SimInspiralChooseTDWaveform(
    float(ml["mass_1"]) * lal.MSUN_SI, float(ml["mass_2"]) * lal.MSUN_SI,
    float(ml["spin_1x"]), float(ml["spin_1y"]), float(ml["spin_1z"]),
    float(ml["spin_2x"]), float(ml["spin_2y"]), float(ml["spin_2z"]),
    float(ml["luminosity_distance"]) * 1e6 * lal.PC_SI,
    float(ml["theta_jn"]), float(ml["phase"]), 0.0, 0.0, 0.0,
    dt, 10.0, 10.0, None, lalsim.IMRPhenomXPHM)
hp_a, hx_a = hp_td.data.data, hx_td.data.data
gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(TC_GEO))

HFD = {}
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
    HFD[det] = np.fft.rfft(hseg) * dt

# 重み w(f) = sum_det |H_det(f)|^2 / Sn_det(f), 帯域内のみ
FB = FREQS[BAND]
W = np.zeros(len(FB))
for det in ["H1", "L1"]:
    W += np.abs(HFD[det][BAND]) ** 2 / Sn_all[det][BAND]
WSUM = float(np.sum(W))
print(f"band bins: {len(FB)}, weight sum = {WSUM:.6e}", flush=True)

# ---------------------------------------------------------------- lens_F (元スクリプトから逐語コピー)
mp.mp.dps = 50
_GMSUN_MP = mp.mpf("4.9254909476412675e-6")

def _F_mpmath_point(M_msun, y, f):
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

# ---------------------------------------------------------------- 診断量
def diagnostics(F_full):
    """帯域内の F から osc_rms / osc_proj 等を返す.

    osc_rms : F - <F>_w の加重 RMS (<F>_w は複素加重平均)
    osc_proj: F-1 を {1, f} (複素係数) へ加重最小二乗射影した残差の加重 RMS
    """
    Fb = F_full[BAND]
    Fmean = np.sum(W * Fb) / WSUM
    osc_rms = float(np.sqrt(np.sum(W * np.abs(Fb - Fmean) ** 2) / WSUM))
    v = Fb - 1.0
    # 基底 phi0 = 1, phi1 = f (実数). Gram 行列は実数, 右辺は複素数.
    S0 = WSUM
    S1 = float(np.sum(W * FB))
    S2 = float(np.sum(W * FB ** 2))
    G = np.array([[S0, S1], [S1, S2]])
    rhs = np.array([np.sum(W * v), np.sum(W * FB * v)], dtype=complex)
    a, b = np.linalg.solve(G, rhs)
    resid = v - (a + b * FB)
    osc_proj = float(np.sqrt(np.sum(W * np.abs(resid) ** 2) / WSUM))
    return dict(F_weighted_mean_abs=float(abs(Fmean)),
                F_weighted_mean_re=float(Fmean.real), F_weighted_mean_im=float(Fmean.imag),
                osc_rms=osc_rms, osc_proj=osc_proj,
                remaining_ratio=float(osc_proj / osc_rms),
                proj_coef_a_re=float(a.real), proj_coef_a_im=float(a.imag),
                proj_coef_b_re=float(b.real), proj_coef_b_im=float(b.imag))

def main():
    saved = json.load(open(f"{WD}/gw231123_inject_recover.json"))
    saved_osc = {(e["M_msun"], e["y"]): e for e in saved["stageB_osc_diagnostic"]}
    saved_g99 = {(r_["M_msun"], r_["y"]): r_["g99"] for r_ in saved["stageB"]["results"]}
    WINDOWS = list(saved["background"].keys())  # plan / matched の順を保存値に合わせる

    GRID_M = [100.0, 200.0, 400.0, 1000.0]
    GRID_Y = [0.3, 0.5, 1.0, 2.0]
    PRED = [(2000.0, 0.3), (2000.0, 0.5), (5000.0, 1.0), (500.0, 0.5)]

    out = {
        "settings": {
            "band_Hz": [F_LO, F_HI],
            "weight": "w(f) = sum_det |H_det(f)|^2 / Sn_det(f), H1+L1, PE sample idx 19435 template, PE PSD",
            "osc_rms": "weighted RMS of F - <F>_w (complex weighted mean removed)",
            "osc_proj": "weighted RMS of residual of (F-1) after weighted least-squares projection onto span{1, f} (complex coefficients)",
            "note": "Stage B injection/recovery was NOT re-run; g99 values are taken from the saved JSON.",
        },
        "reproduction": [], "grid": [], "prediction_points": [],
        "product_check": {},
    }

    max_err_osc, max_err_mean = 0.0, 0.0
    for M in GRID_M:
        for y in GRID_Y:
            F, ierr = lens_F(M, y, FREQS)
            dg = diagnostics(F)
            sv = saved_osc[(M, y)]
            e_osc = abs(dg["osc_rms"] - sv["osc_rms_after_mean_removal"])
            e_mean = abs(dg["F_weighted_mean_abs"] - sv["F_weighted_mean_abs"])
            max_err_osc = max(max_err_osc, e_osc); max_err_mean = max(max_err_mean, e_mean)
            out["reproduction"].append(dict(
                M_msun=M, y=y, osc_rms_saved=sv["osc_rms_after_mean_removal"],
                osc_rms_recomputed=dg["osc_rms"], abs_err=e_osc,
                F_mean_abs_saved=sv["F_weighted_mean_abs"],
                F_mean_abs_recomputed=dg["F_weighted_mean_abs"], abs_err_mean=e_mean,
                F_interp_max_relerr=ierr))
            rec = dict(M_msun=M, y=y, **dg, F_interp_max_relerr=ierr,
                       g99={w: saved_g99[(M, y)][w] for w in WINDOWS})
            out["grid"].append(rec)
            print(f"M={M:.0f} y={y}: osc_rms={dg['osc_rms']:.5f} (saved {sv['osc_rms_after_mean_removal']:.5f}, "
                  f"err {e_osc:.2e}) osc_proj={dg['osc_proj']:.5f} ratio={dg['remaining_ratio']:.3f}", flush=True)
    out["reproduction_max_abs_err_osc_rms"] = max_err_osc
    out["reproduction_max_abs_err_F_mean_abs"] = max_err_mean
    print(f"max reproduction error: osc_rms {max_err_osc:.3e}, |<F>| {max_err_mean:.3e}", flush=True)

    for (M, y) in PRED:
        F, ierr = lens_F(M, y, FREQS)
        dg = diagnostics(F)
        rec = dict(M_msun=M, y=y, **dg, F_interp_max_relerr=ierr,
                   g99_plan_hypothesis_from_osc_proj=float(0.5 / dg["osc_proj"]),
                   g99_plan_hypothesis_note="HYPOTHESIS (Claude v6): g99 ~ 0.5 / osc_proj, plan window. Not a measurement.")
        out["prediction_points"].append(rec)
        print(f"PRED M={M:.0f} y={y}: osc_rms={dg['osc_rms']:.5f} osc_proj={dg['osc_proj']:.5f} "
              f"ratio={dg['remaining_ratio']:.3f} g99_hyp={rec['g99_plan_hypothesis_from_osc_proj']:.3f}", flush=True)

    # 積の一定性の検算 (有限な g99 を持つ格子点のみ)
    for w in WINDOWS:
        rows = []
        for rec in out["grid"]:
            g = rec["g99"][w]
            if g is None:
                continue
            rows.append(dict(M_msun=rec["M_msun"], y=rec["y"], g99=g,
                             osc_proj=rec["osc_proj"], osc_rms=rec["osc_rms"],
                             product_proj=float(rec["osc_proj"] * g),
                             product_rms=float(rec["osc_rms"] * g)))
        def stats(vals):
            v = np.array(vals, dtype=float)
            return dict(n=int(len(v)), mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        min=float(v.min()), max=float(v.max()),
                        cv=float(v.std(ddof=1) / v.mean()) if len(v) > 1 else 0.0)
        out["product_check"][w] = dict(
            points=rows,
            product_proj_stats=stats([r_["product_proj"] for r_ in rows]),
            product_rms_stats=stats([r_["product_rms"] for r_ in rows]))
        sp = out["product_check"][w]["product_proj_stats"]
        sr = out["product_check"][w]["product_rms_stats"]
        print(f"[{w}] n={sp['n']}  osc_proj*g99: mean={sp['mean']:.3f} std={sp['std']:.3f} "
              f"range=[{sp['min']:.3f},{sp['max']:.3f}] CV={sp['cv']:.2f} | "
              f"osc_rms*g99: mean={sr['mean']:.3f} range=[{sr['min']:.3f},{sr['max']:.3f}] CV={sr['cv']:.2f}", flush=True)

    jp = f"{WORK}/gw_osc_diagnostic.json"
    json.dump(out, open(jp, "w"), indent=1)
    print(f"saved {jp}", flush=True)

    # ---------------------------------------------------------------- MD
    L = []
    L.append("# GW231123 osc 診断: 再現照合と射影版 osc_proj")
    L.append("")
    L.append("Stage B 本体（実ストレインへの注入・回収）は再実行していない。"
             "g99 は保存済み結果（`wf_data/gw231123_inject_recover.json`）の値をそのまま使う。"
             "本ファイルは診断量 F(f) の計算のみ。")
    L.append("")
    L.append("## 設定（gw231123_inject_recover.py と同一）")
    L.append("")
    L.append("- 帯域 20–448 Hz（64 s セグメント、df = 1/64 Hz）")
    L.append("- 重み w(f) = Σ_det |H_det(f)|² / Sn_det(f)（H1+L1、PE サンプル idx 19435 のテンプレート、PE PSD。f_eq と同一の SNR 重み）")
    L.append("- osc_rms: F から複素加重平均 ⟨F⟩_w を引いた残りの加重 RMS")
    L.append("- osc_proj: F−1 を {1, f} の張る複素空間へ加重最小二乗射影し、残差の加重 RMS を取る（Claude v6 §4.3(b)）。"
             "{1} のみへの射影が平均除去に相当するので、osc_proj は「平均＋周波数線形成分」を除いた残り。"
             "線形成分はフィットの τ（到着時刻ずれ）が吸収する方向に対応する。")
    L.append("")
    L.append("## 1. 再現照合（16 格子点、保存値 vs 再計算）")
    L.append("")
    L.append(f"- osc_rms の最大絶対誤差: **{max_err_osc:.3e}**")
    L.append(f"- |⟨F⟩_w| の最大絶対誤差: **{max_err_mean:.3e}**")
    L.append("")
    L.append("| M_z | y | osc_rms（保存） | osc_rms（再計算） | 絶対誤差 |")
    L.append("|---|---|---|---|---|")
    for rp in out["reproduction"]:
        L.append(f"| {rp['M_msun']:.0f} | {rp['y']} | {rp['osc_rms_saved']:.5f} | "
                 f"{rp['osc_rms_recomputed']:.5f} | {rp['abs_err']:.2e} |")
    L.append("")
    L.append("## 2. osc_proj（16 格子点）")
    L.append("")
    L.append("| M_z | y | osc_rms | osc_proj | 残る割合 osc_proj/osc_rms |")
    L.append("|---|---|---|---|---|")
    for rec in out["grid"]:
        L.append(f"| {rec['M_msun']:.0f} | {rec['y']} | {rec['osc_rms']:.5f} | "
                 f"{rec['osc_proj']:.5f} | {rec['remaining_ratio']:.3f} |")
    L.append("")
    L.append("## 3. Claude の予測点")
    L.append("")
    L.append("g99 予測は Claude の仮説 g99 ≈ 0.5/osc_proj（計画窓）による外挿であり、**測定ではない**。")
    L.append("")
    L.append("| M_z | y | osc_rms | osc_proj | 残る割合 | g99 予測（仮説） |")
    L.append("|---|---|---|---|---|---|")
    for rec in out["prediction_points"]:
        L.append(f"| {rec['M_msun']:.0f} | {rec['y']} | {rec['osc_rms']:.5f} | "
                 f"{rec['osc_proj']:.5f} | {rec['remaining_ratio']:.3f} | "
                 f"{rec['g99_plan_hypothesis_from_osc_proj']:.2f} |")
    L.append("")
    L.append("## 4. 積 osc_proj × g99 の一定性の検算")
    L.append("")
    L.append("有限な g99 を持つ格子点のみ（g99 が None の点は g=4 でも背景 p99 を超えなかった点で、積は定義できない）。")
    L.append("参考として osc_rms × g99 も併記する。")
    for w in WINDOWS:
        pc = out["product_check"][w]
        sp, sr = pc["product_proj_stats"], pc["product_rms_stats"]
        L.append("")
        L.append(f"### 窓: {w}（n = {sp['n']}）")
        L.append("")
        L.append("| M_z | y | g99 | osc_proj | osc_proj×g99 | osc_rms×g99 |")
        L.append("|---|---|---|---|---|---|")
        for r_ in pc["points"]:
            L.append(f"| {r_['M_msun']:.0f} | {r_['y']} | {r_['g99']:.3f} | {r_['osc_proj']:.5f} | "
                     f"{r_['product_proj']:.3f} | {r_['product_rms']:.3f} |")
        L.append("")
        L.append(f"- osc_proj×g99: 平均 {sp['mean']:.3f}、標準偏差 {sp['std']:.3f}、範囲 [{sp['min']:.3f}, {sp['max']:.3f}]、変動係数 {sp['cv']:.2f}")
        L.append(f"- osc_rms×g99: 平均 {sr['mean']:.3f}、標準偏差 {sr['std']:.3f}、範囲 [{sr['min']:.3f}, {sr['max']:.3f}]、変動係数 {sr['cv']:.2f}")
    L.append("")
    L.append("## 注意")
    L.append("")
    L.append("- g99 は単一ノイズ実現・固定テンプレートでの Stage B 保存値であり、その不確かさは積のばらつきにそのまま乗る。")
    L.append("- 射影基底の f は Hz 単位の生の周波数。射影は加重内積による厳密な最小二乗。")
    L.append("- F のスプライン補間誤差（帯域内ランダム点での直接評価との最大相対誤差）は JSON に点ごとに記録した。")
    mp_ = f"{WORK}/gw_osc_diagnostic.md"
    open(mp_, "w").write("\n".join(L) + "\n")
    print(f"saved {mp_}", flush=True)

if __name__ == "__main__":
    main()
