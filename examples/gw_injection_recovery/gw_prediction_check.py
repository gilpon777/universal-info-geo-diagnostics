#!/usr/bin/env python3
"""GW231123 Stage B 予測点検証: 既存格子の外の 3 点で g99 を実測する.

対象 (ゴーサイン記録 2, 2026-11-16):
  (M_z, y) = (2000, 0.3), (2000, 0.5), (500, 0.5)

予測は osc_proj 物差し仮説 (Claude v6 §4.3, gw_osc_diagnostic.py で実設定計算):
  g99_plan ≈ 0.5 / osc_proj   (計画窓)
    (2000, 0.3): osc_proj = 0.93357 -> 予測 0.536
    (2000, 0.5): osc_proj = 0.75135 -> 予測 0.665
    (500, 0.5) : osc_proj = 0.31622 -> 予測 1.581

設定は既存 Stage B と完全に同一:
  - gw231123_inject_recover.py をモジュールとして読み込み, その前処理・
    テンプレート・lens_F・fit_and_stats・背景 p99・g グリッドをそのまま使う.
  - 注入は必ずフィットの前 (inject_into_strain 経由). PE サンプル idx 19435 固定.
  - g グリッド [0.1, 0.25, 0.5, 1.0, 2.0, 4.0], g99 は cross_frac の線形補間.

!!! 既存結果ファイル wf_data/gw231123_inject_recover.json には書き込まない !!!
出力は gw_prediction_check.json のみ (考察は gw_prediction_check.md に別途).
"""
import json, os, sys
import numpy as np

WORK = os.path.expanduser("~/workspace/info_geo_diag/void_validation")
WD = WORK + "/wf_data"
sys.path.insert(0, WORK)

import gw231123_inject_recover as base  # 前処理・テンプレート構築が走る

WINDOWS = base.WINDOWS
PLAN = "plan_[-1.5,+0.5]"
MATCHED = "matched_[-0.6,+0.3]"
G_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]  # 既存 Stage B と同一

TARGETS = [(2000.0, 0.3), (2000.0, 0.5), (500.0, 0.5)]

# 予測 (gw_osc_diagnostic.json の prediction_points から実数を読む)
diag = json.load(open(f"{WORK}/gw_osc_diagnostic.json"))
pred_diag = {(p["M_msun"], p["y"]): p for p in diag["prediction_points"]}
prod_ref = {w: diag["product_check"][w]["product_proj_stats"] for w in diag["product_check"]}

saved = json.load(open(f"{WD}/gw231123_inject_recover.json"))
saved_S0 = saved["stageB"]["S_baseline"]

# ---------------------------------------------------------------- ベースライン照合
print("=== baseline check (settings identity) ===", flush=True)
fits0, S0 = base.fit_and_stats(base.data["H1"]["x"], base.data["L1"]["x"])
base_diff = {w: float(abs(S0[w] - saved_S0[w])) for w in WINDOWS}
for w in WINDOWS:
    print(f"  [{w}] S0={S0[w]:.6f} (saved {saved_S0[w]:.6f}, diff {base_diff[w]:.2e})", flush=True)

# ---------------------------------------------------------------- 3 点の実行
results = []
for (M, y) in TARGETS:
    print(f"=== (M_z, y) = ({M:.0f}, {y}) ===", flush=True)
    F, interp_err = base.lens_F(M, y, base.FREQS)
    dtd = 4.0 * base.GMSUN_C3 * M * base.delta_T(y)
    mu_tot = (y * y + 2.0) / (y * np.sqrt(y * y + 4.0))
    meanF2 = float(np.mean(np.abs(F[base.BAND]) ** 2))
    num = den = 0.0
    for det in ["H1", "L1"]:
        wt = np.abs(base.HFD[det][base.BAND]) ** 2 / base.Sn_all[det][base.BAND]
        num += float(np.sum(np.abs(F[base.BAND] - 1.0) ** 2 * wt)); den += float(np.sum(wt))
    f_eq = float(np.sqrt(num / den))
    dTD = {}
    for det in ["H1", "L1"]:
        dH = (F - 1.0) * base.HFD[det]
        dTD[det] = np.fft.irfft(dH / base.dt, n=base.NSEG)
    grows = []
    for g in G_GRID:
        # ★ 注入はフィットの前: g*(F-1)h を実ストレインへ加えてからフィット
        xH1 = base.inject_into_strain("H1", g * dTD["H1"])
        xL1 = base.inject_into_strain("L1", g * dTD["L1"])
        fits, S = base.fit_and_stats(xH1, xL1)
        grows.append({"g": g, "S": S})
        print(f"  g={g}: S_plan={S[PLAN]:.4f} S_matched={S[MATCHED]:.4f}", flush=True)
    g99 = {}
    for wname in WINDOWS:
        xs = [0.0] + G_GRID
        ys = [S0[wname]] + [gr["S"][wname] for gr in grows]
        g99[wname] = base.cross_frac(xs, ys, base.P99[wname])

    pd_ = pred_diag[(M, y)]
    osc_proj = float(pd_["osc_proj"]); osc_rms = float(pd_["osc_rms"])
    g99_pred_plan = float(pd_["g99_plan_hypothesis_from_osc_proj"])
    meas_plan = g99[PLAN]; meas_matched = g99[MATCHED]
    rec = dict(
        M_msun=M, y=y,
        delta_t_ms=float(dtd * 1e3), beat_freq_Hz=float(1.0 / dtd),
        mu_total=float(mu_tot), F_interp_max_relerr=interp_err,
        band_mean_absF2=meanF2, f_eq_distortion=f_eq,
        osc_rms=osc_rms, osc_proj=osc_proj,
        g99_predicted_plan=g99_pred_plan,
        g99_measured=g99,
        ratio_measured_over_predicted_plan=(float(meas_plan / g99_pred_plan)
                                            if meas_plan is not None else None),
        implied_product_oscproj_x_g99={
            w: (float(osc_proj * g99[w]) if g99[w] is not None else None) for w in WINDOWS},
        rows=grows,
        S_at_g1={wname: grows[3]["S"][wname] for wname in WINDOWS},
    )
    results.append(rec)
    print(f"  -> g99 plan={meas_plan} (pred {g99_pred_plan:.3f}, "
          f"ratio {rec['ratio_measured_over_predicted_plan']}), matched={meas_matched}", flush=True)

out = dict(
    purpose="Stage B prediction-point check: measured g99 at 3 points outside the original grid, "
            "testing the osc_proj yardstick hypothesis g99_plan ~ 0.5/osc_proj.",
    settings=dict(
        identical_to="gw231123_inject_recover.py run_stage_b (same module imported)",
        sample_idx=base.SAMPLE_IDX, band_Hz=[base.F_LO, base.F_HI],
        g_grid=G_GRID, windows=list(WINDOWS.keys()),
        background={w: dict(n=int(len(base.BG[w])), p99=base.P99[w], max=base.BMAX[w])
                    for w in WINDOWS},
        injection="pre-fit, into real strain via inject_into_strain",
    ),
    baseline_check=dict(S0_measured=S0, S0_saved_stageB=saved_S0, abs_diff=base_diff),
    prediction_source="gw_osc_diagnostic.json prediction_points (osc_proj under actual settings; "
                      "hypothesis g99_plan = 0.5/osc_proj, Claude v6 section 4.3)",
    product_reference_from_existing_grid={
        w: dict(mean=prod_ref[w]["mean"], std=prod_ref[w]["std"], cv=prod_ref[w]["cv"],
                min=prod_ref[w]["min"], max=prod_ref[w]["max"], n=prod_ref[w]["n"])
        for w in prod_ref},
    results=results,
)
jp = f"{WORK}/gw_prediction_check.json"
json.dump(out, open(jp, "w"), indent=1)
print(f"saved {jp}", flush=True)
