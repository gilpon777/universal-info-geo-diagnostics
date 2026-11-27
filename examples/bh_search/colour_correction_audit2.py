#!/usr/bin/env python3
"""色調補正 B′ の完全監査 v2（ChatGPT 第21回・優先1/2/4/5、2026-11-28 Muse）。

v1 の不備（ChatGPT の指摘）: label_with() が M_G_corr と dcolour だけを更新し、
A_G_tge が元の A_0 のまま残っていたため、光度天井が B′ の減光で計算されていなかった。
v2 では B′ の判定入力を完全に更新する:
  A_G_tge' = k_G × A_0、A_G,max' = max(A_G_gsp, A_G_tge')、
  M_G_corr' = M_G - A_G,max'、dcolour' は A_used = k_G × A_0 で計算。
  判定は丸め前の値で行う。
安定性は k_G を細かく走査し、主系列アンカーの折れ点を M_G,0 が横切る k_G も
評価点に含める。r の範囲 [0.531, 0.563] は GSP 実測10件の最小・最大であり、
統計的な信頼区間ではない（感度幅として使う）。
実行: python3 colour_correction_audit2.py
"""
import csv, importlib.util, io, contextlib, json

spec = importlib.util.spec_from_file_location("tab", "dr4_m1m2_table.py")
tab = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(tab)
import dr4_apply_rules_v4 as rules

gsp = {}
with open("gaia_gspphot_13.csv") as fh:
    for row in csv.DictReader(fh):
        gsp[row["source_id"]] = {
            "bp_rp": float(row["bp_rp"]),
            "ag": float(row["ag_gspphot"]) if row["ag_gspphot"] else None,
            "ebp": float(row["ebpminrp_gspphot"]) if row["ebpminrp_gspphot"] else None}

R_MEDIAN = 0.542
R_SPREAD = (0.531, 0.563)   # 実測10件の最小・最大（信頼区間ではない）
K_LO, K_HI = 0.80, 0.88     # 感度分析用の仮定範囲（DR3 公式係数による確定は今後の課題）


def bprime_object(o0, mg, bp, a0, ri, kg):
    """B′ の完全な入力で判定する。戻り値: (判定結果 dict, 使った入力 dict)。"""
    o = dict(o0)
    ag_tge_p = kg * a0
    o["A_G_tge"] = ag_tge_p
    cands = [a for a in (o.get("A_G_gsp"), ag_tge_p) if a is not None]
    agmax = max(cands)
    o["M_G_corr"] = mg - agmax
    a_used = ag_tge_p
    dc = (bp - ri * a_used) - tab.ms_colour(mg - a_used)
    o["dcolour"] = dc
    return rules.label(o), {"A_G_tge_prime": ag_tge_p, "A_G_max_prime": agmax,
                            "M_G_corr_prime": o["M_G_corr"], "dcolour_prime": dc}


def kg_scan_points(mg, a0):
    pts = {K_LO + i * 0.002 for i in range(41)}
    for anch_g in rules.ANCH_G:  # M_G,0 が折れ点を横切る k_G
        kg = (mg - anch_g) / a0
        if K_LO < kg < K_HI:
            pts.add(kg)
    return sorted(pts)


audit = []
for sid, num in sorted(tab.SHORT.items(), key=lambda kv: kv[1]):
    g = gsp[sid]
    mg, bp = tab.MG_TAB[sid], g["bp_rp"]
    a0 = rules.TGE.get(num)
    o0 = rules.build(num)
    rA = rules.label(o0)
    if g["ag"] is not None and g["ebp"] is not None:
        ri, ri_src = g["ebp"] / g["ag"], "GSP実測"
        r_vals = [ri]
    else:
        ri, ri_src = R_MEDIAN, "中央値代用"
        r_vals = [R_SPREAD[0], R_MEDIAN, R_SPREAD[1]]
    rec = {"num": num, "A": {"A_G_gsp": o0.get("A_G_gsp"), "A_G_tge": o0.get("A_G_tge"),
                             "A_G_max": rA["margins"]["A_G_max"] if rA["margins"] else None,
                             "M_G_corr": o0["M_G_corr"], "dcolour": o0["dcolour"],
                             "flip_margin": rA["margins"]["flip_margin"] if rA["margins"] else None,
                             "mass_axis": rA["mass_axis"], "phot_axis": rA["phot_axis"],
                             "label": rA["label"]},
           "Bprime": None}
    if a0 is not None:
        res_mid, used_mid = bprime_object(o0, mg, bp, a0, ri, 0.84)
        labels = set()
        flips = []
        for kg in kg_scan_points(mg, a0):
            for rv in r_vals:
                res, _ = bprime_object(o0, mg, bp, a0, rv, kg)
                labels.add(res["label"])
                if res["margins"]:
                    flips.append(res["margins"]["flip_margin"])
        rec["Bprime"] = {
            "r_i": ri, "r_i_source": ri_src, "at_kG_0.84": {
                **used_mid,
                "flip_margin": res_mid["margins"]["flip_margin"] if res_mid["margins"] else None,
                "mass_axis": res_mid["mass_axis"], "phot_axis": res_mid["phot_axis"],
                "label": res_mid["label"]},
            "labels_over_scan": sorted(labels),
            "label_stable_over_scan": len(labels) == 1,
            "flip_margin_range": [min(flips), max(flips)] if flips else None}
    audit.append(rec)

# --- #8 の臨界減光（精密化）: dcolour(A) = +0.30 となる A を二分法で解く ---
sid8 = [s for s, n in tab.SHORT.items() if n == 8][0]
g8, mg8 = gsp[sid8], tab.MG_TAB[sid8]


def dc8(a):
    return (g8["bp_rp"] - R_MEDIAN * a) - tab.ms_colour(mg8 - a)


lo, hi = 0.0, 2.0
assert dc8(lo) > 0.30 > dc8(hi)
for _ in range(60):
    mid = (lo + hi) / 2
    if dc8(mid) > 0.30:
        lo = mid
    else:
        hi = mid
crit8 = (lo + hi) / 2

out = {"meta": {"version": "audit v2 (complete B-prime inputs, unrounded)",
                "rule_Bprime": "A_G,max' = max(A_G_gsp, k_G*A0); M_G_corr' = M_G - A_G,max'; "
                               "dcolour' uses A_used = k_G*A0",
                "k_G_range": [K_LO, K_HI],
                "k_G_status": "感度分析用の仮定範囲。DR3 公式係数による個別確定は今後の課題",
                "r_spread": list(R_SPREAD),
                "r_spread_status": "GSP 実測10件の最小・最大。統計的な信頼区間ではない"},
       "objects": audit,
       "track_8_critical_A": crit8}
json.dump(out, open("colour_correction_audit2.json", "w"), ensure_ascii=False, indent=1)
for rec in audit:
    b = rec["Bprime"]
    if b:
        m = b["at_kG_0.84"]
        print(f"#{rec['num']:>2} A: flip={rec['A']['flip_margin']} label={rec['A']['label']} | "
              f"B': AGmax={m['A_G_max_prime']:.3f} flip={m['flip_margin']} "
              f"range={b['flip_margin_range']} label={m['label']} stable={b['label_stable_over_scan']}")
    else:
        print(f"#{rec['num']:>2} A: label={rec['A']['label']} | B': TGE なし")
print(f"#8 critical A_used = {crit8:.6f}")
print("saved: colour_correction_audit2.json")
