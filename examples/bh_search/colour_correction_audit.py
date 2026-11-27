#!/usr/bin/env python3
"""色調補正 B′ の完全監査（ChatGPT 第20回・優先1/2/3/5、2026-11-27 Muse）。

B′: E(BP-RP) = r_i × k_G × A_0(TGE)、A_used = k_G × A_0、
dcolour = (bp_rp - E) - ms_colour(M_G - A_used)。
r_i は各天体の GSP 比 ebp/ag（実測）。GSP 欠落の #1・#8・#10 は中央値 0.542 を
代用し、その3件には r の実測ばらつき [0.531, 0.563] も感度幅に含める。
k_G は減光則の範囲 [0.80, 0.88]、代表値 0.84。
全ての中間値を丸めずに JSON へ出力する（再現可能性のため）。
実行: python3 colour_correction_audit.py
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
R_SPREAD = (0.531, 0.563)   # GSP 実測10件の最小・最大
K_RANGE = (0.80, 0.84, 0.88)


def dcolour_full(mg, bp, a_used, e_used):
    mg0 = mg - a_used
    return (bp - e_used) - tab.ms_colour(mg0), mg0, tab.ms_colour(mg0)


def label_with(o0, mg, bp, a_used, e_used):
    dc, mg0, _ = dcolour_full(mg, bp, a_used, e_used)
    o = dict(o0)
    o["M_G_corr"] = round(mg0, 2)
    o["dcolour"] = round(dc, 2)
    return dc, rules.label(o)["label"]


audit = []
for sid, num in sorted(tab.SHORT.items(), key=lambda kv: kv[1]):
    g = gsp[sid]
    mg, bp = tab.MG_TAB[sid], g["bp_rp"]
    a0 = rules.TGE.get(num)
    o0 = rules.build(num)
    if g["ag"] is not None and g["ebp"] is not None:
        ri, ri_src = g["ebp"] / g["ag"], "GSP実測 (ebp/ag)"
    else:
        ri, ri_src = R_MEDIAN, "中央値代用 (GSP欠落)"
    rec = {"num": num, "source_id": sid, "bp_rp": bp, "M_G": mg,
           "A0_TGE": a0, "r_i": ri, "r_i_source": ri_src,
           "dcolour_A": o0["dcolour"], "label_A": rules.label(o0)["label"],
           "Bprime": None}
    if a0 is not None:
        bp_rec = {}
        for kg in K_RANGE:
            E = ri * kg * a0
            A = kg * a0
            dc, mg0, msc = dcolour_full(mg, bp, A, E)
            bp_rec[str(kg)] = {"k_G": kg, "E": E, "A_used": A, "M_G0": mg0,
                               "ms_colour": msc, "dcolour": dc,
                               "flag": bool(abs(dc) > 0.30)}
        dc_mid, lab_mid = label_with(o0, mg, bp, 0.84 * a0, ri * 0.84 * a0)
        # 範囲全域でのラベル安定性: k_G の両端（+ 代用3件は r の両端も）
        r_vals = [ri] if ri_src.startswith("GSP") else [R_SPREAD[0], ri, R_SPREAD[1]]
        labels = set()
        dc_all = []
        for kg in (K_RANGE[0], K_RANGE[-1]):
            for rv in r_vals:
                dc_x, lab_x = label_with(o0, mg, bp, kg * a0, rv * kg * a0)
                labels.add(lab_x)
                dc_all.append(dc_x)
        bp_rec["label_at_mid"] = lab_mid
        bp_rec["labels_over_range"] = sorted(labels)
        bp_rec["label_stable_over_range"] = len(labels) == 1
        bp_rec["dcolour_range_full"] = [min(dc_all), max(dc_all)]
        rec["Bprime"] = bp_rec
    audit.append(rec)

# --- #8 の追跡: 旗が立つ補正量の境界 ---
sid8 = [s for s, n in tab.SHORT.items() if n == 8][0]
g8, mg8 = gsp[sid8], tab.MG_TAB[sid8]
a0_8 = rules.TGE[8]
crit = None
a = 0.0
while a < 3.0:
    dc, _, _ = dcolour_full(mg8, g8["bp_rp"], a, R_MEDIAN * a)
    if dc <= 0.30:
        crit = a
        break
    a += 0.001
track8 = {"A_used_critical_for_flag": crit,
          "A_used_TGE_implied_range": [K_RANGE[0] * a0_8, K_RANGE[-1] * a0_8],
          "gsp_row_state": {k: g8[k] for k in ("bp_rp", "ag", "ebp")},
          "note": "A_used が臨界を超えると旗は消える。TGE 由来の A_used は臨界を大きく上回る。"}

out = {"meta": {"definition": "B': E = r_i * k_G * A0, A_used = k_G * A0, "
                              "dcolour = (bp_rp - E) - ms_colour(M_G - A_used)",
                "k_G_range": list(K_RANGE), "r_spread_measured": list(R_SPREAD),
                "law": "Danielski et al. 2018 (Gaia (E)DR3 extinction law)"},
       "objects": audit, "track_8": track8}
json.dump(out, open("colour_correction_audit.json", "w"), ensure_ascii=False, indent=1)
for rec in audit:
    b = rec["Bprime"]
    if b:
        print(f"#{rec['num']:>2} r={rec['r_i']:.3f}({rec['r_i_source'][:4]}) "
              f"dcA={rec['dcolour_A']} B'@.84={b['0.84']['dcolour']:+.4f} "
              f"range=[{b['dcolour_range_full'][0]:+.3f},{b['dcolour_range_full'][1]:+.3f}] "
              f"label {rec['label_A']} -> {b['label_at_mid']} stable={b['label_stable_over_range']}")
    else:
        print(f"#{rec['num']:>2} TGE なし")
print("#8 track:", track8)
print("saved: colour_correction_audit.json")
