"""ルールチェック v2（ChatGPT 第2回査読 / Claude v5 §7 への対応）。

v1（dr4_rule_check.py）からの変更点:
1. 測光排除の余裕を A_G,max 控除後の式 (M_G − A_G,max) − M_G,MS(M2下限) で
   軌道解のある 8 件すべてに計算する。A_G,max は gaia_extinction_13.json
   （GSP-Phot A_G と TGE a0 の大きい方。A_G ≤ A_V の天井規約、Claude v5 §7E）。
2. #10 は分光質量関数 f_rv で sin^3 i が未知のため M2 の上限は存在しない。
   質量軸は下限のみで評価し、最終ラベルは「判定保留（軌道傾斜角未確定）」。
3. M1 感度列: M1 を [0.3, 5] に振ったとき質量軸の判定が変わるかを記録。
4. 2 軸ラベル（Claude v5 §7A）: 質量軸（支持/非支持/保留/対象外）と
   測光軸（排除済み/排除できない/不要）を分離。M_G,0 < 2 は測光軸を
   「排除できない」とするだけに留め、質量軸の非支持を妨げない。
5. M1 範囲: dr4_m1m2_table.py の従来範囲（下端 0.8×M1(M_G)、上端
   1.3×M1(M_G − A_G_gsp)）。主系列仮定が不適な天体（現状 #13 のみ）は
   [1.0, 3.0] との和集合を採る（Claude v5 §3.2 / §7B）。
6. 表示精度: CI は %.4g で表示し、上限が 0.0 に丸められる v1 のバグを修正。

f 区間は v1 と同じ凍結値: BH1 は全共分散 MC [7.58, 18.14]、他は
nss_orbits_errors.json の独立周辺 MC 95% 区間。#12 は独立周辺 [0.2039, 82.70]
を主値とし、全共分散 [0.078, 90.3] を注記する（どちらでも判定は保留）。
DR3 時点のラベルは暫定（Claude v5 §7F）。
"""
import csv
import json
import numpy as np
from scipy.optimize import brentq

# dr4_rule_check.py v1 と同一: source_id -> (G, M_G, 粗い M1, 短リスト番号)
M1TAB = {
    "4375290679157690368": (8.94, 0.69, 1.6, 13),
    "4373686556112920192": (12.11, 2.89, 1.38, 5),
    "4370723823251080832": (11.72, 3.61, 1.21, 4),
    "4370289481799067008": (10.31, 4.32, 1.06, 9),
    "4374096880110414080": (12.04, 4.79, 0.99, 7),
    "4373814997114111744": (11.32, 5.08, 0.95, 10),
    "4370722517581033088": (13.58, 5.28, 0.93, 11),
    "4373465352415301632": (13.77, 5.35, 0.92, 6),   # BH1
    "4375405814342089344": (13.19, 6.10, 0.83, 3),
    "4370422312249022848": (11.50, 6.17, 0.83, 2),
    "4373878115952155776": (14.30, 6.46, 0.79, 8),
    "4373996966286538240": (10.88, 6.94, 0.75, 1),
    "4370246910083179136": (15.15, 8.41, 0.61, 12),  # #12
}
ANCH_M = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.2, 3.0, 4.0, 5.5, 8.0, 10.0, 15.0, 20.0, 30.0])
ANCH_G = np.array([8.1, 6.4, 4.83, 3.7, 2.7, 0.6, -0.1, -1.0, -1.7, -2.5, -3.0, -4.0, -4.9, -5.7])
BH1_SID = "4373465352415301632"
NO12_SID = "4370246910083179136"
NO10_SID = "4373814997114111744"


def mg_ms(m2):
    return float(np.interp(np.log10(m2), np.log10(ANCH_M), ANCH_G))


def m1_from_mg(mg):
    return float(10 ** np.interp(mg, ANCH_G[::-1], np.log10(ANCH_M[::-1])))


def m2_from_f(f, m1):
    if f <= 0:
        return 0.0
    return brentq(lambda m2: m2**3 / (m1 + m2) ** 2 - f, 1e-6, 1e5)


def g4(x):
    return float("%.4g" % x)


gsp = {}
with open("gaia_gspphot_13.csv") as fh:
    for row in csv.DictReader(fh):
        gsp[row["source_id"]] = float(row["ag_gspphot"]) if row["ag_gspphot"] else None

ext = {r["source_id"]: r for r in json.load(open("gaia_extinction_13.json"))["rows"]}
tab1 = {r["source_id"]: r for r in json.load(open("dr4_m1m2_table.json"))["rows"]}
errs = json.load(open("nss_orbits_errors.json"))["stars"]


def mass_axis(f_lo, f_hi, m1, rv=False):
    """固定した M1 での質量軸判定。rv の場合は上限なし（下限のみ）。"""
    lo = m2_from_f(f_lo, m1)
    if lo > 3.0:
        return "支持"
    if rv:
        return "保留"
    return "非支持" if m2_from_f(f_hi, m1) < 3.0 else "保留"


rows = []
for sid, (G, MG, M1_coarse, num) in sorted(M1TAB.items(), key=lambda kv: kv[1][3]):
    ag_gsp = gsp[sid]
    agmax = ext[sid]["AG_max"]
    # --- M1 範囲（dr4_m1m2_table.py の従来範囲） ---
    m1_low = 0.8 * m1_from_mg(MG)
    mg0_gsp = MG - ag_gsp if ag_gsp is not None else MG
    m1_high = 1.3 * m1_from_mg(mg0_gsp)
    ms_flag = tab1[sid]["MS_flag"]
    ms_unfit = ms_flag.startswith("不適")
    if ms_unfit:
        m1_low, m1_high = min(m1_low, 1.0), max(m1_high, 3.0)
    rec = {"num": num, "sid": sid, "G": G, "M_G": MG,
           "AG_gspphot": ag_gsp, "TGE_a0": ext[sid]["TGE_a0"], "AG_max": agmax,
           "AG_max_source": ext[sid]["AG_max_source"],
           "MS_flag": ms_flag,
           "M1_range": [round(m1_low, 2), round(m1_high, 2)],
           "M1_range_note": "従来範囲と [1.0, 3.0] の和集合" if ms_unfit else "従来範囲"}
    # --- f 区間（v1 と同じ凍結値） ---
    st = errs.get(sid, {})
    kind, fci = None, None
    for key in ("f_ast_indep", "f_rv_indep"):
        if key in st:
            kind, fci = key, st[key]["ci95"]
    if sid == BH1_SID:
        kind, fci = "f_ast_frozen_cov", [7.58, 18.14]
    if fci is None:
        rec.update({"mass_axis": "対象外", "photo_axis": "不要",
                    "final_label": "軌道解なし"})
        rows.append(rec)
        continue
    f_lo, f_hi = fci
    rec["f_kind"] = kind
    rec["f_ci95"] = [g4(f_lo), g4(f_hi)]
    if sid == NO12_SID:
        rec["f_ci95_cov_note"] = [0.078, 90.3]
    rv = (kind == "f_rv_indep")
    # --- M2 範囲 ---
    m2_lo = m2_from_f(f_lo, m1_low)
    rec["M2_low"] = round(m2_lo, 3)
    if rv:
        rec["M2_high"] = None
        rec["M2_high_note"] = "なし（傾斜角未確定）"
    else:
        rec["M2_high"] = round(m2_from_f(f_hi, m1_high), 3)
    # --- 質量軸（M1 範囲の両端で保守的に） ---
    if m2_lo > 3.0:
        axis = "支持"
    elif rv:
        axis = "保留"
    elif m2_from_f(f_hi, m1_high) < 3.0:
        axis = "非支持"
    else:
        axis = "保留"
    rec["mass_axis"] = axis
    # --- 測光軸 ---
    mg_ms_at_m2lo = mg_ms(m2_lo)
    margin = (MG - agmax) - mg_ms_at_m2lo
    rec["MG_MS_at_M2_low"] = round(mg_ms_at_m2lo, 2)
    rec["photometric_margin_mag"] = round(margin, 2)
    rec["M_G0_AGmax"] = round(MG - agmax, 2)
    if rec["M_G0_AGmax"] < 2.0:
        photo = "排除できない"
        rec["photo_axis_note"] = "M_G,0 (= M_G − A_G,max) < 2"
    elif margin >= 1.0:
        photo = "排除済み"
    else:
        photo = "排除できない"
        rec["photo_axis_note"] = "測光余裕 < 1 等"
    rec["photo_axis"] = photo
    # --- M1 感度（[0.3, 5]） ---
    v03 = mass_axis(f_lo, f_hi, 0.3, rv=rv)
    v50 = mass_axis(f_lo, f_hi, 5.0, rv=rv)
    rec["M1_sensitivity"] = {"at_0.3": v03, "at_5": v50,
                             "label_changes": v03 != v50}
    # --- 最終ラベル（2 軸の合成、Claude v5 §7A） ---
    if axis == "支持" and photo == "排除済み":
        final = "BH候補"
    elif axis == "支持":
        final = "判定不能（測光）"
    elif axis == "非支持":
        final = "BH非支持"
        if num == 13:  # Claude v5 §3.1: #13 は測光軸の判定不能を付記する
            final += "（測光軸: 排除できない（M_G,0<2））"
    elif sid == NO10_SID:
        final = "判定保留（軌道傾斜角未確定）"
    else:
        final = "判定保留"
    rec["final_label"] = final
    rows.append(rec)

out = {"method": "v2: photometric margin with A_G,max deducted; #10 RV lower-bound only; "
                 "two-axis labels (Claude v5 §7A); M1 union range for MS-unfit; "
                 "M1 sensitivity over [0.3, 5]; DR3 labels are provisional (§7F)",
       "AG_convention": json.load(open("gaia_extinction_13.json"))["convention"],
       "rows": rows}
json.dump(out, open("dr4_rule_check_v2.json", "w"), indent=2, ensure_ascii=False)
for r in rows:
    print(r["num"], r["sid"], "AGmax", r["AG_max"], "M1", r["M1_range"],
          "f", r.get("f_ci95"), "M2", r.get("M2_low"), r.get("M2_high"),
          "margin", r.get("photometric_margin_mag"),
          "mass", r["mass_axis"], "photo", r["photo_axis"],
          "sens", r.get("M1_sensitivity"), "->", r["final_label"])
print("saved dr4_rule_check_v2.json")
