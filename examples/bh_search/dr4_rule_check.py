"""Claude v3 の提案規則（2.1 測光排除・2.2 M1範囲）を 13 件の実数で検証する。

- M1 は追記文書 §2.4 の粗い推定値を使用。
- f 区間は凍結値: BH1 は全共分散 MC [7.58, 18.14]、他は nss_orbits_errors.json
  の独立周辺 MC 95% 区間（#12 は周辺でも縛れない → 点値と共分散 MC の注記）。
- 主系列の M_G(質量) は Pecaut & Mamajek 系の概略アンカーで補間（±0.5 等級級の近似）。
"""
import json
import numpy as np
from scipy.optimize import brentq

# 追記文書 §2.4: source_id -> (G, M_G, 粗い M1, 短リスト番号)
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


def mg_ms(m2):
    return float(np.interp(np.log10(m2), np.log10(ANCH_M), ANCH_G))


def m2_from_f(f, m1):
    if f <= 0:
        return 0.0
    return brentq(lambda m2: m2**3 / (m1 + m2) ** 2 - f, 1e-6, 1e5)


errs = json.load(open("nss_orbits_errors.json"))["stars"]
rows = []
for sid, (G, MG, M1, num) in sorted(M1TAB.items(), key=lambda kv: kv[1][3]):
    st = errs.get(sid, {})
    fci = None
    kind = None
    for key in ("f_ast_indep", "f_rv_indep"):
        if key in st:
            fci = st[key]["ci95"]
            kind = key
    m1_lo, m1_hi = 0.8 * M1, 1.3 * M1
    rec = {"num": num, "sid": sid, "G": G, "M_G": MG, "M1": M1,
           "M1_range_claude": [round(m1_lo, 2), round(m1_hi, 2)], "f_kind": kind}
    if fci:
        rec["f_ci95"] = [round(fci[0], 4), round(fci[1], 3)]
        # BH 候補判定は最も保守的 = f 下限 × M1 下端で M2 下限
        rec["M2_low_worst"] = round(m2_from_f(fci[0], m1_lo), 2)
        rec["M2_at_f_up_M1_hi"] = round(m2_from_f(fci[1], m1_hi), 2)
        if num == 6:  # BH1 は凍結区間（全共分散）で再評価
            rec["M2_low_frozen"] = round(m2_from_f(7.58, m1_lo), 2)
            rec["M2_frozen_range_M1"] = [round(m2_from_f(7.58, m1_lo), 2),
                                         round(m2_from_f(7.58, m1_hi), 2),
                                         round(m2_from_f(18.14, m1_hi), 2)]
            mg = mg_ms(rec["M2_low_frozen"])
            rec["MG_MS_at_M2_low"] = round(mg, 2)
            rec["photometric_margin_mag"] = round(MG - mg, 2)
    if MG < 2.0:
        rec["claude_clause3"] = "M_G<2 → 判定不能（候補ラベル付けない）"
    rows.append(rec)

out = {"anchors_note": "MS M_G anchors are approximate (Pecaut-Mamajek-like), +-0.5 mag class",
       "rows": rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
json.dump(out, open("dr4_rule_check.json", "w"), indent=2, ensure_ascii=False)
