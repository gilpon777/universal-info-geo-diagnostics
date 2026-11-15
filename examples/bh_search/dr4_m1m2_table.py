"""13 天体の M1・M2 評価表（ChatGPT 追加査読 §2.2 / §8 No.5 への提出資料）。

- 減光: Gaia DR3 GSP-Phot の ag_gspphot / ebpminrp_gspphot（ARI ミラー経由で取得、
  gaia_gspphot_13.csv）。ESA TAP 本体が応答なしのためミラーを使用した。
- M1: 主系列の質量–絶対等級アンカー（概略、Pecaut & Mamajek 系）で逆引き。
  M1_low = 0.8 × M1(M_G 観測値)（減光は M1 を小さく見せる方向のみ）
  M1_cen = M1(M_G − A_G)（A_G が無い天体は観測値のまま、要注記）
  M1_high = 1.3 × M1_cen
- 主系列仮定の適否: 補正後の色 (BP−RP)_0 を、その M_G,0 における主系列の
  色と比較。+0.25 等より赤い場合は「進化星/未分解連星の疑い」として旗を立てる。
- f 区間: BH1 は凍結した全共分散 MC [7.58, 18.14]、#12 は全共分散 MC
  [0.078, 90.3]、他は nss_covariance_mc.json にあればその 95% 区間、
  無ければ nss_orbits_errors.json の独立周辺 MC 95% 区間。
- 判定（ChatGPT §2.1 の 3 分類）: M2_low = M2(f_low, M1_low)、
  M2_high = M2(f_high, M1_high) とし、M2_low>3 → BH候補、
  M2_high<3 → BH非支持、それ以外 → 判定保留。
  ただし系の M_G,0 < 2 の天体は Claude 条項により「判定不能（測光）」を優先する。
"""
import csv
import json
import numpy as np
from scipy.optimize import brentq

SHORT = {  # source_id -> 短リスト番号
    "4373996966286538240": 1, "4370422312249022848": 2, "4375405814342089344": 3,
    "4370723823251080832": 4, "4373686556112920192": 5, "4373465352415301632": 6,
    "4374096880110414080": 7, "4373878115952155776": 8, "4370289481799067008": 9,
    "4373814997114111744": 10, "4370722517581033088": 11, "4370246910083179136": 12,
    "4375290679157690368": 13,
}
MG_TAB = {  # 追記文書 §2.4 の M_G（減光補正なし）
    "4375290679157690368": 0.69, "4373686556112920192": 2.89, "4370723823251080832": 3.61,
    "4370289481799067008": 4.32, "4374096880110414080": 4.79, "4373814997114111744": 5.08,
    "4370722517581033088": 5.28, "4373465352415301632": 5.35, "4375405814342089344": 6.10,
    "4370422312249022848": 6.17, "4373878115952155776": 6.46, "4373996966286538240": 6.94,
    "4370246910083179136": 8.41,
}
ANCH_M = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.2, 3.0, 4.0, 5.5, 8.0, 10.0, 15.0, 20.0, 30.0])
ANCH_G = np.array([8.1, 6.4, 4.83, 3.7, 2.7, 0.6, -0.1, -1.0, -1.7, -2.5, -3.0, -4.0, -4.9, -5.7])
LOCUS_G = np.array([0.2, 2.0, 3.0, 4.0, 4.8, 5.35, 6.0, 7.0, 8.4])
LOCUS_C = np.array([0.02, 0.28, 0.45, 0.62, 0.74, 0.85, 0.97, 1.18, 1.50])


def m1_from_mg(mg):
    return float(10 ** np.interp(mg, ANCH_G[::-1], np.log10(ANCH_M[::-1])))


def ms_colour(mg):
    return float(np.interp(mg, LOCUS_G, LOCUS_C))


def m2_from_f(f, m1):
    if f <= 0:
        return 0.0
    return brentq(lambda m2: m2**3 / (m1 + m2) ** 2 - f, 1e-6, 1e5)


gsp = {}
with open("gaia_gspphot_13.csv") as fh:
    for row in csv.DictReader(fh):
        gsp[row["source_id"]] = {
            "bp_rp": float(row["bp_rp"]) if row["bp_rp"] else None,
            "ag": float(row["ag_gspphot"]) if row["ag_gspphot"] else None,
            "ebp": float(row["ebpminrp_gspphot"]) if row["ebpminrp_gspphot"] else None,
        }

cov = json.load(open("nss_covariance_mc.json"))
errs = json.load(open("nss_orbits_errors.json"))["stars"]


def f_interval(sid):
    """(f_low, f_high, 出典) を返す。軌道解が無ければ None。"""
    if sid == "4373465352415301632":
        return 7.58, 18.14, "凍結・全共分散MC"
    if sid == "4370246910083179136":
        return 0.078, 90.3, "全共分散MC"
    st = cov.get("stars", {}).get(sid) if isinstance(cov, dict) else None
    if st:
        for k in ("f_ast", "f"):
            if k in st and "ci95" in st[k]:
                return st[k]["ci95"][0], st[k]["ci95"][1], "全共分散MC"
    st = errs.get(sid, {})
    for k in ("f_ast_indep", "f_rv_indep"):
        if k in st:
            return st[k]["ci95"][0], st[k]["ci95"][1], "独立周辺MC"
    return None


rows = []
for sid, num in sorted(SHORT.items(), key=lambda kv: kv[1]):
    mg = MG_TAB[sid]
    g = gsp[sid]
    ag = g["ag"]
    mg0 = mg - ag if ag is not None else mg
    m1_obs = m1_from_mg(mg)
    m1_cen = m1_from_mg(mg0)
    m1_low = 0.8 * m1_obs
    m1_high = 1.3 * m1_cen
    col0 = (g["bp_rp"] - g["ebp"]) if (g["bp_rp"] is not None and g["ebp"] is not None) else g["bp_rp"]
    dcol = col0 - ms_colour(mg0) if col0 is not None else None
    if sid == "4375290679157690368":
        ms_flag = "不適（色が主系列より大幅に赤い＝進化星の疑い）"
    elif dcol is not None and dcol > 0.25:
        ms_flag = "要注意（主系列より赤い：進化星/未分解連星の可能性）"
    elif dcol is not None and dcol < -0.25:
        ms_flag = "要注意（主系列より青い：減光推定の過大評価の可能性）"
    else:
        ms_flag = "概ね妥当"
    rec = {"num": num, "source_id": sid, "M_G": mg, "A_G_gspphot": ag,
           "M_G_corr": round(mg0, 2), "colour0": round(col0, 2) if col0 is not None else None,
           "dcolour_vs_MS": round(dcol, 2) if dcol is not None else None,
           "MS_flag": ms_flag,
           "M1_cen": round(m1_cen, 2), "M1_range": [round(m1_low, 2), round(m1_high, 2)]}
    fi = f_interval(sid)
    if fi is None:
        rec["label"] = "軌道解なし（DR4待ち）"
    else:
        flo, fhi, src = fi
        m2lo = m2_from_f(flo, m1_low)
        m2hi = m2_from_f(fhi, m1_high)
        rec["f_ci95"] = [flo, fhi]
        rec["f_source"] = src
        rec["M2_range"] = [round(m2lo, 2), round(m2hi, 2)]
        if mg0 < 2.0:
            rec["label"] = "判定不能（測光：M_G,0<2）"
        elif m2lo > 3.0:
            rec["label"] = "BH候補"
        elif m2hi < 3.0:
            rec["label"] = "BH非支持"
        else:
            rec["label"] = "判定保留"
    rows.append(rec)

out = {"method": "M1 from MS anchors; A_G from Gaia DR3 GSP-Phot (ARI mirror); "
                 "labels per ChatGPT 3-way rule + Claude M_G<2 clause",
       "rows": rows}
json.dump(out, open("dr4_m1m2_table.json", "w"), indent=2, ensure_ascii=False)
for r in rows:
    print(r["num"], r["source_id"], "M_G", r["M_G"], "A_G", r["A_G_gspphot"],
          "M1", r["M1_cen"], r["M1_range"], r["MS_flag"], "->", r["label"],
          r.get("M2_range", ""))
print("saved dr4_m1m2_table.json")
