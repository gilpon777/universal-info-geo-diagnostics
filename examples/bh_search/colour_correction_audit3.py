#!/usr/bin/env python3
"""色調補正の監査 v3（ChatGPT 第22回・優先1/2、2026-11-28 Muse）。

優先1: 減光の使い方を3方式で比較する。
  G = GSP のみ（TGE を使わない。GSP 欠落の天体は構成不可）
  T = TGE 変換値のみ（天井にも色にも k_G × A_0 を使う）
  M = 現行 B′（天井は max(GSP, k_G × A_0)、色は k_G × A_0）
優先2: k_G ∈ [0.80, 0.88] の安定性を、境界を直接解いて区間分割で確認する。
  臨界点 = 区間の端、max の切替点 k = A_G_gsp / A_0、主系列アンカーの折れ点、
  dcolour が旗の境界 (±0.15, ±0.30, +0.60) に達する点、flip が 1.5 に達する点、
  候補側の margin_c = 1.0 と M_G,0 = 2.0 の点。
  dcolour と flip は各小区間では線形なので、境界は線形方程式で厳密に解ける。
  全小区間の中点と全臨界点でラベルを評価し、一定であることを確認する。
実行: python3 colour_correction_audit3.py
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
K_LO, K_HI = 0.80, 0.88


def variant_object(o0, mg, bp, a0, ri, kg, variant):
    o = dict(o0)
    if variant == "G":
        if o0.get("A_G_gsp") is None:
            return None
        o["A_G_tge"] = None
        return rules.label(o), {"agmax": o0["A_G_gsp"], "M_G_corr": o0["M_G_corr"],
                                "dcolour": o0["dcolour"]}
    ag_tge_p = kg * a0
    if variant == "T":
        o["A_G_gsp"] = None
        o["A_G_tge"] = ag_tge_p
        agmax = ag_tge_p
    else:  # M
        o["A_G_tge"] = ag_tge_p
        agmax = max(a for a in (o0.get("A_G_gsp"), ag_tge_p) if a is not None)
    o["M_G_corr"] = mg - agmax
    o["dcolour"] = (bp - ri * ag_tge_p) - tab.ms_colour(mg - ag_tge_p)
    res = rules.label(o)
    return res, {"agmax": agmax, "M_G_corr": o["M_G_corr"], "dcolour": o["dcolour"]}


def quantities(o0, mg, bp, a0, ri, kg, variant):
    """境界計算用: (dcolour, flip, margin_c, M_G_corr) を連続量で返す。"""
    out = variant_object(o0, mg, bp, a0, ri, kg, variant)
    if out is None:
        return None
    res, used = out
    m = res["margins"]
    return {"dcolour": used["dcolour"],
            "flip": (mg - used["agmax"]) - rules.mg_ms(rules.M3_REF),
            "margin_c": m.get("margin_cand") if m else None,
            "M_G_corr": used["M_G_corr"]}


def critical_points(o0, mg, bp, a0, ri, variant):
    pts = {K_LO, K_HI}
    if variant == "M" and o0.get("A_G_gsp") is not None:
        ks = o0["A_G_gsp"] / a0
        if K_LO < ks < K_HI:
            pts.add(ks)
    for anch_g in rules.ANCH_G:
        kg = (mg - anch_g) / a0
        if K_LO < kg < K_HI:
            pts.add(kg)
    base = sorted(pts)
    targets = [("dcolour", t) for t in (0.15, -0.15, 0.30, -0.30, 0.60)]
    targets += [("flip", rules.FLIP_MARGIN_MIN), ("margin_c", rules.MARGIN_CAND_MAG),
                ("M_G_corr", rules.MG0_MIN_CAND)]
    crit = set(base)
    for a, b in zip(base, base[1:]):
        qa, qb = quantities(o0, mg, bp, a0, ri, a, variant), quantities(o0, mg, bp, a0, ri, b, variant)
        if qa is None or qb is None:
            continue
        for key, tv in targets:
            va, vb = qa.get(key), qb.get(key)
            if va is None or vb is None or va == vb:
                continue
            if (va - tv) * (vb - tv) < 0:
                crit.add(a + (tv - va) * (b - a) / (vb - va))
    return sorted(crit)


def stability(o0, mg, bp, a0, ri, variant):
    crit = critical_points(o0, mg, bp, a0, ri, variant)
    labels = {}
    evals = list(crit)
    for a, b in zip(crit, crit[1:]):
        evals.append((a + b) / 2)
    for kg in evals:
        out = variant_object(o0, mg, bp, a0, ri, kg, variant)
        if out is not None:
            labels.setdefault(out[0]["label"], []).append(round(kg, 4))
    return {"n_subintervals": len(crit) - 1, "critical_kG": [round(c, 5) for c in crit],
            "labels_seen": {k: (min(v), max(v)) for k, v in labels.items()},
            "stable": len(labels) == 1}


rows = []
for sid, num in sorted(tab.SHORT.items(), key=lambda kv: kv[1]):
    g = gsp[sid]
    mg, bp = tab.MG_TAB[sid], g["bp_rp"]
    a0 = rules.TGE.get(num)
    o0 = rules.build(num)
    if g["ag"] is not None and g["ebp"] is not None:
        ri, ri_src = g["ebp"] / g["ag"], "GSP実測"
    else:
        ri, ri_src = R_MEDIAN, "中央値代用"
    rec = {"num": num, "r_i_source": ri_src, "variants_at_0.84": {}, "stability": {}}
    for var in ("G", "T", "M"):
        if a0 is None and var != "G":
            rec["variants_at_0.84"][var] = None
            continue
        out = variant_object(o0, mg, bp, a0 if a0 else 0.0, ri, 0.84, var)
        if out is None:
            rec["variants_at_0.84"][var] = "構成不可（GSP欠落）"
        else:
            res, used = out
            rec["variants_at_0.84"][var] = {
                **used,
                "flip_margin": res["margins"]["flip_margin"] if res["margins"] else None,
                "label": res["label"]}
    if a0 is not None:
        for var in ("T", "M"):
            rec["stability"][var] = stability(o0, mg, bp, a0, ri, var)
    rows.append(rec)

json.dump({"meta": {"variants": "G=GSPのみ, T=TGE変換値のみ, M=現行B'(天井max+色TGE)",
                    "k_G_range": [K_LO, K_HI],
                    "note": "r の代用幅は実測10件の最小・最大であり統計的な信頼区間ではない"},
           "objects": rows}, open("colour_correction_audit3.json", "w"),
          ensure_ascii=False, indent=1)
for rec in rows:
    v = rec["variants_at_0.84"]
    def lab(x):
        return x["label"] if isinstance(x, dict) else x
    st = rec["stability"].get("M", {})
    print(f"#{rec['num']:>2} G={lab(v['G'])} T={lab(v['T'])} M={lab(v['M'])} | "
          f"M安定={st.get('stable')} (小区間 {st.get('n_subintervals')}) "
          f"T安定={rec['stability'].get('T', {}).get('stable')}")
print("saved: colour_correction_audit3.json")
