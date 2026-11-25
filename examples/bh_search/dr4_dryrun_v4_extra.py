#!/usr/bin/env python3
"""ドライラン v4 の追加試験（ChatGPT 第12回相当の提案 S4f–S4i、2026-11-21 Muse。
2026-11-21 第13回で警告機構を拡張: 全ての判定余裕を許容誤差つきで比較し、
入力不足への移行を別の警告として扱い、差分を記録する）。
固定対象の dr4_dryrun_v4.py は変更せず、追加ケースだけを別ファイルで実行する。
論点: S4a（#4 の GSP 欠落）は #4 では TGE が天井のため A_G,max が変わらない。
GSP が天井である #12 で欠落させると、ラベルが変わらなくても採用減光と判定余裕が動く。
原則: ラベル不変でも余裕が変われば「警告」として記録する。DR4 当日の判定でも、
この監視を入力データに対してそのまま使う（本体 v4 への警告統合は固定後の v5 候補）。
実行: python3 dr4_dryrun_v4_extra.py
"""
import copy, json
import dr4_apply_rules_v4 as rules

TOL = 1e-9  # 浮動小数点の微小差で誤警告しないための許容誤差
FIELDS = ["A_G_max", "flip_margin", "margin_cand", "M2_lo", "M2_hi"]
base = {n: rules.build(n) for n in range(1, 14)}
CASES = [
    ("S4f", 12, {}, dict(A_G_gsp=None), "#12 の GSP-Phot 欠落（TGE は残る）"),
    ("S4g", 12, {}, dict(A_G_tge=None), "#12 の TGE 欠落（GSP は残る）"),
    ("S4h", 12, {}, dict(A_G_gsp=None, A_G_tge=None), "#12 の減光双方欠落"),
    ("S4g2", 4, {}, dict(A_G_tge=None), "#4 の TGE 欠落（GSP は残る）"),
    ("S4r", 12, dict(A_G_gsp=None, A_G_tge=None), dict(A_G_gsp=None),
     "#12 の減光双方欠落 → TGE が届いて余裕が計算可能になる（逆方向）"),
]

def diff_margins(before, after):
    """余裕の差分と警告種別を返す。警告: label_changed / became_insufficient / margin_changed。"""
    warns, deltas = [], {}
    if before["label"] != after["label"]:
        warns.append("label_changed")
    mb, ma = before["margins"], after["margins"]
    if mb and not ma:
        warns.append("became_insufficient")
    if not mb and ma:
        warns.append("became_sufficient")
    if mb and ma:
        for k in FIELDS:
            vb, va = mb.get(k), ma.get(k)
            if vb is None and va is None:
                continue
            if (vb is None) != (va is None) or abs(vb - va) > TOL:
                deltas[k] = [vb, va]
        if deltas:
            warns.append("margin_changed")
    return warns, deltas

out = []
for cid, num, before_over, over, desc in CASES:
    ob = copy.deepcopy(base[num]); ob.update(before_over)
    o = copy.deepcopy(base[num]); o.update(over)
    r = rules.label(o); b = rules.label(ob)
    warns, deltas = diff_margins(b, r)
    rec = dict(id=cid, num=num, desc=desc, label_before=b["label"], label_after=r["label"],
               warnings=warns, margin_deltas=deltas)
    out.append(rec)
    print(f"{cid} #{num} {desc}: {b['label']} -> {r['label']} | 警告: {warns or 'なし'} | 差分: {deltas}")
json.dump(out, open("dr4_dryrun_v4_extra_results.json", "w"), ensure_ascii=False, indent=1)
print("saved: dr4_dryrun_v4_extra_results.json")


# ---- 色の旗の境界警告（記録のみ、2026-11-25 中谷さんの決定） ----
# 色の旗の閾値 0.30 から ±0.08 以内にある天体を「境界」として記録する。
# ラベルは変えない（自動保留にはしない）。Claude v12 §3.5 の提案を受けた実装。
def colour_boundary_warnings():
    out = []
    for n in sorted(rules.ROWS):
        o = rules.build(n)
        dc = o.get("dcolour")
        if dc is None:
            continue
        if abs(abs(dc) - 0.30) <= 0.08 + 1e-9:
            out.append({"num": n, "dcolour": dc,
                        "distance_to_threshold": round(abs(abs(dc) - 0.30), 3)})
    return out


if __name__ == "__main__" or True:
    cb = colour_boundary_warnings()
    print("colour boundary warnings (record only):", cb)
