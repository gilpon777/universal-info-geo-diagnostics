#!/usr/bin/env python3
"""ドライラン v4 の追加試験（ChatGPT 第12回相当の提案 S4f–S4i、2026-11-21 Muse）。
固定対象の dr4_dryrun_v4.py は変更せず、追加ケースだけを別ファイルで実行する。
論点: S4a（#4 の GSP 欠落）は #4 では TGE が天井のため A_G,max が変わらない。
GSP が天井である #12 で欠落させると、ラベルが変わらなくても採用減光と判定余裕が動く。
S4i の原則: ラベル不変でも余裕が変われば「警告」として記録する。
実行: python3 dr4_dryrun_v4_extra.py
"""
import copy, json
import dr4_apply_rules_v4 as rules

base = {n: rules.build(n) for n in range(1, 14)}
CASES = [
    ("S4f", 12, dict(A_G_gsp=None), "#12 の GSP-Phot 欠落（TGE は残る）"),
    ("S4g", 12, dict(A_G_tge=None), "#12 の TGE 欠落（GSP は残る）"),
    ("S4h", 12, dict(A_G_gsp=None, A_G_tge=None), "#12 の減光双方欠落"),
    ("S4g2", 4, dict(A_G_tge=None), "#4 の TGE 欠落（GSP は残る）"),
]
out = []
for cid, num, over, desc in CASES:
    o = copy.deepcopy(base[num]); o.update(over)
    r = rules.label(o); b = rules.label(copy.deepcopy(base[num]))
    m, mb = r["margins"], b["margins"]
    warn = bool(m and mb and (m.get("A_G_max") != mb.get("A_G_max") or m.get("flip_margin") != mb.get("flip_margin")))
    rec = dict(id=cid, num=num, desc=desc, label_before=b["label"], label_after=r["label"],
               A_G_max_before=mb.get("A_G_max") if mb else None, A_G_max_after=m.get("A_G_max") if m else None,
               flip_before=mb.get("flip_margin") if mb else None, flip_after=m.get("flip_margin") if m else None,
               margin_warning=warn)
    out.append(rec)
    print(f"{cid} #{num} {desc}: {b['label']} -> {r['label']} | A_G_max {rec['A_G_max_before']} -> {rec['A_G_max_after']} | flip {rec['flip_before']} -> {rec['flip_after']} | 余裕変化の警告: {warn}")
json.dump(out, open("dr4_dryrun_v4_extra_results.json", "w"), ensure_ascii=False, indent=1)
print("saved: dr4_dryrun_v4_extra_results.json")
