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
構造: import では何も実行しない。全ての処理は関数化し、main() から実行する
（ChatGPT 第19回の指摘、2026-11-27）。
"""
import copy, json, math
from datetime import datetime, timezone

import dr4_apply_rules_v4 as rules

TOL = 1e-9  # 浮動小数点の微小差で誤警告しないための許容誤差
FIELDS = ["A_G_max", "flip_margin", "margin_cand", "M2_lo", "M2_hi"]
CASES = [
    ("S4f", 12, {}, dict(A_G_gsp=None), "#12 の GSP-Phot 欠落（TGE は残る）"),
    ("S4g", 12, {}, dict(A_G_tge=None), "#12 の TGE 欠落（GSP は残る）"),
    ("S4h", 12, {}, dict(A_G_gsp=None, A_G_tge=None), "#12 の減光双方欠落"),
    ("S4g2", 4, {}, dict(A_G_tge=None), "#4 の TGE 欠落（GSP は残る）"),
    ("S4r", 12, dict(A_G_gsp=None, A_G_tge=None), dict(A_G_gsp=None),
     "#12 の減光双方欠落 → TGE が届いて余裕が計算可能になる（逆方向）"),
]


def diff_margins(before, after):
    """余裕の差分と警告種別を返す。警告: label_changed / became_insufficient /
    became_sufficient / margin_changed。"""
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


def run_extra_cases():
    """S4 追加試験を実行し、結果レコードのリストを返す（ファイル保存はしない）。"""
    base = {n: rules.build(n) for n in range(1, 14)}
    out = []
    for cid, num, before_over, over, desc in CASES:
        ob = copy.deepcopy(base[num]); ob.update(before_over)
        o = copy.deepcopy(base[num]); o.update(over)
        r = rules.label(o); b = rules.label(ob)
        warns, deltas = diff_margins(b, r)
        out.append(dict(id=cid, num=num, desc=desc, label_before=b["label"],
                        label_after=r["label"], warnings=warns,
                        margin_deltas=deltas))
    return out


# ---- 色の旗の境界警告（記録のみ、2026-11-25 中谷さんの決定） ----
# 色の旗の閾値 0.30 から ±0.08 以内にある天体を「境界」として記録する。
# ラベルは変えない（自動保留にはしない）。Claude v12 §3.5 の提案を受けた実装。
# ChatGPT 第18回で JSON 保存・入力引数化を修正、第19回で入力検査とメタデータを追加。

def _is_finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def colour_boundary_warnings(objects=None, threshold=0.30, band=0.08,
                             correction="A (GSP-Phot)",
                             input_source="frozen13 (dr4_apply_rules_v4.ROWS)"):
    """色の旗の境界警告。戻り値は dict:
    {meta, warnings, errors}。判定ラベルには一切触れない。
    objects: 判定入力 dict のリスト（num と dcolour を含む）。
    None のときは凍結した13件を使う。DR4 当日は実入力のリストを渡す。
    不正な dcolour（文字列・無限大など）は無視せず errors に記録する。"""
    if not _is_finite_number(threshold) or threshold <= 0:
        raise ValueError("threshold は正の有限数でなければならない")
    if not _is_finite_number(band) or band < 0:
        raise ValueError("band は 0 以上の有限数でなければならない")
    if objects is None:
        objects = [rules.build(n) for n in sorted(rules.ROWS)]
    warnings, errors = [], []
    for o in objects:
        num = o.get("num")
        dc = o.get("dcolour")
        if dc is None:
            continue
        if not _is_finite_number(dc):
            errors.append({"num": num, "dcolour_repr": repr(dc),
                           "reason": "dcolour が有限の数値でない"})
            continue
        dist = abs(abs(dc) - threshold)
        if dist <= band + 1e-9:
            warnings.append({"num": num, "dcolour": dc,
                             "threshold": threshold,
                             "distance_to_threshold": round(dist, 3),
                             "correction": correction,
                             "note": "境界（記録のみ、ラベル変更なし）"})
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "rules_version": "dr4_apply_rules_v4 (frozen, fixation 3)",
            "input_source": input_source,
            "threshold": threshold, "band": band, "correction": correction}
    return {"meta": meta, "warnings": warnings, "errors": errors}


def _selftest_colour_boundary():
    base = [rules.build(n) for n in sorted(rules.ROWS)]
    labels_before = {o["num"]: rules.label(o)["label"] for o in base}
    res = colour_boundary_warnings(base)
    labels_after = {o["num"]: rules.label(o)["label"] for o in base}
    assert labels_before == labels_after, "警告処理がラベルを変えた"
    assert {r["num"] for r in res["warnings"]} == {4, 5, 7, 12}, res
    assert res["errors"] == []
    # DR4 実入力形式（任意の入力リスト）と不正値の扱い
    fake = [{"num": 901, "dcolour": 0.29}, {"num": 902, "dcolour": 0.05},
            {"num": 903, "dcolour": None}, {"num": 904, "dcolour": "0.25"},
            {"num": 905, "dcolour": float("inf")}]
    rf = colour_boundary_warnings(fake, input_source="selftest")
    assert [r["num"] for r in rf["warnings"]] == [901], rf
    assert {r["num"] for r in rf["errors"]} == {904, 905}, rf
    print("colour boundary selftest: OK (labels unchanged, warnings:",
          [r["num"] for r in res["warnings"]], ", invalid inputs recorded)")


def main():
    out = run_extra_cases()
    for rec in out:
        print(f"{rec['id']} #{rec['num']} {rec['desc']}: "
              f"{rec['label_before']} -> {rec['label_after']} | "
              f"警告: {rec['warnings'] or 'なし'} | 差分: {rec['margin_deltas']}")
    json.dump(out, open("dr4_dryrun_v4_extra_results.json", "w"),
              ensure_ascii=False, indent=1)
    print("saved: dr4_dryrun_v4_extra_results.json")
    _selftest_colour_boundary()
    cb = colour_boundary_warnings()
    json.dump(cb, open("dr4_colour_boundary_warnings.json", "w"),
              ensure_ascii=False, indent=1)
    print("colour boundary warnings (record only):",
          [r["num"] for r in cb["warnings"]], "errors:", cb["errors"])
    print("saved: dr4_colour_boundary_warnings.json")


if __name__ == "__main__":
    main()
