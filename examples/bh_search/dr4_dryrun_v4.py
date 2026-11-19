#!/usr/bin/env python3
"""DR4 判定パイプラインのドライラン【v4: 自動合否判定つき】。
作成: 2026-11-19（Muse、ChatGPT 第8回査読の優先2を受けた改訂）。
参照実装は dr4_apply_rules_v4.py。シナリオ定義は dr4_dryrun.py（初版）と同一。

第8回での訂正と改善:
- 実行件数の訂正: 初版の集計式は S3 の 13 件を数え漏らしており「117 件」と表示していた。
  実際の判定実行は S0〜S5 で 13 × 10 ブロック = **130 件**が正しい（初版・v3 版の報告に
  あった 117 件は誤り。本ファイルでは実行のたびに数えて 130 と照合する）。
- S1〜S5 にも期待結果を設定し、機械的に合否判定する（初版は表示のみで終了コード 0 固定だった）。
  期待結果は、判定ロジックが同一の v3 での実行記録（2026-11-19）から固定した。
- 例外・期待不一致が1件でもあれば終了コード 1。結果は dr4_dryrun_v4_results.json に保存する。

シナリオと期待結果:
- S0 基準: 評価表 v4 のラベルおよび dr4_test_vectors_v4.json の保存出力と全件一致（回帰）。
- S1 精度向上（f 区間の対数幅を半減）: 全件 S0 と同一ラベル。
- S2 BH1 確定（#6 の f95 = [10, 14]）: 全件 S0 と同一ラベル（#6 は BH候補のまま）。
- S3 軌道解の喪失と出現: #4 → 軌道解なし、#5（仮の軌道解）→ 判定保留、他は S0 と同一。
- S4 入力欠損（5小ケース × 13 件）: S4a（#4 の GSP 欠落、TGE が残る）は S0 と同一
  （減光は一方で足りる設計）。S4b（#7 色欠落）・S4c（#9 減光双方欠落）・S4d（#6 対応付け欠落）・
  S4e（#12 M_G_corr 欠落）は対象天体のみ「判定保留（入力不足）」、他は S0 と同一。
- S5 境界（#13 の f 上端を 1.0）: #13 → 判定保留（質量軸 保留）、他は S0 と同一。

実行: python3 dr4_dryrun_v4.py
"""
import copy, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dr4_apply_rules_v4 as rules

HERE = os.path.dirname(os.path.abspath(__file__))
V4_LABELS = {1: "軌道解なし", 2: "軌道解なし", 3: "軌道解なし", 4: "BH非支持", 5: "軌道解なし",
             6: "BH候補", 7: "BH非支持", 8: "判定保留", 9: "BH非支持", 10: "判定保留",
             11: "軌道解なし", 12: "判定保留", 13: "判定保留"}
COUNT = [0]
FAILURES = []


def base_inputs():
    return {n: rules.build(n) for n in range(1, 14)}


def run_all(inputs, tag):
    out = {}
    for n, o in inputs.items():
        COUNT[0] += 1
        try:
            out[n] = rules.label(copy.deepcopy(o))
        except Exception as e:
            out[n] = {"label": f"例外（{type(e).__name__}: {e}）", "mass_axis": "例外",
                      "phot_axis": "例外", "reasons": [], "margins": {}}
            FAILURES.append(f"{tag} #{n}: 例外 {e}")
    return out


def expect(tag, out, expected):
    ng = 0
    for n, lab in expected.items():
        if out[n]["label"] != lab:
            FAILURES.append(f"{tag} #{n}: 期待 {lab} / 実際 {out[n]['label']}")
            ng += 1
    print(f"  {tag}: {'合格' if ng == 0 else '不合格あり'}")


def fmt_res(r):
    m = r.get("margins") or {}
    m2 = f' M2=[{m.get("M2_lo")}, {m.get("M2_hi")}] flip={m.get("flip_margin")} marg_c={m.get("margin_cand")}' if m else ""
    return f'{r["label"]}（質量軸 {r["mass_axis"]} / 測光軸 {r["phot_axis"]}）{m2}'


def main():
    t_start = time.perf_counter()
    results = {}

    print("== S0 基準 ==")
    s0 = run_all(base_inputs(), "S0")
    tv = json.load(open(os.path.join(HERE, "dr4_test_vectors_v4.json")))
    tv_out = {r["num"]: r["output"] for r in tv["real"]}
    s0_ng = 0
    for n in range(1, 14):
        got, saved = s0[n], tv_out[n]
        if not (got["label"] == saved["label"] == V4_LABELS[n]
                and (got["mass_axis"], got["phot_axis"]) == (saved["mass_axis"], saved["phot_axis"])):
            FAILURES.append(f"S0 #{n}: 保存出力・評価表 v4 と不一致")
            s0_ng += 1
    results["S0"] = {n: s0[n]["label"] for n in s0}
    print(f"  S0: {'合格（全件一致）' if s0_ng == 0 else '不合格あり'}")
    s0_labels = {n: s0[n]["label"] for n in s0}

    print("== S1 精度向上（f 区間の対数幅を半減） ==")
    inp = base_inputs()
    for o in inp.values():
        if o["f95"] is not None:
            lo, hi = o["f95"]
            c = math.sqrt(lo * hi); r = math.sqrt(math.sqrt(hi / lo))
            o["f95"] = [c / r, c * r]
    s1 = run_all(inp, "S1")
    expect("S1", s1, s0_labels)
    results["S1"] = {n: s1[n]["label"] for n in s1}

    print("== S2 BH1 確定（#6 の f95 = [10, 14]） ==")
    inp = base_inputs(); inp[6]["f95"] = [10.0, 14.0]
    s2 = run_all(inp, "S2")
    expect("S2", s2, s0_labels)
    results["S2"] = {n: s2[n]["label"] for n in s2}

    print("== S3 軌道解の喪失と出現 ==")
    inp = base_inputs()
    inp[5].update(f95=[0.5, 2.0], has_orbit=True, solution="astrometric")
    inp[4].update(f95=None, has_orbit=False, solution="none")
    s3 = run_all(inp, "S3")
    exp3 = dict(s0_labels); exp3[4] = "軌道解なし"; exp3[5] = "判定保留"
    expect("S3", s3, exp3)
    print(f'    #5 の結果: {fmt_res(s3[5])}')
    results["S3"] = {n: s3[n]["label"] for n in s3}

    print("== S4 入力欠損の混入 ==")
    s4_cases = [
        ("S4a", 4, dict(A_G_gsp=None), "BH非支持"),
        ("S4b", 7, dict(dcolour=None), "判定保留（入力不足）"),
        ("S4c", 9, dict(A_G_gsp=None, A_G_tge=None), "判定保留（入力不足）"),
        ("S4d", 6, dict(match=None), "判定保留（入力不足）"),
        ("S4e", 12, dict(M_G_corr=None), "判定保留（入力不足）"),
    ]
    results["S4"] = {}
    for cid, num, over, exp_lab in s4_cases:
        inp = base_inputs(); inp[num].update(over)
        out = run_all(inp, cid)
        exp = dict(s0_labels); exp[num] = exp_lab
        expect(cid, out, exp)
        results["S4"][cid] = {n: out[n]["label"] for n in out}

    print("== S5 境界ケース（#13 の f 上端を 1.0） ==")
    inp = base_inputs(); inp[13]["f95"] = [inp[13]["f95"][0], 1.0]
    s5 = run_all(inp, "S5")
    exp5 = dict(s0_labels); exp5[13] = "判定保留"
    expect("S5", s5, exp5)
    if s5[13]["mass_axis"] != "保留":
        FAILURES.append(f"S5 #13: 質量軸が保留でない（{s5[13]['mass_axis']}）")
    print(f'    #13 の結果: {fmt_res(s5[13])}')
    results["S5"] = {n: s5[n]["label"] for n in s5}

    total = COUNT[0]
    if total != 130:
        FAILURES.append(f"実行件数が 130 でない（{total}）")
    elapsed = time.perf_counter() - t_start
    status = "OK" if not FAILURES else "FAIL"
    print(f"\n総実行時間: {elapsed:.3f} 秒（S0〜S5、全 {total} 件の判定） status: {status}")
    for f in FAILURES:
        print("  FAIL:", f)
    out = dict(status=status, total_judgements=total, failures=FAILURES, results=results)
    json.dump(rules._sanitize(out), open(os.path.join(HERE, "dr4_dryrun_v4_results.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
