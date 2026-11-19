#!/usr/bin/env python3
"""DR4 判定パイプラインのドライラン（通し試験）。
作成: 2026-11-17（Muse）/ 目的: 2026-12-02 の Gaia DR4 公開当日に、判定パイプライン
（入力 → dr4_apply_rules_v2.label() → ラベル）が混乱なく回ることを本番前に証明する。
dr4_apply_rules_v2.py は import して使うだけで、既存ファイルは一切書き換えない。

シナリオ定義:
- S0 基準: DR3 入力そのまま。出力が dr4_test_vectors_v2.json の保存済み出力および
  評価表 v4（dr4_m1m2_table_v4.md）の最終ラベルと全件一致すること（回帰）。
- S1 精度向上: 軌道解のある全件について f 区間を対数空間で半分の幅に狭める
  （幾何中心は不変。DR4 で誤差が乗算的に縮む想定）。定義を明示するため対数幅半減を採用した。
- S2 BH1 確定: #6 (BH1) の f 区間を [10, 14] に置換。
- S3 軌道解の喪失と出現: #5 に仮の軌道解（f95=[0.5, 2.0], solution=astrometric,
  M_G 等は現行値）を与え、同時に #4 の軌道解を落とす（f95=None, has_orbit=False, solution=none）。
- S4 入力欠損の混入: 小ケースに分け、各小ケースでは対象 1 件だけを変更して 13 件を実行する。
    S4a: #4 の GSP-Phot 欠落（A_G_gsp=None。TGE は残る）
    S4b: #7 の色データ欠落（dcolour=None）
    S4c: #9 の減光双方欠落（A_G_gsp=None かつ A_G_tge=None）
    S4d: #6 の対応付け欠落（match=None）
    S4e: #12 の M_G_corr 欠落（M_G_corr=None）
  ※ MSC（ag_msc）は本パイプラインの入力項目に存在しないため、欠落シナリオは適用不能
    （dr4_dryrun_v3.md の所見参照）。
- S5 境界ケース: #13 の f 上端を 1.0 に置換し、M2 上限が 3 M☉ をまたぐ形にする
  （下端は現行値のまま）。

実行: python3 dr4_dryrun_v3.py （結果は標準出力のみ。ファイルは書き出さない）
"""
import copy, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dr4_apply_rules_v3 as rules

HERE = os.path.dirname(os.path.abspath(__file__))

# 評価表 v4 の最終ラベル（括弧内の限定注記は表の注釈であり、パイプラインのラベル本体ではない）
V4_LABELS = {1: "軌道解なし", 2: "軌道解なし", 3: "軌道解なし", 4: "BH非支持", 5: "軌道解なし",
             6: "BH候補", 7: "BH非支持", 8: "判定保留", 9: "BH非支持", 10: "判定保留",
             11: "軌道解なし", 12: "判定保留", 13: "判定保留"}


def base_inputs():
    return {n: rules.build(n) for n in range(1, 14)}


def run_all(inputs):
    """13 件を実行。例外は握りつぶさず、例外として記録して返す（完走成否の判定材料）。"""
    out = {}
    for n, o in inputs.items():
        try:
            out[n] = rules.label(copy.deepcopy(o))
        except Exception as e:  # label() は内部で受け皿を持つため、ここに来るのは想定外
            out[n] = {"label": f"例外（{type(e).__name__}: {e}）", "mass_axis": "例外",
                      "phot_axis": "例外", "reasons": [], "margins": {}}
    return out


def diff_vs(base_out, scen_out):
    """ラベル・質量軸・測光軸のいずれかが変わった天体を返す。"""
    diffs = []
    for n in range(1, 14):
        b, s = base_out[n], scen_out[n]
        if (b["label"], b["mass_axis"], b["phot_axis"]) != (s["label"], s["mass_axis"], s["phot_axis"]):
            diffs.append((n, b, s))
    return diffs


def fmt_res(r):
    m = r.get("margins") or {}
    m2 = ""
    if m:
        m2 = f' M2=[{m.get("M2_lo")}, {m.get("M2_hi")}] flip={m.get("flip_margin")} marg_c={m.get("margin_cand")}'
    return f'{r["label"]}（質量軸 {r["mass_axis"]} / 測光軸 {r["phot_axis"]}）{m2}'


def show_diff(title, base_out, scen_out):
    diffs = diff_vs(base_out, scen_out)
    if not diffs:
        print(f"  {title}: S0 から変化なし（13 件すべて同一）")
    for n, b, s in diffs:
        print(f"  {title} #{n}: {fmt_res(b)} → {fmt_res(s)}")
        for reason in s.get("reasons", []):
            print(f"      理由: {reason}")
    return diffs


def main():
    t_start = time.perf_counter()
    base = base_inputs()

    # ---------- S0 基準（回帰） ----------
    print("== S0 基準 ==")
    s0 = run_all(base)
    tv = json.load(open(os.path.join(HERE, "dr4_test_vectors_v2.json")))
    tv_out = {r["num"]: r["output"] for r in tv["real"]}
    s0_fail = []
    for n in range(1, 14):
        got, saved = s0[n], tv_out[n]
        same_label = got["label"] == saved["label"] == V4_LABELS[n]
        same_axes = (got["mass_axis"], got["phot_axis"]) == (saved["mass_axis"], saved["phot_axis"])
        if not (same_label and same_axes):
            s0_fail.append(n)
        print(f'  #{n:<2d} {fmt_res(got)}  [v4表: {V4_LABELS[n]} / 保存済み出力と一致: {same_label and same_axes}]')
    print(f"  S0 回帰: {'全件一致' if not s0_fail else '不一致あり #' + str(s0_fail)}")

    # ---------- S1 精度向上（f 区間を対数空間で半分の幅に） ----------
    print("== S1 精度向上（f 区間の対数幅を半減） ==")
    inp = base_inputs()
    for n, o in inp.items():
        if o["f95"] is not None:
            lo, hi = o["f95"]
            c = math.sqrt(lo * hi)              # 幾何中心
            r = math.sqrt(math.sqrt(hi / lo))   # 対数半幅の半減 → 比の 1/4 乗
            o["f95"] = [c / r, c * r]
            print(f'  #{n}: f95 {[round(lo, 5), round(hi, 5)]} → {[round(c / r, 5), round(c * r, 5)]}')
    s1 = run_all(inp)
    show_diff("S1", s0, s1)

    # ---------- S2 BH1 確定 ----------
    print("== S2 BH1 確定（#6 の f95 = [10, 14]） ==")
    inp = base_inputs()
    inp[6]["f95"] = [10.0, 14.0]
    s2 = run_all(inp)
    show_diff("S2", s0, s2)
    print(f'  #6 の結果: {fmt_res(s2[6])}')

    # ---------- S3 軌道解の喪失と出現 ----------
    print("== S3 軌道解の喪失と出現 ==")
    inp = base_inputs()
    inp[5].update(f95=[0.5, 2.0], has_orbit=True, solution="astrometric")   # #5 に仮の軌道解
    inp[4].update(f95=None, has_orbit=False, solution="none")               # #4 の軌道解を落とす
    s3 = run_all(inp)
    show_diff("S3", s0, s3)

    # ---------- S4 入力欠損 ----------
    print("== S4 入力欠損の混入 ==")
    s4_cases = [
        ("S4a", 4, dict(A_G_gsp=None), "GSP-Phot 欠落（TGE は残る）"),
        ("S4b", 7, dict(dcolour=None), "色データ欠落"),
        ("S4c", 9, dict(A_G_gsp=None, A_G_tge=None), "減光双方欠落"),
        ("S4d", 6, dict(match=None), "対応付け欠落"),
        ("S4e", 12, dict(M_G_corr=None), "M_G_corr 欠落"),
    ]
    for cid, num, over, desc in s4_cases:
        inp = base_inputs()
        inp[num].update(over)
        out = run_all(inp)
        d = diff_vs(s0, out)
        others = [n for n, _, _ in d if n != num]
        print(f"  {cid} #{num} {desc}: {fmt_res(s0[num])} → {fmt_res(out[num])}")
        for reason in out[num].get("reasons", []):
            print(f"      理由: {reason}")
        if others:
            print(f"      ※ 対象外の天体にも変化: {others}")

    # ---------- S5 境界ケース ----------
    print("== S5 境界ケース（#13 の f 上端を 1.0 にし M2 上限が 3 をまたぐ形） ==")
    inp = base_inputs()
    inp[13]["f95"] = [inp[13]["f95"][0], 1.0]
    s5 = run_all(inp)
    show_diff("S5", s0, s5)
    print(f'  #13 の結果: {fmt_res(s5[13])}')

    elapsed = time.perf_counter() - t_start
    print(f"\n総実行時間: {elapsed:.3f} 秒（S0〜S5、全 {13 * (3 + len(s4_cases) + 1)} 件の判定）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
