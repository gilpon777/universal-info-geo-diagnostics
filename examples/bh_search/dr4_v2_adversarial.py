#!/usr/bin/env python3
"""dr4_apply_rules_v2 への敵対テスト（ChatGPT 第7回 優先2 の検証用）。
不正値（NaN・inf・負の減光・逆転区間・未知の列挙値）を入れたとき、
v2 が「問題なし」側（BH非支持 / BH候補）へ倒れないかを実測する。
既存ファイルは変更しない。"""
import math, sys
sys.path.insert(0, ".")
import dr4_apply_rules_v2 as v2

nan, inf = float("nan"), float("inf")

def run(name, num, **over):
    o = v2.build(num); o.update(over)
    r = v2.label(o)
    print(f"{name:<44s} -> {r['label']:<16s} margins={r['margins']}")
    return r

print("== 基準（無改変） ==")
run("基準 #9", 9)
run("基準 #8（色の旗 要注意で保留）", 8)
run("基準 S9型 #4 M_G=1.8（天井不足で保留）", 4, M_G=1.8, M_G_corr=1.5, dcolour=0.1)

print("== 敵対ケース ==")
# 1. NaN の減光（GSP 側）。ag リストは (gsp, tge) の順で max を取るため NaN が残る
run("A1 #9 A_G_gsp=NaN", 9, A_G_gsp=nan)
# 2. NaN の色差: #8 は dcolour=0.42（要注意）で保留。NaN だと旗が「なし」に化けるか
run("A2 #8 dcolour=NaN", 8, dcolour=nan)
# 3. 負の減光: S9型（天井不足で保留のはず）が非支持に化けるか
run("A3 S9型 #4 A_G_tge=-2.0", 4, M_G=1.8, M_G_corr=1.5, dcolour=0.1, A_G_tge=-2.0)
# 4. 逆転した f95 区間
run("A4 #4 f95 逆転 [0.0991, 0.0523]", 4, f95=[0.09913539779506524, 0.05233166499420931])
# 5. 負の f95
run("A5 #9 f95=[-0.5, -0.1]", 9, f95=[-0.5, -0.1])
# 6. 未知の解の種類: #10 は SB1（上限なし→保留のはず）。未知文字列だと astrometric 扱いになるか
run("A6 #10 solution='orbital' f95=[2,3]", 10, solution="orbital", f95=[2.0, 3.0])
run("   （比較）#10 solution=SB1 f95=[2,3]", 10, f95=[2.0, 3.0])
# 7. 未知の対応付け
run("A7 #6 match='X'", 6, match="X")
# 8. M_G = inf
run("A8 #9 M_G=inf", 9, M_G=inf)
# 9. M1_range 逆転
run("A9 #9 M1_range=[1.57, 0.9]", 9, M1_range=[1.57, 0.9])
