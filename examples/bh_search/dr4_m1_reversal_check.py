#!/usr/bin/env python3
"""M1 反転境界の自動確認（ChatGPT 第12回相当の改善提案、2026-11-21 Muse）。
f = M2^3/(M1+M2)^2 より、M2 上限が 3 M_sun に達する M1 は M1_crit = sqrt(27/f_hi) - 3。
現在の M1 上端が M1_crit を下回っていれば、M1 範囲の変更では非支持判定は反転しない。
DR4 で新しい軌道解が加わったときに同じ確認を機械的に回すためのスクリプト。
入力: dr4_m1m2_table.json（M1 範囲は dr4_apply_rules_v4.build の値を使う）。
実行: python3 dr4_m1_reversal_check.py
"""
import json, math
import dr4_apply_rules_v4 as rules

rows = {r["num"]: r for r in json.load(open("dr4_m1m2_table.json"))["rows"]}
print(f"{'#':>3} {'M1上端':>7} {'M1_crit':>8} {'差':>7}  判定")
for n in sorted(rows):
    o = rules.build(n)
    if not o["has_orbit"] or o["f95"] is None:
        continue
    crit = math.sqrt(27.0 / o["f95"][1]) - 3
    hi = o["M1_range"][1]
    # 判定は M1 の上下端の両方で決める（ChatGPT 第14回の指摘で修正、2026-11-21）。
    # crit < 0 は「許される正の M1 の全域で M2 が 3 M_sun を超える側」を意味する。
    lo = o["M1_range"][0]
    if crit <= 0 or lo >= crit:
        verdict = "許容範囲全域で3太陽質量超"
    elif hi < crit:
        verdict = "許容範囲全域で3太陽質量未満"
    else:
        verdict = "許容範囲内に反転境界あり"
    print(f"{n:>3} {hi:>7.2f} {crit:>8.2f} {crit - hi:>7.2f}  {verdict}")
