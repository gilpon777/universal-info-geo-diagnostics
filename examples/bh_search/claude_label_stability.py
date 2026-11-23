#!/usr/bin/env python3
"""Claude v12 補助（固定対象外）: ラベルの安定性（維持確率）。
凍結した参照実装 dr4_apply_rules_v4.py をそのまま呼び、入力の系統誤差をモンテカルロで振って「ラベルが維持される確率」を出す。
光度天井の余裕（等）を、閾値からの距離ではなく『誤差を入れたときの維持確率』として読むための補助指標。
摂動モデル（シナリオ。実際の分布ではない）:
  P（測光系統のみ）: M_G, M_G_corr に δ=N(0,0.10)[視差5%相当]+U(-0.5,0.5)[アンカー概略]、A_G は k=U(0.8,1.1) 倍、dcolour に N(0,0.05)。f 区間と M1 範囲は固定。
  D（DR4 想定）  : P に加え、f 区間の対数位置を N(0,0.25) ずらし、対数幅を U(0.5,1.2) 倍。
使い方: 同じフォルダに dr4_apply_rules_v4.py を置いて python3 claude_label_stability.py
"""
import importlib.util, copy, math, collections
import numpy as np
spec = importlib.util.spec_from_file_location("r4", "dr4_apply_rules_v4.py"); r4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r4)
rng = np.random.default_rng(20261122)

def perturb(o, scenario):
    q = copy.deepcopy(o)
    d = rng.normal(0, 0.10) + rng.uniform(-0.5, 0.5)
    q["M_G"] += d; q["M_G_corr"] += d
    k = rng.uniform(0.8, 1.1)
    for key in ("A_G_gsp", "A_G_tge"):
        if q.get(key) is not None: q[key] = q[key] * k
    q["dcolour"] += rng.normal(0, 0.05)
    if scenario == "D":
        lo, hi = q["f95"]; c = 0.5*(math.log(lo)+math.log(hi)) + rng.normal(0, 0.25); w = 0.5*(math.log(hi)-math.log(lo))*rng.uniform(0.5, 1.2)
        q["f95"] = [math.exp(c-w), math.exp(c+w)]
    return q

def stability(num, N=20000, scenario="P", override=None):
    o = r4.build(num)
    if override: o.update(override)
    base = r4.label(o)["label"]; cnt = collections.Counter()
    for _ in range(N): cnt[r4.label(perturb(o, scenario))["label"]] += 1
    return base, cnt[base]/N, cnt

if __name__ == "__main__":
    print("天体  現ラベル            P:維持確率  D:維持確率  P の主な移行先")
    for n in (6, 4, 7, 9, 8, 13, 12, 10):
        b, pP, cP = stability(n, scenario="P"); _, pD, _ = stability(n, scenario="D")
        other = ", ".join(f"{k}:{v/20000:.2f}" for k, v in cP.most_common(3) if k != b)
        print(f"#{n:<3d} {b:<16s}  {pP:6.3f}     {pD:6.3f}     {other}")
    # #5 は軌道解が無いので、#4 型の f 区間を仮置きし、色の旗を外した場合（閾値近傍の例）
    ov = dict(has_orbit=True, solution="astrometric", f95=[0.05, 0.10], M1_range=[1.23, 2.2], dcolour=0.10)
    b, pP, cP = stability(5, scenario="P", override=ov)
    print(f"#5(仮: 軌道解を #4 型の f で仮置き、色の旗なし)  {b}  P:維持確率 {pP:.3f}  移行先: " + ", ".join(f"{k}:{v/20000:.2f}" for k, v in cP.most_common(3)))
