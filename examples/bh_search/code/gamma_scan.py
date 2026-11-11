"""ガンマ線で走査したら何が見えるか — BH のクラス別。

フェルミ LAT 級の感度を物差しに、クラスごとの
ガンマ線光度から検出限界距離を出す。閾値は
高銀緯 3e-12、銀河面は拡散背景で約 10 倍悪い
3e-11 erg/cm2/s と置く (括弧つきの目安)。

要点: 見えるのは「活動している少数派」だけ。
休眠型は光度ゼロで感度以前の問題として見えない。
孤立型の ISM 降着は、楽観側の隅でやっと近傍が届く。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json

import numpy as np

F_HI = 3e-12          # erg/cm2/s, 高銀緯
F_PLANE = 3e-11       # 銀河面 (拡散背景で約 10 倍悪化)
PC = 3.086e18         # cm
RHO_BH = 5e-4         # 個/pc^3 (銀河系 1e8 個 / 円盤体積)


def horizon_pc(L, F):
    if L <= 0:
        return 0.0
    return np.sqrt(L / (4 * np.pi * F)) / PC


CLASSES = [
    ("accreting binary / microquasar (outburst)", 3e36),
    ("accreting binary (quiescent)", 3e32),
    ("isolated, ISM accretion (optimistic)", 1e30),
    ("isolated, ISM accretion (typical)", 1e28),
    ("dormant (Gaia BH type)", 0.0),
]


def main():
    out = {}
    print(f"{'class':<45} {'plane':>10} {'high-lat':>10}")
    for name, L in CLASSES:
        hp = horizon_pc(L, F_PLANE)
        hh = horizon_pc(L, F_HI)
        out[name] = {"L": L, "horizon_plane_pc": hp,
                     "horizon_hilat_pc": hh}
        print(f"{name:<45} {hp:>8.0f} pc {hh:>8.0f} pc")
    # 孤立型: 限界距離内にある個数 (存在するだけ。光るとは限らない)
    for L in (1e30, 1e28):
        d = horizon_pc(L, F_PLANE)
        n = RHO_BH * 4 / 3 * np.pi * d**3
        print(f"isolated L={L:.0e}: {n:.1f} BHs exist within the "
              f"{d:.0f} pc horizon (if all shone at that L)")
        out[f"count_within_L{L:.0e}"] = n
    with open(str(HERE / "gamma_scan_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved gamma_scan_numbers.json")


if __name__ == "__main__":
    main()
