"""残差チャンネルの試し走査 — Gaia DR3、BH1 のいる箱。

条件: RA 262-265、Dec -2..+1、Plx>1 mas、G<17、
RUWE>5 の 293 件。既知の休眠 BH (Gaia BH1) が
「星のエラー」の順位でどこに出るかを見る。
転記の検証として、取得側が報告した帯域別件数
(RUWE>20: 19、15-20: 31、10-15: 74、5-10: 169) と
NSS 内訳 (1: 13、2: 1、3: 3) を照合する。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import csv
import json

BH1 = "4373465352415301632"


def main():
    with open(str(DATA / "gaia_patch.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("RAdeg", "DEdeg", "Plx", "Gmag", "RUWE", "AEN"):
            r[k] = float(r[k])
        r["NSS"] = int(r["NSS"])
    out = {"n": len(rows)}
    assert len(rows) == 293, len(rows)

    bands = {">20": sum(1 for r in rows if r["RUWE"] > 20),
             "15-20": sum(1 for r in rows if 15 < r["RUWE"] <= 20),
             "10-15": sum(1 for r in rows if 10 < r["RUWE"] <= 15),
             "5-10": sum(1 for r in rows if 5 < r["RUWE"] <= 10)}
    assert bands == {">20": 19, "15-20": 31, "10-15": 74, "5-10": 169}, bands
    out["bands"] = bands
    from collections import Counter
    nss = Counter(r["NSS"] for r in rows if r["NSS"] > 0)
    # 取得側メモは {1:13, 2:1, 3:3} だったが、転記を
    # 2 回見直した結果は {1:10, 2:1, 3:2}。RUWE の
    # 帯域件数は完全一致したので、NSS の食い違いは
    # 取得側メモの数え違いの可能性が高い。断定せず
    # 記録に残す。
    out["nss_counts"] = dict(nss)
    out["nss_note"] = "fetch-side memo said 17 flagged; " \
                      "verified transcription has 13"

    rows.sort(key=lambda r: -r["RUWE"])
    bh1 = next(r for r in rows if r["source_id"] == BH1)
    rank = rows.index(bh1) + 1
    out["bh1"] = {"rank_ruwe": rank, "ruwe": bh1["RUWE"],
                  "aen": bh1["AEN"], "plx": bh1["Plx"],
                  "dist_pc": 1000.0 / bh1["Plx"], "nss": bh1["NSS"]}
    print(f"BH1: RUWE rank {rank} of {len(rows)} "
          f"(RUWE {bh1['RUWE']}, AEN {bh1['AEN']} mas, "
          f"{1000.0 / bh1['Plx']:.0f} pc, NSS {bh1['NSS']})")

    flagged = [r for r in rows if r["NSS"] > 0]
    frank = flagged.index(bh1) + 1 if bh1 in flagged else None
    out["bh1_rank_within_nss"] = frank
    print(f"NSS-flagged: {len(flagged)} of {len(rows)}; "
          f"BH1 is #{frank} of them by RUWE")

    for r in rows:  # 物理スケールの代理指標 (AEN は残差ノイズ
        r["a_proxy"] = r["AEN"] / r["Plx"]  # であって軌道振幅ではない)
    by_a = sorted(rows, key=lambda r: -r["a_proxy"])
    out["bh1_a_proxy"] = bh1["a_proxy"]
    out["bh1_rank_a_proxy"] = by_a.index(bh1) + 1
    print(f"BH1 AEN/Plx proxy {bh1['a_proxy']:.2f} AU, "
          f"rank {out['bh1_rank_a_proxy']} (crude, see docstring)")

    out["top5"] = [{"source_id": r["source_id"], "ruwe": r["RUWE"],
                    "aen": r["AEN"], "plx": r["Plx"],
                    "gmag": r["Gmag"], "nss": r["NSS"]}
                   for r in rows[:5]]
    with open(str(HERE / "residual_scan_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved residual_scan_numbers.json")


if __name__ == "__main__":
    main()
