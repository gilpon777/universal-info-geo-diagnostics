"""X 線走査の照合 — eRASS1 パッチ x BlackCAT 既知 BH。

パッチ: RA 245-265 度、Dec -55..-40 度 (eRASS1 の
公開半球側)。パッチ内の BlackCAT 天体を数え、
フラックス >= 1e-12 の点状源リスト (58 件) と
60 秒角で照合する。GX 339-4 はコーン検索で別途
確認済み (EXT フラグで点状リストから漏れた)。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import csv
import json

import numpy as np

PATCH = (245.0, 265.0, -55.0, -40.0)       # RA lo/hi, Dec lo/hi
MATCH_R = 60.0                              # arcsec


def hms_to_deg(ra, dec):
    h, m, s = [float(x) for x in ra.split(":")]
    rad = 15.0 * (h + m / 60 + s / 3600)
    sign = -1 if dec.strip().startswith("-") else 1
    d, m2, s2 = [float(x) for x in dec.replace("-", "").split(":")]
    return rad, sign * (d + m2 / 60 + s2 / 3600)


def _unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


# 銀河中心方向と北銀河極の単位ベクトル (J2000 赤道座標)
_C = _unit(266.405, -28.936)
_P = _unit(192.859508, 27.128336)
_Y = np.cross(_P, _C)
_Y = _Y / np.linalg.norm(_Y)
_P = np.cross(_C, _Y)          # 直交化 (向きは元の北銀河極を保つ)


def lb(ra_deg, dec_deg):
    """J2000 赤道 -> 銀河座標 (度)。"""
    s = _unit(ra_deg, dec_deg)
    l = np.degrees(np.arctan2(s @ _Y, s @ _C)) % 360
    b = np.degrees(np.arcsin(s @ _P))
    return l, b


def sep_arcsec(a, b, c, d):
    dra = np.radians(a - c) * np.cos(np.radians((b + d) / 2))
    ddec = np.radians(b - d)
    return np.degrees(np.hypot(dra, ddec)) * 3600


def main():
    out = {}
    with open(str(DATA / "blackcat.csv")) as f:
        bc = [(r["Name"],) + hms_to_deg(r["RA"], r["Dec"])
              for r in csv.DictReader(f)]
    inside = [x for x in bc
              if PATCH[0] <= x[1] <= PATCH[1]
              and PATCH[2] <= x[2] <= PATCH[3]]
    print(f"BlackCAT BHs inside the patch: {len(inside)}")
    for name, ra, dec in inside:
        print(f"  {name}: RA {ra:.3f}, Dec {dec:.3f}")
    out["patch_bhs"] = [x[0] for x in inside]

    with open(str(DATA / "erass_patch.csv")) as f:
        er = [(r["IAUName"], float(r["RAdeg"]), float(r["DEdeg"]),
               float(r["flux"])) for r in csv.DictReader(f)]
    print(f"eRASS1 bright point sources in patch: {len(er)}")

    matches = {}
    for name, ra, dec in inside:
        best = min(er, key=lambda s: sep_arcsec(ra, dec, s[1], s[2]))
        s = sep_arcsec(ra, dec, best[1], best[2])
        if s <= MATCH_R:
            matches[name] = {"erass": best[0], "sep_arcsec": s,
                             "flux": best[3]}
            print(f"  MATCH {name} <-> {best[0]} ({s:.0f} arcsec, "
                  f"flux {best[3]:.2e})")
        else:
            print(f"  no bright match for {name} (nearest {s:.0f} arcsec)")
    out["matches"] = matches
    matched_src = {v["erass"] for v in matches.values()}
    unmatched = [s for s in er if s[0] not in matched_src]
    unmatched.sort(key=lambda s: -s[3])
    print("top unmatched (follow-up list):")
    cand = []
    for name, ra, dec, flux in unmatched[:8]:
        l, b = lb(ra, dec)
        cand.append({"name": name, "ra": ra, "dec": dec,
                     "l": l, "b": b, "flux": flux})
        print(f"  {name}: flux {flux:.2e}, l {l:.1f}, b {b:+.1f}")
    out["candidates"] = cand
    with open(str(HERE / "xray_scan_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved xray_scan_numbers.json")


if __name__ == "__main__":
    main()
