"""抽出点と Gaia の照合: 位置の床と完全性の実測。

image_points.npz (DECaPS2 集約画像からの抽出点) と
gaia_baade.csv (同じ空の Gaia DR3) を突き合わせる。
- 明るい星での位置オフセットの散らばり = この画像の
  点の位置の再現性の床 (2 エポック差分が相手にする量)
- G 等級別の回収率 = 素朴な抽出器が混雑場でどれだけ
  点を取りこぼすか (公式パイプラインの同時 PSF
  フィットとの差)
"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json

import numpy as np

z = np.load(str(HERE / "image_points.npz"))
ra, dec = z["ra"], z["dec"]
g = np.genfromtxt(str(DATA / "gaia_baade.csv"),
                  delimiter=",", names=True)
gra, gdec, gmag = g["RA_ICRS"], g["DE_ICRS"], g["Gmag"]

# 画像フットプリント内 (抽出点の範囲で近似) の Gaia 星だけ
m_in = (gra >= ra.min()) & (gra <= ra.max()) & \
       (gdec >= dec.min()) & (gdec <= dec.max())
gra, gdec, gmag = gra[m_in], gdec[m_in], gmag[m_in]
print(f"Gaia stars inside footprint: {len(gra)}")

dra = (ra[:, None] - gra[None, :]) * np.cos(np.radians(dec[:, None]))
ddec = dec[:, None] - gdec[None, :]
d = np.hypot(dra, ddec) * 3600.0
j = np.argmin(d, axis=1)
dm = d[np.arange(len(ra)), j]
m = dm < 1.0
print(f"extracted points matched to Gaia (<1 arcsec): {m.sum()} / {len(ra)}")

off = np.stack([dra[m, j[m]], ddec[m, j[m]]], axis=1) * 3.6e6  # mas
gm = gmag[j[m]]
sc = 1.4826 * np.median(np.abs(off - np.median(off, axis=0)), axis=0)
res = {"gaia_in_footprint": int(len(gra)),
       "extracted_matched": int(m.sum()),
       "offset_median_mas": np.median(off, axis=0).tolist(),
       "offset_scatter_mas_per_axis": sc.tolist(),
       "offset_scatter_total_mas": float(np.hypot(*sc))}
bright = gm < 18
if bright.sum() > 30:
    ob = off[bright]
    scb = 1.4826 * np.median(np.abs(ob - np.median(ob, axis=0)), axis=0)
    res["bright_Glt18_n"] = int(bright.sum())
    res["bright_scatter_mas_per_axis"] = scb.tolist()
    res["bright_scatter_total_mas"] = float(np.hypot(*scb))

# Gaia 側から見た回収率 (等級別)
gj = np.argmin(d, axis=0)
gdm = d[gj, np.arange(len(gra))]
gm_match = gdm < 1.0
comp = {}
for lo, hi in [(10, 16), (16, 18), (18, 19), (19, 20), (20, 22)]:
    sel = (gmag >= lo) & (gmag < hi)
    if sel.sum():
        comp[f"G{lo}-{hi}"] = {"n": int(sel.sum()),
                               "recovered": float(gm_match[sel].mean())}
res["completeness_from_gaia"] = comp
print(json.dumps(res, indent=1))

prev = json.load(open(
    str(HERE / "image_points_numbers.json")))
prev.update(res)
with open(str(HERE / "image_points_numbers.json"),
          "w") as f:
    json.dump(prev, f, indent=2)
print("updated image_points_numbers.json")
