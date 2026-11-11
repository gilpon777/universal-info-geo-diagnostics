"""集約画像 1 枚への点抽出の試行。

DECaPS2 のような「何千枚もの露出を 1 枚に集約した画像」に
点抽出 (集点) を当てると何が起きるかを見る。
1 枚からは位置と明るさしか出ない (時刻は平均化されて
消えている) — 出た点の位置が、公開カタログ (Gaia) と
比べてどれだけ再現するかで「この画像の点の位置の床」
= 2 エポック差分が相手にするノイズの大きさを実測する。

FITS 読みは numpy だけの最小実装 (astropy なし環境のため)。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json
import sys

import numpy as np
from scipy import ndimage

DTYPES = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8",
          -32: ">f4", -64: ">f8"}


def read_fits(path):
    """(header, data) のリストを返す最小 FITS リーダ。"""
    raw = open(path, "rb").read()
    out, pos = [], 0
    while pos < len(raw):
        hdr = {}
        start = pos
        while True:
            block = raw[pos:pos + 2880]
            if len(block) < 2880:
                return out
            pos += 2880
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    break
                if card[8:10] == "= ":
                    val = card[10:].split("/")[0].strip()
                    try:
                        hdr[key] = float(val) if any(
                            c in val for c in ".ED") else int(val)
                    except ValueError:
                        hdr[key] = val.strip("'").strip()
            if "END" in block.decode("ascii", "replace")[::80][:36]:
                pass
            # END カードを含むブロックで終わる
            txt = raw[start:pos].decode("ascii", "replace")
            cards = [txt[i:i + 80] for i in range(0, len(txt), 80)]
            if any(cd[:8].strip() == "END" for cd in cards):
                break
        n = 1
        for ax in range(1, int(hdr.get("NAXIS", 0)) + 1):
            n *= int(hdr.get(f"NAXIS{ax}", 0))
        bp = int(hdr.get("BITPIX", 8))
        if n == 0:
            out.append((hdr, None))
            continue
        itemsize = abs(bp) // 8
        nbytes = n * itemsize
        data = np.frombuffer(raw[pos:pos + nbytes],
                             dtype=np.dtype(DTYPES[bp])).astype(np.float64)
        data = data * hdr.get("BSCALE", 1.0) + hdr.get("BZERO", 0.0)
        n1, n2 = int(hdr.get("NAXIS1", 0)), int(hdr.get("NAXIS2", 0))
        data = data[:n1 * n2].reshape(n2, n1)
        out.append((hdr, data))
        pos += (nbytes + 2879) // 2880 * 2880
        if pos >= len(raw):
            break
    return out


class WCS:
    """TAN 投影の最小 WCS (CD 行列 or CDELT+PC)。"""

    def __init__(self, hdr):
        self.crval = np.array([hdr["CRVAL1"], hdr["CRVAL2"]])
        self.crpix = np.array([hdr["CRPIX1"], hdr["CRPIX2"]])
        if "CD1_1" in hdr:
            self.cd = np.array([[hdr["CD1_1"], hdr["CD1_2"]],
                                [hdr["CD2_1"], hdr["CD2_2"]]])
        else:
            cd1, cd2 = hdr.get("CDELT1", 1.0), hdr.get("CDELT2", 1.0)
            self.cd = np.array([[cd1, 0.0], [0.0, cd2]])

    def pix2sky(self, x, y):
        d = np.array([x - (self.crpix[0] - 1.0),
                      y - (self.crpix[1] - 1.0)])
        xi, eta = self.cd @ d
        xi, eta = np.radians(xi), np.radians(eta)
        ra0, dec0 = np.radians(self.crval)
        rho = np.hypot(xi, eta)
        c_ = np.arctan(rho)
        dec = np.arcsin(np.cos(c_) * np.sin(dec0)
                        + eta * np.sin(c_) * np.cos(dec0) / np.where(
                            rho == 0, 1.0, rho))
        ra = ra0 + np.arctan2(xi * np.sin(c_),
                              rho * np.cos(dec0) * np.cos(c_)
                              - eta * np.sin(dec0) * np.sin(c_))
        return np.degrees(ra) % 360.0, np.degrees(dec)


def extract(img, thresh_sig=5.0, fwhm_px=5):
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    sig = 1.4826 * mad
    thr = med + thresh_sig * sig
    mx = ndimage.maximum_filter(img, size=fwhm_px)
    ys, xs = np.nonzero((img == mx) & (img > thr))
    pts = []
    r = 3
    for x, y in zip(xs, ys):
        x0, x1 = max(0, x - r), min(img.shape[1], x + r + 1)
        y0, y1 = max(0, y - r), min(img.shape[0], y + r + 1)
        win = img[y0:y1, x0:x1] - med
        win = np.clip(win, 0, None)
        tot = win.sum()
        if tot <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pts.append((float((xx * win).sum() / tot),
                    float((yy * win).sum() / tot),
                    float(img[y, x]), float(tot)))
    return np.array(pts), med, sig


def main(fits_path, gaia_csv=None):
    hdus = read_fits(fits_path)
    img = hdr = None
    for hd, data in hdus:
        if data is not None and data.ndim == 2 and data.size > 10000:
            hdr, img = hd, data
            break
    assert img is not None, "no image HDU found"
    pts, med, sig = extract(img)
    wcs = WCS(hdr)
    ra, dec = wcs.pix2sky(pts[:, 0], pts[:, 1])
    pixscale = abs(wcs.cd[0, 0]) * 3600.0
    area_arcmin2 = (img.shape[1] * pixscale / 60.0) * \
        (img.shape[0] * pixscale / 60.0)
    out = {"image_shape": list(img.shape), "background": med,
           "sigma_mad": sig, "n_points": int(len(pts)),
           "pixscale_arcsec": float(pixscale),
           "area_arcmin2": float(area_arcmin2),
           "density_per_arcmin2": float(len(pts) / area_arcmin2)}
    print(json.dumps(out, indent=1))
    np.savez(str(HERE / "image_points.npz"),
             x=pts[:, 0], y=pts[:, 1], ra=ra, dec=dec,
             peak=pts[:, 2], flux=pts[:, 3])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    disp = np.log10(np.clip(img - med, sig * 0.5, None))
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(disp, origin="lower", cmap="gray_r",
              vmin=np.log10(sig), vmax=np.log10(sig) + 3.2)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3, mfc="none",
            mec="red", mew=0.6, alpha=0.65)
    ax.set_title(f"One consolidated DECaPS2 image (Baade's Window) — "
                 f"{len(pts)} extracted points")
    fig.tight_layout()
    fig.savefig(str(HERE / "image_points.png"),
                dpi=110)

    if gaia_csv:
        g = np.genfromtxt(gaia_csv, delimiter=",", names=True)
        gra, gdec = g["RA_ICRS"], g["DE_ICRS"]
        dra = (ra[:, None] - gra[None, :]) * np.cos(np.radians(dec[:, None]))
        ddec = dec[:, None] - gdec[None, :]
        d = np.hypot(dra, ddec) * 3600.0
        j = np.argmin(d, axis=1)
        dm = d[np.arange(len(ra)), j]
        m = dm < 1.0
        off = np.stack([dra[m, j[m]], ddec[m, j[m]]], axis=1) * 3.6e6
        sc = 1.4826 * np.median(np.abs(off - np.median(off, axis=0)),
                                 axis=0)
        out["gaia_matched"] = int(m.sum())
        out["gaia_total"] = int(len(gra))
        out["offset_median_mas"] = np.median(off, axis=0).tolist()
        out["offset_scatter_mas"] = sc.tolist()
        out["offset_scatter_total_mas"] = float(np.hypot(*sc))
        print("Gaia match:", json.dumps(out, indent=1))
    with open(str(HERE / "image_points_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved image_points.png / .npz / _numbers.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
