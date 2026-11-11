"""波長掃引 — ガンマ線から長い方へ、静かな BH が
どのバンドで見えるか。

基準天体は静穏時 BH 連星 A0620-00 (距離 1.06 kpc、
最もよく測られた静かな BH)。各バンドの代表的な
サーベイ感度と基準フラックスから検出限界距離を出す。
数値は桁の目安 (基準天体の実測値ベース、バンドで
x3 程度の揺れはある)。順位はその揺れでは動かない。

チャンネルは 3 つ: 降着の光、伴星の力学 (位置測定/
視線速度)、ジェット。レンズは無色なので別行で
「監視している源の数」で比べる。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json

import numpy as np

D0_PC = 1060.0


def horizon(d0, flux, sens):
    return d0 * np.sqrt(flux / sens)


# name, lambda_m, fiducial flux at 1.06 kpc, survey sensitivity,
# flux unit, channel
BANDS = [
    ("gamma (Fermi-LAT, plane)", 1e-12, 8e-13, 3e-11, "erg/cm2/s", "accretion"),
    ("X-ray (eROSITA all-sky)", 5e-10, 4.4e-14, 1e-13, "erg/cm2/s", "accretion"),
    ("UV (GALEX-like, plane)", 2e-7, 3e-14, 1e-13, "erg/cm2/s", "companion+disk"),
    ("optical (Gaia astrometry)", 6e-7, None, None, "-", "companion dynamics"),
    ("NIR (AO / VVV-like)", 2.2e-6, None, None, "-", "companion dynamics"),
    ("MIR (WISE)", 1.2e-5, 0.3, 0.1, "mJy", "companion (faint)"),
    ("submm (ALMA pointed)", 1e-3, 0.05, 0.02, "mJy", "jet tail"),
    ("radio (VLA/SKA-pathfinder)", 6e-2, 0.05, 0.01, "mJy", "jet"),
]

# 力学チャンネルは伴星の明るさと減光で決まる (別扱いの目安)
DYNAMICS_HORIZON = {"optical (Gaia astrometry)": 8000,
                    "NIR (AO / VVV-like)": 15000}

SOURCES_MONITORED = [   # レンズ用: 監視源の数 (桁)
    ("gamma (Fermi)", 7e3), ("X-ray (eROSITA)", 1e6),
    ("optical (Gaia/OGLE)", 1e9), ("radio (VLBI surveys)", 1e4),
]


def main():
    out = {}
    print("=== detection horizon for a quiescent BH (A0620-00-like) ===")
    rows = []
    for name, lam, flux, sens, unit, ch in BANDS:
        if name in DYNAMICS_HORIZON:
            h = DYNAMICS_HORIZON[name]
        else:
            h = horizon(D0_PC, flux, sens)
        rows.append((name, lam, h, ch))
        out[name] = {"lambda_m": lam, "horizon_pc": h, "channel": ch}
        print(f"  {name:<32} lambda {lam:>8.0e} m  horizon {h:>8.0f} pc  [{ch}]")
    print("=== lensing channel: monitored sources (rate ∝ count) ===")
    for name, n in SOURCES_MONITORED:
        print(f"  {name:<28} {n:.0e} sources")
    out["sources"] = dict(SOURCES_MONITORED)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    names = [r[0].split(" (")[0] for r in rows]
    ax.barh(names, [np.log10(r[2]) for r in rows],
            color=["#b44" if r[3] == "accretion" else
                   "#36c" if "dynamics" in r[3] else
                   "#888" if "jet" in r[3] else "#ca8" for r in rows])
    for i, r in enumerate(rows):
        ax.text(np.log10(r[2]) + 0.05, i, f"{r[2]:,.0f} pc", va="center",
                fontsize=9)
    ax.set_xlabel("detection horizon, log10(pc) — red: accretion glow, "
                  "blue: companion dynamics, grey: jet")
    ax.set_title("Wavelength sweep: where a quiet black hole becomes visible "
                 "(fiducial A0620-00, order-of-magnitude)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(str(HERE / "wavelength_sweep.png"),
                dpi=130)
    with open(str(HERE / "wavelength_sweep_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved wavelength_sweep.png, wavelength_sweep_numbers.json")


if __name__ == "__main__":
    main()
