"""wavelength_sweep のテスト。Run: python3 test_wavelength_sweep.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import wavelength_sweep as ws


def test_sweep():
    dyn = ws.DYNAMICS_HORIZON
    g = ws.horizon(ws.D0_PC, 8e-13, 3e-11)
    x = ws.horizon(ws.D0_PC, 4.4e-14, 1e-13)
    radio = ws.horizon(ws.D0_PC, 0.05, 0.01)
    assert dyn["NIR (AO / VVV-like)"] > dyn["optical (Gaia astrometry)"]
    assert dyn["optical (Gaia astrometry)"] > 10 * g     # 力学が桁で勝つ
    assert radio > x > g                                # 順位: 電波 > X線 > ガンマ
    src = dict(ws.SOURCES_MONITORED)
    assert src["optical (Gaia/OGLE)"] > 1e5 * src["gamma (Fermi)"]


if __name__ == "__main__":
    test_sweep()
    print("all tests passed")
