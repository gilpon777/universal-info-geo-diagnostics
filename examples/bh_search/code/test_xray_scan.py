"""xray_scan のテスト。Run: python3 test_xray_scan.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import numpy as np

import xray_scan as xs


def test_lb():
    for ra, dec, l0, b0 in ((255.7058, -48.7897, 338.939, -4.326),
                            (95.6854, -0.3458, 209.956, -6.540),
                            (306.0160, 33.8670, 73.119, -2.091)):
        l, b = xs.lb(ra, dec)
        assert abs(l - l0) < 0.05 and abs(b - b0) < 0.05


def test_recovery():
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xs.main()
    import json
    out = json.load(open(str(HERE / "xray_scan_numbers.json")))
    assert len(out["patch_bhs"]) == 7
    assert set(out["matches"]) == {"MAXI J1631-479", "4U 1630-472"}
    assert out["matches"]["4U 1630-472"]["sep_arcsec"] < 5
    assert out["candidates"][0]["flux"] > 1e-10


if __name__ == "__main__":
    test_lb()
    test_recovery()
    print("all tests passed")
