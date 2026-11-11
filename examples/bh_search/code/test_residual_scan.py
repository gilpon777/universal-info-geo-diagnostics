"""residual_scan のテスト。Run: python3 test_residual_scan.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import io
import json
import contextlib
import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))
import residual_scan


def test_all():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        residual_scan.main()
    out = json.load(open(str(HERE / "residual_scan_numbers.json")))
    assert out["n"] == 293
    assert out["bands"] == {">20": 19, "15-20": 31,
                            "10-15": 74, "5-10": 169}
    assert out["bh1"]["rank_ruwe"] == 179
    assert out["bh1"]["nss"] == 1
    assert out["bh1_rank_within_nss"] == 6
    assert abs(out["bh1"]["dist_pc"] - 483) < 2


if __name__ == "__main__":
    test_all()
    print("all tests passed")
