"""gamma_scan のテスト。Run: python3 test_gamma_scan.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import gamma_scan as gs


def test_gamma():
    assert gs.horizon_pc(0.0, gs.F_PLANE) == 0.0        # 休眠型は見えない
    assert gs.horizon_pc(3e36, gs.F_PLANE) > 8000       # 爆発型は銀河全体
    assert gs.horizon_pc(1e28, gs.F_PLANE) < 20         # 孤立型 (典型) は近傍のみ
    assert gs.horizon_pc(3e32, gs.F_HI) > gs.horizon_pc(3e32, gs.F_PLANE)


if __name__ == "__main__":
    test_gamma()
    print("all tests passed")
