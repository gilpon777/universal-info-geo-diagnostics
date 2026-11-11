"""wavelength_bh のテスト。Run: python3 test_wavelength_bh.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import numpy as np

import wavelength_bh as wb


def test_templates():
    assert wb.t_jet(np.array([1.5e8]))[0] == 1.0
    assert wb.t_corona(np.array([1.0e15]))[0] == 0.0
    assert abs(wb.t_corona(np.array([2.42e17]))[0] - 1.0) < 1e-12
    J = wb.templates_matrix()
    assert J.shape == (len(wb.NU), 5)
    # 電波はジェットのみ、X 線はコロナのみが担う
    assert abs(J[0, 0]) < 1e-8 and J[0, 3] == 1.0
    assert J[18, 4] > 0 and abs(J[18, 0]) < 1e-6


def test_states():
    q = wb.run_state(wb.THETA_Q, wb.AMP_Q, "q")
    hd = wb.run_state(wb.THETA_H, wb.AMP_H, "h")
    assert 3.2 < q["scaled_r_eff"] < 3.4
    assert 2.9 < hd["scaled_r_eff"] < 3.2
    assert q["greedy"][0]["band"] == "optical 550nm"
    assert hd["greedy"][0]["band"] == "optical 440nm"
    assert "radio 150MHz" in [g["band"] for g in hd["greedy"][:3]]
    # 円盤と輝点の縮退が両状態で最強クラス
    for st in (q, hd):
        worst = min(p[2] for p in st["correlations"])
        assert worst < -0.7
    # ハード状態では円盤が締まり、伴星が緩む
    assert hd["cr_sigma_lnamp"]["disk"] < q["cr_sigma_lnamp"]["disk"]
    assert hd["cr_sigma_lnamp"]["companion"] > q["cr_sigma_lnamp"]["companion"]


if __name__ == "__main__":
    test_templates()
    test_states()
    print("all tests passed")
