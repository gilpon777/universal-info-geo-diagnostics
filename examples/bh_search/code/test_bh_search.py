"""bh_search のテスト。Run: python3 test_bh_search.py"""

from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import sys

sys.path.insert(0, str(_Path(__file__).resolve().parent))
import os as _os
_fw = _os.environ.get("INFO_GEO_DIAG_DIR")
if not _fw:
    for _cand in (_Path(__file__).resolve().parents[3],
                  _Path(__file__).resolve().parents[3] / "info_geo_diag",
                  _Path.home() / "workspace" / "info_geo_diag"):
        if (_cand / "universal_info_geo_diagnostics_v3.py").exists():
            _fw = str(_cand)
            break
if _fw:
    sys.path.insert(0, _fw)

import numpy as np

import bh_search as bs
from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3 as Diag)


def test_search():
    d = Diag(bs.model, param_scales=bs.SCALES)
    J = d.compute_jacobian(bs.THETA0)
    cov = np.diag(np.full(bs.N_CAND, bs.SIG_V**2))
    recs = d.greedy_select(bs.THETA0, cov, max_points=6)
    greedy = [r["Index"] for r in recs]
    uniform = list(np.linspace(0, bs.N_CAND - 1, 6).round().astype(int))
    sg = bs.sigma_for(greedy, J)
    su = bs.sigma_for(uniform, J)
    assert sg[0] < su[0] and sg[1] < su[1]          # 貪欲が両方で勝つ
    peri = (bs.THETA0[3] / bs.P_DAYS) % 1
    phases = bs.EPOCHS[greedy] / bs.P_DAYS
    near = np.mean([min(abs(p - peri), 1 - abs(p - peri)) < 0.2
                    for p in phases])
    assert near >= 0.6                              # 近点通過に集まる


if __name__ == "__main__":
    test_search()
    print("all tests passed")
