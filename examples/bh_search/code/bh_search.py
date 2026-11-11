"""光学検索コードの玩具 — 休眠ブラックホール候補の
フォローアップ観測点選定。

状況設定: Gaia の位置測定が「明るい星 + 見えない重い
伴星」の候補を挙げた (Gaia BH1 型)。周期は位置測定で
分かっている。残る仕事は分光フォローアップで伴星の
視線速度曲線を測り、質量関数 f ∝ K^3 (1-e^2)^{3/2}
から暗い伴星の質量を確定すること — そして偽物
(普通の二重星) を落とすこと。

観測点 = 分光する夜 (エポック)。候補 36 夜の中から
N 夜を選ぶとき、均等に置くか、ユーザーの v3
フレームワークで貪欲に選ぶかで、σ(K) と σ(e) が
どれだけ違うかを測る。

パラメータ: theta = (K [km/s], e, omega [rad], T0 [d])。
周期 P = 185 d 固定 (位置測定で既知という設定)。
ノイズ: 1 エポック 0.30 km/s。
"""
from __future__ import annotations


from pathlib import Path as _Path
HERE = _Path(__file__).resolve().parent
DATA = HERE.parent / "data"

import json
import sys

import numpy as np

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
from universal_info_geo_diagnostics_v3 import (  # noqa: E402
    UniversalInformationGeometricDiagnosticsV3 as Diag)

P_DAYS = 185.0
N_CAND = 36
SIG_V = 0.30                       # km/s per epoch
EPOCHS = np.linspace(0, P_DAYS, N_CAND, endpoint=False)
THETA0 = np.array([65.0, 0.45, 1.0, 40.0])      # K, e, omega, T0
SCALES = np.array([30.0, 0.30, 1.0, 30.0])


def rv_curve(theta, t):
    K, e, om, T0 = theta
    M = 2 * np.pi * (t - T0) / P_DAYS
    E = M.copy()
    for _ in range(12):                          # Newton でケプラー方程式
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    return K * (np.cos(om + nu) + e * np.cos(om))


def model(theta):
    return rv_curve(theta, EPOCHS)


def sigma_for(idxs, J, sig=SIG_V):
    """選んだエポック集合での周辺化 σ (K と e)。"""
    Js = J[list(idxs)] / sig
    F = Js.T @ Js + 1e-10 * np.eye(J.shape[1])
    C = np.linalg.inv(F)
    return np.sqrt(C[0, 0]), np.sqrt(C[1, 1])


def main():
    out = {}
    d = Diag(model, param_scales=SCALES)
    J = d.compute_jacobian(THETA0)
    cov = np.diag(np.full(N_CAND, SIG_V**2))

    res = d.evaluate(THETA0, covariance=cov)
    print(f"full set: Scaled r_eff "
          f"{res['Scaled']['EffectiveRank']:.2f} / 4")

    for N in (4, 6, 8):
        recs = d.greedy_select(THETA0, cov, max_points=N)
        greedy = [r["Index"] for r in recs]
        uniform = list(np.linspace(0, N_CAND - 1, N).round().astype(int))
        rng = np.random.default_rng(7)
        rand = list(rng.choice(N_CAND, N, replace=False))
        sg = sigma_for(greedy, J)
        su = sigma_for(uniform, J)
        sr = sigma_for(rand, J)
        phases = np.sort(EPOCHS[greedy] / P_DAYS)
        print(f"N={N}: sigma_K greedy {sg[0]:.2f} vs uniform {su[0]:.2f} "
              f"vs random {sr[0]:.2f} km/s | sigma_e greedy {sg[1]:.3f} "
              f"vs uniform {su[1]:.3f}")
        print(f"      greedy phases: {np.round(phases, 2)} "
              f"(periastron at phase {(THETA0[3] / P_DAYS) % 1:.2f})")
        out[f"N{N}"] = {"greedy_idx": greedy,
                        "sigmaK_g": sg[0], "sigmaK_u": su[0],
                        "sigmaK_r": sr[0], "sigmae_g": sg[1],
                        "sigmae_u": su[1],
                        "gain_K": su[0] / sg[0], "gain_e": su[1] / sg[1]}
    with open(str(HERE / "bh_search_numbers.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("saved bh_search_numbers.json")


if __name__ == "__main__":
    main()
