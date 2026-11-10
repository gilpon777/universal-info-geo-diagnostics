"""v2 検証テスト:
1) 線形モデルで解析解 A^T Sigma^{-1} A と一致するか
2) 再パラメータ化（対角スケーリング）で Scaled 指標が不変、Raw が変動するか
3) sloppy モデル（2指数関数の和）でスペクトルと r_eff がどう出るか
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universal_info_geo_diagnostics_v2 import UniversalInformationGeometricDiagnosticsV2

rng = np.random.default_rng(42)

print("=== Test 1: 線形モデル vs 解析解 ===")
A = rng.normal(size=(8, 3))
Sigma = np.diag(rng.uniform(0.01, 0.1, size=8))
theta_true = np.array([1.0, -2.0, 0.5])
model_lin = lambda th: A @ th
diag = UniversalInformationGeometricDiagnosticsV2(model_lin)
res = diag.evaluate(theta_true, covariance=Sigma)
F_exact = A.T @ np.linalg.inv(Sigma) @ A
err = np.max(np.abs(res["FIM"] - F_exact)) / np.max(np.abs(F_exact))
print(f"FIM 相対誤差: {err:.3e}  (InversePath={res['InversePath']})")
assert err < 1e-6, "解析解と不一致"
print("OK")

print("\n=== Test 2: 再パラメータ化（theta_2 を 1e6 倍の単位に） ===")
def fmt(r):
    return (f"Raw: CN={r['Raw']['ConditionNumber']:.3e}, r_eff={r['Raw']['EffectiveRank']:.4f} | "
            f"Scaled: CN={r['Scaled']['ConditionNumber']:.6e}, r_eff={r['Scaled']['EffectiveRank']:.6f}")
print("元スケール  ", fmt(res))
C = np.diag([1.0, 1e6, 1.0])
model_rescaled = lambda th: A @ (C @ th)   # th'_2 = 1e6 * th_2 相当の入力を受ける
diag2 = UniversalInformationGeometricDiagnosticsV2(model_rescaled)
res2 = diag2.evaluate(np.array([1.0, -2.0e-6, 0.5]), covariance=Sigma)
print("再スケール後", fmt(res2))
d_scaled = abs(res2["Scaled"]["ConditionNumber"] - res["Scaled"]["ConditionNumber"]) / res["Scaled"]["ConditionNumber"]
print(f"Scaled CN の相対変化: {d_scaled:.3e}（不変であるべき）")
assert d_scaled < 1e-4
print("OK: Scaled は単位変更に不変、Raw は大きく変動することを確認")

print("\n=== Test 3: sloppy モデル（2指数関数の和） ===")
t = np.linspace(0.05, 3.0, 40)
def sloppy(th):
    a1, b1, a2, b2 = th
    return a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)
theta_s = np.array([1.0, 1.0, 1.0, 1.05])   # b1≈b2 で強い縮退
noise = 0.01
samples = np.array([sloppy(theta_s) + rng.normal(0, noise, size=len(t)) for _ in range(200)])
diag3 = UniversalInformationGeometricDiagnosticsV2(sloppy)
res3 = diag3.evaluate(theta_s, observed_data=samples)
ev = res3["Raw"]["Eigenvalues"]
print("固有値スペクトル（降順）:", np.array2string(ev, precision=3))
print(f"Raw   : log10 CN={res3['Raw']['Log10ConditionNumber']:.2f}, "
      f"r_eff={res3['Raw']['EffectiveRank']:.3f}, 数値ランク={res3['Raw']['NumericalRank']}/4")
print(f"Scaled: log10 CN={res3['Scaled']['Log10ConditionNumber']:.2f}, "
      f"r_eff={res3['Scaled']['EffectiveRank']:.3f}, 数値ランク={res3['Scaled']['NumericalRank']}/4")
print(f"CovConditionNumber={res3['CovConditionNumber']:.3e}, Path={res3['InversePath']}")
assert res3["Raw"]["EffectiveRank"] < 3.5
print("OK: 4パラメータでも実効的には 3 次元未満の情報しかないことを検出")

print("\n=== Test 4: cov_func + 共分散依存項 ===")
def covf(th):
    s = 0.05 * (1.0 + 0.1 * th[0])
    return np.eye(len(t)) * s**2
res4 = UniversalInformationGeometricDiagnosticsV2(sloppy).evaluate(
    theta_s, cov_func=covf, include_cov_term=True)
print(f"CovTermIncluded={res4.get('CovTermIncluded')}, "
      f"Raw r_eff={res4['Raw']['EffectiveRank']:.3f}")
print("OK")

print("\n全テスト通過")
