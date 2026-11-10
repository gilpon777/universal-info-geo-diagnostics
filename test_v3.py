"""v3 tests: v2 equivalence, absolute scale, nuisance profiling /
marginalization, and the greedy attention breakdown on the real void data."""
import json
import os

import numpy as np

from universal_info_geo_diagnostics_v2 import (
    UniversalInformationGeometricDiagnosticsV2)
from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3)

rng = np.random.default_rng(42)
ok = []

# --- 1. v2 equivalence on a linear model --------------------------------
A = rng.normal(size=(30, 4))
theta_t = np.array([1.0, -2.0, 0.5, 3.0])
model = lambda th: A @ th
cov = np.diag(rng.uniform(0.01, 0.09, 30))
r2 = UniversalInformationGeometricDiagnosticsV2(model).evaluate(
    theta_t, covariance=cov)
r3 = UniversalInformationGeometricDiagnosticsV3(model).evaluate(
    theta_t, covariance=cov)
assert np.allclose(r2["Raw"]["Eigenvalues"], r3["Raw"]["Eigenvalues"])
assert np.allclose(r2["Scaled"]["Eigenvalues"], r3["Scaled"]["Eigenvalues"])
assert np.allclose(r2["FIM"], r3["FIM"])
ok.append("v2 equivalence")

# --- 2. absolute scale against analytic values ---------------------------
F_analytic = A.T @ np.linalg.solve(cov, A)
ev = np.sort(np.linalg.eigvalsh(F_analytic))[::-1]
ab = UniversalInformationGeometricDiagnosticsV3.absolute_scale(ev, 1.0)
assert np.allclose(ab["DirectionSNR"], np.sqrt(ev))
assert np.allclose(ab["DirectionSigmaCR"], 1.0 / np.sqrt(ev))
assert np.allclose(ab["InfoMultiplierForTargetSNR"], 1.0 / ev)
# FIM doubles when the same observations are repeated twice
A2 = np.vstack([A, A]); cov2 = np.diag(np.tile(np.diag(cov), 2))
r3b = UniversalInformationGeometricDiagnosticsV3(
    lambda th: A2 @ th).evaluate(theta_t, covariance=cov2)
assert np.allclose(np.sort(r3b["Raw"]["Eigenvalues"])[::-1], 2 * ev)
ok.append("absolute scale (analytic + doubling)")

# --- 3. nuisance basis profiling absorbs a constant-amplitude direction --
x = np.linspace(0, 1, 25)
B = np.ones((25, 1))
model2 = lambda th: th[0] * np.ones(25) + th[1] * x
cov2b = 0.04 * np.eye(25)
d3 = UniversalInformationGeometricDiagnosticsV3(model2)
res = d3.evaluate(np.array([2.0, 1.0]), covariance=cov2b,
                  nuisance_basis=B)
prof = res["NuisanceProfiled"]["FIM"]
assert abs(prof[0, 0]) < 1e-6 and abs(prof[0, 1]) < 1e-6, prof
assert prof[1, 1] > 1.0
unprof = res["FIM"]
assert unprof[0, 0] > 100
ok.append("nuisance basis absorption (constant amplitude -> zero)")

# --- 4. Schur marginalization on a known 2x2 ------------------------------
F = np.array([[2.0, 1.0], [1.0, 2.0]])
Fm = UniversalInformationGeometricDiagnosticsV3.marginalize_fim(F, [1])
assert np.allclose(Fm, [[1.5]])
ok.append("Schur marginalization")

# --- 5. greedy breakdown reproduces the void centre story -----------------
VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "numbers")
va = np.load(f"{VAL}/validation_arrays.npz")
prof_data, cov_v = va["profile"], va["cov"]
xc = va["xc"]
summ = json.load(open(f"{VAL}/summary.json"))
th_fit = np.array(summ["theta_fit"])

def hamaus_profile(x_, th):
    dc_, s, alpha, beta = th
    return 1.0 + dc_ * (1.0 - (x_ / s) ** alpha) / (1.0 + x_ ** beta)

d3v = UniversalInformationGeometricDiagnosticsV3(
    lambda th: hamaus_profile(xc, th))
recs = d3v.greedy_select(th_fit, cov_v, max_points=7)
first = recs[0]
assert first["Index"] == 0, first
assert first["Driver"] == "noise", first
assert 10.0 < first["DiagScoreMargin"] < 20.0, first
assert abs(first["DiagScore"] - 118220) / 118220 < 0.05, first
order = [r["Index"] for r in recs]
assert order[:3] == [0, 5, 9], order  # centre, 0.79 Rv, 1.36 Rv
ok.append(f"void greedy breakdown (order {order[:3]}, "
          f"driver={first['Driver']}, score margin "
          f"{first['DiagScoreMargin']:.1f}x)")

print("ALL V3 TESTS PASSED:", "; ".join(ok))
