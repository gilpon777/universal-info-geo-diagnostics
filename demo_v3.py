"""v3 demonstration on the two real cases where v2 stayed silent.

(1) Void stack: the new greedy breakdown states, natively, that the
    centre bin wins on NOISE (driver + 14.2x score margin).
(2) GW projected observable Sigma(b) at realistic noise: v2 reported
    r_eff = 1.004 and stopped; v3 additionally reports the absolute
    scale -- stiffest-direction SNR 0.019 and the ~2,700x information
    multiplier (~16,000 crossings) needed to reach SNR 1.
"""
import json
import os

import numpy as np

from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3 as Diag)

VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "numbers")
out = {}

# ---------------- (1) void ----------------
va = np.load(f"{VAL}/validation_arrays.npz")
profile, cov_v, xc = va["profile"], va["cov"], va["xc"]
summ = json.load(open(f"{VAL}/summary.json"))
th = np.array(summ["theta_fit"])


def hamaus_profile(x_, t):
    dc_, s, alpha, beta = t
    return 1.0 + dc_ * (1.0 - (x_ / s) ** alpha) / (1.0 + x_ ** beta)


d = Diag(lambda t: hamaus_profile(xc, t))
res = d.evaluate(th, covariance=cov_v)
recs = d.greedy_select(th, cov_v, max_points=7)
out["void"] = {
    "scaled_direction_snr": np.round(
        res["Absolute"]["Scaled"]["DirectionSNR"], 2).tolist(),
    "greedy_first3": [
        {"index": r["Index"], "xc_Rv": float(xc[r["Index"]]),
         "driver": r["Driver"],
         "diag_score_margin": round(r["DiagScoreMargin"], 1),
         "gain_margin": round(r["GainMargin"], 2)} for r in recs[:3]],
}

# ---------------- (2) GW Sigma(b), realistic noise ----------------
def hamaus_contrast(x_, t):
    dc_, s, alpha, beta = t
    return dc_ * (1.0 - (x_ / s) ** alpha) / (1.0 + x_ ** beta)


def sigma_proj(b, t):
    smax = np.sqrt(max(64.0 - b * b, 0.0))
    sg = np.linspace(0.0, smax, 1201)
    r = np.sqrt(b * b + sg ** 2)
    return float(2.0 * np.trapz(hamaus_contrast(r, t), sg))


bs6 = [0.972, 0.088, 0.146, 0.953, 0.374, 0.802]
base_v = np.array([sigma_proj(b, th) for b in bs6])
sigma_real = 0.12 / 2.838e-5  # fractional error ~4200x (from collapse memo)
cov_gw = np.diag((sigma_real * np.abs(base_v)) ** 2)
model_gw = lambda t: np.array([sigma_proj(b, t) for b in bs6])
res_gw = Diag(model_gw).evaluate(th, covariance=cov_gw)
ab = res_gw["Absolute"]["Raw"]
out["gw_sigma_b"] = {
    "r_eff_raw": float(res_gw["Raw"]["EffectiveRank"]),
    "stiffest_direction_snr": float(ab["StiffestDirectionSNR"]),
    "info_multiplier_stiffest": float(
        ab["InfoMultiplierForTargetSNR"][0]),
    "equivalent_crossings_for_snr1": float(
        ab["InfoMultiplierForTargetSNR"][0] * 6),
}

print(json.dumps(out, indent=1))
with open(f"{VAL}/v3_demo_numbers.json", "w") as f:
    json.dump(out, f, indent=1)
