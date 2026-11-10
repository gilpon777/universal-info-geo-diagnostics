"""Non-void application example for the publication package:
a first-order reaction chain A -> B -> C -> D (biochemistry-style
kinetics).  Observable: concentrations of B and C at 12 candidate
sampling times.  Question, structurally identical to the void case:
which sampling times should be measured (observation-point selection),
and which rate constants can the data identify at all?

Model (analytic, A(0)=1):
  B(t) = k1/(k2-k1) (e^{-k1 t} - e^{-k2 t})
  C(t) = k1 k2 [ e^{-k1 t}/((k2-k1)(k3-k1)) + e^{-k2 t}/((k1-k2)(k3-k2))
                 + e^{-k3 t}/((k1-k3)(k2-k3)) ]
theta = (k1, k2, k3, A0).  Noise: sigma = 0.01 + 0.03*value per point
(heteroscedastic, diagonal).
"""
import json

import numpy as np

from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3 as Diag)

TIMES = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                  18.0, 28.0])
TH = np.array([0.9, 0.35, 0.12, 1.0])  # k1, k2, k3, A0


def chain(th):
    k1, k2, k3, A0 = th
    t = TIMES
    B = A0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    C = A0 * k1 * k2 * (
        np.exp(-k1 * t) / ((k2 - k1) * (k3 - k1))
        + np.exp(-k2 * t) / ((k1 - k2) * (k3 - k2))
        + np.exp(-k3 * t) / ((k1 - k3) * (k2 - k3)))
    return np.concatenate([B, C])


y = chain(TH)
sig = 0.01 + 0.03 * np.abs(y)
cov = np.diag(sig ** 2)

d = Diag(chain)
res = d.evaluate(TH, covariance=cov)
recs = d.greedy_select(TH, cov, max_points=8)

labels = [f"B@{t:g}" for t in TIMES] + [f"C@{t:g}" for t in TIMES]
picks = [{"point": labels[r["Index"]], "driver": r["Driver"],
          "gain_margin": (round(r["GainMargin"], 2)
                          if np.isfinite(r["GainMargin"]) else None),
          "diag_score_margin": (round(r["DiagScoreMargin"], 2)
                                if np.isfinite(r["DiagScoreMargin"]) else None)}
         for r in recs]
out = {
    "model": "A->B->C->D first-order chain, theta=(k1,k2,k3,A0)",
    "theta": TH.tolist(),
    "r_eff_raw": float(res["Raw"]["EffectiveRank"]),
    "r_eff_scaled": float(res["Scaled"]["EffectiveRank"]),
    "log10CN_scaled": float(res["Scaled"]["Log10ConditionNumber"]),
    "scaled_direction_snr": np.round(
        res["Absolute"]["Scaled"]["DirectionSNR"], 2).tolist(),
    "param_corr_k1_k2": float(res["FIM_correlation"][0, 1]),
    "greedy_picks": picks,
}
print(json.dumps(out, indent=1))
with open("/home/hatch/workspace/info_geo_diag/biochem_example_numbers.json",
          "w") as f:
    json.dump(out, f, indent=1)
