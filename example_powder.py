"""使用例（通常観測）: 分光の「チャンネル」が観測点。どの波長
チャンネルが組成を決めるかを選ぶ、観測点選定の例。

Non-void application example 2: powder mixture composition.

A 4-component oxide powder (SiO2 / Al2O3 / Fe2O3 / TiO2 -- ceramic and
glaze-like materials).  Observable: a reflectance spectrum in 40
channels (300-700 nm); each pure component has a fixed spectral shape
and the mixture spectrum is the fraction-weighted sum (the standard
linear-mixture approximation used as the first step of XRF/NIR-style
composition work).  Fractions live on the simplex, so the free
parameters are (f1, f2, f3) with f4 = 1 - f1 - f2 - f3.

Questions, same structure as the void bins and the sampling times:
which channels actually determine the composition (channel selection),
and which component trade-offs stay degenerate (identifiability).
"""
import json
import os

import numpy as np

from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3 as Diag)

LAM = np.linspace(300, 700, 40)  # nm


def gauss(x, mu, s, a):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


PURE = np.stack([
    0.55 + gauss(LAM, 620, 90, 0.10),                    # SiO2: flat-ish
    0.30 + gauss(LAM, 385, 28, 0.55),                    # Al2O3
    0.12 + gauss(LAM, 430, 35, 0.65) + gauss(LAM, 535, 45, 0.45),  # Fe2O3
    0.25 + gauss(LAM, 340, 16, 0.80),                    # TiO2: sharp UV
])
NAMES = ["SiO2", "Al2O3", "Fe2O3", "TiO2"]
F_TRUE = np.array([0.50, 0.25, 0.15, 0.10])


def mixture(th):
    f = np.array([th[0], th[1], th[2], 1.0 - th.sum()])
    return f @ PURE


th0 = F_TRUE[:3]
y = mixture(th0)
sig = 0.004 + 0.01 * np.abs(y)
cov = np.diag(sig ** 2)

d = Diag(mixture)
res = d.evaluate(th0, covariance=cov)
recs = d.greedy_select(th0, cov, max_points=8)

# Cramér-Rao sigma for the four fractions (delta method for f4)
F = res["FIM"]
cov_th = np.linalg.inv(F)
sig_f = np.sqrt(np.diag(cov_th))
sig_f4 = float(np.sqrt(np.sum(cov_th)))  # Var(1 - f1 - f2 - f3)

out = {
    "components": NAMES,
    "true_fractions": F_TRUE.tolist(),
    "r_eff_raw": float(res["Raw"]["EffectiveRank"]),
    "r_eff_scaled": float(res["Scaled"]["EffectiveRank"]),
    "log10CN_scaled": float(res["Scaled"]["Log10ConditionNumber"]),
    "scaled_direction_snr": np.round(
        res["Absolute"]["Scaled"]["DirectionSNR"], 2).tolist(),
    "cramer_rao_sigma_fractions": np.round(
        np.concatenate([sig_f, [sig_f4]]), 4).tolist(),
    "param_corr_matrix_free": np.round(res["FIM_correlation"], 2).tolist(),
    "greedy_channels_nm": [
        {"lambda_nm": float(LAM[r["Index"]]), "driver": r["Driver"],
         "gain_margin": (round(r["GainMargin"], 2)
                         if np.isfinite(r["GainMargin"]) else None)}
        for r in recs],
}
print(json.dumps(out, indent=1))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "powder_example_numbers.json"),
          "w") as f:
    json.dump(out, f, indent=1)
