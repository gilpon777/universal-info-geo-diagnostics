"""Wavelength selection for ordinary observation: what is inside a void?

The user's framework used as originally intended -- no events needed.
Observation points = wavelength bands (radio to X-ray).  Model: the
line-of-sight contents of a cosmic void as a sum of emission
components, each with its own spectral template:

  dust        modified blackbody, beta=1.8, T free (fiducial 20 K)
  stars       4000 K blackbody (old stellar population proxy)
  hot gas     bremsstrahlung, kT = 0.5 keV (shape fixed)
  synchrotron power law nu^-0.8
  free-free   nu^-0.1 (warm ionised gas)
  HI          21 cm line (only the line band carries flux)

Parameters theta = (lnA_dust, T_dust, lnA_star, lnA_gas, lnA_sync,
lnA_ff, lnA_HI).  Each template is normalised to 1 at its own
reference band and fiducial amplitudes are 1, so the STRUCTURE
(which band sees which component, degeneracies, minimal band sets)
is the result; absolute levels are stated assumptions, and an
equal-noise ablation separates template geometry from the assumed
band depths.

Noise per band (same normalised flux units) = assumed typical deep
survey depths: radio 0.01, HI line 0.05, mm 0.02, FIR 0.03
(confusion), MIR 0.02, NIR/optical 0.005, UV 0.01, X-ray 0.02.

Validation: (V1) template peaks vs Wien / shape expectations;
(V2) for the linear amplitude sub-problem (T fixed) the framework's
FIM eigenvalues must match the analytic J^T Sigma^-1 J.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universal_info_geo_diagnostics_v3 import (  # noqa: E402
    UniversalInformationGeometricDiagnosticsV3 as Diag)

import os
VAL = os.path.dirname(os.path.abspath(__file__))
h, kB, eV = 6.62607015e-34, 1.380649e-23, 1.602176634e-19

# ---------------- bands: (name, nu_Hz, sigma, kind) -------------------------
BANDS = [
    ("radio 150MHz", 1.50e8, 0.010, "cont"),
    ("radio 408MHz", 4.08e8, 0.010, "cont"),
    ("HI line 1.420GHz", 1.420e9, 0.050, "line"),
    ("radio 1.4GHz", 1.40e9, 0.010, "cont"),
    ("radio 5GHz", 5.00e9, 0.010, "cont"),
    ("radio 15GHz", 1.50e10, 0.010, "cont"),
    ("mm 90GHz", 9.00e10, 0.020, "cont"),
    ("mm 150GHz", 1.50e11, 0.020, "cont"),
    ("mm 230GHz", 2.30e11, 0.020, "cont"),
    ("FIR 350um", 8.57e11, 0.030, "cont"),
    ("FIR 250um", 1.20e12, 0.030, "cont"),
    ("FIR 160um", 1.87e12, 0.030, "cont"),
    ("FIR 100um", 3.00e12, 0.030, "cont"),
    ("MIR 24um", 1.25e13, 0.020, "cont"),
    ("NIR 2.2um", 1.36e14, 0.005, "cont"),
    ("optical 550nm", 5.45e14, 0.005, "cont"),
    ("UV 150nm", 2.00e15, 0.010, "cont"),
    ("X-ray 1keV", 2.42e17, 0.020, "cont"),
    ("X-ray 5keV", 1.21e18, 0.020, "cont"),
]
NU = np.array([b[1] for b in BANDS])
SIG = np.array([b[2] for b in BANDS])
NAMES = [b[0] for b in BANDS]
LINE_IDX = [i for i, b in enumerate(BANDS) if b[3] == "line"]


def Bnu(nu, T):
    x = h * nu / (kB * T)
    return nu ** 3 / np.expm1(np.minimum(x, 300.0))


# templates normalised to 1 at their reference band --------------------------
def t_dust(nu, T):
    v = nu ** 1.8 * Bnu(nu, T)
    ref = (8.57e11) ** 1.8 * float(Bnu(np.array([8.57e11]), T)[0])
    return v / ref


def t_star(nu):
    ref = Bnu(np.array([1.36e14]), 4000.0)[0]
    return Bnu(nu, 4000.0) / ref


def t_gas(nu):
    kT = 0.5e3 * eV
    v = np.exp(-h * nu / kT)
    return v / np.exp(-h * 2.42e17 / kT)


def t_sync(nu):
    return (nu / 1.50e8) ** -0.8


def t_ff(nu):
    return (nu / 1.50e8) ** -0.1


def t_hi(nu):
    out = np.zeros_like(np.asarray(nu, dtype=float))
    return out  # continuum bands: no HI; line handled below


def model(theta):
    lnAd, Td, lnAs, lnAg, lnAsy, lnAff, lnAhi = theta
    f = (np.exp(lnAd) * t_dust(NU, Td) + np.exp(lnAs) * t_star(NU)
         + np.exp(lnAg) * t_gas(NU) + np.exp(lnAsy) * t_sync(NU)
         + np.exp(lnAff) * t_ff(NU))
    for i in LINE_IDX:  # HI line flux lives only in the line band
        f[i] += np.exp(lnAhi)
    return f


THETA0 = np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
COV = np.diag(SIG ** 2)

# ---------------- V1: template shape checks ---------------------------------
nu_fine = np.logspace(7, 19, 4000)
pk_dust = nu_fine[np.argmax(t_dust(nu_fine, 20.0))]
pk_star = nu_fine[np.argmax(t_star(nu_fine))]
print(f"V1 dust peak: {pk_dust:.2e} Hz (lambda {2.998e8/pk_dust*1e6:.0f} um; "
      f"expect FIR ~100-350 um)")
print(f"V1 star peak: {pk_star:.2e} Hz (lambda {2.998e8/pk_star*1e6:.2f} um; "
      f"expect NIR ~1.3 um)")
print(f"V1 gas: F(5keV)/F(1keV) = {t_gas(np.array([1.21e18]))[0]:.5f} "
      f"(expect exp(-8) ~ 0.00034) sync ratio 5GHz/150MHz = "
      f"{t_sync(np.array([5e9]))[0]:.3f} (expect (33.3)^-0.8 = 0.060)")

# ---------------- V2: analytic FIM for linear amplitudes --------------------
def templates_matrix(Td=20.0):
    cols = [t_dust(NU, Td), t_star(NU), t_gas(NU), t_sync(NU), t_ff(NU),
            np.array([1.0 if i in LINE_IDX else 0.0 for i in range(len(NU))])]
    return np.stack(cols, axis=1)  # bands x 6 amplitudes

J = templates_matrix()
FIM_analytic = J.T @ np.diag(1 / SIG ** 2) @ J
ev_analytic = np.sort(np.linalg.eigvalsh(FIM_analytic))[::-1]

d_lin = Diag(lambda a: templates_matrix() @ a)
res_lin = d_lin.evaluate(np.ones(6), covariance=COV)
ev_fw = np.sort(res_lin["Raw"]["Eigenvalues"])[::-1]
print("V2 analytic eigenvalues:", np.round(ev_analytic, 1))
print("V2 framework eigenvalues:", np.round(ev_fw, 1))
print("V2 max rel diff:", float(np.max(np.abs(ev_analytic - ev_fw) / ev_analytic)))

# ---------------- main run ---------------------------------------------------
d = Diag(model)
res = d.evaluate(THETA0, covariance=COV)
print("\nScaled r_eff:", res["Scaled"]["EffectiveRank"],
      " Raw r_eff:", res["Raw"]["EffectiveRank"])
print("Scaled direction SNR:", np.round(
    res["Absolute"]["Scaled"]["DirectionSNR"], 2))
print("Scaled direction sigma (CR):", np.round(
    res["Absolute"]["Scaled"]["DirectionSigmaCR"], 3))

recs = d.greedy_select(THETA0, COV, max_points=10)
greedy = [{"band": NAMES[r["Index"]], "driver": r["Driver"],
           "margin": r.get("Margin")} for r in recs]
print("\nGreedy band order:")
for g in greedy:
    print("  ", g)

# per-component best band (component alone vs band noise)
comp_names = ["dust", "stars", "hot gas", "synchrotron", "free-free", "HI line"]
best = {}
for k, cn in enumerate(comp_names):
    snr_b = np.abs(J[:, k]) / SIG
    best[cn] = {"band": NAMES[int(np.argmax(snr_b))],
                "snr": float(np.max(snr_b)),
                "top3": [NAMES[i] for i in np.argsort(snr_b)[::-1][:3]]}
    print(f"best band {cn}: {best[cn]['band']} (SNR {best[cn]['snr']:.1f})")

# equal-noise ablation: does the greedy order survive?
COV_EQ = np.diag(np.full(len(NU), 0.01 ** 2))
recs_eq = d.greedy_select(THETA0, COV_EQ, max_points=6)
print("\nEqual-noise greedy:", [NAMES[r["Index"]] for r in recs_eq])
res_eq = d.evaluate(THETA0, covariance=COV_EQ)
print("Equal-noise Scaled r_eff:", res_eq["Scaled"]["EffectiveRank"])

out = {"bands": NAMES, "sigma": SIG.tolist(),
       "scaled_r_eff": float(res["Scaled"]["EffectiveRank"]),
       "raw_r_eff": float(res["Raw"]["EffectiveRank"]),
       "scaled_direction_snr": res["Absolute"]["Scaled"]["DirectionSNR"].tolist(),
       "greedy": greedy, "best_band_per_component": best,
       "snr_matrix": (np.abs(J) / SIG[:, None]).round(2).tolist(),
       "components": comp_names,
       "equal_noise": {"scaled_r_eff": float(res_eq["Scaled"]["EffectiveRank"]),
                       "greedy6": [NAMES[r["Index"]] for r in recs_eq]},
       "validation": {"analytic_eigs": ev_analytic.tolist(),
                      "framework_eigs": ev_fw.tolist()}}
with open(f"{VAL}/wavelength_void_numbers.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nsaved wavelength_void_numbers.json")
