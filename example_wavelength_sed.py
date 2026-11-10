"""Wavelength selection for ordinary observation: galaxy SED decomposition.

Companion to wavelength_void.py (same skeleton, user's request: both).
Observation points = photometric bands (radio to X-ray).  Model: a
star-forming galaxy with a modest AGN, as a sum of components:

  old stars    4000 K blackbody (proxy)
  young stars  12000 K blackbody (proxy, UV-bright)
  cold dust    modified blackbody beta=1.8, T free (fiducial 30 K)
  warm dust    modified blackbody beta=1.8, 150 K (MIR bump / torus+PAH proxy)
  AGN          power law nu^-0.7 (UV to X-ray)
  synchrotron  power law nu^-0.8

Parameters theta = (lnA_old, lnA_young, lnA_cdust, T_cdust, lnA_wdust,
lnA_agn, lnA_sync); fiducial amplitudes (0, ln0.3, 0, 30, ln0.5, ln0.2, 0)
with templates normalised to 1 at their reference bands.
Noise: 5% relative photometry at the fiducial flux per band,
sigma_b = 0.05 * F_b(theta0), plus a 0.002 absolute floor.

Validation: (V1) template peaks vs Wien expectations;
(V2) linear amplitude sub-problem: framework FIM vs analytic.
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
h, kB = 6.62607015e-34, 1.380649e-23

BANDS = [
    ("radio 150MHz", 1.50e8), ("radio 1.4GHz", 1.40e9), ("radio 5GHz", 5.00e9),
    ("FIR 250um", 1.20e12), ("FIR 160um", 1.87e12), ("FIR 100um", 3.00e12),
    ("FIR 70um", 4.28e12), ("MIR 24um", 1.25e13), ("MIR 8um", 3.75e13),
    ("NIR 3.6um", 8.33e13), ("NIR 2.2um", 1.36e14), ("NIR 1.2um", 2.50e14),
    ("optical 900nm", 3.33e14), ("optical 550nm", 5.45e14),
    ("optical 440nm", 6.81e14), ("NUV 230nm", 1.30e15), ("FUV 150nm", 2.00e15),
    ("X-ray 1keV", 2.42e17), ("X-ray 5keV", 1.21e18),
]
NU = np.array([b[1] for b in BANDS])
NAMES = [b[0] for b in BANDS]


def Bnu(nu, T):
    x = h * nu / (kB * T)
    return nu ** 3 / np.expm1(np.minimum(x, 300.0))


def mbb(nu, T, ref_nu):
    v = nu ** 1.8 * Bnu(nu, T)
    ref = ref_nu ** 1.8 * float(Bnu(np.array([ref_nu]), T)[0])
    return v / ref


def bb(nu, T, ref_nu):
    return Bnu(nu, T) / float(Bnu(np.array([ref_nu]), T)[0])


def t_old(nu): return bb(nu, 4000.0, 1.36e14)
def t_young(nu): return bb(nu, 12000.0, 2.00e15)
def t_cdust(nu, T): return mbb(nu, T, 1.87e12)
def t_wdust(nu): return mbb(nu, 150.0, 1.25e13)
def t_agn(nu):
    # disk/corona power law spans UV to X-ray only; it does NOT extend
    # to radio (radio belongs to the synchrotron/jet component).  An
    # unrestricted power law would swamp the radio bands by ~1e4 and
    # make synchrotron formally invisible -- a template artefact,
    # caught in review and fixed here.
    return np.where(nu >= 1e11, (nu / 5.45e14) ** -0.7, 0.0)
def t_sync(nu): return (nu / 1.50e8) ** -0.8


def templates_matrix(T=30.0):
    cols = [t_old(NU), t_young(NU), t_cdust(NU, T), t_wdust(NU),
            t_agn(NU), t_sync(NU)]
    return np.stack(cols, axis=1)


AMP0 = np.array([1.0, 0.3, 1.0, 0.5, 0.2, 1.0])
THETA0 = np.array([0.0, np.log(0.3), 0.0, 30.0, np.log(0.5),
                   np.log(0.2), 0.0])


def model(theta):
    lnAo, lnAy, lnAcd, T, lnAwd, lnAagn, lnAsy = theta
    return (np.exp(lnAo) * t_old(NU) + np.exp(lnAy) * t_young(NU)
            + np.exp(lnAcd) * t_cdust(NU, T) + np.exp(lnAwd) * t_wdust(NU)
            + np.exp(lnAagn) * t_agn(NU) + np.exp(lnAsy) * t_sync(NU))


F0 = model(THETA0)
SIG = 0.05 * np.abs(F0) + 0.002
COV = np.diag(SIG ** 2)

# V1 -------------------------------------------------------------------------
nu_fine = np.logspace(7, 19, 4000)
for nm, fn, exp in [("old stars", lambda v: bb(v, 4000., 1.36e14), "~1.3 um"),
                    ("young stars", lambda v: bb(v, 12000., 2e15), "~0.42 um"),
                    ("cold dust 30K", lambda v: mbb(v, 30., 1.87e12), "~100 um"),
                    ("warm dust 150K", lambda v: mbb(v, 150., 1.25e13), "~20 um")]:
    pk = nu_fine[np.argmax(fn(nu_fine))]
    print(f"V1 {nm}: peak {2.998e8/pk*1e6:.2f} um (expect {exp})")

# V2 -------------------------------------------------------------------------
J = templates_matrix()
Jln = J * AMP0[None, :]  # d model/d lnA_k = A_k * template_k
FIMa = Jln.T @ np.diag(1 / SIG ** 2) @ Jln
# framework on ln-amplitudes at fixed T:
d_lin = Diag(lambda la: templates_matrix() @ np.exp(la))
res_lin = d_lin.evaluate(np.log(AMP0), covariance=COV)
ev_a = np.sort(np.linalg.eigvalsh(FIMa))[::-1]
ev_f = np.sort(res_lin["Raw"]["Eigenvalues"])[::-1]
print("V2 max rel diff (ln-amp FIM):",
      float(np.max(np.abs(ev_a - ev_f) / ev_a)))

# main -----------------------------------------------------------------------
d = Diag(model)
res = d.evaluate(THETA0, covariance=COV)
print("\nScaled r_eff:", res["Scaled"]["EffectiveRank"],
      " Raw r_eff:", res["Raw"]["EffectiveRank"])
print("Scaled direction SNR:", np.round(res["Absolute"]["Scaled"]["DirectionSNR"], 2))
recs = d.greedy_select(THETA0, COV, max_points=10)
greedy = [{"band": NAMES[r["Index"]], "driver": r["Driver"]} for r in recs]
print("Greedy:", greedy)

comp = ["old stars", "young stars", "cold dust", "warm dust", "AGN", "synchrotron"]
FIM = FIMa
covm = np.linalg.inv(FIM)
sig = np.sqrt(np.diag(covm))
corr = covm / np.outer(sig, sig)
print("\nper-component CR sigma (ln amplitude):")
for c, s in zip(comp, sig):
    print(f"  {c}: {s:.3f}")
print("strongest correlations:")
for i in range(6):
    for j in range(i + 1, 6):
        if abs(corr[i, j]) > 0.3:
            print(f"  {comp[i]} x {comp[j]}: {corr[i, j]:+.2f}")
print("\nleave-one-band-out (sigma growth, top3 per component):")
for k, c in enumerate(comp):
    fac = []
    for b in range(len(NU)):
        m = np.ones(len(NU), bool); m[b] = False
        W = Jln[m] / SIG[m][:, None]
        F2 = W.T @ W
        fac.append(np.sqrt(np.linalg.pinv(F2)[k, k]) / sig[k])
    order = np.argsort(fac)[::-1][:3]
    print(f"  {c}: " + ", ".join(f"{NAMES[i]} x{fac[i]:.2f}" for i in order))

best = {}
snr_mat = np.abs(J * AMP0[None, :]) / SIG[:, None]
for k, c in enumerate(comp):
    best[c] = NAMES[int(np.argmax(snr_mat[:, k]))]
print("\nnaive best band per component:", best)

out = {"bands": NAMES, "components": comp,
       "scaled_r_eff": float(res["Scaled"]["EffectiveRank"]),
       "raw_r_eff": float(res["Raw"]["EffectiveRank"]),
       "scaled_direction_snr": res["Absolute"]["Scaled"]["DirectionSNR"].tolist(),
       "greedy": greedy, "cr_sigma_lnamp": sig.tolist(),
       "snr_matrix": snr_mat.round(2).tolist(), "best_band_naive": best,
       "validation": {"analytic_eigs": ev_a.tolist(), "framework_eigs": ev_f.tolist()}}
with open(f"{VAL}/wavelength_sed_numbers.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nsaved wavelength_sed_numbers.json")
