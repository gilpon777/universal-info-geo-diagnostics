"""(M, y) detectability map for point-lens wave-optics signatures.

Forward-looking companion to lens_observable_check.py.  That check
translated the MEASURED residual floor f99 into the (M, y) plane
(backward-looking: what the residual method already excluded).
This code asks the forward question with the user's framework:
treating (ln M, y) as parameters, build the Fisher matrix of the
lensed waveform h_L = F(f; M, y) * h(f), profile out the nuisance
directions the pipeline absorbs (overall amplitude, phase, time
shift -- Schur complement, same construction as v3), and report

  * rho_lens(M, y)  = rho_event * || P_perp (F - 1) ||_W
      the matched-filter SNR of the lensing difference waveform
      after nuisance projection.  By construction rho_lens / rho
      must reproduce the resid_grid of lens_observable_check
      (regression test T2).
  * sigma_lnM, sigma_y : Cramer-Rao errors from the profiled
      2x2 Fisher matrix (v3-style absolute scale).
  * direction SNRs sqrt(eig) of the profiled Fisher.

Weighting: same inspiral weight W = f^-7/3 on a log grid as the
previous check (NOT a real PSD -- comparability is deliberate).
Nuisance set is amplitude/phase/time ONLY: degeneracies with
intrinsic source parameters (chirp mass, spins, ...) are NOT
included, so every number here is an optimistic (ideal-template)
bound.  The gap to the measured f99 boundary is reported as-is.

F: Takahashi & Nakamura exact form (w <= 8, validated against
|F(w,0)|^2 = pi w / (1 - exp(-pi w))), geometric-optics two-image
form for w > 8.  Derivatives wrt w and y: central differences,
step-stability checked in T4.
"""
import json

import numpy as np
import mpmath as mp

mp.mp.dps = 25
import os
VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "numbers")
GMsun = 4.9254909476412675e-6  # s


# ---------- F (copied from lens_observable_check.py, validated there) ----
def F_exact(w, y):
    if w < 1e-9:
        return 1.0 + 0.0j
    xm = 0.5 * (y + np.sqrt(y * y + 4.0))
    phi_m = 0.5 * (xm - y) ** 2 - np.log(xm)
    F = (mp.e ** (mp.pi * w / 4)
         * mp.e ** (0.5j * w * (mp.log(w / 2) - 2 * phi_m))
         * mp.gamma(1 - 0.5j * w)
         * mp.hyper([0.5j * w], [1], 0.5j * w * y * y))
    return complex(F)


def mu_pm(y):
    s = np.sqrt(y * y + 4.0)
    base = (y * y + 2.0) / (2.0 * y * s)
    return 0.5 + base, 0.5 - base


def dtau(y):
    s = np.sqrt(y * y + 4.0)
    return 0.5 * y * s + np.log((s + y) / (s - y))


def F_geo(w, y):
    mp_, mm_ = mu_pm(y)
    return np.sqrt(mp_) - 1j * np.sqrt(abs(mm_)) * np.exp(1j * w * dtau(y))


def F_any(w, y):
    return F_exact(w, y) if w <= 8.0 else F_geo(w, y)


STEP_SCALE = 1.0


def dF_dw(w, y, rel=1e-3):
    dw = rel * STEP_SCALE * max(w, 0.02)
    return (F_any(w + dw, y) - F_any(w - dw, y)) / (2 * dw)


def dF_dy(w, y, h=1e-3):
    h = h * STEP_SCALE
    return (F_any(w, y + h) - F_any(w, y - h)) / (2 * h)


# ---------- per-point quantities -------------------------------------------
def point_quantities(M, y, zl, f_lo, f_hi, rho, nf=None):
    """Return dict with resid fraction, rho_lens, sigma_lnM, sigma_y,
    profiled-Fisher eigenvalues (direction SNRs squared).

    nf is adaptive: at large M the geometric-optics fringes oscillate
    many times across the band and a fixed grid aliases (nf=40/48
    disagree by ~0.2 in frac at M~1.8e5, y=0.1; nf>=96 converges).
    The high-M branch is the cheap numpy F_geo, so oversampling
    there costs little."""
    if nf is None:
        w_max = 8 * np.pi * GMsun * M * (1 + zl) * f_hi
        nf = 48 if w_max <= 8 else int(min(384, 48 + w_max / 4))
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), nf)
    w = 8 * np.pi * GMsun * M * (1 + zl) * f
    W = f ** (-7 / 3)
    Wn = W / np.sum(W)
    F = np.array([F_any(wi, y) for wi in w])
    fbar = np.sum(Wn * f)
    u = (f - fbar) / fbar

    # residual fraction: projection onto span{1, i u} (same as before)
    A = np.stack([np.ones_like(f), 1j * u], axis=1)
    Aw = A * np.sqrt(Wn)[:, None]
    coef, *_ = np.linalg.lstsq(Aw, F * np.sqrt(Wn), rcond=None)
    resid = F - A @ coef
    frac = float(np.sqrt(np.sum(Wn * np.abs(resid) ** 2)))

    # Fisher in F-space. Basis factors (h = factor * h0, <h0,h0> = rho^2):
    #   nuisance: lnA -> F ; phi -> iF ; tau -> i u F
    #   interest: lnM -> w dF/dw ; y -> dF/dy
    dFw = np.array([dF_dw(wi, y) for wi in w])
    dFy = np.array([dF_dy(wi, y) for wi in w])
    fac = [F, 1j * F, 1j * u * F, w * dFw, dFy]
    G = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            G[i, j] = rho ** 2 * float(np.real(np.sum(Wn * np.conj(fac[i]) * fac[j])))
    Gnn, Gin = G[:3, :3], G[:3, 3:]
    Gii = G[3:, 3:]
    try:
        Geff = Gii - Gin.T @ np.linalg.solve(Gnn, Gin)
    except np.linalg.LinAlgError:
        Geff = Gii - Gin.T @ np.pinv(Gnn) @ Gin
    ev = np.linalg.eigvalsh(Geff)
    cov = np.linalg.pinv(Geff)
    sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {"frac": frac, "rho_lens": rho * frac,
            "sigma_lnM": float(sig[0]), "sigma_y": float(sig[1]),
            "dir_snr": [float(np.sqrt(max(e, 0.0))) for e in ev],
            "cond_eff": float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")}


EVENTS = {
    "GW231123_135430": {"rho": 21.8, "f99_best": 0.47, "zl": 0.042, "band": (20, 90)},
    "GW250114_082203": {"rho": 78.6, "f99_best": 0.24, "zl": 0.05, "band": (25, 250)},
    "GW240925_005809": {"rho": 31.2, "f99_best": 0.36, "zl": 0.01, "band": (25, 300)},
    "GW241011_233834": {"rho": 35.8, "f99_best": 0.37, "zl": 0.01, "band": (25, 300)},
}
MS = np.logspace(1, 5.5, 19)
YS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
RHO_DET = 8.0

# ---------- validation ------------------------------------------------------
print("== T1: F(w,0) analytic ==")
for w in [0.5, 1.0, 2.0, 5.0]:
    num = abs(F_exact(w, 0.0))
    ana = float(np.sqrt(np.pi * w / (1 - np.exp(-np.pi * w))))
    print(f"  w={w}: numeric {num:.12f} analytic {ana:.12f} diff {abs(num-ana):.2e}")

print("== T4: derivative step stability (GW250114 band) ==")
for (M, y) in [(300.0, 1.0), (3000.0, 0.5), (30.0, 1.5)]:
    a = point_quantities(M, y, 0.05, 25, 250, 78.6)
    globals()["STEP_SCALE"] = 0.5
    b = point_quantities(M, y, 0.05, 25, 250, 78.6)
    globals()["STEP_SCALE"] = 1.0
    print(f"  M={M} y={y}: sigma_lnM {a['sigma_lnM']:.4f} vs half-step {b['sigma_lnM']:.4f}; "
          f"sigma_y {a['sigma_y']:.4f} vs {b['sigma_y']:.4f}; rho_lens {a['rho_lens']:.3f}")

print("== T3: no-lens limits ==")
q = point_quantities(10.0, 3.0, 0.05, 25, 250, 78.6)
print(f"  M=10, y=3: frac {q['frac']:.4f} rho_lens {q['rho_lens']:.3f} "
      f"sigma_y {q['sigma_y']:.2f} (expect small frac, large sigma)")

# ---------- main grid --------------------------------------------------------
saved = json.load(open(f"{VAL}/lens_observable_numbers.json"))
out = {"events": {}, "rho_det": RHO_DET}
t2_max = 0.0
t2_core = 0.0
for name, ev in EVENTS.items():
    grids = {"rho_lens": [], "sigma_lnM": [], "sigma_y": [], "frac": []}
    bound_ideal, bound_meas = {}, {}
    for y in YS:
        row = [point_quantities(M, y, ev["zl"], *ev["band"], ev["rho"]) for M in MS]
        for k in grids:
            grids[k].append([round(r[k], 4) for r in row])
        hit_i = [float(M) for M, r in zip(MS, row) if r["rho_lens"] >= RHO_DET]
        hit_m = [float(M) for M, r in zip(MS, row) if r["rho_lens"] >= ev["rho"] * ev["f99_best"]]
        bound_ideal[str(y)] = ([min(hit_i), max(hit_i)] if hit_i else None)
        bound_meas[str(y)] = ([min(hit_m), max(hit_m)] if hit_m else None)
        # T2/T6 regression vs saved resid grid and excluded ranges
        saved_row = saved["events"][name]["resid_grid"][str(y)]
        for M, r, s in zip(MS, row, saved_row):
            t2_max = max(t2_max, abs(r["frac"] - s))
            if M <= 1e4:
                t2_core = max(t2_core, abs(r["frac"] - s))
    out["events"][name] = {**{k: ev[k] for k in ("rho", "f99_best", "zl")},
                           "band_Hz": ev["band"], "masses_Msun": [float(m) for m in MS],
                           "y_values": YS, "grids": grids,
                           "bound_ideal_M": bound_ideal, "bound_meas_M": bound_meas,
                           "gap_factor": round(ev["rho"] * ev["f99_best"] / RHO_DET, 2),
                           "saved_excluded_M": saved["events"][name]["excluded_M_range_best"]}
    print(name, "ideal:", bound_ideal)
    print(name, "meas :", bound_meas, " saved:", saved["events"][name]["excluded_M_range_best"],
          " gap x", out["events"][name]["gap_factor"])

print(f"== T2: max |frac - saved resid| = {t2_max:.4f} overall; "
      f"{t2_core:.4f} for M <= 1e4 (tolerance 0.02; boundaries T6 are decisive) ==")
out["validation"] = {"T2_max_frac_diff": t2_max, "T2_max_frac_diff_M_le_1e4": t2_core}
with open(f"{VAL}/lens_detectability_numbers.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("saved lens_detectability_numbers.json")
