#!/usr/bin/env python3
"""階層三重系（主星 M1 + 内側連星 [光る星 Ms + 暗い天体 Mc] が「伴星」）の全探索と、内側軌道への追加制約。
作成: 2026-11-17（Claude）。Claude v8 §2.3 の探索の完全なコード。入力: dr4_m1m2_table.json（Muse）。
モデル: 系の光 = L(M1) + L(Ms)（主系列）、β = L_s/(L1+L_s)、M2 = Ms + Mc、B = M2/(M1+M2)、
        光心軌道の質量関数 f_phot = (M1+M2)|B-β|^3 が f 区間（凍結の 95% 区間）に入り、系の M_G,0 が観測の M_G_corr ± TOL に一致する解を全て数える。
グリッド: M1 0.30–5.00（刻み 0.05）、Ms 0.10–5.00（刻み 0.05）、Mc 0.05–15.00（刻み 0.05）。許容 TOL=0.3 等。
アンカー: OLD = dr4_beta_scan3.py（主系列、高質量側は旧値）、EEM = 2.2–5.5 M☉ を P&M EEM（2022.04.16）の値に置換（Muse の extinction_anchor_check: 2.2:+0.98, 3.0:+0.29, 4.0:-0.42, 5.5:-1.19）。8 M☉ 以上は EEM 値を確認できていないため旧値のまま。
追加制約（スポット解）: Mardling–Aarseth 窓、内側軌道による光心の揺れの振幅 A_in = a_in·Mc/(Ms+Mc)·ϖ、Roche 溢流の下限、Ms の視線速度半振幅。
実行: python3 dr4_triple_search.py <dr4_m1m2_table.json があるディレクトリ> → triple_search_summary.json と triple_solutions_{old,eem}.csv を書き出す。
"""
import json, sys, csv
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
TAB = {r["num"]: r for r in json.load(open(f"{D}/dr4_m1m2_table.json"))["rows"]}
AM = [0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.0,1.3,1.6,2.2,3.0,4.0,5.5,8.0,10.0,15.0,20.0,30.0]
AG_OLD = [14.26,13.35,12.6519,11.6832,11.0332,10.5958,9.67,8.82,8.1,6.4,4.83,3.7,2.7,0.6,-0.1,-1.0,-1.7,-2.5,-3.0,-4.0,-4.9,-5.7]
AG_EEM = list(AG_OLD); AG_EEM[AM.index(2.2)] = 0.98; AG_EEM[AM.index(3.0)] = 0.29; AG_EEM[AM.index(4.0)] = -0.42; AG_EEM[AM.index(5.5)] = -1.19
M1G = np.round(np.arange(0.30, 5.0001, 0.05), 2); MSG = np.round(np.arange(0.10, 5.0001, 0.05), 2); MCG = np.round(np.arange(0.05, 15.0001, 0.05), 2)

def search(num, ag, tol=0.3):
    r = TAB[num]; flo, fhi = r["f_ci95"]; mg0 = r["M_G_corr"]
    mgf = lambda m: np.interp(np.log10(m), np.log10(AM), ag)
    sols = []
    for m1 in M1G:
        L1 = 10**(-0.4*mgf(m1))
        for ms in MSG:
            Ls = 10**(-0.4*mgf(ms))
            if abs(-2.5*np.log10(L1+Ls) - mg0) > tol: continue
            beta = Ls/(L1+Ls); M2 = ms + MCG; B = M2/(m1+M2); f = (m1+M2)*np.abs(B-beta)**3
            for i in np.where((f >= flo) & (f <= fhi))[0]:
                sols.append((float(m1), float(ms), float(MCG[i]), float(beta), float(f[i])))
    return sols

def wobble(M1, Ms, Mc, Pout_d, eout, G, MG, f95, Rs):
    """スポット解の内側軌道への制約。ϖ は G − M_G（減光補正なし）から（距離指数）。"""
    d = 10**((G-MG+5)/5); plx = 1000/d; Pout = Pout_d/365.25; Min = Ms+Mc
    a0 = ((np.sqrt(f95[0]*f95[1])*Pout**2)**(1/3))*plx               # 外側の光心軌道の半長軸 [mas]
    aout = ((M1+Ms+Mc)*Pout**2)**(1/3); qout = M1/Min
    Y0 = 2.8*((1+qout)*(1+eout)/np.sqrt(1-eout))**0.4; ain_max = aout*(1-eout)/Y0
    q = Ms/Mc; q13 = q**(1/3); rl = 0.49*q13**2/(0.6*q13**2+np.log(1+q13))       # Eggleton
    ain_roche = Rs*0.00465047/rl
    frac = Mc/Min; Pin = lambda a: np.sqrt(a**3/Min)*365.25
    Kms = lambda a: 2*np.pi*a*1.495978707e8/(Pin(a)*86400)*frac     # km/s
    out = dict(d_pc=d, plx_mas=plx, a0_outer_mas=a0, MA_ain_max=ain_max, MA_Pin_max_d=Pin(ain_max),
               A_in_at_MA_max_mas=ain_max*frac*plx, Roche_ain_min=ain_roche, Roche_Pin_min_d=Pin(ain_roche), windows={})
    for cap in (0.15, 0.3, 0.5):
        hi = min(cap/(frac*plx), ain_max); lo = ain_roche
        out["windows"][str(cap)] = None if hi < lo else dict(Pin_d=[Pin(lo), Pin(hi)], K_Ms_kms=[Kms(hi), Kms(lo)])
    return out

if __name__ == "__main__":
    summ = {}
    for name, ag in (("old", AG_OLD), ("eem", AG_EEM)):
        summ[name] = {}
        for n in (4, 7, 8, 9, 13):
            s = search(n, ag); big = [x for x in s if x[2] >= 3.0]
            e = dict(n_solutions=len(s), n_Mc_ge_3=len(big))
            if big:
                b = np.array(big)
                e.update(beta_range=[float(b[:,3].min()), float(b[:,3].max())], M1_range=[float(b[:,0].min()), float(b[:,0].max())],
                         Ms_over_M1=[float((b[:,1]/b[:,0]).min()), float((b[:,1]/b[:,0]).max())], Mc_max=float(b[:,2].max()))
            summ[name][str(n)] = e
            with open(f"triple_solutions_{name}_{n}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["M1", "Ms", "Mc", "beta", "f_phot"]); w.writerows(s)
    # Muse のスポット解がグリッド解に含まれるか（old アンカー）
    spots = {"4": (0.95, 1.30, 14.75), "9": (0.85, 1.10, 8.50)}
    summ["spot_in_grid"] = {}
    for n, (m1, ms, mc) in spots.items():
        S = search(int(n), AG_OLD)
        hit = [x for x in S if abs(x[0]-m1) < 1e-6 and abs(x[1]-ms) < 1e-6 and abs(x[2]-mc) < 1e-6]
        summ["spot_in_grid"][n] = bool(hit)
    summ["wobble"] = {"4": wobble(0.95, 1.30, 14.75, 1129.96, 0.408, 11.72, 3.61, TAB[4]["f_ci95"], 1.5),
                      "9": wobble(0.85, 1.10, 8.50, 1431.09, 0.940, 10.31, 4.32, TAB[9]["f_ci95"], 1.1)}
    json.dump(summ, open("triple_search_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summ[k] for k in ("old", "eem", "spot_in_grid")}, ensure_ascii=False, indent=1))
    for n, w in summ["wobble"].items():
        print(f"#{n}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items() if k != "windows"}); print("   windows:", w["windows"])
