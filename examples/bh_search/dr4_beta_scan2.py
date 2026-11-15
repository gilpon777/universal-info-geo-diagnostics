import json, numpy as np
ANCH_M = np.array([0.6,0.8,1.0,1.3,1.6,2.2,3.0,4.0,5.5,8.0,10.0,15.0,20.0,30.0])
ANCH_G = np.array([8.1,6.4,4.83,3.7,2.7,0.6,-0.1,-1.0,-1.7,-2.5,-3.0,-4.0,-4.9,-5.7])
def mg_of(m): return np.interp(np.log10(m), np.log10(ANCH_M), ANCH_G)
d = json.load(open('dr4_m1m2_table.json'))
M1g = np.arange(0.30, 5.01, 0.02); M2g = np.concatenate([np.arange(0.05,3.0,0.01), np.arange(3.0,15.01,0.05)])
L1 = 10**(-0.4*mg_of(M1g)); 
for r in d['rows']:
    if 'f_ci95' not in r or r['num'] == 10: continue
    flo, fhi = r['f_ci95']; mg0 = r['M_G_corr']
    sols = []
    L2 = 10**(-0.4*mg_of(M2g))
    for i, m1 in enumerate(M1g):
        comb = -2.5*np.log10(L1[i] + L2)
        okL = np.abs(comb - mg0) <= 0.30
        if not okL.any(): continue
        B = M2g/(m1+M2g); beta = L2/(L1[i]+L2)
        fph = (m1+M2g)*np.abs(B-beta)**3
        ok = okL & (fph >= flo) & (fph <= fhi)
        if ok.any(): sols.append((m1, M2g[ok].min(), M2g[ok].max()))
    if sols:
        print(f"#{r['num']} M_G,0={mg0}: joint solutions M1 {sols[0][0]:.2f}..{sols[-1][0]:.2f}, M2 max = {max(s[2] for s in sols):.2f}, M2>3 exists: {max(s[2] for s in sols) > 3}")
    else:
        print(f"#{r['num']} M_G,0={mg0}: no joint MS-MS solution")
