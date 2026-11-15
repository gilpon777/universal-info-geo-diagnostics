import json, numpy as np
ANCH_M = np.array([0.6,0.8,1.0,1.3,1.6,2.2,3.0,4.0,5.5,8.0,10.0,15.0,20.0,30.0])
ANCH_G = np.array([8.1,6.4,4.83,3.7,2.7,0.6,-0.1,-1.0,-1.7,-2.5,-3.0,-4.0,-4.9,-5.7])
def mg_of(m): return float(np.interp(np.log10(m), np.log10(ANCH_M), ANCH_G))
def beta_ms(m1, m2):
    l1 = 10**(-0.4*mg_of(m1)); l2 = 10**(-0.4*mg_of(m2))
    return l2/(l1+l2)
d = json.load(open('dr4_m1m2_table.json'))
m2grid = np.concatenate([np.arange(0.05, 3.0, 0.01), np.arange(3.0, 15.01, 0.05)])
for r in d['rows']:
    if 'f_ci95' not in r or r['num'] == 10: continue
    flo, fhi = r['f_ci95']; bands_all = []
    for m1 in [r['M1_range'][0], r['M1_cen'], r['M1_range'][1]]:
        B = m2grid/(m1+m2grid)
        beta = np.array([beta_ms(m1, m) for m in m2grid])
        fph = (m1+m2grid)*np.abs(B-beta)**3
        ok = (fph >= flo) & (fph <= fhi)
        bands = []
        if ok.any():
            idx = np.where(ok)[0]; start = idx[0]; prev = idx[0]
            for i in idx[1:]:
                if i > prev+1: bands.append((m2grid[start], m2grid[prev])); start = i
                prev = i
            bands.append((m2grid[start], m2grid[prev]))
        bands_all.append((round(m1,2), [(round(a,2), round(b,2)) for a,b in bands]))
    print(f"#{r['num']} f=[{flo:.4g},{fhi:.4g}] label_now={r['label']}")
    for m1, bands in bands_all: print(f"   M1={m1}: MS-companion M2 solution bands = {bands}")
