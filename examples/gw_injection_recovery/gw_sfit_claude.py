#!/usr/bin/env python3
"""GW231123 Stage B: S(g) の2次フィット（Claude v8 の方法の完全な再掲）と、方法ごとの感度。
作成: 2026-11-17（Claude）。入力: gw231123_inject_recover.json, gw_prediction_check.json（Muse の保存 JSON）、
osc_proj は gw_osc_diagnostic.md の実設定値（帯域 20–448 Hz、SNR 重み）。注入・回収は再実行していない。
方法:
  CL  = Claude v8 の方法: S0 固定、(S^2 - S0^2) = b g + q g^2 を、行ごとの重み 1/max(S^2,1e-3) を A と y の両方に掛けた最小二乗（lstsq(A*w, y*w)）で解く。
        ※ 重みは lstsq の仕様上 2 乗で効く（実効重みは 1/S^4）。この重みが Muse の M1（無重み OLS）との差の主因。
  M1  = 無重み OLS（ChatGPT・Muse の M1）。
有意点（SIG）: S(g=4) が背景 p99 を超え、かつ q>0 の点。
回帰: log a vs log osc_proj。OLS 傾き（SE 付き）と TLS（直交回帰）傾きを併記。
"""
import json, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
J = json.load(open(f"{D}/gw231123_inject_recover.json")); P = json.load(open(f"{D}/gw_prediction_check.json"))
BG = {w: J["background"][w]["p99"] for w in J["background"]}
S0 = {w: J["stageA_baseline"]["S"][w] for w in J["stageA_baseline"]["S"]}
WIN = list(S0.keys())
OSC = {(100,0.3):0.02080,(100,0.5):0.02699,(100,1.0):0.03856,(100,2.0):0.05388,(200,0.3):0.05931,(200,0.5):0.08193,(200,1.0):0.12111,(200,2.0):0.12028,
       (400,0.3):0.16400,(400,0.5):0.23536,(400,1.0):0.27496,(400,2.0):0.16557,(1000,0.3):0.56326,(1000,0.5):0.59858,(1000,1.0):0.40414,(1000,2.0):0.17362,
       (2000,0.3):0.93357,(2000,0.5):0.75135,(500,0.5):0.31622}
PTS = {}
for src, R in (("grid", J["stageB"]["results"]), ("added", P["results"])):
    for r in R: PTS[(int(r["M_msun"]), r["y"])] = (src, r["rows"])

def fit(rows, w, method):
    g = np.array([r["g"] for r in rows]); S = np.array([r["S"][w] for r in rows])
    A = np.vstack([g, g**2]).T; y = S**2 - S0[w]**2
    if method == "CL":
        wt = 1/np.maximum(S**2, 1e-3); b, q = np.linalg.lstsq(A*wt[:, None], y*wt, rcond=None)[0]
    else:
        b, q = np.linalg.lstsq(A, y, rcond=None)[0]
    a = np.sqrt(q) if q > 0 else float("nan")
    cth = b/(2*a*S0[w]) if q > 0 else float("nan")
    return b, q, a, cth, S[-1]

def ols(x, y):
    X = np.vstack([x, np.ones(len(x))]).T; cf = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X@cf
    se = np.sqrt(r@r/(len(x)-2)*np.linalg.inv(X.T@X)[0, 0]); return cf[0], se
def tls(x, y):
    xm, ym = x.mean(), y.mean(); C = np.cov(np.vstack([x-xm, y-ym])); w, V = np.linalg.eigh(C); v = V[:, -1]; return v[1]/v[0]

out = {"methods": {}, "note": __doc__.strip().splitlines()[0]}
for m in ("CL", "M1"):
    out["methods"][m] = {}
    for w in WIN:
        tab = []
        for k, (src, rows) in sorted(PTS.items()):
            b, q, a, c, S4 = fit(rows, w, m)
            tab.append(dict(M=k[0], y=k[1], src=src, osc_proj=OSC[k], b=b, q=q, a=a, cos=c, S4=S4, sig=bool(S4 > BG[w] and q > 0)))
        sig = [t for t in tab if t["sig"]]
        x = np.log([t["osc_proj"] for t in sig]); yv = np.log([t["a"] for t in sig])
        s, se = ols(x, yv); st = tls(x, yv)
        cs = np.array([t["cos"] for t in tab if t["q"] > 0])
        out["methods"][m][w] = dict(table=tab, n_sig=len(sig), slope_ols=s, se_ols=se, slope_tls=st,
                                    cos_all=dict(n=int(len(cs)), mean=float(cs.mean()), sd=float(cs.std(ddof=1)), maxabs=float(np.abs(cs).max())),
                                    n_negative_q=int(sum(1 for t in tab if t["q"] <= 0)))
# 窓間の応答比 a_整合 / a_計画（両窓で有意な点）
for m in ("CL", "M1"):
    T = out["methods"][m]; p, c = T[WIN[0]]["table"], T[WIN[1]]["table"]
    r = np.array([cc["a"]/pp["a"] for pp, cc in zip(p, c) if pp["sig"] and cc["sig"]])
    out["methods"][m]["a_ratio_matched_over_plan"] = dict(n=int(len(r)), mean=float(r.mean()), sd=float(r.std(ddof=1)), se=float(r.std(ddof=1)/np.sqrt(len(r))), median=float(np.median(r)))
json.dump(out, open("gw_sfit_claude_coeffs.json", "w"), ensure_ascii=False, indent=1, default=float)
for m in ("CL", "M1"):
    print(f"\n===== method {m} =====")
    for w in WIN:
        T = out["methods"][m][w]
        print(f"[{w}] n_sig={T['n_sig']}  a-space slope: OLS {T['slope_ols']:.3f} ± {T['se_ols']:.3f}, TLS {T['slope_tls']:.3f};  cos(all q>0): n={T['cos_all']['n']} mean {T['cos_all']['mean']:+.3f} sd {T['cos_all']['sd']:.3f} max|cos| {T['cos_all']['maxabs']:.2f};  q<=0 points: {T['n_negative_q']}")
    r = out["methods"][m]["a_ratio_matched_over_plan"]; print(f"a_matched/a_plan: n={r['n']} mean {r['mean']:.2f} sd {r['sd']:.2f} (se {r['se']:.2f}) median {r['median']:.2f}")
print("\n(400,1.0) matched, M1:", [ (t['b'],t['q']) for t in out['methods']['M1'][WIN[1]]['table'] if (t['M'],t['y'])==(400,1.0)])
print("(400,1.0) plan, M1 cos:", [ t['cos'] for t in out['methods']['M1'][WIN[0]]['table'] if (t['M'],t['y'])==(400,1.0)])
print("(2000,0.5) matched, M1 cos:", [ t['cos'] for t in out['methods']['M1'][WIN[1]]['table'] if (t['M'],t['y'])==(2000,0.5)])
