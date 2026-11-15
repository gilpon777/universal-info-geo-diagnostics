"""光心 β 同時走査 v3 — ChatGPT 第3回査読（問題A: 補間の端値固定 / 問題B: 許容幅）への回答。

dr4_beta_scan2.py と同一の走査（M1・M2・系全体の光度 M_G,0 と f_phot 区間を同時に課す）を、
次の2点だけ変更して再実行する。凍結文書・scan1/scan2 のファイルは変更しない。

1. 質量–光度アンカーの低質量側を拡張する（問題A の修正）。
   scan2 のアンカーは M=0.6 から始まるのに走査グリッド M1 は 0.30 からで、
   np.interp が範囲外を端値 (M_G=8.1) に固定していた。0.6 未満に実在アンカーを追加する。
   追加分の出典: Pecaut & Mamajek (2013) の主系列テーブル（Mamajek の EEM dwarf table、
   "supplemented online" 版）を再録した Kurtenkov (2022, Bulgarian Astron. J. 37) Table 1
   でウェブ確認した値（2026-11-16 に原典 PDF で照合）:
     M=0.50 (M1V) -> M_G=8.82, M=0.40 (M2.5V) -> 9.67, M=0.18 (M4.5V) -> 12.04,
     M=0.12 (M5.5V) -> 13.35, M=0.10 (M6V) -> 14.26   … 直接の表値
     M=0.15, 0.20, 0.25, 0.30 は上記表値の間の対数–対数補間（コード自身の補間方式と同一）
     で求めた: 0.15->12.65, 0.20->11.68, 0.25->11.03, 0.30->10.60。
   既存アンカー (0.6 以上) は scan2 から一切変更しない。
   これらは全て「概略アンカー（±0.5 等級級）」であり、精密な質量–光度関係ではない。
   残存する注意: M2 グリッドの下端 0.05–0.10 M_sun は依然として端値 (14.26) 固定だが、
   この質量域の伴星の光は系全体に対して無視できる水準であり、β への影響は小さい。

2. 光度許容幅を ±0.1 / ±0.3 / ±0.5 / ±1.0 等の4通りで走査する（問題B の感度）。
   比較のため、旧アンカー（scan2 と同一）の結果も同じ4許容幅で併記する。
   ChatGPT の感度表は旧アンカー（端値固定あり）での計算なので、突き合わせは
   旧アンカー列に対して行う。

対象: #4, #7, #8, #9, #13, #6 (BH1)。前提は scan2 と同じ主系列同士モデル。
f 区間と M_G,0 は dr4_m1m2_table.json の凍結値を使う。
"""
import json
import numpy as np

# ---- 旧アンカー (scan2 と同一) ----
OLD_M = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.2, 3.0, 4.0, 5.5, 8.0, 10.0, 15.0, 20.0, 30.0])
OLD_G = np.array([8.1, 6.4, 4.83, 3.7, 2.7, 0.6, -0.1, -1.0, -1.7, -2.5, -3.0, -4.0, -4.9, -5.7])

# ---- Pecaut & Mamajek (2013) EEM table の低質量エントリ (Kurtenkov 2022 Table 1 で照合) ----
PM = {0.57: 8.16, 0.54: 8.44, 0.50: 8.82, 0.47: 8.98, 0.44: 9.29, 0.40: 9.67,
      0.37: 10.05, 0.27: 10.87, 0.23: 11.21, 0.18: 12.04, 0.16: 12.45,
      0.12: 13.35, 0.10: 14.26}


def pm_interp(m):
    xs = np.array(sorted(PM))
    ys = np.array([PM[x] for x in xs])
    return float(np.interp(np.log10(m), np.log10(xs), ys))


ADD_M = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ADD_G = [PM[0.10], PM[0.12], pm_interp(0.15), pm_interp(0.20),
         pm_interp(0.25), pm_interp(0.30), PM[0.40], PM[0.50]]
NEW_M = np.array(ADD_M + list(OLD_M))
NEW_G = np.array(ADD_G + list(OLD_G))

TARGETS = [4, 7, 8, 9, 13, 6]
TOLS = [0.1, 0.3, 0.5, 1.0]


def scan(flo, fhi, mg0, tol, anch_m, anch_g):
    """scan2 と同一の同時走査。解があれば dict、無ければ None を返す。"""
    def mg_of(m):
        return np.interp(np.log10(m), np.log10(anch_m), anch_g)
    M1g = np.arange(0.30, 5.01, 0.02)
    M2g = np.concatenate([np.arange(0.05, 3.0, 0.01), np.arange(3.0, 15.01, 0.05)])
    L1 = 10 ** (-0.4 * mg_of(M1g))
    L2 = 10 ** (-0.4 * mg_of(M2g))
    sols = []
    for i, m1 in enumerate(M1g):
        comb = -2.5 * np.log10(L1[i] + L2)
        okL = np.abs(comb - mg0) <= tol
        if not okL.any():
            continue
        B = M2g / (m1 + M2g)
        beta = L2 / (L1[i] + L2)
        fph = (m1 + M2g) * np.abs(B - beta) ** 3
        ok = okL & (fph >= flo) & (fph <= fhi)
        if ok.any():
            sols.append((float(m1), float(M2g[ok].min()), float(M2g[ok].max())))
    if not sols:
        return None
    return {"M1_min": sols[0][0], "M1_max": sols[-1][0],
            "M2_min": min(s[1] for s in sols), "M2_max": max(s[2] for s in sols)}


def main():
    d = json.load(open('dr4_m1m2_table.json'))
    rows = {r['num']: r for r in d['rows']}
    out = {
        "method": "dr4_beta_scan2.py と同一の同時走査。変更点は (1) 低質量アンカー追加、"
                  "(2) 許容幅4通り、のみ。旧アンカー結果も併記。",
        "anchors_old_M": list(OLD_M), "anchors_old_G": list(OLD_G),
        "anchors_added": [
            {"M": m, "M_G": round(g, 2),
             "source": ("Pecaut & Mamajek (2013) EEM table 直接値 (Kurtenkov 2022 Table 1 で照合)"
                        if m in PM else
                        "Pecaut & Mamajek (2013) EEM table 隣接エントリ間の対数-対数補間")}
            for m, g in zip(ADD_M, ADD_G)],
        "anchors_note": "追加分を含む全アンカーは概略 (±0.5 等級級)。既存アンカー (0.6 以上) は無変更。",
        "tolerances_mag": TOLS,
        "results": {},
    }
    for num in TARGETS:
        r = rows[num]
        flo, fhi = r['f_ci95']
        mg0 = r['M_G_corr']
        rec = {"source_id": r['source_id'], "f_ci95": [flo, fhi], "M_G_corr": mg0,
               "old_anchors": {}, "new_anchors": {}}
        for tol in TOLS:
            rec["old_anchors"][str(tol)] = scan(flo, fhi, mg0, tol, OLD_M, OLD_G)
            rec["new_anchors"][str(tol)] = scan(flo, fhi, mg0, tol, NEW_M, NEW_G)
        out["results"][str(num)] = rec

    json.dump(out, open('dr4_beta_scan3.json', 'w'), indent=2, ensure_ascii=False)

    def fmt(res):
        return "解なし" if res is None else f"{res['M2_max']:.2f} (M2 {res['M2_min']:.2f}–{res['M2_max']:.2f})"
    for num in TARGETS:
        rec = out["results"][str(num)]
        print(f"#{num}  M_G,0={rec['M_G_corr']}  f=[{rec['f_ci95'][0]:.4g}, {rec['f_ci95'][1]:.4g}]")
        for label, key in (("旧アンカー", "old_anchors"), ("新アンカー", "new_anchors")):
            cells = " | ".join(f"±{t}: {fmt(rec[key][str(t)])}" for t in TOLS)
            print(f"   {label}: {cells}")
    print("saved dr4_beta_scan3.json")


if __name__ == "__main__":
    main()
