"""光心 β 同時走査 v4 — ChatGPT 第4回査読・第1段階（再現性の確保）への回答。

dr4_beta_scan3.py と同一のアンカー集合（旧14点 + 追加8点 = 22点）と同一の同時走査を、
補間方式だけを変えて再実行し、M2 最大値を比較する。凍結文書群・scan1–3 は無変更。

補間方式（アンカー間の M_G(M) の決め方）:
  方式A（現行, scan2/scan3 と同一）: M_G を log10(M) に対して線形補間
      （いわゆる半対数補間。ChatGPT 第4回査読はこれを「対数–対数」と表現したが、
        実態は M_G そのものを log M に対して補間する方式である）
  方式B（光度基準）: アンカーを相対光度 L ∝ 10^(−0.4·M_G) に変換し、
      log L を log M に対して線形補間（= 区間ごとの冪乗則）してから M_G に戻す
  方式C（参考）: L を log M に対して線形補間してから M_G に戻す

対象: #4, #7, #8, #9, #13, #6 (BH1)。許容幅は ±0.3 等を主とし、
#13 と #8 は ±0.5 等も併記する。前提は scan2/scan3 と同じ主系列同士モデル。
f 区間と M_G,0 は dr4_m1m2_table.json の凍結値を使う。
方式A の結果が dr4_beta_scan3.json の新アンカー結果と一致することを実行時に照合する。
"""
import json
import numpy as np

import dr4_beta_scan3 as s3   # アンカー集合を scan3 と同一にするため import する

ANCH_M = s3.NEW_M            # 22 点（追加8 + 旧14）
ANCH_G = s3.NEW_G
LOGM = np.log10(ANCH_M)
LOGL = -0.4 * ANCH_G         # log10(L/L0)（L ∝ 10^(−0.4 M_G)）
LUM = 10 ** LOGL

TARGETS = [4, 7, 8, 9, 13, 6]
TOLS_MAIN = [0.3]
TOLS_EXTRA = {8: [0.5], 13: [0.5]}   # #8 と #13 は ±0.5 も


def mg_a(m):
    """方式A: M_G を log10(M) に対して線形補間（現行）。"""
    return np.interp(np.log10(m), LOGM, ANCH_G)


def mg_b(m):
    """方式B: log L を log M に対して線形補間 → M_G に戻す。"""
    return -2.5 * np.interp(np.log10(m), LOGM, LOGL)


def mg_c(m):
    """方式C: L を log M に対して線形補間 → M_G に戻す。"""
    return -2.5 * np.log10(np.interp(np.log10(m), LOGM, LUM))


METHODS = {"A_semi_log_MG": mg_a, "B_logL_logM": mg_b, "C_L_logM": mg_c}


def scan(flo, fhi, mg0, tol, mg_of):
    """scan2/scan3 と同一の同時走査。mg_of だけを差し替える。"""
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


# ---- アンカー全点の出典表（作業2: 訂正資料） ----
ADDED_SRC = {a["M"]: a["source"] for a in
             json.load(open('dr4_beta_scan3.json'))["anchors_added"]}
ANCHOR_TABLE = []
for m, g in zip(ANCH_M, ANCH_G):
    if m in ADDED_SRC:
        src = "追加アンカー（scan3）: " + ADDED_SRC[m]
    else:
        src = "既存アンカー（scan2 以来無変更、Pecaut & Mamajek 系の概略値）"
    ANCHOR_TABLE.append({"M": float(m), "M_G": float(g), "source": src})


def main():
    d = json.load(open('dr4_m1m2_table.json'))
    rows = {r['num']: r for r in d['rows']}
    s3res = json.load(open('dr4_beta_scan3.json'))["results"]

    out = {
        "method": "scan3 と同一アンカー（22点）・同一走査で補間方式 A/B/C のみを変更。"
                  "方式A は M_G を log M に対して線形補間する半対数方式（現行）であり、"
                  "「対数–対数補間」ではない。",
        "anchors_all": ANCHOR_TABLE,
        "anchors_correction_note": "scan3 の追加アンカーは 8 点 "
            "[0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]。"
            "M=0.18 (M_G=12.04) は Pecaut & Mamajek 元表の値として scan3 報告 (§1 の表) "
            "に参照掲載されたが、アンカーとしては使用していない。"
            "「追加9点」という説明は誤りで、正しくは 8 点。",
        "crosscheck_methodA_vs_scan3": {},
        "results": {},
    }

    for num in TARGETS:
        r = rows[num]
        flo, fhi = r['f_ci95']
        mg0 = r['M_G_corr']
        tols = TOLS_MAIN + TOLS_EXTRA.get(num, [])
        rec = {"source_id": r['source_id'], "f_ci95": [flo, fhi], "M_G_corr": mg0,
               "by_method": {}}
        for mname, mfun in METHODS.items():
            rec["by_method"][mname] = {}
            for tol in tols:
                rec["by_method"][mname][str(tol)] = scan(flo, fhi, mg0, tol, mfun)
        out["results"][str(num)] = rec
        # 方式A と scan3 新アンカー結果の照合（±0.3 と、あれば ±0.5）
        cc = {}
        for tol in tols:
            ref = s3res[str(num)]["new_anchors"][str(tol)]
            got = rec["by_method"]["A_semi_log_MG"][str(tol)]
            if ref is None or got is None:
                cc[str(tol)] = (ref is None and got is None)
            else:
                cc[str(tol)] = abs(ref["M2_max"] - got["M2_max"]) < 1e-9
        out["crosscheck_methodA_vs_scan3"][str(num)] = cc

    json.dump(out, open('dr4_beta_scan4.json', 'w'), indent=2, ensure_ascii=False)

    def fmt(res):
        return "解なし" if res is None else f"{res['M2_max']:.2f}"
    for num in TARGETS:
        rec = out["results"][str(num)]
        print(f"#{num}  M_G,0={rec['M_G_corr']}  照合A=scan3: "
              f"{out['crosscheck_methodA_vs_scan3'][str(num)]}")
        for tol in TOLS_MAIN + TOLS_EXTRA.get(num, []):
            cells = " | ".join(
                f"{name.split('_')[0]}: {fmt(rec['by_method'][name][str(tol)])}"
                for name in METHODS)
            print(f"   ±{tol}: {cells}")
    print("saved dr4_beta_scan4.json")


if __name__ == "__main__":
    main()
