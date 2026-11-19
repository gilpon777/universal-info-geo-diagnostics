#!/usr/bin/env python3
"""DR4 判定規則（統合規則）の参照実装とテストベクトル【v4: 型検査＋JSON 標準化版】。
作成: 2026-11-19（Muse、v3 を基に改訂）/ 状態: 追記8・第3回固定（2026-11-22）の参照実装。
入力: 各天体の f 区間・M1 範囲・M_G・減光（GSP-Phot, TGE）・色の旗・軌道解の種類・対応付け。
出力: 質量軸 / 測光軸 / 最終ラベルと理由。標準ライブラリ＋numpy のみ。
実行: python3 dr4_apply_rules_v4.py  → dr4_test_vectors_v4.json を書き出し、全ケースが期待どおりなら status: OK。

v4 の変更点（ChatGPT 第8回査読を受けた修正。判定ロジック自体は v3 と同一）:
- has_orbit の型検査を追加: 厳密な真偽値でない値（0、1、"false" 等）は「判定保留（入力不正）」に
  倒す。v3 は None かどうかだけを見ており、文字列 "false" は真として、整数 0 は偽として
  処理され得た。敵対ケース A12・A13 を追加。
- テストベクトル JSON を標準 JSON に変更: 非有限値（NaN / ±Infinity）は保存時に文字列
  "NaN" / "Infinity" / "-Infinity" へ置き換える（実行時の値は実数として扱う。v3 の出力は
  Python 拡張の NaN / Infinity リテラルを含み、厳密な JSON パーサーで読めなかった）。

v3 の変更点（ChatGPT 第7回査読の優先1・優先2を受けた修正。判定ロジック自体は v2 と同一）:
- 優先1: 3 M☉ アンカーを −0.1 → +0.30 に統一（第6回で確定した正本 = Pecaut & Mamajek EEM
  2022.04.16 の補間値 +0.29 の運用値。extinction_anchor_check.md §3）。v2 は報告書側だけが
  +0.30 になり、コードの ANCH_G が −0.1 のまま残っていた。これで flip_margin は一律 0.40 小さくなる。
  なお高質量側アンカー全体（2.2–5.5 M☉）が EEM より 0.38–0.58 等明るい系統差は別途記録済みの
  未解決事項（β走査側）であり、ここでは天井条件が使う 3 M☉ 点のみを正本に合わせる。
- 優先2: 「欠損」だけでなく「不正値」へのフェイルセーフを追加。NaN・無限大・負の減光・
  逆転した区間・負の f95・未知の解の種類／対応付けは、計算に入る前に検査し、
  「判定保留（入力不正）」に倒す（v2 では NaN の減光・色差が黙って BH非支持 を出し得た。
  検証: dr4_v2_adversarial.py）。
- テストに敵対ケース（不正値）と境界ケース（flip = 1.5 ちょうど等）を追加。

v1 からの変更点（ChatGPT の指摘を受けたフェイルセーフ化。判定ロジック自体は v1 と同一）:
- v1 は入力の欠落を「問題なし」側の初期値で埋めていた（match 欠落 → 対応A とみなす、
  has_orbit 欠落 → 軌道解ありとみなす、dcolour 欠落 → 色の旗「なし」、solution 欠落 → astrometric）。
  v2 は必須入力が欠けていれば、その天体を自動的に「判定保留（入力不足）」とし、不足項目を理由に出力する。
- v1 で例外になっていた経路（減光が GSP・TGE の双方とも無い場合の max([])、M_G 等のキー欠落、
  has_orbit=True で f95 が None、dcolour が None）を、例外ではなくフェイルセーフ側の判定に倒す。
  想定外の形式の入力についても、最後の受け皿で「判定不能」に倒し、例外で落ちないようにする。
- テストベクトルの入力（build）では、13 天体の対応付けが既知であることを明示するため match="A" を
  入力として与える（v1 の build は match を一切設定しておらず、全件が暗黙に "A" 扱いだった）。
規則の出典（追記番号）はコメントに示す。★は本案での新規または明確化（採否は Muse・ChatGPT・中谷さんの確認が必要）。
"""
import json, math, sys
import numpy as np

# --- 主系列の M_G(M) アンカー（dr4_beta_scan3.py と同一。0.6 未満は追記5の Pecaut & Mamajek 追加分）---
ANCH_M = [0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.0,1.3,1.6,2.2,3.0,4.0,5.5,8.0,10.0,15.0,20.0,30.0]
ANCH_G = [14.26,13.35,12.6519,11.6832,11.0332,10.5958,9.67,8.82,8.1,6.4,4.83,3.7,2.7,0.6,0.30,-1.0,-1.7,-2.5,-3.0,-4.0,-4.9,-5.7]  # ★v3: 3.0 M☉ を −0.1 → +0.30（P&M EEM 正本）
M3_REF = 3.0  # BH 候補/非支持の境界質量 [M☉]（追記1 §2.3）

def mg_ms(m):
    """主系列星の M_G。下側の範囲外は例外（端値固定の欠陥を再発させない）。上側の範囲外は端値（保守的：実際はより明るい）。"""
    if m < ANCH_M[0]:
        raise ValueError(f"M={m} は低質量アンカー {ANCH_M[0]} 未満")
    m = min(m, ANCH_M[-1])
    return float(np.interp(np.log10(m), np.log10(ANCH_M), ANCH_G))

def m2_from_f(f, m1):
    """f = M2^3/(M1+M2)^2 の逆変換（M2 について単調増加）。"""
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if mid**3/(m1+mid)**2 < f: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

# --- 閾値（追記2〜3 の条文。★は本案）---
MARGIN_CAND_MAG   = 1.0   # 追記2 §1: 候補側の測光余裕
MG0_MIN_CAND      = 2.0   # 追記2 §3: M_G,0 < 2 は候補ラベルを付けない
FLIP_MARGIN_MIN   = 1.5   # ★ 非支持側の光度天井の余裕（本案。アンカー概略±0.5 + 余裕1.0）
FLAG_WEAK, FLAG_ATT, FLAG_BAD = 0.15, 0.30, 0.60  # 色の旗: 追記3 §5（0.15/0.30）。0.60 は「不適」の実務境界（★明確化）

def colour_flag(dc):
    a = abs(dc)
    if a >= FLAG_BAD and dc > 0: return "不適"
    if a > FLAG_ATT: return "要注意"
    if a >= FLAG_WEAK: return "弱い注意"
    return "なし"

def _insufficient(missing):
    """★ v2 新規: 入力不足時のフェイルセーフ出力。不足項目を理由に挙げ、判定保留（入力不足）に倒す。"""
    return dict(mass_axis="判定不能", phot_axis="判定不能", label="判定保留（入力不足）",
                reasons=["入力不足のため判定を保留する: " + "、".join(missing) + " が無い"], margins={})

def _invalid(problems):
    """★ v3 新規: 入力不正時のフェイルセーフ出力。不正項目を理由に挙げ、判定保留（入力不正）に倒す。"""
    return dict(mass_axis="判定不能", phot_axis="判定不能", label="判定保留（入力不正）",
                reasons=["入力が不正のため判定を保留する: " + "、".join(problems)], margins={})

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def _validate(o):
    """★ v3 新規: 値そのものの検査（欠損検査を通過した入力が対象）。
    NaN・無限大・負の減光や f95・逆転区間・未知の列挙値を、計算に入る前に拾う。
    問題が無ければ空リストを返す。"""
    bad = []
    def _interval(key, name, positive=True):
        v = o.get(key)
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            bad.append(f"{name} が2要素の区間でない"); return
        lo, hi = v
        if not (_is_num(lo) and _is_num(hi)):
            bad.append(f"{name} に有限でない値がある"); return
        if positive and not (lo > 0 and hi > 0):
            bad.append(f"{name} に正でない値がある")
        if lo > hi:
            bad.append(f"{name} の下限が上限を超えている")
    _interval("f95", "f 95% 区間（f95）")
    _interval("M1_range", "M1 範囲（M1_range）")
    if o.get("f95_frozen") is not None:
        _interval("f95_frozen", "凍結 f 区間（f95_frozen）")
    for key, name in (("M_G", "M_G"), ("M_G_corr", "M_G_corr"), ("dcolour", "色の差（dcolour）")):
        if not _is_num(o.get(key)):
            bad.append(f"{name} が有限の数値でない")
    for key, name in (("A_G_gsp", "減光 A_G_gsp"), ("A_G_tge", "減光 A_G_tge")):
        v = o.get(key)
        if v is not None:
            if not _is_num(v):
                bad.append(f"{name} が有限の数値でない")
            elif v < 0:
                bad.append(f"{name} が負である")
    if o.get("solution") not in ("astrometric", "spectroscopic_SB1"):
        bad.append(f"軌道解の種類（solution）が未知の値である: {o.get('solution')!r}")
    return bad

def label(o):
    """o: dict。返り値: dict(mass_axis, phot_axis, label, reasons, margins)。
    ★ v2: 必須入力の欠落は「問題なし」側に倒さず、判定保留（入力不足）にする。"""
    # --- フェイルセーフ 1: 対応付け（v1 は欠落時に "A" とみなしていた）---
    if o.get("match") is None:
        return _insufficient(["対応付け（match）"])
    if o["match"] not in ("A", "B", "C", "D"):   # ★ v3: 未知の対応付けを黙って通さない
        return _invalid([f"対応付け（match）が未知の値である: {o['match']!r}"])
    if o["match"] == "D":
        return dict(mass_axis="対象外", phot_axis="対象外", label="対応不明", reasons=["対応 D（確認できない）"], margins={})
    # --- フェイルセーフ 2: 軌道解の有無（v1 は欠落時に「あり」とみなしていた）---
    if o.get("has_orbit") is None:
        return _insufficient(["軌道解の有無（has_orbit）"])
    if type(o["has_orbit"]) is not bool:   # ★ v4: 型検査（"false" は真、0 は偽として処理され得た）
        return _invalid(["軌道解の有無（has_orbit）が真偽値でない"])
    if not o["has_orbit"]:
        return dict(mass_axis="対象外", phot_axis="対象外", label="軌道解なし", reasons=["軌道解が無い"], margins={})
    # --- フェイルセーフ 3: 軌道解がある場合の必須入力（v1 は初期値で埋めるか例外になっていた）---
    missing = []
    if o.get("f95") is None:      missing.append("f 95% 区間（f95）")
    if o.get("M1_range") is None: missing.append("M1 範囲（M1_range）")
    if o.get("M_G") is None:      missing.append("M_G")
    if o.get("M_G_corr") is None: missing.append("M_G_corr")
    if o.get("solution") is None: missing.append("軌道解の種類（solution）")
    if o.get("dcolour") is None:  missing.append("色の差（dcolour）")
    if o.get("A_G_gsp") is None and o.get("A_G_tge") is None:
        missing.append("減光（A_G_gsp・A_G_tge の双方）")
    if missing:
        return _insufficient(missing)
    # --- フェイルセーフ 4（★ v3）: 値そのものの検査。不正値は「問題なし」側に倒さない ---
    bad = _validate(o)
    if bad:
        return _invalid(bad)
    # --- 本体の判定（v1 と同一ロジック）。想定外の形式は例外で落とさず「判定不能」に倒す ---
    try:
        return _label_core(o)
    except Exception as e:
        return dict(mass_axis="判定不能", phot_axis="判定不能", label="判定不能",
                    reasons=[f"入力の形式が想定外で計算できない（{type(e).__name__}: {e}）"], margins={})

def _label_core(o):
    """v1 の label() 本体（完全な入力がある場合の判定。ロジックは v1 と同一）。"""
    R = []
    flo, fhi = o["f95"]; m1lo, m1hi = o["M1_range"]
    sb1 = (o["solution"] == "spectroscopic_SB1")
    # 質量軸（追記3 §1, §3）。M2 下限は M1 下端、上限は M1 上端で評価
    m2lo = m2_from_f(flo, m1lo)
    m2hi = None if sb1 else m2_from_f(fhi, m1hi)       # SB1 は sin i 未知のため上限なし（追記3 §3）
    if m2lo > M3_REF: mass = "支持"
    elif (m2hi is not None) and m2hi < M3_REF: mass = "非支持"
    else: mass = "保留"
    # 減光の天井（追記3 §2）。フェイルセーフで少なくとも一方は存在することが保証されている
    ag = [a for a in (o.get("A_G_gsp"), o.get("A_G_tge")) if a is not None]
    agmax = max(ag)
    MG = o["M_G"]; MG0 = o["M_G_corr"]
    margin_c = (MG - agmax) - mg_ms(m2lo) if m2lo >= ANCH_M[0] else float("nan")
    flip = (MG - agmax) - mg_ms(M3_REF)                # ★ 光度天井: 系が M2=3 M☉ の主系列星より何等暗いか
    flag = colour_flag(o["dcolour"])
    margins = dict(A_G_max=round(agmax,4), margin_cand=round(margin_c,2), flip_margin=round(flip,2), M2_lo=round(m2lo,3), M2_hi=None if m2hi is None else round(m2hi,3), colour_flag=flag)
    # 光度条件
    if mass == "支持":
        phot = "排除済み" if (margin_c >= MARGIN_CAND_MAG and MG0 >= MG0_MIN_CAND) else "排除できない"
        if phot == "排除済み": lab = "BH候補"; R.append(f"M2下限 {m2lo:.2f} > 3、測光余裕 {margin_c:.2f} ≥ {MARGIN_CAND_MAG}、M_G,0 {MG0} ≥ {MG0_MIN_CAND}")
        else: lab = "判定不能（測光）"; R.append(f"M2下限 {m2lo:.2f} > 3 だが測光余裕 {margin_c:.2f} / M_G,0 {MG0} が条件未達")
        if flag in ("要注意","不適"): R.append(f"色の旗 {flag}（記録のみ。候補側には適用しない）")
    elif mass == "非支持":
        phot = "光度天井で確認" if flip >= FLIP_MARGIN_MIN else "光度天井が不足"
        if flip < FLIP_MARGIN_MIN:
            lab = "判定保留"; R.append(f"M2上限 {m2hi:.2f} < 3 だが光度天井の余裕 {flip:.2f} < {FLIP_MARGIN_MIN}（M2>3 の主系列伴星を排除できない）")
        elif flag in ("要注意","不適"):
            lab = "判定保留"; R.append(f"M2上限 {m2hi:.2f} < 3、光度余裕 {flip:.2f} は十分だが、色の旗 {flag}（主系列前提が弱い。追記6 §3）")
        else:
            lab = "BH非支持"; R.append(f"M2上限 {m2hi:.2f} < 3、光度天井の余裕 {flip:.2f} ≥ {FLIP_MARGIN_MIN}、色の旗 {flag}")
    else:
        phot = "対象外"; lab = "判定保留"
        R.append("M2区間が 3 をまたぐ" if m2hi is not None else "M2 の上限が原理的に定まらない（SB1：傾斜角未確定）")
    # 凍結区間との不整合（追記1 ⑤）。f95_frozen は任意入力（与えられた場合のみ照合する。v1 と同一）
    fz = o.get("f95_frozen")
    if fz and (flo > fz[1] or fhi < fz[0]):
        lab = "判定保留"; R.append("DR4 区間が凍結区間と重ならない：対応付けと解の種類を再確認し、説明が付くまでラベルを動かさない（追記1 ⑤）")
    return dict(mass_axis=mass, phot_axis=phot, label=lab, reasons=R, margins=margins)

# --- 13 天体の入力（dr4_m1m2_table_v3 と gaia_extinction_bracket_13 から転記）---
TGE = {1:1.1139,2:1.0468,3:1.0608,4:0.9811,5:1.1048,6:1.4387,7:1.1139,8:1.2589,9:1.4664,10:1.0004,11:0.9811,12:1.4664,13:0.929}
GSP = {1:None,2:0.207,3:0.5942,4:0.4042,5:0.5465,6:0.5575,7:0.5238,8:None,9:0.3036,10:None,11:0.615,12:2.6713,13:0.5131}
ROWS = {  # num: (M_G, M_G_corr, dcolour, M1_range, f95(frozen main), solution)
 1:(6.94,6.94,0.39,[0.58,0.95],None,None), 2:(6.17,5.96,0.05,[0.66,1.11],None,None), 3:(6.10,5.51,0.03,[0.67,1.18],None,None),
 4:(3.61,3.21,0.24,[1.06,1.87],[0.05233166499420931,0.09913539779506524],"astrometric"),
 5:(2.89,2.34,0.38,[1.23,2.2],None,None),
 6:(5.35,4.79,0.13,[0.74,1.31],[7.58,18.14],"astrometric"),
 7:(4.79,4.27,0.23,[0.81,1.48],[0.03591066662087977,0.06671779870462954],"astrometric"),
 8:(6.46,6.46,0.42,[0.63,1.03],[0.024709736466311003,0.0766031920234604],"astrometric"),
 9:(4.32,4.02,0.12,[0.9,1.57],[0.014168281215786137,0.024917291774521833],"astrometric"),
 10:(5.08,5.08,0.19,[0.77,1.25],[5.496613959429415e-05,0.00021338160352113884],"spectroscopic_SB1"),
 11:(5.28,4.67,0.21,[0.75,1.35],None,None),
 12:(8.41,5.74,-0.32,[0.48,0.95],[0.078,90.3],"astrometric"),   # M1 範囲は追記7（A_G=TGE で再計算）
 13:(0.69,0.18,1.01,[1.0,3.45],[0.04255985715166325,0.07198666889242408],"astrometric")}  # 和集合（追記3 §5）
EXPECT = {1:"軌道解なし",2:"軌道解なし",3:"軌道解なし",4:"BH非支持",5:"軌道解なし",6:"BH候補",7:"BH非支持",8:"判定保留",9:"BH非支持",10:"判定保留",11:"軌道解なし",12:"判定保留",13:"判定保留"}

def build(num):
    MG,MG0,dc,m1,f95,sol = ROWS[num]
    # ★ v2: 対応付けは入力として明示する（13 天体は source_id で同定済み = 対応A。v1 は未設定で暗黙の初期値に依存していた）
    return dict(num=num, match="A", has_orbit=f95 is not None, solution=sol or "none", f95=f95, M1_range=m1, M_G=MG, M_G_corr=MG0,
                A_G_gsp=GSP[num], A_G_tge=TGE[num], dcolour=dc)

# --- 合成テストベクトル（DR4 を想定した分岐の網羅用。予測ではない）---
SYN = [
 dict(id="S1", note="BH1 が精度向上して区間が狭まる", num=6, over=dict(f95=[9.0,12.0]), expect="BH候補"),
 dict(id="S2", note="#12 が高い f に絞られる（候補側。色の旗は記録のみ）", num=12, over=dict(f95=[6.0,15.0]), expect="BH候補"),
 dict(id="S3", note="#12 が低い f に絞られる（非支持。色の旗 要注意 のため保留）", num=12, over=dict(f95=[0.06,0.12]), expect="判定保留"),
 dict(id="S4", note="#9 が精度向上して f が小さいまま（非支持を維持）", num=9, over=dict(f95=[0.016,0.023]), expect="BH非支持"),
 dict(id="S5", note="M2 区間が 3 をまたぐ（保留）", num=4, over=dict(f95=[1.2,2.0]), expect="判定保留"),
 dict(id="S6", note="SB1 で M2 の最小質量が 3 を超える（下限のみ。傾斜角未確定でも支持側は成立）", num=10, over=dict(f95=[6.0,9.0], M_G=9.0, M_G_corr=9.0), expect="BH候補"),
 dict(id="S7", note="SB1 で f が小さい（上限が無いため非支持にしない）", num=10, over=dict(f95=[1e-4,2e-4]), expect="判定保留"),
 dict(id="S8", note="支持だが系の M_G,0 < 2（測光で判定不能）", num=6, over=dict(f95=[9.0,12.0], M_G=1.5, M_G_corr=1.2), expect="判定不能（測光）"),
 dict(id="S9", note="非支持だが光度天井の余裕が不足（#13 型）", num=4, over=dict(M_G=1.8, M_G_corr=1.5, dcolour=0.1), expect="判定保留"),
 dict(id="S10", note="対応 D", num=6, over=dict(match="D"), expect="対応不明"),
 dict(id="S11", note="DR4 区間が凍結区間の上に外れる（⑤：説明が付くまで保留）", num=6, over=dict(f95=[22.0,30.0], f95_frozen=[7.58,18.14]), expect="判定保留"),
 dict(id="S12", note="軌道解なし", num=1, over=dict(), expect="軌道解なし"),
]

# --- ★ v3 追加: 敵対テスト（不正値。全件「判定保留（入力不正）」が期待値）---
NAN, INF = float("nan"), float("inf")
ADV = [
 dict(id="A1", note="NaN の減光（GSP 側）。v2 では BH非支持 になった", num=9, over=dict(A_G_gsp=NAN)),
 dict(id="A2", note="NaN の色差。v2 では旗が「なし」に化けて #8 が BH非支持 になった", num=8, over=dict(dcolour=NAN)),
 dict(id="A3", note="負の減光（TGE のみ）。v2 では S9 型が BH非支持 になった", num=4, over=dict(M_G=1.8, M_G_corr=1.5, dcolour=0.1, A_G_gsp=None, A_G_tge=-2.0)),
 dict(id="A4", note="逆転した f95 区間", num=4, over=dict(f95=[0.09913539779506524, 0.05233166499420931])),
 dict(id="A5", note="負の f95。v2 では M2=0 のまま BH非支持 になった", num=9, over=dict(f95=[-0.5, -0.1])),
 dict(id="A6", note="未知の解の種類。v2 では astrometric 扱いで BH非支持 になった", num=10, over=dict(solution="orbital", f95=[0.5, 0.8])),
 dict(id="A7", note="未知の対応付け。v2 では黙って BH候補 になった", num=6, over=dict(match="X")),
 dict(id="A8", note="M_G = +inf。v2 では flip=inf のまま BH非支持 になった", num=9, over=dict(M_G=INF)),
 dict(id="A9", note="逆転した M1_range。v2 では M2_lo > M2_hi のまま出力された", num=9, over=dict(M1_range=[1.57, 0.9])),
 dict(id="A10", note="dcolour = -inf", num=9, over=dict(dcolour=-INF)),
 dict(id="A11", note="f95 に文字列が混入", num=9, over=dict(f95=["0.02", 0.03])),
 dict(id="A12", note="has_orbit = 0（真偽値でない）。v3 では偽として扱われ「軌道解なし」になった", num=9, over=dict(has_orbit=0)),
 dict(id="A13", note="has_orbit = \"false\"（真偽値でない）。v3 では真として扱われ通常判定された", num=9, over=dict(has_orbit="false")),
]

# --- ★ v3 追加: 境界テスト（正常系。境界のどちら側に倒れるかを固定する）---
def _b1_mg():
    # flip が 1.5 に達する最小の M_G を 1 ulp 単位で探す。
    # 10 進の「ちょうど 1.5」は 2 進浮動小数では表せないため、境界テストは
    # 「flip >= 1.5 になる最小の M_G では非支持、その 1 ulp 下では保留」という形で
    # 比較演算（>=）の意味を固定する。
    agmax, a3 = 0.9811, mg_ms(M3_REF)
    mg = (FLIP_MARGIN_MIN + a3) + agmax
    while (mg - agmax) - a3 < FLIP_MARGIN_MIN:
        mg = math.nextafter(mg, math.inf)
    return mg
def _b1_over():
    mg = _b1_mg()
    return dict(M_G=mg, M_G_corr=mg, dcolour=0.1)
def _b2_over():
    mg = math.nextafter(_b1_mg(), -math.inf)
    return dict(M_G=mg, M_G_corr=mg, dcolour=0.1)
BND = [
 dict(id="B1", note="flip が 1.5 に達する最小の M_G（>= 側: 非支持）", num=4, over_fn=_b1_over, expect="BH非支持"),
 dict(id="B2", note="その 1 ulp 下（< 側: 保留）", num=4, over_fn=_b2_over, expect="判定保留"),
 dict(id="B3", note="f95 の下限 = 上限（点区間は正常）", num=9, over=dict(f95=[0.02, 0.02]), expect="BH非支持"),
 dict(id="B4", note="減光 0.0 は正常", num=9, over=dict(A_G_gsp=None, A_G_tge=0.0), expect="BH非支持"),
 dict(id="B5", note="dcolour = 0.30 ちょうど（弱い注意。非支持は維持）", num=9, over=dict(dcolour=0.30), expect="BH非支持"),
 dict(id="B6", note="対応 B は正常系として処理する", num=6, over=dict(match="B"), expect="BH候補"),
]

def _sanitize(x):
    """★ v4: 保存用に非有限値を文字列へ置き換え、標準 JSON にする（実行時の値は変えない）。"""
    if isinstance(x, float):
        if math.isnan(x): return "NaN"
        if math.isinf(x): return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, dict): return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_sanitize(v) for v in x]
    return x

def main():
    out = dict(real=[], synthetic=[], adversarial=[], boundary=[], anchor_unit=None, status="OK")
    # ★ v3: アンカーの単体テスト（3.0 M☉ で正本 +0.30 が返ること）
    a3 = mg_ms(3.0)
    out["anchor_unit"] = dict(mg_ms_3Msun=a3, expected=0.30, pass_=abs(a3 - 0.30) < 1e-12)
    if not out["anchor_unit"]["pass_"]: out["status"] = "FAIL"
    for n in range(1,14):
        o = build(n); r = label(o); ok = (r["label"] == EXPECT[n])
        out["real"].append(dict(num=n, input=o, output=r, expected=EXPECT[n], pass_=ok))
        if not ok: out["status"] = "FAIL"
    for s in SYN:
        o = build(s["num"]); o.update(s["over"])
        if "f95" in s["over"]: o["has_orbit"] = True
        r = label(o); ok = (r["label"] == s["expect"])
        out["synthetic"].append(dict(id=s["id"], note=s["note"], input=o, output=r, expected=s["expect"], pass_=ok))
        if not ok: out["status"] = "FAIL"
    for s in ADV:
        o = build(s["num"]); o.update(s["over"])
        r = label(o); ok = (r["label"] == "判定保留（入力不正）")
        out["adversarial"].append(dict(id=s["id"], note=s["note"], input=o, output=r, expected="判定保留（入力不正）", pass_=ok))
        if not ok: out["status"] = "FAIL"
    for s in BND:
        o = build(s["num"]); o.update(s["over"] if "over" in s else s["over_fn"]())
        r = label(o); ok = (r["label"] == s["expect"])
        out["boundary"].append(dict(id=s["id"], note=s["note"], input=o, output=r, expected=s["expect"], pass_=ok))
        if not ok: out["status"] = "FAIL"
    json.dump(_sanitize(out), open("dr4_test_vectors_v4.json","w"), ensure_ascii=False, indent=1, default=float)
    print("status:", out["status"], "| anchor mg_ms(3.0) =", a3)
    for r in out["real"]:
        m = r["output"]["margins"]
        print(f'#{r["num"]:<2d} {r["output"]["label"]:<14s} mass={r["output"]["mass_axis"]:<6s} phot={r["output"]["phot_axis"]:<10s} {m}')
    for s in out["synthetic"]:
        print(s["id"], s["output"]["label"], "OK" if s["pass_"] else "FAIL", "-", s["note"])
    for s in out["adversarial"]:
        print(s["id"], s["output"]["label"], "OK" if s["pass_"] else "FAIL", "-", s["note"])
    for s in out["boundary"]:
        print(s["id"], s["output"]["label"], "OK" if s["pass_"] else "FAIL", "-", s["note"])
    return 0 if out["status"]=="OK" else 1

if __name__ == "__main__":
    sys.exit(main())
