#!/usr/bin/env python3
"""Claude v12 補助（固定対象外）: 統合規則の本文だけからの独立再実装（blind）と、参照実装との差分試験。
目的: 規則の本文（dr4_consolidated_rules.md）が、コードを見なくても同じ判定を再現できる自己完結した仕様かを確かめる。
ロジックは参照実装 dr4_apply_rules_v4.py から写していない（アンカー表のみ、本文がコードに委ねているため転記）。
使い方: 同じフォルダに dr4_apply_rules_v4.py と dr4_test_vectors_v4.json を置いて  python3 claude_blind_rules_check.py
結果（2026-11-22）: テストベクトル 43/44 一致（不一致 1 件: 数値文字列の扱い）／有効入力 20 万件の差分 0 件／異常入力 228 件中 4 群で不一致。
"""
import importlib.util, json, math, random, copy, collections
import numpy as np

ANCH_M=[0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.0,1.3,1.6,2.2,3.0,4.0,5.5,8.0,10.0,15.0,20.0,30.0]
ANCH_G=[14.26,13.35,12.6519,11.6832,11.0332,10.5958,9.67,8.82,8.1,6.4,4.83,3.7,2.7,0.6,0.30,-1.0,-1.7,-2.5,-3.0,-4.0,-4.9,-5.7]
def mg_ms(m):
    if m<ANCH_M[0]: raise ValueError("below low-mass anchor")
    return float(np.interp(math.log10(min(m,ANCH_M[-1])),np.log10(ANCH_M),ANCH_G))
def m2(f,m1):
    lo,hi=0.0,1e4
    for _ in range(300):
        mid=(lo+hi)/2
        if mid**3/(m1+mid)**2<f: lo=mid
        else: hi=mid
    return (lo+hi)/2
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def num(x):
    if isinstance(x,str):           # "NaN"/"Infinity" 文字列は JSON 標準化の表記として数値化。数値文字列も（本文に禁止がないため）数値として読む＝後述の不一致の原因
        try: return float(x)
        except ValueError: return x
    return x
def label(o):
    if o.get("match","A") not in ("A","B","C","D"): return "判定保留（入力不正）"
    if o.get("match","A")=="D": return "対応不明"
    ho=o.get("has_orbit")
    if not isinstance(ho,bool): return "判定保留（入力不正）"
    if not ho: return "軌道解なし"
    sol=o.get("solution")
    if sol not in ("astrometric","spectroscopic_SB1"): return "判定保留（入力不正）"
    if any(o.get(k) is None for k in ["f95","M1_range","M_G","M_G_corr","dcolour"]): return "判定保留（入力不足）"
    if o.get("A_G_gsp") is None and o.get("A_G_tge") is None: return "判定保留（入力不足）"
    f=[num(x) for x in o["f95"]]; m1=[num(x) for x in o["M1_range"]]
    sc=[num(o["M_G"]),num(o["M_G_corr"]),num(o["dcolour"])]
    ag=[num(o[k]) for k in ("A_G_gsp","A_G_tge") if o.get(k) is not None]
    if not all(finite(v) for v in f+m1+sc+ag): return "判定保留（入力不正）"
    if any(a<0 for a in ag) or f[0]>f[1] or f[0]<0 or m1[0]>m1[1] or m1[0]<=0: return "判定保留（入力不正）"
    fz=o.get("f95_frozen")
    if fz and (f[0]>fz[1] or f[1]<fz[0]): return "判定保留"
    lo=m2(f[0],m1[0]); hi=None if sol=="spectroscopic_SB1" else m2(f[1],m1[1])
    mass="支持" if lo>3 else ("非支持" if (hi is not None and hi<3) else "保留")
    agmax=max(ag); MG,MG0,dc=sc
    flag="不適" if dc>=0.60 else ("要注意" if abs(dc)>0.30 else ("弱い注意" if abs(dc)>=0.15 else "なし"))
    if mass=="支持": return "BH候補" if ((MG-agmax)-mg_ms(lo)>=1.0 and MG0>=2.0) else "判定不能（測光）"
    if mass=="非支持": return "BH非支持" if ((MG-agmax)-mg_ms(3.0)>=1.5 and flag in ("なし","弱い注意")) else "判定保留"
    return "判定保留"

def load_ref():
    spec=importlib.util.spec_from_file_location("r4","dr4_apply_rules_v4.py"); r4=importlib.util.module_from_spec(spec); spec.loader.exec_module(r4); return r4

def check_vectors():
    T=json.load(open("dr4_test_vectors_v4.json")); tot=0; bad=[]
    for g in ("real","synthetic","adversarial","boundary"):
        for c in T[g]:
            tot+=1
            try: got=label(c["input"])
            except Exception as e: got="EXC:"+type(e).__name__
            if got!=c["expected"]: bad.append((g,c.get("id",c.get("num")),c["expected"],got))
    print(f"テストベクトル: {tot-len(bad)}/{tot} 一致"); [print("  不一致",b) for b in bad]

def fuzz_valid(N=200000):
    r4=load_ref(); rng=random.Random(20261122); mism=0
    for _ in range(N):
        sb1=rng.random()<0.15; flo=10**rng.uniform(-4.5,1.8); fhi=flo*10**rng.uniform(0.05,3.0); m1lo=rng.uniform(0.3,2.0)
        gsp=rng.choice([None,rng.uniform(0,3.0)]); tge=rng.uniform(0.3,1.6)
        o=dict(num=1,match=rng.choice("ABC"),has_orbit=True,solution="spectroscopic_SB1" if sb1 else "astrometric",f95=[flo,fhi],
               M1_range=[m1lo,m1lo*rng.uniform(1.0,3.0)],M_G=0,M_G_corr=0,A_G_gsp=gsp,A_G_tge=tge,dcolour=rng.uniform(-0.8,1.3))
        o["M_G"]=rng.uniform(-1.0,10.0); o["M_G_corr"]=o["M_G"]-rng.uniform(0,1.5)
        if rng.random()<0.2:
            a=10**rng.uniform(-4,1.5); o["f95_frozen"]=[a,a*10**rng.uniform(0.1,2)]
        if r4.label(o)["label"]!=label(o): mism+=1
    print(f"有効入力の差分試験: {N} 件中 不一致 {mism}")

def fuzz_invalid():
    r4=load_ref(); base=lambda n: dict(r4.build(n))
    F={"f95 欠落":lambda o:o.pop("f95",None),"f95=None":lambda o:o.__setitem__("f95",None),"M_G 欠落":lambda o:o.pop("M_G",None),
       "減光が両方 None":lambda o:(o.__setitem__("A_G_gsp",None),o.__setitem__("A_G_tge",None)),"NaN の TGE":lambda o:o.__setitem__("A_G_tge",float("nan")),
       "inf の M_G":lambda o:o.__setitem__("M_G",float("inf")),"負の GSP":lambda o:o.__setitem__("A_G_gsp",-0.2),
       "f95 逆転":lambda o:o.__setitem__("f95",list(reversed(o["f95"]))),"f95 負":lambda o:o.__setitem__("f95",[-0.01,o["f95"][1]]),
       "f95 下限=0":lambda o:o.__setitem__("f95",[0.0,o["f95"][1]]),"数値文字列":lambda o:o.__setitem__("f95",[str(o["f95"][0]),o["f95"][1]]),
       "bool の M_G":lambda o:o.__setitem__("M_G",True),"未知の解":lambda o:o.__setitem__("solution","acceleration"),
       "未知の対応":lambda o:o.__setitem__("match","Z"),"has_orbit=1":lambda o:o.__setitem__("has_orbit",1),
       "M1 下限<=0":lambda o:o.__setitem__("M1_range",[0.0,o["M1_range"][1]]),"M1 逆転":lambda o:o.__setitem__("M1_range",list(reversed(o["M1_range"]))),
       "f95 長さ1":lambda o:o.__setitem__("f95",[o["f95"][0]]),"f95 長さ3":lambda o:o.__setitem__("f95",o["f95"]+[1.0])}
    C={"軌道あり":lambda o:o,"軌道なし":lambda o:(o.__setitem__("has_orbit",False),o)[1],"対応D":lambda o:(o.__setitem__("match","D"),o)[1]}
    tot=0; groups=collections.OrderedDict()
    for cn,cf in C.items():
        for fn,ff in F.items():
            for n in (9,4,12,10):
                o=cf(copy.deepcopy(base(n))); ff(o); tot+=1
                try: a=r4.label(o)["label"]
                except Exception as e: a="EXC:"+type(e).__name__
                try: b=label(o)
                except Exception as e: b="EXC:"+type(e).__name__
                if a!=b: groups.setdefault((cn,fn),(a,b))
    print(f"異常入力の差分試験: {tot} 件、不一致の群 {len(groups)}")
    for (cn,fn),(a,b) in groups.items(): print(f"  [{cn}] {fn}: 参照={a} / blind={b}")

if __name__=="__main__":
    check_vectors(); fuzz_valid(); fuzz_invalid()
