#!/usr/bin/env python3
"""三重系の安定窓の e_out 感度（ChatGPT 第7回 B① の検証用）。
Mardling & Aarseth (2001) の安定条件（共面・順行を基準）で、内側周期の上限が
外側離心率にどれだけ敏感かを、スポット解（#4・#9）について走査する。
既存ファイルは変更しない。"""
import math

def pin_max(P_out, e_out, m_outer, m_in, i_mut_deg=0.0):
    """内側周期の上限 [日]。m_in は内側連星の合計質量、m_outer は外側の星の質量。"""
    q_out = m_outer / m_in
    crit = (2.8 / (1.0 - e_out)) * (((1.0 + q_out) * (1.0 + e_out)) / math.sqrt(1.0 - e_out)) ** 0.4 \
           * (1.0 - 0.3 * i_mut_deg / 180.0)
    m_tot = m_outer + m_in
    return P_out * crit ** -1.5 * math.sqrt(m_tot / m_in)

CASES = {
    "#9": dict(P_out=1431.1, e_obs=0.940, m_outer=0.85, m_in=1.1 + 8.5),    # スポット解 M1=0.85, Ms=1.1, Mc=8.5
    "#4": dict(P_out=1130.0, e_obs=0.408, m_outer=0.95, m_in=1.3 + 14.75),  # スポット解 M1=0.95, Ms=1.3, Mc=14.75
}
for name, c in CASES.items():
    print(f"== {name}（P_out={c['P_out']} d, 観測 e_out={c['e_obs']}）==")
    for e in [0.0, 0.3, 0.5, 0.7, 0.9, c["e_obs"]]:
        print(f"  e_out={e:<5} -> P_in,max = {pin_max(c['P_out'], e, c['m_outer'], c['m_in']):9.2f} d")
    p0 = pin_max(c["P_out"], c["e_obs"], c["m_outer"], c["m_in"], i_mut_deg=0.0)
    p90 = pin_max(c["P_out"], c["e_obs"], c["m_outer"], c["m_in"], i_mut_deg=90.0)
    print(f"  相互傾斜角の影響（観測 e_out）: i_mut=0° {p0:.2f} d / i_mut=90° {p90:.2f} d（比 {p90/p0:.2f}）")
