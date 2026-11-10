# 汎用・情報幾何学的診断フレームワーク
# Universal Information-Geometric Diagnostics

ブラックボックスなモデルに対して、「そのデータではどのパラメータの組み合わせが
決まるのか」「観測点（測定点）をどこに置けば情報が増えるのか」を、
Fisher 情報行列の固有値診断で答えるための軽量な Python フレームワーク。

*An English summary follows the Japanese text.*

## これは何か

- モデル関数（θ → 予測値）とノイズ共分散さえ与えれば、内部構造を知らなくても
  診断できる単一クラス `UniversalInformationGeometricDiagnosticsV3`。
- 出発点は「宇宙ボイド（銀河の少ない領域）の内部で、どこを観測すべきかを
  狭める」という作成者の観測構想。ただし診断そのものは分野を選ばず、
  本リポジトリには生化学・粉末素材・天の川銀河・ボイド＋重力波の使用例が入る。
- 位置づけは正直に: FIM 診断や最適実験計画の部品は確立したものであり、
  先行例（PyOED、SloppyCell、GWFish 等）との差分は
  [docs/related_work.md](docs/related_work.md) に明記した。

## クイックスタート

```python
import numpy as np
from universal_info_geo_diagnostics_v3 import (
    UniversalInformationGeometricDiagnosticsV3)

diag = UniversalInformationGeometricDiagnosticsV3(model_func)
res = diag.evaluate(theta, covariance=cov)

res["Raw"]["EffectiveRank"]        # 生の実効情報ランク r_eff
res["Scaled"]["EffectiveRank"]     # スケーリング不変な r_eff
res["Absolute"]["Raw"]["DirectionSNR"]          # 方向ごとの絶対 SNR (v3)
res["Absolute"]["Raw"]["InfoMultiplierForTargetSNR"]  # SNR 1 までの情報倍率

picks = diag.greedy_select(theta, cov, max_points=7)  # 観測点の貪欲選定
picks[0]["Driver"]   # "gradient" / "noise" / "structure"（勝因の内訳, v3）
picks[0]["GainMargin"]  # 次点との差（ほぼ同点かどうかが分かる）
```

検証: `python3 test_v2.py` / `python3 test_v3.py`（いずれも全テスト付き）

## バージョン

- **v1**（作成者オリジナル）: 前方差分ヤコビアン、F = JᵀΣ⁻¹J、条件数指標、
  エントロピー実効ランク。
- **v2**: 相関化（Scaled）スペクトルの併記、相対ステップ＋中心差分、
  条件数ゲート付き pinv フォールバック、相対リッジ、数値ランク併記。
- **v3**: 絶対スケール（方向 SNR / 情報倍率）、邪魔者パラメータの
  プロファイル／周辺化、貪欲選定の勝因内訳とマージン。
  詳細は [docs/v3_notes.md](docs/v3_notes.md)。

## 使用例（すべて実行可能なスクリプトと数値つき）

| 例 | スクリプト | 結果の要点 |
|---|---|---|
| 合成 sloppy モデル | `demo_v2.py`, `demo_v3.py` | 固有値が数桁に広がる典型構造を検出 |
| 生化学（反応連鎖 A→B→C→D） | `example_biochem.py` | 測定時刻の選定が時定数ごとに散らばる。マージンはほぼ同点 |
| 粉末素材（酸化物4成分の組成） | `example_powder.py` | チャンネル選定が各成分の「指紋」に対応。勝因は勾配 |
| 天の川銀河（実データ, Eilers+2019） | `results/reports/milkyway_validation.md` | Scaled r_eff 1.74/5。円盤–ハロー縮退が支配的 |
| ボイド＋重力波（実データ） | `results/reports/` 一式 | 下記 |

## 実データ検証の要点（ボイド＋重力波）

- VAST（SDSS DR7）ボイドカタログの 386 ボイドをスタックし、Hamaus プロファイルを
  フィット（χ²=8.3/10 dof）。Scaled r_eff 3.42/4、動径ビン上位7点で情報の 98.3%。
  第1位の中心ビンはノイズ構造（σ の窪み）で勝っており、その σ はボイド間の
  個体差の抑制として物理的に裏付けた（`noise_formula_check.md`）。
- ボイドを横切る重力波イベント 4 件（GW231123, GW250114, GW240925, GW241011）に
  波形残差解析を行い、いずれも有意な残差なし＝**感度の上限**として報告。
  点レンズ (M, y) 平面への翻訳では、観測が触れたのは「大質量 × 小衝突径数」の
  隅のみで、100 太陽質量級（ボイド内の「針」）は未検証のままである
  （`lens_observable_check.md`）。
- 結論は一貫して「不在の証明ではなく、測れた範囲の地図」として書かれている。

データ出典（再配布はしていない）: VAST ボイドカタログ（Douglass et al. 2023,
Zenodo）、GWOSC（LIGO–Virgo–KAGRA 公開データ）、Eilers et al. 2019 回転曲線。

## 限界

- 診断は評価点の近傍の局所線形化であり、フィット点を動かせば順位は変わりうる。
- ノイズ共分散は既知として扱う。共分散自体の正体（本検証では個体差が支配的
  だった）は、別途検証が要る。
- 重力波側の結果は上限であり、レンズのテンプレート型解析は未実施
  （次の一手として残してある）。

## ライセンス

- コード: MIT License（`LICENSE`）
- 報告書・図（`results/`）: CC-BY-4.0 で公開する想定

## 著者

Kenichi Nakatani（中谷健一）

---

## English summary

A lightweight, black-box Fisher-information diagnostics framework: given a
model function and a noise covariance, it reports which parameter
combinations a dataset can identify (raw and scaling-invariant effective
information rank), how large the effects are in absolute terms
(per-direction SNR and the information multiplier needed to reach a target
SNR), which nuisance directions are absorbed by fitting, and which
observation points to measure — with, for each greedy pick, a margin and
an attribution of whether it won on signal gradient or on noise structure.

It grew out of the author's project to choose observation points inside
cosmic voids, and is validated on public data: a stacked void profile from
the VAST SDSS DR7 catalogue (386 voids; scaled r_eff 3.42/4), a Milky Way
rotation-curve control (Eilers et al. 2019; scaled r_eff 1.74/5), and
waveform-residual upper limits for four gravitational-wave events whose
sightlines cross voids (GW231123, GW250114, GW240925, GW241011), including
a point-lens (M, y) sensitivity map. Worked examples for biochemistry
kinetics and powder-mixture composition are included. The components
(FIM spectra, optimal experimental design) are established; the
positioning against prior tools (PyOED, SloppyCell, GWFish, Pyomo.DoE)
and the specific differences of this implementation are documented in
`docs/related_work.md`. All null results are reported as upper limits,
not as evidence of absence.
