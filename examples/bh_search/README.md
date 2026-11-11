# 天の川銀河のブラックホール探索 — 観測検証の実例集

情報幾何診断フレームワーク (universal-info-geo-diagnostics v3) と
公開アーカイブデータを、天の川銀河内のブラックホール (BH) 探索という
一つの問いに適用した実例集。「どのチャンネルで、どこまで見えるか」を
数値で示し、**null (不検出) は感度の上限として返す**ことを全編の
作法とした。観測計画の事前検証と、既存データの再走査の雛形として
使えることを意図している。

## コード一覧

| ファイル | 問い | 入力 |
|---|---|---|
| `code/bh_search.py` | 休眠 BH 候補の追観測 (視線速度) は何夜・どの位相に配るべきか | v3 FIM、Gaia BH1 型の軌道モデル |
| `code/gamma_scan.py` | ガンマ線で静かな BH はどこまで見えるか | Fermi-LAT 級の感度 |
| `code/wavelength_sweep.py` | 波長を変えると検出限界はどう動くか | A0620-00 (1.06 kpc) の実測フラックス |
| `code/wavelength_bh.py` | BH 連星の SED 成分はどのバンドで切り分けられるか | v3 FIM、22 バンド×5 成分 |
| `code/xray_scan.py` | X 線カタログ走査は既知 BH を回収できるか | BlackCAT 73 件 × eRASS1 実カタログ |
| `code/residual_scan.py` | 星の「位置のエラー」の尾に既知 BH はいるか | Gaia DR3 実カタログ (RUWE) |
| `code/image_points.py` 他 | 集約画像 1 枚に点抽出を当てると何が測れるか | DECaPS2 実画像 × Gaia DR3 |

全テスト (`code/test_*.py`) 通過済み。数値は `results/numbers/`、
報告書は `results/reports/`、図は `results/figures/`。

## 結果の表

### 1. 波長掃引 — 静かな BH (A0620-00 型) の検出限界距離

| バンド | チャンネル | 限界距離 |
|---|---|---|
| ガンマ線 (Fermi-LAT) | 降着の光 | 173 pc |
| X 線 (eROSITA) | 降着の光 | 703 pc |
| 紫外 (GALEX 級) | 伴星+円盤 | 581 pc |
| 可視 (Gaia 位置測定) | 伴星の力学 | 8,000 pc |
| 近赤外 | 伴星の力学 | 15,000 pc |
| 中間赤外 (WISE) | 伴星 (暗い) | 1,836 pc |
| サブミリ (ALMA) | ジェットの裾 | 1,676 pc |
| 電波 (VLA 級) | ジェット | 2,370 pc |

窓は 3 つ: X 線 (降着)、可視・近赤外 (伴星の力学)、電波 (ジェット)。

### 2. ガンマ線の限界 (銀河面 / 高銀緯)

| 状態 | 光度 (erg/s) | 限界距離 |
|---|---|---|
| アウトバースト中のマイクロクエーサー | 3×10³⁶ | 28.9 / 91.4 kpc |
| 静穏時の降着連星 | 3×10³² | 289 / 914 pc |
| 孤立 BH・ISM 降着 (楽観) | 10³⁰ | 17 / 53 pc |
| 孤立 BH・ISM 降着 (典型) | 10²⁸ | 1.7 / 5.3 pc |
| 休眠 (Gaia BH 型) | 0 | 0 — 原理的に見えない |

### 3. 追観測の配分 (FIM 貪欲 vs 一様配置、視線速度 σ_K)

| 夜数 N | 貪欲 | 一様 | ランダム | 利得 |
|---|---|---|---|---|
| 4 | 0.18 km/s | 1.12 | 26.4 | 6.3× |
| 6 | 0.16 | 0.59 | 9.8 | 3.7× |
| 8 | 0.12 | 0.28 | 5.0 | 2.2× |

貪欲は近点通過の位相に観測を集める。N=6 で質量関数が約 1% まで締まる。

### 4. 波長選択 BH 版 (v3、SED 成分分解)

| | 静穏時 (A0620 型) | ハード状態 (GX 339-4 型) |
|---|---|---|
| Scaled r_eff | 3.29 / 6 | 3.06 / 6 |
| 貪欲の順位 | 550nm → 2.2µm → 150MHz → NUV → X 0.5keV | 440nm → 150MHz → 900nm → 3.6µm → X 0.5keV |
| 伴星 σ (ln 振幅) | 0.029 | 0.162 |
| 円盤 σ | 0.089 | 0.046 |
| 輝点 σ | 0.232 | 0.356 |
| ジェット σ | 0.061 | 0.053 |
| コロナ σ | 0.146 | 0.145 |

最強の縮退は両状態とも円盤×輝点 (−0.75 / −0.79): 光学の余分な光は
測光だけでは分けられず、変動という別の観測が要る。方向 SNR が 1 を
超えるのは 6 方向中 3 つ — 同種のバンド追加では直らない方向がある。

### 5. X 線実カタログ走査 (eRASS1 パッチ、既知 BH での検証)

- パッチ内源 5,117 → 明るい点源 58 に既知 BH 7 件中 2 件を回収
  (MAXI J1631-479: 7.6×10⁻¹²、4U 1630-472: 4.3×10⁻¹² erg/cm²/s、
  分離 < 0.5″)
- GX 339-4 はカタログにある (5.48×10⁻¹⁰) が「拡がった源」フラグで
  点源リストから漏れる — カタログ走査の落とし穴として記録
- 未照合の明るい源は SIMBAD で全て既知の中性子星 LMXB と同定。
  例外 1 件: 1RXS J170855.6-440653 はコンパクト天体の種別が
  未分類で BlackCAT にもない
- 静穏時 BH は 10⁻¹³–10⁻¹⁴ の桁でカットの下。休眠 BH は X 線では
  原理的にゼロ — X 線走査の役割は活動中の掘り起こしと確認

### 6. 残差走査の試行 (Gaia DR3、BH1 のいる空、RUWE>5)

| 項目 | 値 |
|---|---|
| 候補数 (1 kpc 以内、G<17) | 293 |
| Gaia BH1 の RUWE 順位 | 179 位 (RUWE 7.638) |
| 軌道解 (NSS) 付き | 13 件のみ、BH1 はその中で 6 位 |
| 揺れの代理指標 (AEN/視差) での BH1 順位 | 235 位 (使い物にならない) |

RUWE は候補の「山」を作る指標で、BH を直接指す指標ではない。
質量は軌道解と追観測から来る (表 3 の仕事)。

### 7. 集約画像 1 枚の点抽出 (DECaPS2、バーデの窓、約 20 arcmin²)

| 項目 | 値 |
|---|---|
| 抽出点 | 3,284 (うち 98% が Gaia の実星と一致) |
| 位置の床 (明るい星、Gaia との散らばり) | 約 66 mas (全点では約 97 mas) |
| 回収率 | G<18: 97%、G18–19: 72%、G19–20: 12% |

BH1 の揺れ (約 2.7 mas) は床の約 1/25 — 集約画像の点同士の差分では
拾えない。回収率の崖は画像ではなく素朴な抽出器の限界 (混雑場では
背景推定が星明かりに汚染される)。公式パイプラインの同時 PSF
フィットとの差がここに出る。

### 8. 文献との照合 (齟齬の確認)

| 比較対象 | 文献の値 | 本実例集 |
|---|---|---|
| DECaPS2 公式の位置精度 (Gaia 照合) | 中心部 約100 mas / 外側 約18 mas | 実測 66–97 mas (中心部寄りの視野) — 同じ帯 |
| VVV の高精度処理 (HST 級手法) | 単一露出 約12 mas、固有運動 <1 mas/yr | 「正しい処理では床は下がる」方向と一致 |
| DECaPS2 の深さ | 単一露出 r 23.5 等、33.2 億天体 | 回収率の崖は抽出器側、と整合 |

## 動かし方

- 必要: Python 3、numpy、scipy、matplotlib
- `bh_search.py` と `wavelength_bh.py` は v3 フレームワーク
  (`universal_info_geo_diagnostics_v3.py`、本リポジトリ本体) を使う。
  環境変数 `INFO_GEO_DIAG_DIR` で場所を指定できる
- データ (`data/`): BlackCAT (VizieR J/A+A/587/A61)、eRASS1
  (J/A+A/682/A34)、Gaia DR3 (I/355) の抽出。`image_points.py` の
  入力画像は DECaPS2 ビューワの FITS カットアウト
  (RA 270.70, Dec −29.80, r バンド、取得 URL は報告書に記載)

## 限界 (正直なところ)

- 限界距離と SED テンプレートは桁の目安・代理モデル。順位と
  チャンネルの構造はその揺れでは動かないが、絶対値は較正用ではない
- FIM 診断は基準点の近傍の局所評価。状態が変われば順位は動く
  (表 4 で 2 状態を併記した理由)
- 点抽出器は意図的に素朴な実装で、混雑場の本番処理
  (同時 PSF フィット) の代替にはならない。測ったのは
  「公開データ＋素朴な道具での床」であって方法の天井ではない

## English summary

Worked examples applying an information-geometric diagnostics
framework (Fisher information, greedy observation-point selection)
plus archival-data trials to one question: how would you search for
black holes in the Milky Way, and what can each channel actually
see? Headline results: follow-up radial-velocity epochs chosen by
greedy Fisher selection beat uniform placement by 2–6× in σ_K;
a wavelength-selection version of the same framework decomposes a
BH binary SED into companion/disk/hot-spot/jet/corona and shows the
disk–hot-spot degeneracy (−0.75) is the photometric bottleneck;
real-catalog trials recover known BHs in eRASS1 X-ray data (with a
documented extended-flag pitfall), place Gaia BH1 at rank 179/293
by RUWE alone but #6 among the 13 stars with orbital solutions,
and measure a ~66 mas astrometric floor for point extraction from
a single consolidated DECaPS2 image — about 25× larger than
Gaia BH1's wobble, consistent with the survey's own published
astrometric performance. Nulls are reported as sensitivity upper
limits throughout.

Author: Kenichi Nakatani. License: see the repository root LICENSE.
