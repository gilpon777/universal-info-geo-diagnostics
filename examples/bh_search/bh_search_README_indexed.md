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

---

# ファイル索引（2026-11-28 追加）

このフォルダの全ファイルの案内。ファイル自体は移動していない（固定記録と
外部からの参照リンクを壊さないため）。状態の記号: ◎ = 現行の正本、
○ = 現行の補助、記 = 記録（経緯の保存）、旧 = 旧版（後継あり）。

## まずここ — DR4 当日の正本

| ファイル | 内容 | 状態 |
|---|---|---|
| `dr4_freeze_index_v4.md` | 第3回固定の索引（15ファイル、ハッシュ付き） | ◎ |
| `dr4_consolidated_rules.md` | 統合規則（凍結版） | ◎ |
| `dr4_apply_rules_v4.py` | 参照実装 v4（DR4 の入力はこれに適用する） | ◎ |
| `dr4_test_vectors_v4.json` | テストベクトル v4 | ◎ |
| `dr4_dryrun_v4.py` / `dr4_dryrun_v4_report.md` / `dr4_dryrun_v4_results.json` | ドライラン v4（130判定）とその報告・結果 | ◎ |
| `dr4_fixation_record3.md` / `dr4_fixation_record3_addendum.md` | 第3回固定の記録と追補 | 記 |

## 固定の経緯（第1回〜第3回）

| ファイル | 内容 | 状態 |
|---|---|---|
| `dr4_freeze.md` / `dr4_freeze.sha256` / `dr4_freeze.sha256.txt` | 固定の宣言とハッシュ | 記 |
| `dr4_freeze_index.md` / `dr4_freeze_index_v3.md` | 索引の旧版（v4 が現行） | 旧 |
| `dr4_fixation_record.md` / `dr4_fixation_record2.md` | 第1回・第2回固定の記録 | 記 |
| `dr4_freeze_supplement.md`〜`dr4_freeze_supplement8.md`（各 `.sha256` 共） | 追記1〜8（固定後の追加記録） | 記 |
| `dr4_freeze_supplement8_draft.md` / `dr4_freeze_supplement8_revised.md` | 追記8の下書きと改訂版 | 記 |
| `go_sign_record.md` | 段階ごとの実施許可の記録 | 記 |
| `muse_notice_2026-11-19_fixation.md` | 固定に関する通知文 | 記 |

## 判定コードと検証（DR4 判定まわり）

| ファイル | 内容 | 状態 |
|---|---|---|
| `dr4_apply_rules.py` / `_v2.py` / `_v3.py` | 参照実装の旧版（v4 が現行） | 旧 |
| `dr4_m1m2_table.py` / `.json` / `.md`、`_v2.md` / `_v3.md` / `_v4.md` / `_v4_claude.md` | M1・M2 表の生成と各版（v4 が現行） | ○ |
| `dr4_m1_reversal_check.py`、`_before_v14.txt` / `_after_v14.txt` | M1 反転境界の検査と前後の出力 | ○ |
| `dr4_dryrun.md` / `dr4_dryrun_v3.py` | ドライランの旧版 | 旧 |
| `dr4_dryrun_v4_extra.py` / `dr4_dryrun_v4_extra_results.json` | 余裕の監視＋色境界警告（記録のみ） | ○ |
| `dr4_colour_boundary_warnings.json` | 色境界警告の記録（warnings/errors/missing） | ○ |
| `dr4_extra_execution_log_2026-11-27.txt` | 追加検証の実行ログ | 記 |
| `dr4_rule_check.py` / `.json` / `_v2.py` / `_v2.json` | 規則の自己検査（旧） | 旧 |
| `dr4_beta_scan.py` / `2.py` / `3.py` / `3.json` / `3.md` / `4.py` / `4.json` / `4.md`、`dr4_beta_scan.md` | β 走査の一連（探索的な検証） | 記 |
| `dr4_v2_adversarial.py` | v2 時点の敵対的テスト | 記 |
| `dr4_triple_search.py`、`triple_search_summary.json`、`triple_solutions_*.csv`、`triple_stability_check.md`、`triple_stability_escan.py` | 階層三重系の探索と安定性 | ○ |
| `failsafe_check.md` / `extinction_anchor_check.md` | フェイルセーフと減光アンカーの検査記録 | 記 |
| `dr4_consolidated_rules_draft.md` | 統合規則の下書き（凍結版が正本） | 旧 |

## 色調補正の検証（2026-11-25〜）

| ファイル | 内容 | 状態 |
|---|---|---|
| `colour_correction_candidates_2026-11-25.md` | 補正候補 A・B・C・D の初回メモ（B は後に誤りと判明） | 記 |
| `colour_correction_comparison_2026-11-27.md` | A と B′ の比較（末尾に訂正あり） | ○ |
| `colour_correction_audit.py` / `.json` | 監査 v1（判定入力の更新漏れあり、v2 が後継） | 旧 |
| `colour_correction_audit2.py` / `.json` | 監査 v2（完全な B′、折れ点走査、現行） | ◎ |
| `colour_correction_audit3.py` / `.json` | 監査 v3（減光3方式の比較、境界を解いた区間分割の安定性） | ◎ |

## Muse 返信 — v1〜v10

| ファイル | 内容 | 状態 |
|---|---|---|
| `muse_reply_2026-11-15.md` / `_v2.md`、`muse_reply_2026-11-16.md` / `_v2.md` / `_v3.md` | 初期の返信（固定の準備期） | 記 |
| `muse_opinion_2026-11-15.md` | Muse の考察（11/15） | 記 |
| `muse_reply_2026-11-17_v4.md` / `_v4_addendum.md` / `_v5.md` | 第8ラウンド前後の裁定（表 v4 の確定など） | 記 |
| `muse_reply_2026-11-19_v6.md` / `_v7.md` | ChatGPT 第7〜9回への裁定 | 記 |
| `muse_reply_2026-11-20_v8.md` / `_v9.md` / `_v10.md` | GW 重みの二重計上の解消など | 記 |
| `muse_verification_2026-11-19_claude_code.md` | Claude のコードの検証記録 | 記 |

## Muse 返信 — v11〜v20

| ファイル | 内容 | 状態 |
|---|---|---|
| `muse_reply_2026-11-20_v11.md` | ChatGPT 第12回への裁定 | 記 |
| `muse_reply_2026-11-21_v12.md` / `_v13.md` / `_v14.md` | 第13・14回への裁定（GW 帯域の訂正を含む） | 記 |
| `muse_reply_2026-11-22_v15.md` | Claude v11 への返信（量ごとの統合案） | 記 |
| `muse_reply_2026-11-22_v16.md` / `_v17.md` | 第15・16回への裁定、K′ 入力データの配置 | 記 |
| `muse_reply_2026-11-23_v18.md` | Claude v12 への返信（重み散布の再現、脆さの主因の更新） | 記 |
| `muse_reply_2026-11-25_v19.md` | 本人の判断3件の記録（統合パッケージ不採用など） | 記 |
| `muse_reply_2026-11-26_v20.md` | 第18回への裁定（候補 B の誤りの発見と B′ への訂正） | 記 |

## Muse 返信 — v21〜

| ファイル | 内容 | 状態 |
|---|---|---|
| `muse_reply_2026-11-27_v21.md` | 第19回への実施（監査の初版） | 記 |
| `muse_reply_2026-11-27_v22.md` | 第20回への実施（幅の分解、#8 の臨界減光） | 記 |
| `muse_reply_2026-11-28_v23.md` | 第21回への裁定（「測光軸は不変」の訂正、完全 B′ の再計算） | 記 |
| `muse_reply_2026-11-28_v24.md` | 第22回への実施（#12 の減光3方式比較、境界の直接検査） | 記 |

## Claude の査読

| ファイル | 内容 | 状態 |
|---|---|---|
| `claude_review_v8_original_2026-11-17.md` | Claude v8（原文） | 記 |
| `claude_review_v9_2026-11-20.md` / `v10_2026-11-20.md` / `v11_2026-11-21.md` / `v12_2026-11-22.md` | Claude v9〜v12 | 記 |
| `claude_blind_rules_check.py` / `claude_label_stability.py` / `claude_weight_check.py` | Claude v12 付属の検証コード | 記 |

## κ 行列の仕様

| ファイル | 内容 | 状態 |
|---|---|---|
| `kappa_K_spec_pointer.md` | 正本（collaboration/kappa_K_spec.md）への案内 | ◎ |
| `kappa_K_spec.md` | 仕様書の旧コピー（正本は collaboration 側） | 旧 |

## データ

| ファイル | 内容 | 状態 |
|---|---|---|
| `data/gaia_patch.csv`（293件）/ `gaia_baade.csv` / `erass_patch.csv` / `blackcat.csv` | 走査の入力カタログ | ○ |
| `gaia_extinction_13.json` / `gaia_extinction_bracket_13.json` / `gaia_extinction_bracket_13_errata.md` / `gaia_gspphot_percentiles_13.json` | 13件の減光推定と括り、訂正記録 | ○ |
| `gaia_gspphot_13.csv`（Drive・ローカル保持分） | GSP-Phot の取得値 | ○ |

## 探索コードと結果

`code/`（走査コードとテスト）、`results/numbers/`（数値）、
`results/reports/`（報告書、`nss_shortlist.md` が13件の原簿）、
`results/figures/`（図）。詳細は本書の前半（コード一覧・結果の表）を参照。
