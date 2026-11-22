# K 独立検証用 入力データ・マニフェスト

作成: 2026-11-22（Muse）/ 宛先: ChatGPT（K′ 独立構築用）・Claude

ChatGPT の「K独立検証・第1段階指示書」A 項への回答。ボイドの順序は全ファイルで共通（カタログ選択: VIDE/NSA zobovoids、0.02<z<0.114、R_v ≥ 12 h⁻¹ Mpc、|b|>20°、441 件、カタログ順）。

## ファイル 1: kappa_stage2_kmatrix.npz

SHA-256: `de80b01967984594cded77d8d1bf38b959340e92cd45df7e4817af1fa24b8a3e`

| 配列 | 形状 | dtype | 単位・意味 |
|---|---|---|---|
| rows, cols, vals | (34054,) | int32, int32, float64 | K の疎行列（COO）。K_ij = vals（無次元）、対角含む |
| gamma | (34054,) | float64 | ペア中心間距離（deg） |
| theta | (441,) | float64 | 各ボイドの角半径 θ（deg） |
| t_vec | (441,) | float64 | テンプレート補償値 t_i（無次元、κ の補償開口値） |
| ra, dec | (441,) | float64 | 中心座標（deg、赤道座標） |
| zz | (441,) | float64 | 赤方偏移 |
| Rv | (441,) | float64 | 有効半径（Mpc、h=0.674 で換算） |

欠損値: なし。

## ファイル 2: kappa_stage2_per_void.npz

SHA-256: `da31776248c7864ac878cb155e790882e88af76882ec8470f8389b57a89ba15d`

| 配列 | 形状 | dtype | 単位・意味 |
|---|---|---|---|
| t_vec | (441,) | float64 | 上と同じ |
| K_diag | (441,) | float64 | K の対角成分（無次元） |
| X_i | (441,) | float64 | 混線量 Σ_{j≠i} K_ij t_j（κ 単位） |
| R_i | (441,) | float64 | 応答比 (Σ_j K_ij t_j)/t_i（無次元） |
| pred_dcomp | (441,) | float64 | 全441同時注入に対する K 予測応答（κ 単位） |
| actual_dcomp | (441,) | float64 | 全441同時注入の実測応答（κ 単位、実パイプライン） |
| n_neigh | (441,) | int64 | 相互作用する隣接ボイド数 |

欠損値: なし。

## 備考

- プロファイル形状 g(u)、測定演算子、ビーム等の定義は仕様書 `kappa_K_spec.md`（正本は GitHub の collaboration/ 版）v1.2＋追補2–4 にある。
- 対照 A v2 の実現データ `kappa_stage2_controlA2.npz`（200 実現 × 441）は C の推定用であり、K′ 構築そのものには不要。必要なら追加で提供する。
