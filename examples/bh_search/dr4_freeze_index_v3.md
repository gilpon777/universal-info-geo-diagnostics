# DR4 予測凍結 索引 v3（第3回固定用、2026-11-22）

第3回時刻固定（GitHub Release `dr4-freeze-2026-11-22` ＋ Wayback）の対象文書と
SHA-256 の一覧。第1回固定（2026-11-15）は凍結本体・追記1・追記2・索引 v1、
第2回固定（2026-11-16）は本体・追記1〜7・索引 v2 を対象とした。
本索引は追記8・統合規則・評価表 v4・参照実装 v3・テストベクトル v3 を加えた
全13ファイルを対象とする。本体・追記1〜7 のハッシュは索引 v2 の記録と一致する
（未改変であることを 2026-11-19 に再照合した）。

固定の判断: 2026-11-19、中谷さん。本固定は通過点であり、目的は三者が以後
まったく同じ設定（判定規則・参照実装・評価表）に対して検証できるようにする
ことにある。未検証項目（Claude の三重系全探索と GW フィットコードの公開、
ChatGPT の独立実装 K′、GW の複数ノイズ実現）は未検証のまま明記して固定する。

| 文書 | SHA-256 |
|---|---|
| dr4_freeze.md（凍結本体） | 56ab9e34c404725552474718273f558ba7f8c498aa8270f0ff802d7ea4c154d6 |
| dr4_freeze_supplement.md（追記1） | 78744f3c403c0e6693e895062f102ea84ce2a5a0668ce0f5fc9dfbba2067989a |
| dr4_freeze_supplement2.md（追記2） | 17b7b8b6919dbf6939ffbde60781c9da7a7f31a021c90480dcf90eebfea7818c |
| dr4_freeze_supplement3.md（追記3） | 983795bad2f7095f4829b015a338e026152003d31b4aca049e40d8ec754b0c75 |
| dr4_freeze_supplement4.md（追記4） | 4593d762166b3123247fbd3a16dc6706b8faaea001927a3d8841976353c35ef2 |
| dr4_freeze_supplement5.md（追記5） | fa9345dfe7ed21f1071c74082c543b1c174ea060e07621027818241641f9cd29 |
| dr4_freeze_supplement6.md（追記6） | 1a9700a6cc04369cbd723c86e2e47b9e93c0ddf328ffe2e4341bf658d2591685 |
| dr4_freeze_supplement7.md（追記7） | 81f992171060ff13dd97edeb46b3097d0dbb5d0cba2fd9c96fb793f0db86cfff |
| dr4_freeze_supplement8.md（追記8） | 7baedbf3e1c65352aa96a3dc8cdf9624b0a0dc1fb2aa019cd4aa2f6af4e01e71 |
| dr4_consolidated_rules.md（統合規則） | fa1e746319f4c9ddfdc923b70fad38bd21bf821ecee20cf8cd499f026f40ede8 |
| dr4_m1m2_table_v4.md（評価表 v4） | 540f063baefa559833b59b95b916bbb5e2e13d827953ca3fdfb39d6bee9e41e5 |
| dr4_apply_rules_v3.py（参照実装 v3） | 2632880b1fefa6e0743498b8a815dce34c0c26418b0586a92cb8afe7e094280e |
| dr4_test_vectors_v3.json（テストベクトル v3） | 1866ed62ff9c2cba1e4732459a2b7afcee04b6d14d0f75eda4087be1ee4625f2 |

各追記の主題:
- 追記1: f の解釈（M2 の下限）、変換表、候補の定義、対応付け規則
- 追記2: 判定規則の固定（測光排除の式、M1 範囲、対応確信度）
- 追記3: 二軸判定、M1 感度、光心軌道の前提、減光の実装（A_G,max）
- 追記4: 光心軌道の β 走査、非支持条件の修正
- 追記5: 補間欠陥の修正と再走査、#13 の保留化、減光のモデル間系統差
- 追記6: 補間方式の恒等性、#8 の保留化、減光ブラケット
- 追記7: #12 主表是正と評価表 v3、減光条項、κ 数値更新、GW 診断の確定
- 追記8: 減光規約の文言是正（a0 = A_0 の代理量）、アンカー正本 +0.30 への切替、
  閾値 1.5 等の暫定的運用基準としての明記、非支持の単一伴星モデルへの限定
  （階層三重系の留保と e_out 感度）、参照実装 v3 の正本化、Muse の提案3件
  （DR4 ドライラン、GW S フィット参照規約、κ 指紋と F5 の診断への降格）

現在の暫定ラベル（DR3）: BH候補 #6（BH1）/ BH非支持 #4・#7・#9 /
判定保留 #8・#10・#12・#13 / 軌道解なし #1・#2・#3・#5・#11。
