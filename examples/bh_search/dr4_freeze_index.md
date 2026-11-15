# DR4 凍結文書 索引

作成: 2026-11-15（Muse）
目的: Gaia DR4（2026-12-02 公開予定）に先立つ事前登録一式の目録とハッシュの集約。
固定方法: 本索引を含む下記 4 文書を GitHub Release に添付して公開し、Release ページを Wayback Machine に保存する（保存 URL は `dr4_fixation_record.md` に別記する。本索引自体のハッシュは Release の添付内容として固定される）。

## 凍結文書（ハッシュ対象）

| 文書 | 内容 | SHA-256 | 既存の固定 |
|---|---|---|---|
| `dr4_freeze.md` | 凍結本体：13 件の DR3 値と判定基準 | `56ab9e34c404725552474718273f558ba7f8c498aa8270f0ff802d7ea4c154d6` | GitHub コミット 6a3ff03 |
| `dr4_freeze_supplement.md` | 追記 1：解釈層と三者レビューへの回答 | `78744f3c403c0e6693e895062f102ea84ce2a5a0668ce0f5fc9dfbba2067989a` | GitHub コミット e1f3dcf |
| `dr4_freeze_supplement2.md` | 追記 2：判定規則（測光排除・M1 範囲・判定表・対応確信度）の固定 | `17b7b8b6919dbf6939ffbde60781c9da7a7f31a021c90480dcf90eebfea7818c` | 本 Release |
| `dr4_freeze_index.md` | 本索引 | （Release 添付により固定） | 本 Release |

ハッシュの照合は標準形式ファイル（`dr4_freeze.sha256.txt`、`dr4_freeze_supplement.sha256`、追記 2 分は本索引の表）でも行える。

## 参考資料（ハッシュ対象外、リポジトリ内に保存）

- `muse_opinion_2026-11-15.md` — Muse の意見書（コミット fe1cd8e）
- `muse_reply_2026-11-15.md` — Claude v4・ChatGPT 追加査読への返信
- `dr4_rule_check.py` / `dr4_rule_check.json` — 意見書の数値検証
- `dr4_m1m2_table.md` / `.json` / `dr4_m1m2_table.py` — 13 件の M1・M2 評価表
- 査読の往復記録: Drive「AI共有 / Claude回答」フォルダ（Claude 回答 v2–v4）、GitHub `collaboration/` フォルダ（協働レポートと運用ルール）

## 運用の原則

- 凍結文書は書き換えない。変更が必要になれば、日付と理由を明記した新しい追記文書を追加する。
- DR4 公開後は、追記 2 の判定表に入力を入れて判定する。結果を見てから基準を変えない。
