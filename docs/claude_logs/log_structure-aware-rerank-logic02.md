# structure-aware rerank logic02 作業ログ

## 2026-05-01: chunking_logic_06 用 reranking_logic_02 の実装

### 目的

距離検索で取得した数十件の候補（chunking_logic_06 由来。同一論理チャンクが
`full_context_plain` / `local_context_plain` の 2 variant に展開される、また最大
文字数で `parent` / `child` が分割される）を、階層 metadata に基づき LLM が読み
やすい件数まで圧縮する。

### 結論サマリ

- `backend/app/rag/logic/reranking/reranking_logic_02.py` を実装。
- `search_logic_03.py` および `vector_db.py` の修正は不要だった
  （metadata は既に `RetrievedChunk.metadata` へ正しく流れている）。

### metadata の流通確認

- `chunking_logic_06.default_metadata_builder` が `chunk_role` / `level` /
  `path_text` / `chunk_id` を生成。
- `flatten_chunks` が `parent_chunk_id` / `child_chunk_id` /
  `grandchild_chunk_id` を追記。
- `vector_db._expand_chunks_for_vector_records` が variant 展開時に
  `logical_chunk_id` / `vector_record_id` / `vector_text_variant` を補完。
- `vectordb.chroma.store` 側で予約キー `{doc_id, chunk_id, source, chunk_text}`
  以外を `custom_meta` として復元 → `RetrievedChunk.metadata` に格納。
- `RetrievedChunk.chunk_id` は **logical_chunk_id**（同一チャンク由来の variant
  を束ねる ID）。

### 再ランキングの段階処理

| step | 処理 |
| ---- | ---- |
| 0 | 元データ < 5 件は再ランキングをスキップしてそのまま返す（少数データに対する集約は LLM 文脈を痩せさせるため） |
| 0 | `new_top_k = max(5, top_k // 6)` |
| 1 | `final_score` の全体平均を保持 |
| 2 | 同一 `chunk_id` を最高スコアのみへ集約（variant 重複の解消） |
| 3 | 同一 `parent_chunk_id` 内に `parent` がいれば、それより低スコアの `child`/`grandchild` を削除 |
| 4 | 同一 `child_chunk_id` 内に `child` がいれば、それより低スコアの `grandchild` を削除 |
| 5 | 同一 `child_chunk_id` の grandchild 最高スコアを代表 child へ引き継ぎ、grandchild は全削除（child が複数ある場合は元スコア最高の child のみが受け取り、他の child は維持） |
| 6 | 同一 `parent_chunk_id` に child が 2 件以上残っていて parent が居る場合、child 最高スコアを代表 parent に引き継ぎ、child は全削除（parent が複数ある場合は元スコア最高 parent が受け取り、他の parent は維持） |
| 7 | step 1 の平均未満を削除 |
| 8 | 残数 < `new_top_k` なら `new_top_k` を残数に再修正、そうでなければ上位 `new_top_k` 件を返す |

### 実装上の意思決定

- **「元データ < 5 件はそのまま返す」**: プロンプトの「その件数のままで渡してくだ
  さい」を文字どおりに解釈し、再ランキング自体をスキップ。少数候補に平均カット
  / 階層集約を効かせると LLM への入力が極端に痩せる懸念があるため。
- **スコア引き継ぎは max 採用**: `donor_score > recipient.score` の場合のみ上書き。
  上位ロールがすでに強い場合に下位ロールのスコアで上書きすると関連性評価が劣化
  するため、ロジックの「引き継ぎ」を `max` で実装。
- **互換性**: `parent_chunk_id` / `child_chunk_id` / `chunk_role` を持たない
  チャンク（chunking_logic_01〜05 など）は、グループ化対象から外し（`others`
  バケットへ）削除や引き継ぎを受けない。step 1/2/7/8 のみ適用される。
- **スコア集約用の値**: `final_score` を採用（vector hit では
  `vector_score_norm` と同値、keyword hit でも `keyword_weight *
  keyword_score_norm` として 0..1 に揃っている）。

### 変更ファイル

- `backend/app/rag/logic/reranking/reranking_logic_02.py`: 上記 8 ステップを実装。

---

## 2026-05-01: 子省略（merge_child）ケースの孤立孫問題を修正

### 問題

`第5章介護休暇` 配下の条（子）が 1 件のみの場合 `merge_child=True` となり、子
チャンクは emit されず親チャンクが兼ねる。しかし子配下の孫（各項）は独立して
emit される。

| チャンク | chunk_role | child_chunk_id |
| ------- | ---------- | -------------- |
| 親 (第5章) | parent | 自身の chunk_id（自己参照） |
| 孫 (各項) | grandchild | `chunk_art13`（= emit されていない子の ID） |

step 5 は `child_chunk_id` で `role="child"` のチャンクを探すが存在しないため
孫が削除されず、親と孫が両方コンテキストに残る重複が生じていた。

### 修正方針

`chunking_logic_06.py` の修正は不要。`reranking_logic_02.py` に **step 5b** を
追加し、「孤立孫（`child_chunk_id` に対応する chunk が存在しない grandchild）」
を検出して `parent_chunk_id` をキーに親へ引き継ぎ・削除する。

- 孤立検出: 現ワークセットの全 `chunk_id` 集合に `child_chunk_id` が含まれない grandchild
- 親を `parent_chunk_id == chunk_id` で逆引き
- donor_score（孤立孫の最大スコア）が親スコアを上回る場合のみ親スコアを更新
- 対応する親チャンクが取得されていなければ孤立孫をそのまま残す（コンテキスト保護）

### 変更ファイル

- `backend/app/rag/logic/reranking/reranking_logic_02.py`: `_promote_orphan_grandchildren` を追加、step 5b として呼び出しを挿入。

---

## 2026-05-01: step 5b のバグ修正（chunk_id 名前空間の不一致）

### 問題

前回の step 5b 実装が機能していなかった。原因は chunk_id の名前空間の不一致：

| 値 | 形式 | 生成元 |
| -- | ---- | ------ |
| `RetrievedChunk.chunk_id` | `"{doc_id}:{i}"` (連番) | `chunker.py` |
| `parent_chunk_id` / `child_chunk_id` (metadata) | `"chunk_sec_{xxx}"` (section-based) | `chunking_logic_06.py` |

前実装では `by_chunk_id`（連番 ID でインデックス）に対して `parent_chunk_id`（section ID）で検索していたため、常に `None` が返り、孤立孫が全件維持されていた。

### 修正

`by_chunk_id` による逆引きをやめ、`parent_by_pk` に切り替える。

**親チャンクの見つけ方:**
- 親チャンク（role="parent"）は `meta("parent_chunk_id")` が自身の section-based base_id と同値（自己参照）という性質を持つ
- 孤立孫の `meta("parent_chunk_id")` も同じ section-based ID
- → `parent_by_pk` を `meta("parent_chunk_id")` でインデックスして照合 ✓

また、step 5 通過後に残る grandchild は「step 5 で対応 child が見つからなかった孤立孫」のみのため、`chunk_id_set` による孤立判定は不要と判断し削除した（step 3-5 は meta 値同士の比較のみなので名前空間問題の影響なし）。

### 変更ファイル

- `backend/app/rag/logic/reranking/reranking_logic_02.py`: `_promote_orphan_grandchildren` を `parent_by_pk` ルックアップ方式で修正。
