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
