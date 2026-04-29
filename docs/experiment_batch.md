# RAG 実験バッチ（research_pair / Docker）

## ディレクトリの責務

| パス | 役割 |
|------|------|
| `research_pairs/` | 実験条件 1 本 = YAML（または JSON）1 ファイル。`research_pair_id` とファイル名を対応させる。 |
| `qa_datasets/` | 質問リスト JSON。`research_pair` の `qa_dataset` でファイル名指定。 |
| `ingest_document/<document_set_id>/` | その実験で取り込む PDF（再帰的に探索）。 |
| `outputs/csv/` | 実験 CSV の出力先（CLI が UTC タイムスタンプ付きファイル名で保存）。 |
| `outputs/json/` | 入力（質問）と推論 LLM 出力のみを抜き出した JSON の出力先。CSV と同一ステムで対応。 |

バックエンドのコード・設定は `backend/`、Docker 全体はルートの `compose.yaml` を参照してください。

## research_pair の書き方

- 形式: YAML（`.yaml` / `.yml`）または JSON（`.json`）
- 必須フィールド（Pydantic で検証）:
  - `research_pair_id`, `search_logic_id`, `chunking_logic_id`, `reranking_logic_id`, `tokenizer_logic_id`
  - `top_k`（1〜500）
  - `qa_dataset`（`qa_datasets/` 内のファイル名）
  - `document_set_id`（`ingest_document/` 直下のディレクトリ名）
- 任意: `llm_model`, `llm_api_base_url`, `llm_temperature`, `llm_request_timeout_seconds`
- 任意: `embedding_provider`, `embedding_base_url`, `embedding_model`
- `ragas_enabled`: `true` / `false`（OFF 時も CSV の RAGAS 列は空で残る）
- 任意: `rag_hybrid_delegate`（ランタイムの `Settings.rag_hybrid_delegate` を上書き、`vector_search` | `keyword_search`）

`search_logic_id` の例:

- `logic_01`: ベクトル検索のみ（通常の RAG 取り込みありの前提）。
- `logic_02`: 検索結果常に空（コンテキスト無し。ingest 済みでも検索はスキップ）。

通常の検索窓口（実験以外の単体テスト等）は `app.rag.logic.search` の `search_documents`（ベクトル検索）。キーワード検索は `app.rag.logic.keyword_search.search_keyword_chunks`。実験バッチでは `logic/search/search_logic_XX.py` の `retrieve` が使われます。

`logic_id` は **`logic_01`** の形式（英小文字・数字・アンダースコア）。新しいロジックは `backend/app/rag/logic/<category>/<category>_logic_XX.py` に追加し、所定の関数を実装してください。

## QA データセット

`qa_datasets/*.json` の例:

```json
{
  "dataset_name": "任意名",
  "items": [ { "question": "..." } ]
}
```

ルートが配列のみの JSON でも読み込み可能です。

## ドキュメントセット

`ingest_document/<document_set_id>/**/*.pdf` を読み込みます（拡張子 `.pdf` / `.PDF`）。

## 実行コマンド（ローカル）

リポジトリルートで、依存サービス（少なくとも vector DB）が起動している状態で:

```bash
docker compose run --rm experiment_runner --research-pair RP-0001
```

成功時、CSV の絶対パスが 1 行で標準出力に出ます。

## GitHub Actions / self-hosted runner から

Actions 本体は本リポジトリ外です。runner 上では例えば次のように **同一リポジトリをチェックアウトしたディレクトリ** で実行します。

```bash
docker compose -f compose.yaml run --rm experiment_runner --research-pair RP-0001
```

事前に `.env`（ホストの LLM / Langfuse / `VECTOR_STORE_SERVER_HOST` 等）を整え、`research_pairs` / `qa_datasets` / `ingest_document` に実データを配置してください。CSV は `outputs/csv/`、JSON は `outputs/json/` にマウントされたホスト側ディレクトリに残ります（同一ステム = `{research_pair_id}_{UTC タイムスタンプ}` で対応）。

## JSON 出力

`outputs/json/{research_pair_id}_{ts}.json` に、入力（質問）と推論 LLM 出力のみを抜き出した JSON を保存します。CSV と同一ステムです。

```json
{
  "dataset_name": "...",
  "items": [
    { "question": "<入力>", "answer": "<推論 LLM 出力>" }
  ]
}
```

## CSV 列

固定列に加え、`top_k` ごとに次が繰り返し列として付きます（1 行あたり列数固定）。

- `retrieved_source_i`, `retrieved_chunk_id_i`, `retrieved_distance_i`, `retrieved_text_i`（`i = 1..top_k`）

固定列には `research_pair_id`, **`research_pair_spec`**（比較用の JSON 文字列）, `question_index`, `document_set_id`, `dataset_name`, 入出力、レイテンシ、`top_k`, 各 logic_id, LLM / embedding 名, `ragas_faithfulness`, `ragas_answer_relevancy` が含まれます。

## 環境変数（`.env`）

API 接続・ホスト・Langfuse・ベクトルストアは従来どおり `.env` に記載します。実験用ディレクトリは必要に応じて `EXPERIMENT_*` で上書きできます（`.env.example` 参照）。

## 注意

- 1 回の実験で **コレクションをリセットしてから全 PDF を ingest し、その後 QA 全件を推論** します（QA ごとの再 ingest はしません）。
- 本 compose は **vector_store + experiment_runner のみ** です。他プロセスが同一 Chroma に同時書き込みしないでください。
