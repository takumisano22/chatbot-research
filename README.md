# chatbot-research

RAG 実験バッチ用のリポジトリです。`research_pair` を 1 つ指定して、**Chroma リセット -> 文書取り込み -> QA 一括推論 -> CSV/JSON 出力**を Docker で実行できます。

## できること

- `research_pairs/*.yaml`（または JSON）で実験条件を定義
- `qa_datasets/*.json` の質問セットを一括実行
- `ingest_document/<document_set_id>/**/*.pdf` を取り込み
- 結果を `outputs/csv/` と `outputs/json/` に保存

## クイックスタート

1. `.env.example` を `.env` にコピーして接続先を設定する（LLM / embedding / Langfuse / vector store）。
2. `qa_datasets/` に質問データ、`ingest_document/<document_set_id>/` に PDF を配置する。
3. `research_pairs/*.yaml` に `research_pair_id`, `qa_dataset`, `document_set_id`, `top_k` などを記述する。
4. リポジトリルートで実行する。

```bash
docker compose run --rm experiment_runner --research-pair RP-0001
```

成功時は、生成された CSV の絶対パスが標準出力に 1 行で表示されます。

## GitHub Actions での実行

- GitHub Actions（self-hosted runner）からも同じコマンドで実行できます。
- runner 上で本リポジトリをチェックアウトし、`.env` と実験データ（`research_pairs/`, `qa_datasets/`, `ingest_document/`）を配置して実行します。
- ワークフロー例は `.github/workflows/run-batch-and-store.yaml` を参照してください。

```bash
docker compose -f compose.yaml run --rm experiment_runner --research-pair RP-0001
```

## ディレクトリ概要

- `backend/`: 実験バッチ本体（`python -m app.experiment.runner`）
- `compose.yaml`: `vector_store` + `experiment_runner` の実行定義
- `research_pairs/`: 実験条件ファイル
- `qa_datasets/`: 質問 JSON
- `ingest_document/`: 取り込み対象 PDF
- `outputs/`: 実験結果（CSV / JSON）


## 詳細ドキュメント

実験設定、`research_pair` スキーマ、CSV 列の詳細は [docs/experiment_batch.md](docs/experiment_batch.md) を参照してください。
