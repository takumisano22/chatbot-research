# chatbot-research 作業メモ

このファイルは Codex がこのリポジトリで作業するときの補助メモです。実装が変わる可能性があるため、必ず実際のコード・workflow・設定ファイルを正として確認してください。不整合があれば、このファイルを更新します。

## 応答・作業方針

- 日本語で回答し、コードコメントも日本語を基準にする。
- ユーザーから明示されない限り、テストコードは新規追加しない。
- KISS を優先し、実験ロジックや workflow に関係しない先回りのリファクタリングは避ける。
- 既存ファイルには文字化けした日本語コメントもある。UTF-8で読んだ上で、意味が不明なコメントを根拠にせず、実装本体と周辺テストを優先して読む。

## GitHub Actions 起点の処理

- `.github/workflows/run-batch-and-store.yaml`
  - 手動実行で `pair_file` を受け取り、self-hosted Windows runner 上で動く。
  - research リポジトリを checkout し、runner 側の秘密ディレクトリから `.env`、`ingest_document/DOCSET-0001`、`qa_datasets` をコピーする。
  - `docker compose build` 後、`docker compose run --rm experiment_runner --research-pair <pair_file>` を実行する。
  - 実行後に scoring リポジトリを checkout し、`outputs/csv/` と `outputs/json/` の該当成果物を `chatbot-scoring/input/` 以下へコピーして commit/push する。
- `.github/workflows/all-run-batch-store.yaml`
  - 手動実行で、`research_pairs/*.yaml` を順に処理する。
  - `research_pairs/executed_log/executed_pairs.txt` に記録済みのファイルはスキップする。
  - 成功した pair の CSV/JSON を scoring リポジトリへコピーし、実行済みログも research 側へ commit/push する。
  - 失敗した pair がある場合はそこで workflow を失敗させる。

## Docker / 実験ランナー

- `compose.yaml` は RAG 実験用で、主に `vector_store`（Chroma）と `experiment_runner` を起動する。
- `experiment_runner` の entrypoint は `python -m app.experiment.runner`。
- `research_pairs/`、`qa_datasets/`、`ingest_document/`、`outputs/` はコンテナへ volume mount される。
- `backend/Dockerfile` は `backend` を中心に、ローカルパッケージ `llm_definitions` と `vector_stores` を file dependency としてインストールする。

## バッチ実行の主な流れ

1. `backend/app/experiment/runner.py`
  - `--research-pair` から research_pair YAML/JSON を解決する。
  - `qa_datasets` から質問一覧を読み、`ingest_document/<document_set_id>` から PDF を読み込む。
  - `run_research_pair_batch` を呼び、結果を `outputs/csv/` と `outputs/json/` に同じ stem で保存する。
2. `backend/app/experiment/batch_runner.py`
  - research_pair の各 logic id を事前検証する。
  - research_pair の LLM / embedding / top_k / Langfuse などを `Settings` に上書きする。
  - 実験ごとに Chroma コレクションを `rag_reset_collection` でリセットする。
  - PDF を取り込み、質問ごとに retrieve -> rerank -> prompt 組み立て -> LLM 呼び出し -> CSV/JSON 行生成を行う。
  - `ragas_enabled` が true の場合だけ RAGAS 指標を計算する。
3. `backend/app/rag/ingest_batch.py` と `backend/app/rag/ingest_pipeline/`
  - 通常経路は convert -> normalize -> chunking -> vector DB 書き込み。
  - `ingest_pipeline_id` が指定されると `app.rag.ingest_pipeline.library.<id>` を動的 import し、`SUPERSEDES` に含まれる段階を既存経路から差し替える。
  - `docling_library` は convert / normalize / chunking をまとめて置き換える特殊経路。
4. `backend/app/experiment/logic_registry.py`
  - `research_pair` の `logic_01` などを `app.rag.logic.<category>.<category>_logic_XX` として動的 import する。
  - chunking は `split_for_rag_with_metadata`、tokenizer は `tokenize_query`、search は `retrieve`、reranking は `rerank`、prompt は `RAG_SYSTEM_MESSAGE` が契約。

## 実装時に注意すること

- research_pair は実験条件の中心。`research_pair_id`、logic id、`qa_dataset`、`document_set_id`、`top_k`、LLM/embedding 上書き値が runner の挙動を決める。
- `search_logic_01` はベクトル検索、`search_logic_02` は検索なしのベースライン用途。
- `reranking_logic_01` は現状 no-op。
- `chunking_logic_04` は構造認識系で大きく複雑。変更時は、よく読んで出力形式に注意する。
- `RetrievedChunk` と CSV 列は scoring 側との接点になりやすい。列名や JSON 形式の変更は workflow 後段に影響する。
- Chroma への書き込みでは embedding 次元不一致時にコレクションをリセットして再試行する実装がある。
- `Settings` は `.env` とデフォルト値から生成され、Docker 内では Langfuse の localhost を `host.docker.internal` に寄せる処理がある。
- workflow は self-hosted Windows runner 前提で `cmd` と固定パスを使う。PowerShell 前提に変えると Actions 側の挙動が変わる。

## よく見るファイル

- `.github/workflows/*.yaml`: Actions 入口。
- `compose.yaml`: runner と Chroma の起動定義。
- `backend/app/experiment/runner.py`: CLI 入口。
- `backend/app/experiment/batch_runner.py`: 実験全体の制御。
- `backend/app/experiment/research_pair_schema.py`: research_pair のスキーマ。
- `backend/app/experiment/logic_registry.py`: logic id の解決。
- `backend/app/rag/ingest_batch.py`: 入力ファイルの取り込み制御。
- `backend/app/rag/vectorstore/vector_db.py`: vector DB への薄い境界。
- `backend/app/rag/logic/`: chunking / tokenizer / search / reranking / prompt の実験差し替え単位。

