# chatbot-research

FastAPI ベースのチャット / RAG 実験用リポジトリです。通常の API 起動に加え、**research_pair を 1 つ指定した一括 RAG 実験**を Docker から実行できます。

## クイックスタート（実験バッチ）

1. `.env.example` を `.env` にコピーし、ホスト上の LLM（例: Ollama）や Langfuse、ベクトルストア接続を設定する。
2. `ingest_document/<document_set_id>/` に PDF を置く。
3. `qa_datasets/` に質問 JSON を置き、`research_pairs/*.yaml` で `qa_dataset` / `document_set_id` / `top_k` 等を指定する。
4. 実行:

```bash
docker compose run --rm experiment_runner --research-pair RP-0001
```

詳細は [docs/experiment_batch.md](docs/experiment_batch.md) を参照してください。

RAG 検索の実装窓口は `backend/app/rag/logic/search/__init__.py` の `search_documents` です（`retrieval_service` は互換の再エクスポート）。

## その他

- アプリ本体・API: `backend/`
- Compose: `compose.yaml`
