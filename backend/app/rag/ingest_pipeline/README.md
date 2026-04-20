# ingest_pipeline

RAG 用ドキュメント取り込みのドメインロジック。

| 入口 | 場所 |
|------|------|
| HTTP | `POST/GET /api/v1/rag/ingest/jobs`（`app/api/routes/ingest.py`）→ `ingest_pipeline.service` |
| DB キューワーカー | `python -m app.rag.ingest_pipeline.jobs` |
| CLI | （現状未実装。ジョブ API / ワーカーを利用） |
| キュー操作 | `app/repositories/ingestion_job_repository.py` |

## レイヤ

- **`service.py`** … DB 起票・ワーカー用 `run_queued_upload_batch`・`ingest_job_status_from_row`。
- **`runner.py`** … `ingest_plain_text`（正規化・チャンク化 → `vector_db`）。
- **`registry.py`** … 拡張子 → `converters/*` への振り分けのみ。
- **`processors/`** … `upload_multipart`（multipart 読取・上限）。
- **`app/rag/logic/normalizer.py`** … チャンク前の全文正規化（`runner` が利用）。
- **`stages/`** … 段階別モジュール用（現行は空に近い。チャンクは `runner` に集約）。
- **`converters/`** … バイト列 → 本文（PDF は MarkItDown）。

## 流れ（アップロード）

`service`（拡張子ゲート・例外境界）→ `registry.convert_upload_bytes_to_text` → `converters` → `runner.ingest_plain_text` → `vectorstore`。

## 流れ（非同期ジョブ）

`jobs` → `service.run_queued_upload_batch`（1 ジョブ内で拡張子ごとに convert〜ingest）→ 上記と同じ。

## 流れ（アップロード：詳細）

フロントエンドが `POST /api/v1/rag/ingest/jobs` で multipart 送信し、別プロセスのワーカーがキューを処理してベクトル DB に保存するまで。

### 1. HTTP 受付〜DB 起票（API プロセス）

1. 取り込みジョブ登録の HTTP を受ける — `app/api/routes/ingest.py` — `post_ingest_jobs`
2. multipart を読み取り順序・件数・サイズを検証する — `processors/upload_multipart.py` — `collect_mixed_ordered_items`  
   - 拡張子から PDF / txt_md を判定する — `processors/upload_multipart.py` — `classify_upload_kind`
3. キュー起票をサービスへ渡す — `service.py` — `enqueue_upload_job`
4. DB にジョブ行とペイロードを保存する — `app/repositories/ingestion_job_repository.py` — `enqueue_rag_upload_job`

### 2. ワーカー — ジョブ取得とペイロード読込

1. キューから 1 件ずつ処理する — `jobs.py` — `process_next_job_once`
2. 待機中ジョブを実行中へ claim する — `app/repositories/ingestion_job_repository.py` — `claim_next_queued_job_id`
3. ジョブ行を読み種別を確定して処理へ進む — `jobs.py` — `dispatch_claimed_job`（`RagIngestJobRow` の読取・`job_kind` の解決を含む）
4. 保存済みペイロードをファイル列に復元する — `app/repositories/ingestion_job_repository.py` — `load_job_upload_items`

### 3. ワーカー — 各ファイルの変換〜ベクトル DB 書込

1. ジョブ内の各ファイルを順に取り込み結果を集める — `service.py` — `run_queued_upload_batch`  
   - ベクトル DB 書き込み用セッションを開く — `app/rag/vectorstore/vector_db.py` — `rag_write_session`
2. 拡張子に応じてバイト列を本文へ変換する — `registry.py` — `convert_upload_bytes_to_text`  
   - PDF をテキスト相当へ変換する — `.pdf` → `converters/pdf_converter.py` — `convert_pdf_bytes`  
   - UTF-8 テキストとして本文を得る — `.txt` / `.md` → `converters/text_converter.py` — `convert_text_bytes`
3. チャンク化からストア反映までをまとめる — `runner.py` — `ingest_plain_text`  
   - チャンク前に全文を正規化する — `app/rag/logic/normalizer.py` — `normalize_document_text`  
   - 設定に従いチャンクを生成する — `app/rag/vectorstore/chunker.py` — `build_chunks_for_source`
4. 同一ソースの既存データを消す — `app/rag/vectorstore/vector_db.py` — `RagWriteSession.delete_by_source`  
5. 新チャンクをベクトル DB に書く — `app/rag/vectorstore/vector_db.py` — `RagWriteSession.add_chunks`

### 4. ワーカー — 完了記録

1. 成功結果を DB に書き戻す — `jobs.py` — `dispatch_claimed_job`（成功時の 2 つ目の DB セッション内）
2. ジョブを成功状態に確定する — `app/repositories/ingestion_job_repository.py` — `mark_job_succeeded`
