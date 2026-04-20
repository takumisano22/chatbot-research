from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: 取り込みジョブ種別（DB rag_ingest_jobs.job_kind）の定数定義。
# 主な呼び出し元: ingestion_job_repository、ingest_pipeline.jobs。
# 流れ: 行の job_kind と照合し、旧値はワーカー側で rag_upload に正規化される。
# -----------------------------------------------------------------------------

JOB_KIND_RAG_UPLOAD = "rag_upload"
