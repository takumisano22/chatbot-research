from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db import get_db_session
from app.repositories.ingestion_job_repository import get_job_row
from app.rag.ingest_pipeline import service as ingest_svc
from app.rag.ingest_pipeline.processors.upload_multipart import collect_mixed_ordered_items
from app.rag.schemas import IngestJobCreatedResponse, IngestJobStatusResponse

# -----------------------------------------------------------------------------
# 役割: RAG 取り込みジョブの受付と状態参照（マルチパートアップロード）。
# 主な呼び出し元: main がルーターをマウントし、クライアントが /api/v1/rag/ingest/jobs を利用する。
# 流れ: ファイル検証・読込 → ingest_pipeline.service で DB キュー起票 → GET で状態取得。
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/rag", tags=["rag-ingest"])


@router.post("/ingest/jobs", status_code=202, response_model=IngestJobCreatedResponse)
async def post_ingest_jobs(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db_session),
) -> IngestJobCreatedResponse:
    ordered = await collect_mixed_ordered_items(files, settings)
    items = [(n, d) for n, d, _kind in ordered]
    job_id = await ingest_svc.enqueue_upload_job(session, items)
    return IngestJobCreatedResponse(job_id=job_id, status="queued")


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> IngestJobStatusResponse:
    row = await get_job_row(session, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return ingest_svc.ingest_job_status_from_row(row)
