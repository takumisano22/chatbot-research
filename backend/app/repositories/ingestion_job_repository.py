from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_ingest_job import RagIngestJobRow
from app.rag.ingest_pipeline.enums import JOB_KIND_RAG_UPLOAD

# -----------------------------------------------------------------------------
# 役割: rag_ingest_jobs の enqueue・claim・結果更新とペイロードの encode/decode。
# 主な呼び出し元: ingest API / ingest_pipeline.service、ingest_pipeline.jobs ワーカー。
# 流れ: enqueue → claim_next_queued_job_id → load_job_upload_items → mark_job_*。
# -----------------------------------------------------------------------------


async def enqueue_rag_upload_job(session: AsyncSession, items: list[tuple[str, bytes]]) -> UUID:
    job = RagIngestJobRow(
        status="queued",
        job_kind=JOB_KIND_RAG_UPLOAD,
        payload_summary=_batch_upload_summary(len(items)),
        payload_items=_encode_payload_items(items),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job.id


async def get_job_row(session: AsyncSession, job_id: UUID) -> RagIngestJobRow | None:
    return await session.get(RagIngestJobRow, job_id)


async def claim_next_queued_job_id(session: AsyncSession) -> UUID | None:
    now = datetime.now(timezone.utc)
    next_id = (
        select(RagIngestJobRow.id)
        .where(RagIngestJobRow.status == "queued")
        .order_by(RagIngestJobRow.created_at.asc(), RagIngestJobRow.id.asc())
        .limit(1)
        .scalar_subquery()
    )
    stmt = (
        update(RagIngestJobRow)
        .where(RagIngestJobRow.id == next_id)
        .values(status="running", updated_at=now)
        .returning(RagIngestJobRow.id)
    )
    result = await session.execute(stmt)
    row = result.fetchone()
    await session.commit()
    if row is None:
        return None
    raw_id = row[0]
    return raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))


async def load_job_upload_items(session: AsyncSession, job_id: UUID) -> list[tuple[str, bytes]]:
    job = await session.get(RagIngestJobRow, job_id)
    if job is None:
        return []
    return _decode_payload_items(job.payload_items)


async def mark_job_succeeded(
    session: AsyncSession,
    job_id: UUID,
    results: list[dict[str, Any]],
) -> None:
    payload = [
        {
            "source_name": str(r["source_name"]),
            "ok": bool(r["ok"]),
            "error": r.get("error"),
            "chunks_written": int(r.get("chunks_written", 0)),
        }
        for r in results
    ]
    job = await session.get(RagIngestJobRow, job_id)
    if job is None:
        return
    job.status = "succeeded"
    job.result_json = payload
    job.error_message = None
    job.payload_items = None
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()


async def mark_job_failed(session: AsyncSession, job_id: UUID, message: str) -> None:
    job = await session.get(RagIngestJobRow, job_id)
    if job is None:
        return
    job.status = "failed"
    job.error_message = message[:8000]
    job.payload_items = None
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()


# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------


def _batch_upload_summary(file_count: int) -> str:
    return f"RAG upload ×{file_count}"


def _encode_payload_items(items: list[tuple[str, bytes]]) -> list[dict[str, str]]:
    return [
        {
            "filename": Path(name).name,
            "body_b64": base64.b64encode(data).decode("ascii"),
        }
        for name, data in items
    ]


def _decode_payload_items(raw: object) -> list[tuple[str, bytes]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, bytes]] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        fn = str(x.get("filename") or "")
        b64 = x.get("body_b64")
        if not fn or not isinstance(b64, str):
            continue
        try:
            data = base64.b64decode(b64.encode("ascii"), validate=True)
        except (ValueError, binascii.Error):
            continue
        out.append((fn, data))
    return out
