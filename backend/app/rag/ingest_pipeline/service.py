from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.rag_ingest_job import RagIngestJobRow
from app.repositories.ingestion_job_repository import enqueue_rag_upload_job
from app.rag.ingest_pipeline.registry import convert_upload_bytes_to_text
from app.rag.ingest_pipeline.runner import ingest_plain_text
from app.rag.schemas import IngestJobFileResult, IngestJobStatusResponse
from app.rag.vectorstore.vector_db import RagWriteSession, rag_write_session

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 役割: 取り込みジョブの DB 起票・状態組み立てと、ワーカー向けバッチ実行（アップロード本文の取り込み）。
# 主な呼び出し元: ingest API ルート、ingest_pipeline.jobs ワーカー。
# 流れ: enqueue_upload_job →（ワーカー）run_queued_upload_batch → registry/runner 経由でストア反映。
# -----------------------------------------------------------------------------


def ingest_job_status_from_row(row: RagIngestJobRow) -> IngestJobStatusResponse:
    results: list[IngestJobFileResult] | None = None
    if row.status == "succeeded" and row.result_json is not None:
        raw = row.result_json
        if isinstance(raw, list):
            results = [
                IngestJobFileResult.model_validate(
                    {
                        "source_name": x.get("source_name") or "",
                        "ok": bool(x.get("ok")),
                        "error": x.get("error"),
                        "chunks_written": int(x.get("chunks_written", 0)),
                    }
                )
                for x in raw
                if isinstance(x, dict)
            ]
    return IngestJobStatusResponse(
        job_id=row.id,
        status=row.status,
        payload_summary=row.payload_summary,
        error_message=row.error_message,
        results=results,
    )


async def enqueue_upload_job(session: AsyncSession, items: list[tuple[str, bytes]]) -> UUID:
    """DB に 1 ジョブを起票する。items はアップロード順を保つ。"""
    return await enqueue_rag_upload_job(session, items)


def run_queued_upload_batch(settings: Settings, items: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    """ワーカー用: 混在拡張子を順に取り込み、result_json 用の dict 列を返す。"""
    session = rag_write_session(settings)
    out: list[dict[str, Any]] = []
    for name, data in items:
        sn, ok, err, n = _ingest_single_upload(settings, session, name, data)
        out.append({"source_name": sn, "ok": ok, "error": err, "chunks_written": n})
    return out


@dataclass(frozen=True)
class _UploadIngestPolicy:
    ## 拡張子ゲートと convert〜ingest の例外境界（PDF は一体・txt/md は分割＋ UTF-8 特化）。
    allowed_suffixes: frozenset[str]
    rejected_extension_message: str
    wrap_convert_and_ingest_together: bool


_POLICY_PDF = _UploadIngestPolicy(
    allowed_suffixes=frozenset({".pdf"}),
    rejected_extension_message="拡張子が .pdf ではありません",
    wrap_convert_and_ingest_together=True,
)
_POLICY_TXT_MD = _UploadIngestPolicy(
    allowed_suffixes=frozenset({".txt", ".md"}),
    rejected_extension_message="拡張子は .txt または .md のみ対応です",
    wrap_convert_and_ingest_together=False,
)


def _policy_for_file(safe_name: str) -> _UploadIngestPolicy | None:
    suf = Path(safe_name).suffix.lower()
    if suf == ".pdf":
        return _POLICY_PDF
    if suf in (".txt", ".md"):
        return _POLICY_TXT_MD
    return None


def _ingest_single_upload(
    settings: Settings,
    session: RagWriteSession,
    name: str,
    data: bytes,
) -> tuple[str, bool, str | None, int]:
    safe_name = Path(name).name
    pol = _policy_for_file(safe_name)
    if pol is None:
        return name, False, "未対応の拡張子です（.pdf / .txt / .md のみ）", 0
    return _ingest_one_with_policy(settings, session, name, data, pol)


def _ingest_one_with_policy(
    settings: Settings,
    session: RagWriteSession,
    name: str,
    data: bytes,
    policy: _UploadIngestPolicy,
) -> tuple[str, bool, str | None, int]:
    safe_name = Path(name).name
    suf = Path(safe_name).suffix.lower()
    if suf not in policy.allowed_suffixes:
        return name, False, policy.rejected_extension_message, 0
    source = f"uploaded/{safe_name}"
    if policy.wrap_convert_and_ingest_together:
        try:
            full_text = convert_upload_bytes_to_text(safe_name, data, settings)
            n = ingest_plain_text(settings, session, source, full_text)
        except Exception as e:
            logger.exception("アップロード取り込み失敗: %s", safe_name)
            return safe_name, False, str(e), 0
        return safe_name, True, None, n
    try:
        full_text = convert_upload_bytes_to_text(safe_name, data, settings)
    except UnicodeDecodeError as e:
        logger.exception("UTF-8 デコード失敗: %s", safe_name)
        return safe_name, False, f"UTF-8 でないか破損: {e}", 0
    try:
        n = ingest_plain_text(settings, session, source, full_text)
    except Exception as e:
        logger.exception("アップロード取り込み失敗: %s", safe_name)
        return safe_name, False, str(e), 0
    return safe_name, True, None, n
