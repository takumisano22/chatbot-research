from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.rag.ingest_pipeline.registry import convert_upload_bytes_to_text
from app.rag.ingest_pipeline.runner import ingest_plain_text
from app.rag.vectorstore.vector_db import RagWriteSession, rag_write_session

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 役割: DB を介さず (ファイル名, bytes) を順にベクトルストアへ書き込む（実験バッチ専用）。
# 流れ: rag_write_session → 拡張子ごとに convert → ingest_plain_text。
# -----------------------------------------------------------------------------


def run_upload_items_batch(settings: Settings, items: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
    session = rag_write_session(settings)
    out: list[dict[str, Any]] = []
    for name, data in items:
        sn, ok, err, n = _ingest_single_upload(settings, session, name, data)
        out.append({"source_name": sn, "ok": ok, "error": err, "chunks_written": n})
    return out


@dataclass(frozen=True)
class _UploadIngestPolicy:
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
            logger.exception("取り込み失敗: %s", safe_name)
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
        logger.exception("取り込み失敗: %s", safe_name)
        return safe_name, False, str(e), 0
    return safe_name, True, None, n
