from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.normalizer import normalize_document_text
from app.rag.vectorstore.chunker import ChunkForStore, build_chunks_for_source
from app.rag.vectorstore.vector_db import RagWriteSession

# -----------------------------------------------------------------------------
# 役割: プレーンテキスト全文のチャンク化と vector DB への反映（ファイル種別は上位で解決済み）。
# 主な呼び出し元: ingest_pipeline.service。
# 流れ: _run_chunk_stage → delete_by_source → add_chunks → 件数返却。
# -----------------------------------------------------------------------------


def ingest_plain_text(
    settings: Settings,
    session: RagWriteSession,
    repo_relative_source: str,
    full_text: str,
) -> int:
    chunks = _run_chunk_stage(
        repo_relative_source=repo_relative_source,
        full_text=full_text,
        settings=settings,
    )
    session.delete_by_source(repo_relative_source)
    session.add_chunks(chunks)
    return len(chunks)


# --- 補助（チャンク列の組み立て。正規化と分割は既存モジュールに委譲） ---


def _run_chunk_stage(
    *,
    repo_relative_source: str,
    full_text: str,
    settings: Settings,
) -> list[ChunkForStore]:
    # チャンク境界を安定させるため、分割前に全文を正規化してから chunker へ渡す。
    normalized = normalize_document_text(full_text)
    return build_chunks_for_source(
        repo_relative_source=repo_relative_source,
        full_text=normalized,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
