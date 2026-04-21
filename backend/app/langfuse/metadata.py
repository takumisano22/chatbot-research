from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.langfuse.constants import CHUNKING_STRATEGY_RECURSIVE_CHARACTER_TEXT_SPLITTER
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: Settings と RAG 実行文脈から Langfuse metadata / 入力サマリを組み立てる。
# 主な呼び出し元: app.langfuse.tracer（各観測点は dict を手書きしすぎないようにする）。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_DEFAULT_TRUNCATE = 1200
_CHUNK_TEXT_PREVIEW = 200
_MAX_HIT_ROWS = 8


def truncate_for_observation(text: str, *, max_chars: int = _DEFAULT_TRUNCATE) -> str:
    t = text if isinstance(text, str) else str(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…(truncated)"


def build_common_metadata(
    settings: Settings,
    *,
    use_rag: bool | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "app_env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "vector_store_provider": settings.vector_store_provider,
        "vector_db_adapter_subpackage": settings.vector_db_adapter_subpackage,
        "rag_collection_name": settings.rag_collection_name,
        "chunking_strategy": CHUNKING_STRATEGY_RECURSIVE_CHARACTER_TEXT_SPLITTER,
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
    }
    if use_rag is not None:
        meta["use_rag"] = use_rag
    env = (settings.langfuse_environment or "").strip()
    if env:
        meta["langfuse_environment"] = env
    return meta


def summarize_retrieved_chunks(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ch in chunks[:_MAX_HIT_ROWS]:
        score_val = ch.vector_score_raw
        rows.append(
            {
                "source": truncate_for_observation(ch.source, max_chars=240),
                "chunk_id": ch.chunk_id,
                "distance_or_score": float(score_val) if score_val is not None else float(ch.final_score),
                "chunk_text_preview": truncate_for_observation(
                    ch.chunk_text, max_chars=_CHUNK_TEXT_PREVIEW
                ),
            }
        )
    cap = min(len(chunks), 32)
    return {
        "hit_count": len(chunks),
        "retrieved_count": len(chunks),
        "chunk_ids": [c.chunk_id for c in chunks[:cap]],
        "sources": [truncate_for_observation(c.source, max_chars=160) for c in chunks[:cap]],
        "top_hits_preview": rows,
    }
