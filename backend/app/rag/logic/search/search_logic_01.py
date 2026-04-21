from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RagSearchMode, RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: SEARCH logic_01 — ベクトル検索のみ（通常 RAG 想定）。rag_search_mode は実質未使用。
# -----------------------------------------------------------------------------


def retrieve(
    settings: Settings,
    query: str,
    *,
    top_k: int | None,
    rag_search_mode: RagSearchMode,
) -> list[RetrievedChunk]:
    _ = rag_search_mode
    return search_vector_chunks(settings, query, top_k=top_k)
