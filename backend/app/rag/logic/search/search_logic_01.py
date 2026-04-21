from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: SEARCH logic_01 — ベクトル検索のみ（通常 RAG 想定）。
# -----------------------------------------------------------------------------


def retrieve(
    settings: Settings,
    query: str,
    *,
    top_k: int | None,
) -> list[RetrievedChunk]:
    return search_vector_chunks(settings, query, top_k=top_k)
