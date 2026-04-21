from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.keyword_search import search_keyword_chunks
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RagSearchMode, RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RAG 検索の窓口（vector / keyword / hybrid をここで分岐）。retrieval_service は本モジュールを再エクスポート。
# 実験用の差し替えは logic/search/search_logic_*.py の retrieve を logic_registry から呼ぶ。
# -----------------------------------------------------------------------------


def search_documents(
    settings: Settings,
    query: str,
    top_k: int | None = None,
    *,
    rag_search_mode: RagSearchMode = "vector_search",
) -> list[RetrievedChunk]:
    if rag_search_mode == "keyword_search":
        return search_keyword_chunks(settings, query, top_k=top_k)
    if rag_search_mode == "hybrid_search":
        if settings.rag_hybrid_delegate == "keyword_search":
            return search_keyword_chunks(settings, query, top_k=top_k)
        return search_vector_chunks(settings, query, top_k=top_k)
    return search_vector_chunks(settings, query, top_k=top_k)


__all__ = ["search_documents"]
