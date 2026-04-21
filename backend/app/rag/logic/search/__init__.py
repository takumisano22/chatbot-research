from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RAG 検索の窓口（ベクトル検索）。retrieval_service は本モジュールを再エクスポート。
# キーワード検索は app.rag.logic.keyword_search.search_keyword_chunks を直接使う。
# 実験用の差し替えは logic/search/search_logic_*.py の retrieve を logic_registry から呼ぶ。
# -----------------------------------------------------------------------------


def search_documents(
    settings: Settings,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    return search_vector_chunks(settings, query, top_k=top_k)


__all__ = ["search_documents"]
