from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.hybrid_search import search_hybrid_chunks
from app.rag.logic.keyword_search import search_keyword_chunks
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RagSearchMode, RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RAG 検索の入口。検索方式の詳細は logic 層へ委譲し、本モジュールは窓口に徹する。
# 主な呼び出し元: rag ルート、conversation_chat_service。
# 流れ: rag_search_mode に応じて vector / keyword / hybrid（内部で振り分け）を呼び分ける。
# -----------------------------------------------------------------------------


def search_documents(
    settings: Settings,
    query: str,
    top_k: int | None = None,
    *,
    rag_search_mode: RagSearchMode = "vector_search",
) -> list[RetrievedChunk]:
    if rag_search_mode == "vector_search":
        return search_vector_chunks(settings=settings, query=query, top_k=top_k)
    if rag_search_mode == "keyword_search":
        return search_keyword_chunks(settings=settings, query=query, top_k=top_k)
    return search_hybrid_chunks(settings=settings, query=query, top_k=top_k)
