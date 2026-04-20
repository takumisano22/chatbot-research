from __future__ import annotations

from app.core.config import Settings
from app.langfuse.tracer import observe_vector_query_embedding
from app.rag.logic.embedding import EmbeddingService, build_embedding_service
from app.rag.schemas import RetrievedChunk
from app.rag.vectorstore.vector_db import rag_search_by_vector

# -----------------------------------------------------------------------------
# 役割: クエリを埋め込み、vectordb アダプタ経由で近傍チャンクを取得する（ストア実装は backend 外）。
# 主な呼び出し元: rag.retrieval_service（vector モード）、rag.logic.hybrid_search。
# 流れ: embed_texts → rag_search_by_vector → list[RetrievedChunk]。
# -----------------------------------------------------------------------------


def search_vector_chunks(
    settings: Settings,
    query: str,
    top_k: int | None = None,
    embedding_service: EmbeddingService | None = None,
) -> list[RetrievedChunk]:
    if not query.strip():
        return []

    k = settings.rag_vector_top_k if top_k is None else top_k
    service = embedding_service or build_embedding_service(settings)
    query_vector = observe_vector_query_embedding(
        settings,
        query,
        lambda: service.embed_texts([query])[0],
    )
    return rag_search_by_vector(settings, query_vector, k)
