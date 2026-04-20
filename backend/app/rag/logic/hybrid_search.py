from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.keyword_search import search_keyword_chunks
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: hybrid_search モードの窓口。実際の統合ロジックは後から差し替え可能。
# 主な呼び出し元: rag.retrieval_service（mode=hybrid_search）。
# 流れ: Settings.rag_hybrid_delegate に応じて keyword または vector へ委譲（将来ここを本番 hybrid に置換）。
# 要点: 実験 manifest では rag_search_mode で vector/keyword/hybrid を切替え、hybrid 時は rag_hybrid_delegate で委譲先を指定。
# -----------------------------------------------------------------------------


def search_hybrid_chunks(
    settings: Settings,
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    if not query.strip():
        return []
    if settings.rag_hybrid_delegate == "keyword_search":
        return search_keyword_chunks(settings, query, top_k=top_k)
    return search_vector_chunks(settings, query, top_k=top_k)
