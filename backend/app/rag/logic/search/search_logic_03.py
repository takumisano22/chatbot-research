from __future__ import annotations

from app.core.config import Settings
from app.rag.logic.vector_search import search_vector_chunks
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: SEARCH logic_03 - 1 論理チャンクに複数の検索用ベクトルがある前提の検索。
# 保存時に vector_texts が物理レコードへ展開されるため、通常の Chroma TopK で
# full/local どちらのベクトルも同じ候補集合に入る。既存コレクションでは logic_01 と同じ挙動。
# -----------------------------------------------------------------------------


def retrieve(
    settings: Settings,
    query: str,
    *,
    top_k: int | None,
) -> list[RetrievedChunk]:
    return search_vector_chunks(settings, query, top_k=top_k)
