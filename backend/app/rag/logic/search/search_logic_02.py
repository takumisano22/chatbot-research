from __future__ import annotations

from app.core.config import Settings
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: SEARCH logic_02 — 検索なし（常に空）。コンテキスト無しの応答パス用。
# -----------------------------------------------------------------------------


def retrieve(
    settings: Settings,
    query: str,
    *,
    top_k: int | None,
) -> list[RetrievedChunk]:
    _ = (settings, query, top_k)
    return []
