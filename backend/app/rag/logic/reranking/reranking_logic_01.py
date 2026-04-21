from __future__ import annotations

from app.core.config import Settings
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RERANKING logic_01 — no-op（入力チャンクをそのまま返す）。
# -----------------------------------------------------------------------------


def rerank(
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    _ = (settings, query)
    return list(chunks)
