from __future__ import annotations

from app.core.config import Settings
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RERANKING logic_01 — no-op（入力チャンクと effective_top_k をそのまま返す）。
# -----------------------------------------------------------------------------


def rerank(
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> tuple[list[RetrievedChunk], int]:
    _ = (settings, query)
    return list(chunks), top_k
