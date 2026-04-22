from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.core.config import Settings
from app.langfuse.client import get_langfuse_client, safe_flush
from app.langfuse.metadata import (
    build_common_metadata,
    summarize_retrieved_chunks,
    truncate_for_observation,
)
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: 他層から呼ぶ Langfuse 観測の薄い API（SDK の import はこのモジュールと client に閉じる）。
# 例外は握りつぶし、業務処理の例外は再送出しない（観測の try が業務を汚染しない）。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def safe_span_update(span: Any, **kwargs: Any) -> None:
    if span is None:
        return
    try:
        span.update(**kwargs)
    except Exception:
        logger.exception("Langfuse span.update に失敗しました（無視します）")


def observe_vector_query_embedding(
    settings: Settings,
    query_preview: str,
    run: Callable[[], list[float]],
) -> list[float]:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True)
    try:
        cm = client.start_as_current_observation(
            name="rag.embed_query",
            as_type="embedding",
            input={"query": truncate_for_observation(query_preview)},
            metadata={
                **meta,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
            },
            model=settings.embedding_model,
        )
    except Exception:
        logger.exception("Langfuse rag.embed_query に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span:
            vec = run()
            safe_span_update(
                span,
                output={"vector_dim": len(vec), "vector_sent": False},
            )
            return vec
    finally:
        safe_flush(client)


def observe_vector_store_query(
    settings: Settings,
    top_k: int,
    run: Callable[[], list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True)
    try:
        cm = client.start_as_current_observation(
            name="rag.vector_store.query",
            as_type="retriever",
            input={"top_k": top_k},
            metadata={
                **meta,
                "vector_store_provider": settings.vector_store_provider,
                "vector_db_adapter_subpackage": settings.vector_db_adapter_subpackage,
                "rag_collection_name": settings.rag_collection_name,
            },
        )
    except Exception:
        logger.exception("Langfuse rag.vector_store.query に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span:
            chunks = run()
            safe_span_update(span, output=summarize_retrieved_chunks(chunks))
            return chunks
    finally:
        safe_flush(client)


def observe_llm_chat_turn(
    settings: Settings,
    messages: list[dict[str, str]],
    run: Callable[[], str],
) -> str:
    """実験バッチの 1 往復チャットを Langfuse に載せる。無効時は run のみ（オーバーヘッド最小）。"""
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True)
    input_payload = {
        "messages": [
            {
                "role": (m.get("role") or ""),
                "content": truncate_for_observation(str(m.get("content", ""))),
            }
            for m in messages
        ]
    }
    try:
        cm = client.start_as_current_observation(
            name="rag.llm_chat",
            as_type="generation",
            input=input_payload,
            metadata={**meta, "llm_model": settings.llm_model},
            model=settings.llm_model,
        )
    except Exception:
        logger.exception("Langfuse rag.llm_chat に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span:
            text = run()
            safe_span_update(span, output={"answer": truncate_for_observation(text)})
            return text
    finally:
        safe_flush(client)


def observe_keyword_retrieval(
    settings: Settings,
    query: str,
    top_k: int,
    run: Callable[[], list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True)
    try:
        cm = client.start_as_current_observation(
            name="rag.keyword_retrieval",
            as_type="retriever",
            input={"query": truncate_for_observation(query), "top_k": top_k},
            metadata=meta,
        )
    except Exception:
        logger.exception("Langfuse rag.keyword_retrieval に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span:
            chunks = run()
            safe_span_update(span, output=summarize_retrieved_chunks(chunks))
            return chunks
    finally:
        safe_flush(client)
