from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langfuse.types import TraceContext

from app.core.config import Settings
from app.langfuse.client import get_langfuse_client, safe_flush
from app.langfuse.metadata import (
    build_common_metadata,
    stateless_chat_input_summary,
    summarize_retrieved_chunks,
    truncate_for_observation,
)
from app.rag.schemas import RagSearchMode, RetrievedChunk

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


## langfuse:チャット開始時にメッセージを取得して観測を行う。
def observe_stateless_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    run: Callable[[], str], ## 最終的にはrun_chat()をコールバックする。
) -> str:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=False)
    meta["chat_kind"] = "stateless"
    try:
        cm = client.start_as_current_observation( ## 親観測単位の作成、cmはcontext managerの意らしい（通例？）
            name="chat.turn",
            as_type="span", ## langfuseの観測単位の種類。チャット全体なのでspan。
            input=stateless_chat_input_summary(messages),
            metadata=meta,
        )
    except Exception:
        logger.exception("Langfuse chat.turn 開始に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span: ## チャット観測単位にコールバックの実行結果を挿入
            out = run()
            safe_span_update(span, output={"assistant": truncate_for_observation(out)})
            return out
    finally:
        safe_flush(client)


def observe_conversation_llm_generation(
    settings: Settings,
    conversation_id: UUID,
    user_content: str,
    payload: list[dict[str, str]],
    run: Callable[[], str],
    *,
    use_rag: bool,
    rag_search_mode: RagSearchMode | None,
) -> str:
    """
    会話チャットの最終 LLM 呼び出しを generation として記録する。
    親の conversation.chat.turn（SSE ルート）配下で呼ぶこと。usage は LangChain 側の取り出しは未配線。
    """
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(
        settings,
        use_rag=use_rag,
        rag_search_mode=rag_search_mode if use_rag else None,
    )
    meta["conversation_id"] = str(conversation_id)
    meta["chat_kind"] = "conversation_sse"
    meta["generation_model"] = settings.llm_model
    last_preview = truncate_for_observation(
        (payload[-1].get("content", "") if payload else "") or "",
        max_chars=800,
    )
    gen_input = {
        "user_message": truncate_for_observation(user_content),
        "conversation_id": str(conversation_id),
        "message_count": len(payload),
        "last_message_preview": last_preview,
    }
    try:
        cm = client.start_as_current_observation(
            name="llm.generation",
            as_type="generation",
            model=settings.llm_model,
            input=gen_input,
            metadata=meta,
        )
    except Exception:
        logger.exception("Langfuse llm.generation 開始に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as gen:
            out = run()
            safe_span_update(gen, output=truncate_for_observation(out))
            return out
    finally:
        safe_flush(client)


def observe_conversation_rag_retrieval(
    settings: Settings,
    conversation_id: UUID,
    user_content: str,
    rag_search_mode: RagSearchMode,
    rag_pipeline_id: str,
    run: Callable[[], list[RetrievedChunk]],
    *,
    top_k_effective: int,
) -> list[RetrievedChunk]:
    """
    RAG 検索のみを span として記録する。ルートの conversation.chat.turn 配下で呼ぶ。
    """
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True, rag_search_mode=rag_search_mode)
    meta["conversation_id"] = str(conversation_id)
    meta["rag_pipeline_id"] = rag_pipeline_id
    meta["top_k"] = top_k_effective
    try:
        cm = client.start_as_current_observation(
            name="rag.retrieval",
            as_type="span",
            input={
                "query": truncate_for_observation(user_content),
                "rag_search_mode": rag_search_mode,
                "top_k": top_k_effective,
            },
            metadata=meta,
        )
    except Exception:
        logger.exception("Langfuse rag.retrieval 開始に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as search_span:
            chunks = run()
            summary = summarize_retrieved_chunks(chunks)
            safe_span_update(
                search_span,
                output=summary,
                metadata={**meta, "retrieved_count": summary.get("retrieved_count", len(chunks))},
            )
            return chunks
    finally:
        safe_flush(client)


def observe_rag_search_endpoint(
    settings: Settings,
    query: str,
    top_k: int | None,
    rag_search_mode: RagSearchMode,
    run: Callable[[], list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True, rag_search_mode=rag_search_mode)
    meta["endpoint"] = "POST /api/v1/rag/search"
    try:
        cm = client.start_as_current_observation(
            name="rag.search",
            as_type="span",
            input={
                "query": truncate_for_observation(query),
                "top_k": top_k,
                "rag_search_mode": rag_search_mode,
            },
            metadata=meta,
        )
    except Exception:
        logger.exception("Langfuse rag.search（endpoint）に失敗しました（観測をスキップします）")
        return run()
    try:
        with cm as span:
            chunks = run()
            safe_span_update(span, output=summarize_retrieved_chunks(chunks))
            return chunks
    finally:
        safe_flush(client)


def observe_rag_ask(
    settings: Settings,
    query: str,
    top_k: int | None,
    rag_search_mode: RagSearchMode,
    run_search: Callable[[], list[RetrievedChunk]],
    run_answer: Callable[[list[RetrievedChunk]], str],
) -> tuple[str, list[RetrievedChunk]]:
    client = get_langfuse_client(settings)
    if client is None:
        chunks = run_search()
        return run_answer(chunks), chunks
    try:
        trace_id = client.create_trace_id()
    except Exception:
        trace_id = None
    meta = build_common_metadata(settings, use_rag=True, rag_search_mode=rag_search_mode)
    meta["endpoint"] = "POST /api/v1/rag/ask"
    if trace_id:
        meta["rag_ask_trace_id"] = trace_id
    try:
        cm_root = client.start_as_current_observation(
            name="rag.ask.turn",
            as_type="span",
            input={"query": truncate_for_observation(query)},
            metadata=meta,
            trace_context=TraceContext(trace_id=trace_id) if trace_id else None,
        )
    except Exception:
        logger.exception("Langfuse rag.ask.turn に失敗しました（観測をスキップします）")
        chunks = run_search()
        return run_answer(chunks), chunks
    try:
        with cm_root:
            try:
                cm_search = client.start_as_current_observation(
                    name="rag.search",
                    as_type="span",
                    input={
                        "query": truncate_for_observation(query),
                        "top_k": top_k,
                        "rag_search_mode": rag_search_mode,
                    },
                    metadata=meta,
                )
            except Exception:
                logger.exception("Langfuse rag.search（ask）に失敗しました（観測をスキップします）")
                chunks = run_search()
                return run_answer(chunks), chunks
            with cm_search as sspan:
                chunks = run_search()
                safe_span_update(sspan, output=summarize_retrieved_chunks(chunks))
            try:
                cm_comp = client.start_as_current_observation(
                    name="rag.complete",
                    as_type="generation",
                    model=settings.llm_model,
                    metadata=meta,
                )
            except Exception:
                logger.exception("Langfuse rag.complete（ask）に失敗しました（観測をスキップします）")
                return run_answer(chunks), chunks
            with cm_comp as gen:
                answer = run_answer(chunks)
                safe_span_update(gen, output=truncate_for_observation(answer))
                return answer, chunks
    finally:
        safe_flush(client)


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


def observe_keyword_retrieval(
    settings: Settings,
    query: str,
    top_k: int,
    run: Callable[[], list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    client = get_langfuse_client(settings)
    if client is None:
        return run()
    meta = build_common_metadata(settings, use_rag=True, rag_search_mode="keyword_search")
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
