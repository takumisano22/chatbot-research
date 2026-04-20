# -----------------------------------------------------------------------------
# 役割: 会話 API の SSE 1 リクエストを Langfuse 上で 1 trace（ルート span）にまとめる。
# 下流の RAG / LLM 観測は同一の「現在の観測」コンテキスト配下で子 span としてぶら下がる。
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langfuse.types import TraceContext

from app.core.config import Settings
from app.langfuse.client import get_langfuse_client, safe_flush
from app.langfuse.metadata import build_common_metadata, truncate_for_observation
from app.langfuse.tracer import safe_span_update
from app.rag.schemas import RagSearchMode

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurnHandle:
    """ルート span へ最終応答を書き戻すための薄いハンドル（Langfuse 無効時は no-op）。"""

    _span: Any | None

    def set_assistant_output(self, text: str) -> None:
        if self._span is None:
            return
        safe_span_update(
            self._span,
            output={"assistant": truncate_for_observation(text)},
        )


@contextmanager
def observe_conversation_sse_turn(
    settings: Settings,
    conversation_id: UUID,
    user_content: str,
    *,
    use_rag: bool,
    rag_search_mode: RagSearchMode | None,
) -> Iterator[ConversationTurnHandle]:
    """
    POST .../conversations/{id}/chat の 1 往復を conversation.chat.turn として束ねる。
    ブロック内で rag.retrieval と llm.generation が同一 trace に載る。
    """
    client = get_langfuse_client(settings)
    if client is None:
        yield ConversationTurnHandle(None)
        return

    trace_id: str | None = None
    try:
        trace_id = client.create_trace_id()
    except Exception:
        logger.exception("Langfuse create_trace_id に失敗しました（trace_id なしで続行します）")

    meta = build_common_metadata(
        settings,
        use_rag=use_rag,
        rag_search_mode=rag_search_mode if use_rag else None,
    )
    meta["conversation_id"] = str(conversation_id)
    meta["chat_kind"] = "conversation_sse"
    if trace_id:
        meta["trace_id"] = trace_id

    try:
        cm = client.start_as_current_observation(
            name="conversation.chat.turn",
            as_type="span",
            input={
                "user_message": truncate_for_observation(user_content),
                "conversation_id": str(conversation_id),
            },
            metadata=meta,
            trace_context=TraceContext(trace_id=trace_id) if trace_id else None,
        )
    except Exception:
        logger.exception("Langfuse conversation.chat.turn（ルート）開始に失敗しました（観測をスキップします）")
        yield ConversationTurnHandle(None)
        return

    try:
        with cm as span:
            yield ConversationTurnHandle(span)
    finally:
        safe_flush(client)
