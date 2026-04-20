# -----------------------------------------------------------------------------
# 役割: Langfuse 観測の公開面（他モジュールはここまたは tracer / metadata の薄い API のみ利用する）。
# -----------------------------------------------------------------------------

from app.langfuse.client import get_langfuse_client, safe_flush
from app.langfuse.conversation_turn_trace import ConversationTurnHandle, observe_conversation_sse_turn
from app.langfuse.metadata import (
    build_common_metadata,
    summarize_retrieved_chunks,
    truncate_for_observation,
)
from app.langfuse.tracer import (
    observe_conversation_llm_generation,
    observe_conversation_rag_retrieval,
    observe_keyword_retrieval,
    observe_rag_ask,
    observe_rag_search_endpoint,
    observe_stateless_chat,
    observe_vector_store_query,
    observe_vector_query_embedding,
    safe_span_update,
)

__all__ = [
    "ConversationTurnHandle",
    "build_common_metadata",
    "get_langfuse_client",
    "observe_conversation_llm_generation",
    "observe_conversation_rag_retrieval",
    "observe_conversation_sse_turn",
    "observe_keyword_retrieval",
    "observe_rag_ask",
    "observe_rag_search_endpoint",
    "observe_stateless_chat",
    "observe_vector_store_query",
    "observe_vector_query_embedding",
    "safe_flush",
    "safe_span_update",
    "summarize_retrieved_chunks",
    "truncate_for_observation",
]
