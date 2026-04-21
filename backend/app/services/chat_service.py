from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from uuid import UUID

from app.core.config import Settings
from app.experiment.logic_registry import load_rag_system_message
from app.langfuse.tracer import observe_conversation_llm_generation, observe_conversation_rag_retrieval
from app.rag.prompts import build_rag_user_message
from app.rag.retrieval_service import search_documents
from app.rag.schemas import RagSearchMode
# -----------------------------------------------------------------------------
# 役割: LangChain ChatModel への 1 往復チャット（メッセージ dict から応答本文を取り出す）。
# 主な呼び出し元: API ルート（chat / rag）、conversation_chat_service。
# 流れ: run_chat → messages_from_payload → invoke → content を str に正規化。
# -----------------------------------------------------------------------------


def run_chat(llm: BaseChatModel, messages: list[dict[str, str]]) -> str:
    lc_messages = messages_from_payload(messages)
    result = llm.invoke(lc_messages)
    if isinstance(result, AIMessage):
        text = result.content
    else:
        text = getattr(result, "content", str(result))
    # content がブロックの list になり得るため連結する。
    if isinstance(text, list):
        return "".join(str(part) for part in text)
    return str(text)


# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------

_ROLE_MAP: dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


def messages_from_payload(messages: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "").lower().strip()
        content = m.get("content", "")
        if role not in _ROLE_MAP:
            raise ValueError(f"Unknown message role: {m.get('role')!r}")
        out.append(_ROLE_MAP[role](content=content))
    return out

def _effective_rag_top_k(settings: Settings, mode: RagSearchMode) -> int:
    if mode == "keyword_search":
        return settings.rag_top_k
    if mode == "hybrid_search":
        return max(settings.rag_top_k, settings.rag_vector_top_k)
    return settings.rag_vector_top_k


def chat_turn_phase(
    llm: BaseChatModel,
    settings: Settings,
    conversation_id: UUID,
    user_content: str,
    payload: list[dict[str, str]],
    *,
    use_rag: bool = False,
    rag_search_mode: RagSearchMode | None = None,
) -> str:
    assistant_text = observe_conversation_llm_generation(
        settings,
        conversation_id,
        user_content,
        payload,
        lambda: run_chat(llm, payload),
        use_rag=use_rag,
        rag_search_mode=rag_search_mode,
    )
    return assistant_text


def rag_phase(
    settings: Settings,
    user_content: str,
    *,
    conversation_id: UUID,
    rag_search_mode: RagSearchMode = "vector_search",
    rag_pipeline_id: str = "http_chat",
) -> list[dict[str, str]]:
    top_k_effective = _effective_rag_top_k(settings, rag_search_mode)
    chunks = observe_conversation_rag_retrieval(
        settings,
        conversation_id,
        user_content,
        rag_search_mode,
        rag_pipeline_id,
        lambda: search_documents(
            settings,
            user_content,
            top_k=None,
            rag_search_mode=rag_search_mode,
        ),
        top_k_effective=top_k_effective,
    )

    augmented = build_rag_user_message(user_content, chunks)
    system_message = load_rag_system_message(settings.rag_prompt_logic_id)

    payload = (
        [{"role": "system", "content": system_message}]
        + [{"role": "user", "content": augmented}]
    )
    return payload
