from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# -----------------------------------------------------------------------------
# 役割: LangChain ChatModel への 1 往復チャット（メッセージ dict から応答本文を取り出す）。
# 主な呼び出し元: experiment batch_runner（LLM 応答生成）。
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
