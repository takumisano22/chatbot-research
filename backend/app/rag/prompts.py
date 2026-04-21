from __future__ import annotations

from app.experiment.logic_registry import load_rag_system_message
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RAG 用の定型返答と、検索チャンクからユーザー文を組み立てる。システム文は prompt ロジックから解決する。
# 主な呼び出し元: rag / conversations ルート、conversation_chat_service、experiment batch_runner。
# 流れ: rag_system_message_for_logic → load_rag_system_message / build_rag_user_message → chunks_to_context_lines。
# -----------------------------------------------------------------------------

RAG_NO_DOCUMENTS_REPLY = "参照ドキュメントから該当箇所が見つかりませんでした。"


def rag_system_message_for_logic(logic_id: str) -> str:
    return load_rag_system_message(logic_id)


RAG_SYSTEM_MESSAGE = rag_system_message_for_logic("logic_01")


def build_rag_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context = chunks_to_context_lines(chunks)
    return f"コンテキスト:\n{context}\n\n質問:\n{question}"


# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------


def chunks_to_context_lines(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for c in chunks:
        blocks.append(f"---\n出典: {c.source}\n{c.chunk_text}\n")
    return "\n".join(blocks)
