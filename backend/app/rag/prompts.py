from __future__ import annotations

from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RAG 用の固定システム文・定型返答と、検索チャンクからユーザー文を組み立てる。
# 主な呼び出し元: rag / conversations ルート、conversation_chat_service。
# 流れ: build_rag_user_message → chunks_to_context_lines で出典付きコンテキストを連結。
# -----------------------------------------------------------------------------

RAG_SYSTEM_MESSAGE = (
"""
##役割
-あなたは**与えられたコンテキストのみを根拠に回答するアシスタント**です。
    
##回答ルール
-正確かつ簡潔な回答をしてください。
-推測で補わないでください。
-回答には出典を簡潔に示してください。
#重要なルール
-**コンテキストに情報がない場合は、分からないと答えてください。**    
"""
)

RAG_NO_DOCUMENTS_REPLY = "参照ドキュメントから該当箇所が見つかりませんでした。"


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
