from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.langfuse.tracer import observe_rag_ask, observe_rag_search_endpoint
from app.rag.prompts import RAG_NO_DOCUMENTS_REPLY, RAG_SYSTEM_MESSAGE, build_rag_user_message
from app.rag.retrieval_service import search_documents
from app.rag.schemas import RagSearchMode, RetrievedChunk
from app.services.chat_service import run_chat
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 役割: キーワード検索と、検索結果を用いた単発 RAG 回答の HTTP API。
# 主な呼び出し元: main がルーターをマウントし、クライアントが /api/v1/rag を利用する（会話側 RAG は別ルート）。
# 流れ: /search は search_documents（rag_search_mode 可）。/ask は検索 → プロンプト組み立て → run_chat。
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, description="検索クエリ")
    k: int | None = Field(
        None,
        ge=1,
        le=50,
        description="省略時はキーワード検索なら RAG_TOP_K、ベクトル検索なら RAG_VECTOR_TOP_K",
    )
    rag_search_mode: RagSearchMode = Field(
        "vector_search",
        description="vector / keyword / hybrid。省略時は vector_search。",
    )


class SearchResponse(BaseModel):
    chunks: list[RetrievedChunk]


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, description="質問（検索クエリ兼用）")
    k: int | None = Field(
        None,
        ge=1,
        le=50,
        description="省略時はキーワード検索なら RAG_TOP_K、ベクトル検索なら RAG_VECTOR_TOP_K",
    )
    rag_search_mode: RagSearchMode = Field(
        "vector_search",
        description="vector / keyword / hybrid。省略時は vector_search。",
    )


class AskResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]


@router.post("/search", response_model=SearchResponse)
def post_rag_search(
    body: SearchRequest, settings: Settings = Depends(get_settings)
) -> SearchResponse:
    chunks = observe_rag_search_endpoint(
        settings,
        body.q,
        body.k,
        body.rag_search_mode,
        lambda: search_documents(
            settings,
            body.q,
            top_k=body.k,
            rag_search_mode=body.rag_search_mode,
        ),
    )
    return SearchResponse(chunks=chunks)


@router.post("/ask", response_model=AskResponse)
def post_rag_ask(body: AskRequest, settings: Settings = Depends(get_settings)) -> AskResponse:
    # 会話履歴なし。永続化チャットの RAG は conversations ルートの use_rag。
    try:
        llm = get_chat_model(settings)
    except ValueError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e

    def _search() -> list[RetrievedChunk]:
        return search_documents(
            settings,
            body.q,
            top_k=body.k,
            rag_search_mode=body.rag_search_mode,
        )

    def _answer(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return RAG_NO_DOCUMENTS_REPLY
        user = build_rag_user_message(body.q, chunks)
        return run_chat(
            llm,
            [{"role": "system", "content": RAG_SYSTEM_MESSAGE}, {"role": "user", "content": user}],
        )

    answer, chunks = observe_rag_ask(
        settings,
        body.q,
        body.k,
        body.rag_search_mode,
        _search,
        _answer,
    )
    return AskResponse(answer=answer, chunks=chunks)
