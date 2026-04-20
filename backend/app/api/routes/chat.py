from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.rag.schemas import RagSearchMode
from app.services.chat_service import chat_turn_phase, rag_phase, run_chat
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 役割: ステートレスな単発チャット HTTP API（/api/v1/chat）。
# 主な呼び出し元: main がルーターをマウントし、クライアントが POST /chat を叩く。
# 流れ: 検証 → get_chat_model →（任意で rag_phase）→ chat_turn_phase → 応答本文。
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1)
    use_rag: bool = Field(False, description="true で RAG 検索のあと LLM 応答")
    rag_search_mode: RagSearchMode = Field(
        "vector_search",
        description="RAG 用: vector / keyword / hybrid",
    )


class ChatResponse(BaseModel):
    content: str


@router.post("/chat", response_model=ChatResponse)
def post_chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    try:
        llm = get_chat_model(settings)
    except ValueError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e

    conversation_id = uuid4()
    if body.use_rag:
        try:
            payload = rag_phase(
                settings=settings,
                user_content=body.content,
                conversation_id=conversation_id,
                rag_search_mode=body.rag_search_mode,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        payload = [{"role": "user", "content": body.content}]

    try:
        content = chat_turn_phase(
            llm=llm,
            settings=settings,
            conversation_id=conversation_id,
            user_content=body.content,
            payload=payload,
            use_rag=body.use_rag,
            rag_search_mode=body.rag_search_mode if body.use_rag else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ChatResponse(content=content)
