# 埋め込み API のリクエスト/レスポンス形のみを定義する。

from typing import Literal

from pydantic import BaseModel

EmbeddingInputType = Literal["query", "document", "raw"]


class EmbedRequest(BaseModel):
    texts: list[str]
    input_type: EmbeddingInputType = "document"
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    vectors: list[list[float]]
