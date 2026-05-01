from __future__ import annotations

from typing import Literal, Protocol, cast

from app.core.adapters import load_embedding_provider_adapter
from app.core.config import Settings

# -----------------------------------------------------------------------------
# 役割: ベクトル化（Embedding）を provider 依存から分離し、差し替え境界を固定する。
# 主な呼び出し元: rag.logic.vector_search（将来は ingest 時の埋め込み作成でも利用）。
# 流れ: build_embedding_service で実装を選び、embed_texts 経由でベクトルを取得する。
# 各プロバイダの EmbeddingParams（同一フィールド）へ base_url / model を渡す（normalize は dataclass 既定）。
# -----------------------------------------------------------------------------


EmbeddingInputType = Literal["document", "query", "raw"]


class EmbeddingService(Protocol):
    def embed_texts(
        self, texts: list[str], *, input_type: EmbeddingInputType = "document"
    ) -> list[list[float]]:
        ...


def build_embedding_service(settings: Settings) -> EmbeddingService:
    provider = settings.embedding_provider.strip().lower()
    adapter = load_embedding_provider_adapter(provider)
    params = adapter.EmbeddingParams(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    service = adapter.build_embedding_service(params)
    return cast(EmbeddingService, service)
