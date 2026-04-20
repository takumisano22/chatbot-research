from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse, urlunparse

from langchain_ollama import OllamaEmbeddings

# -----------------------------------------------------------------------------
# 役割: Ollama の Embeddings を LangChain 経由で呼ぶ（backend は base_url / model を渡す）。
# 流れ: build_ollama_embedding_service → OllamaEmbeddingService.embed_texts。
# -----------------------------------------------------------------------------


class EmbeddingService(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingParams:
    base_url: str = ""
    model: str = ""
    normalize: bool = True


class OllamaEmbeddingService:
    def __init__(self, embeddings: OllamaEmbeddings) -> None:
        self._embeddings = embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


def build_ollama_embedding_service(params: EmbeddingParams) -> EmbeddingService:
    if not (params.base_url or "").strip():
        raise ValueError("ollama では EMBEDDING_BASE_URL（base_url）が必要です。")
    if not (params.model or "").strip():
        raise ValueError("ollama では EMBEDDING_MODEL（model）が必要です。")
    base_url = _resolve_localhost_base_url_for_docker(params.base_url.strip())
    embeddings = OllamaEmbeddings(
        base_url=base_url.rstrip("/"),
        model=params.model.strip(),
    )
    return OllamaEmbeddingService(embeddings)


build_embedding_service = build_ollama_embedding_service

__all__ = [
    "EmbeddingParams",
    "EmbeddingService",
    "OllamaEmbeddingService",
    "build_embedding_service",
    "build_ollama_embedding_service",
]


# -----------------------------------------------------------------------------
# 補助（Docker 内から localhost を叩く場合の補正）
# -----------------------------------------------------------------------------


def _resolve_localhost_base_url_for_docker(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return base_url
    if not Path("/.dockerenv").exists():
        return base_url
    host = "host.docker.internal"
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))
