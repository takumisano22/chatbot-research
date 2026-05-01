from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse, urlunparse

import httpx

# -----------------------------------------------------------------------------
# 役割: Ruri 系の POST /embed を叩く（model は応答との照合用）。
# 流れ: build_ruri_http_embedding_service → RuriHttpEmbeddingService.embed_texts。
# Ruri v3 の検索精度に合わせ、文書・クエリの prefix は HTTP 送信前にここで付ける。
# -----------------------------------------------------------------------------

DOCUMENT_PREFIX = "文章: "
QUERY_PREFIX = "クエリ: "


EmbeddingInputType = Literal["document", "query", "raw"]


class EmbeddingService(Protocol):
    def embed_texts(
        self, texts: list[str], *, input_type: EmbeddingInputType = "document"
    ) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingParams:
    base_url: str = ""
    model: str = ""
    normalize: bool = True


class RuriHttpEmbeddingService:
    def __init__(
        self,
        *,
        base_url: str,
        expected_model: str,
        normalize: bool,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._expected_model = expected_model
        self._normalize = normalize
        self._timeout = timeout_seconds

    def embed_texts(
        self, texts: list[str], *, input_type: EmbeddingInputType = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embed"
        prefixed_texts = [_apply_input_prefix(text, input_type) for text in texts]
        payload: dict[str, Any] = {
            "texts": prefixed_texts,
            "input_type": "raw",
            "normalize": self._normalize,
        }
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        if self._expected_model and data.get("model") != self._expected_model:
            raise ValueError(
                "埋め込み API の model が設定と一致しません。"
                f" 期待: {self._expected_model!r} 実際: {data.get('model')!r}"
            )
        vectors = data.get("vectors")
        if not isinstance(vectors, list):
            raise ValueError("埋め込み API の応答に vectors がありません。")
        return vectors


def build_ruri_http_embedding_service(params: EmbeddingParams) -> EmbeddingService:
    if not (params.base_url or "").strip():
        raise ValueError("ruri_http では EMBEDDING_BASE_URL（base_url）が必要です。")
    if not (params.model or "").strip():
        raise ValueError("ruri_http では EMBEDDING_MODEL（model）が必要です（API 応答との照合用）。")
    resolved = _resolve_localhost_base_url_for_docker(params.base_url.strip())
    return RuriHttpEmbeddingService(
        base_url=resolved,
        expected_model=params.model.strip(),
        normalize=params.normalize,
        timeout_seconds=120.0,
    )


build_embedding_service = build_ruri_http_embedding_service

__all__ = [
    "EmbeddingParams",
    "EmbeddingService",
    "RuriHttpEmbeddingService",
    "build_embedding_service",
    "build_ruri_http_embedding_service",
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


def _apply_input_prefix(text: str, input_type: EmbeddingInputType) -> str:
    if input_type == "query":
        return f"{QUERY_PREFIX}{text}"
    if input_type == "document":
        return f"{DOCUMENT_PREFIX}{text}"
    return text
