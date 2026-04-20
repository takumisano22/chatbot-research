from __future__ import annotations

import pytest

from app.core import adapters
from app.core.config import Settings
from app.rag.logic.embedding import build_embedding_service


def _embedding_settings(
    *,
    provider: str = "ollama",
    base_url: str = "http://127.0.0.1:11434",
    model: str = "nomic-embed-text",
) -> Settings:
    return Settings.model_construct(
        embedding_provider=provider,
        embedding_base_url=base_url,
        embedding_model=model,
    )


def test_load_embedding_provider_adapter_ollama() -> None:
    class FakeEmbeddingAdapter:
        EmbeddingParams = object

        @staticmethod
        def build_embedding_service(_params: object) -> object:
            return object()

    adapters.load_embedding_provider_adapter.cache_clear()
    original = adapters.importlib.import_module
    adapters.importlib.import_module = lambda _: FakeEmbeddingAdapter  # type: ignore[assignment]
    try:
        adapter = adapters.load_embedding_provider_adapter("ollama")
    finally:
        adapters.importlib.import_module = original  # type: ignore[assignment]
        adapters.load_embedding_provider_adapter.cache_clear()
    assert adapter is FakeEmbeddingAdapter


def test_build_embedding_service_switches_model(monkeypatch) -> None:
    captured_models: list[str] = []

    class FakeParams:
        def __init__(self, *, base_url: str = "", model: str = "", **_: object) -> None:
            self.base_url = base_url
            self.model = model

    class FakeService:
        def __init__(self, model: str) -> None:
            self._model = model

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            value = 1.0 if self._model == "nomic-embed-text" else 2.0
            return [[value] for _ in texts]

    class FakeAdapter:
        EmbeddingParams = FakeParams

        @staticmethod
        def build_embedding_service(params: FakeParams) -> FakeService:
            assert params.base_url == "http://127.0.0.1:11434"
            captured_models.append(params.model)
            return FakeService(params.model)

    monkeypatch.setattr("app.rag.logic.embedding.load_embedding_provider_adapter", lambda _: FakeAdapter)

    service_a = build_embedding_service(_embedding_settings(model="nomic-embed-text"))
    service_b = build_embedding_service(_embedding_settings(model="ruri-v3"))

    assert service_a.embed_texts(["hello"]) == [[1.0]]
    assert service_b.embed_texts(["hello"]) == [[2.0]]
    assert captured_models == ["nomic-embed-text", "ruri-v3"]


def test_build_embedding_service_unsupported_provider() -> None:
    settings = _embedding_settings(provider="openai")
    with pytest.raises(ValueError, match="読み込めませんでした"):
        build_embedding_service(settings)
