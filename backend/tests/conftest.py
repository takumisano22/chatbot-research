# pytest 共通: RAG 取り込み系で Ollama 実体に依存せずベクトル書き込みが完走するよう埋め込みを固定次元のダミーに差し替える。
from __future__ import annotations

import pytest

from app.rag.vectorstore import vector_db as vdb

_OFFLINE_EMBEDDING_DIM = 768


@pytest.fixture(autouse=True)
def _offline_embedding_for_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_embedding"):
        return

    class _FakeEmbedding:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.01] * _OFFLINE_EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(
        vdb,
        "build_embedding_service",
        lambda _settings: _FakeEmbedding(),
    )
