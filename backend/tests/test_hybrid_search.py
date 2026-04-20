from __future__ import annotations

import pytest

from app.core import field_defaults as FD
from app.core.config import Settings
from app.rag.logic import hybrid_search as hs
from app.rag.schemas import RetrievedChunk


def _vec(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="d1",
        chunk_id=chunk_id,
        source="a.md",
        chunk_text="v",
        keyword_score_raw=0.0,
        keyword_score_norm=0.0,
        vector_score_raw=0.1,
        vector_score_norm=1.0,
        final_score=1.0,
        retrieval_type="vector",
    )


def test_hybrid_facade_delegates_to_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def fake_vec(
        _settings: Settings, _query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        called["mode"] = "vector"
        _ = top_k
        return [_vec("c1")]

    def fake_kw(*_a: object, **_k: object) -> list[RetrievedChunk]:
        called["mode"] = "keyword"
        return []

    monkeypatch.setattr(hs, "search_vector_chunks", fake_vec)
    monkeypatch.setattr(hs, "search_keyword_chunks", fake_kw)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="vector_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    out = hs.search_hybrid_chunks(settings, "q", top_k=4)
    assert called["mode"] == "vector"
    assert len(out) == 1
    assert out[0].retrieval_type == "vector"


def test_hybrid_facade_delegates_to_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def fake_kw(
        _settings: Settings, _query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        called["mode"] = "keyword"
        _ = top_k
        return []

    monkeypatch.setattr(hs, "search_vector_chunks", lambda *a, **k: [])
    monkeypatch.setattr(hs, "search_keyword_chunks", fake_kw)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="keyword_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    hs.search_hybrid_chunks(settings, "q", top_k=4)
    assert called["mode"] == "keyword"


def test_search_hybrid_empty_query() -> None:
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="vector_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    assert hs.search_hybrid_chunks(settings, "   ") == []
