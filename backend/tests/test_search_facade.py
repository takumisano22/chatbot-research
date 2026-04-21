from __future__ import annotations

import pytest

from app.core import field_defaults as FD
from app.core.config import Settings
import app.rag.logic.search as search_mod
from app.rag.logic.search import search_documents
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


def test_search_facade_hybrid_delegates_to_vector(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(search_mod, "search_vector_chunks", fake_vec)
    monkeypatch.setattr(search_mod, "search_keyword_chunks", fake_kw)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="vector_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    out = search_documents(settings, "q", top_k=4, rag_search_mode="hybrid_search")
    assert called["mode"] == "vector"
    assert len(out) == 1
    assert out[0].retrieval_type == "vector"


def test_search_facade_hybrid_delegates_to_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def fake_kw(
        _settings: Settings, _query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        called["mode"] = "keyword"
        _ = top_k
        return []

    monkeypatch.setattr(search_mod, "search_vector_chunks", lambda *a, **k: [])
    monkeypatch.setattr(search_mod, "search_keyword_chunks", fake_kw)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="keyword_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    search_documents(settings, "q", top_k=4, rag_search_mode="hybrid_search")
    assert called["mode"] == "keyword"


def test_search_facade_hybrid_empty_query() -> None:
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_hybrid_delegate="vector_search",
        rag_top_k=4,
        rag_vector_top_k=4,
    )
    assert search_documents(settings, "   ", top_k=4, rag_search_mode="hybrid_search") == []
