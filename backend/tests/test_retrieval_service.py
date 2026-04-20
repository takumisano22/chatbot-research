# [bolt-005] retrieval_service: スコア・Min-Max・final_score（ストア行はモック）
from __future__ import annotations

import pytest

from app.core import field_defaults as FD
from app.core.config import Settings
from app.rag import retrieval_service as rs
from app.rag.logic import keyword_search as ks
from app.rag.logic import vector_search as vs_mod
from app.rag.schemas import RetrievedChunk


def test_search_documents_final_score_is_norm_times_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 上位候補が 1 件だけなら Min-Max で norm=1.0 → final = keyword_weight。
    def fake_rows(_settings: Settings, _tokens: list[str]) -> tuple[list, list, list]:
        return (
            ["only"],
            [
                {
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "source": "a.md",
                    "chunk_text": "Python の説明",
                },
            ],
            ["python の説明"],
        )

    monkeypatch.setattr(ks, "rag_load_keyword_rows", fake_rows)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_top_k=4,
        rag_keyword_weight=0.75,
    )
    out = rs.search_documents(
        settings, "python", top_k=4, rag_search_mode="keyword_search"
    )
    assert len(out) == 1
    assert out[0].keyword_score_norm == 1.0
    assert out[0].final_score == pytest.approx(0.75)
    assert out[0].retrieval_type == "keyword"
    assert out[0].vector_score_norm is None


def test_search_documents_min_max_orders_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2 件で raw が異なり、norm と final_score が設計どおりになる。
    def fake_rows(_settings: Settings, _tokens: list[str]) -> tuple[list, list, list]:
        return (
            ["hi", "lo"],
            [
                {
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "source": "a.md",
                    "chunk_text": "Python Python Java",
                },
                {
                    "doc_id": "d2",
                    "chunk_id": "c2",
                    "source": "b.md",
                    "chunk_text": "Java",
                },
            ],
            ["python python java", "java"],
        )

    monkeypatch.setattr(ks, "rag_load_keyword_rows", fake_rows)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_top_k=10,
        rag_keyword_weight=1.0,
    )
    out = rs.search_documents(
        settings, "python java", top_k=10, rag_search_mode="keyword_search"
    )
    assert len(out) == 2
    # raw 高い順
    assert out[0].chunk_id == "c1"
    assert out[0].keyword_score_norm == 1.0
    assert out[0].final_score == 1.0
    assert out[1].keyword_score_norm == 0.0
    assert out[1].final_score == 0.0


def test_search_documents_empty_tokens_returns_empty() -> None:
    # トークンが空のときはストアに接続せず空リスト（retrieval_service 先頭の早期 return）。
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_top_k=4,
        rag_keyword_weight=1.0,
    )
    assert (
        rs.search_documents(settings, "   ", top_k=4, rag_search_mode="keyword_search")
        == []
    )


def test_search_documents_vector_mode_empty_query_returns_empty() -> None:
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_vector_top_k=4,
    )
    assert (
        rs.search_documents(settings, "   ", top_k=4, rag_search_mode="vector_search")
        == []
    )


def test_search_documents_vector_mode_wires_embed_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeEmb:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            captured["texts"] = texts
            return [[0.5, 0.25]]

    def fake_rag_search_by_vector(
        _settings: Settings, qv: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        captured["qv"] = qv
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(vs_mod, "build_embedding_service", lambda _s: FakeEmb())
    monkeypatch.setattr(vs_mod, "rag_search_by_vector", fake_rag_search_by_vector)
    settings = Settings.model_construct(
        vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER,
        rag_vector_top_k=3,
    )
    out = rs.search_documents(settings, "hello", top_k=None, rag_search_mode="vector_search")
    assert out == []
    assert captured["texts"] == ["hello"]
    assert captured["qv"] == [0.5, 0.25]
    assert captured["top_k"] == 3
