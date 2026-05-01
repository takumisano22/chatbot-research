from __future__ import annotations

import pytest

from app.core import field_defaults as FD
from app.core.config import Settings
import app.rag.logic.chunking.chunking_logic_01 as ch
import app.rag.logic.reranking.reranking_logic_01 as rr
import app.rag.logic.search.search_logic_01 as s1
import app.rag.logic.search.search_logic_02 as s2
import app.rag.logic.tokenizer.tokenizer_logic_01 as tok
from app.rag.schemas import RetrievedChunk


def test_search_logic_01_calls_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_vec(
        settings: Settings, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        called["q"] = query
        called["k"] = top_k
        _ = settings
        return []

    monkeypatch.setattr(s1, "search_vector_chunks", fake_vec)
    settings = Settings.model_construct(vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER)
    s1.retrieve(settings, "hello", top_k=3)
    assert called["q"] == "hello"
    assert called["k"] == 3


def test_search_logic_02_returns_empty() -> None:
    settings = Settings.model_construct(vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER)
    assert s2.retrieve(settings, "x", top_k=4) == []


def test_tokenizer_logic_01_noop() -> None:
    assert tok.tokenize_query("  ") == []
    assert tok.tokenize_query("Hello") == ["hello"]
    assert tok.tokenize_query(" a B ") == ["a b"]


def test_chunking_logic_01_fixed_size() -> None:
    items = ch.split_for_rag_with_metadata(text="abcdefghij", chunk_size=4, chunk_overlap=0)
    assert [c["text"] for c in items] == ["abcd", "efgh", "ij"]
    assert all(c["metadata"] == {} for c in items)


def test_reranking_logic_01_identity() -> None:
    settings = Settings.model_construct(vector_store_provider=FD.DEFAULT_VECTOR_STORE_PROVIDER)
    c = RetrievedChunk(
        doc_id="d",
        chunk_id="c",
        source="s",
        chunk_text="t",
        keyword_score_raw=1.0,
        keyword_score_norm=1.0,
        vector_score_raw=None,
        vector_score_norm=None,
        final_score=1.0,
        retrieval_type="keyword",
    )
    chunks, k_eff = rr.rerank(settings, "q", [c], top_k=2)
    assert k_eff == 2
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c"


def test_preflight_import_search_logic_modules() -> None:
    from app.experiment.logic_registry import import_logic_module

    import_logic_module("search", "logic_01")
    import_logic_module("search", "logic_02")


def test_prompt_logic_01_system_message_non_empty() -> None:
    from app.experiment.logic_registry import load_rag_system_message

    text = load_rag_system_message("logic_01")
    assert "コンテキスト" in text
    assert text.strip()
