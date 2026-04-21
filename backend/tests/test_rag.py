# RAG 取り込みとキーワード検索のスモーク（HTTP なし）
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.rag.ingest_pipeline.runner import ingest_plain_text
from app.rag.retrieval_service import search_documents
from app.rag.vectorstore.chunker import to_repo_relative
from app.rag.vectorstore.vector_db import rag_write_session


def _ingest_txt_md_tree(settings: Settings, source_root: Path, repo_root: Path) -> tuple[int, int]:
    if not source_root.is_dir():
        return 0, 0
    paths = sorted(
        p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
    session = rag_write_session(settings)
    total_chunks = 0
    for fp in paths:
        rel = to_repo_relative(fp, repo_root)
        text = fp.read_text(encoding="utf-8")
        total_chunks += ingest_plain_text(settings, session, rel, text)
    return len(paths), total_chunks


@pytest.fixture
def rag_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    uid = uuid.uuid4().hex[:8]
    monkeypatch.setenv("VECTOR_STORE_SERVER_HOST", "")
    monkeypatch.setenv("VECTOR_STORE_PERSIST_DIR", str(tmp_path / "vs_data"))
    monkeypatch.setenv("RAG_COLLECTION_NAME", f"rag_test_{uid}")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_ingest_and_keyword_search(rag_env: Path) -> None:
    src = rag_env / "docs_in"
    src.mkdir()
    (src / "note.md").write_text(
        "Python はプログラミング言語です。RAG のテスト用の文です。",
        encoding="utf-8",
    )
    settings = get_settings()
    n_files, n_chunks = _ingest_txt_md_tree(settings, src, rag_env)
    assert n_files == 1
    assert n_chunks >= 1

    chunks = search_documents(settings, "Python", top_k=4, rag_search_mode="keyword_search")
    assert len(chunks) >= 1
    row = chunks[0]
    assert row.retrieval_type == "keyword"
    assert row.vector_score_norm is None
    assert row.keyword_score_norm is not None


def test_rag_search_japanese_sentence_query_hits(rag_env: Path) -> None:
    src = rag_env / "docs_in_ja"
    src.mkdir()
    (src / "note.md").write_text(
        "この資料は社内手続きの説明です。申請方法と承認フローを記載します。",
        encoding="utf-8",
    )
    settings = get_settings()
    n_files, n_chunks = _ingest_txt_md_tree(settings, src, rag_env)
    assert n_files == 1
    assert n_chunks >= 1

    chunks = search_documents(
        settings,
        "申請方法と承認フロー",
        top_k=4,
        rag_search_mode="keyword_search",
    )
    assert len(chunks) >= 1


def test_vector_store_provider_must_be_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "opensearch")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings()
    get_settings.cache_clear()
