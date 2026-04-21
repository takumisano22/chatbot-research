# PDF バッチ取り込み（ingest_batch）とキーワード検索
from __future__ import annotations

import io
from pathlib import Path
import pytest
from pypdf import PdfWriter

from app.core.config import Settings, get_settings
from app.rag.ingest_batch import run_upload_items_batch
from app.rag.logic.keyword_search import search_keyword_chunks


def _tiny_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    w.write(buf)
    return buf.getvalue()


def test_run_upload_items_batch_pdf_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE_SERVER_HOST", "")
    monkeypatch.setenv("VECTOR_STORE_PERSIST_DIR", str(tmp_path / "vs_data"))
    monkeypatch.setenv("RAG_COLLECTION_NAME", "rag_pdf_test")

    def _fake_convert_pdf_bytes(data: bytes, *, settings: Settings) -> str:
        _ = (data, settings)
        return "## Page 1\n\npage marker for rag pdf test\n"

    monkeypatch.setattr(
        "app.rag.ingest_pipeline.registry.convert_markit_pdf_bytes",
        _fake_convert_pdf_bytes,
    )
    get_settings.cache_clear()
    pdf = _tiny_pdf_bytes()
    settings = get_settings()
    results = run_upload_items_batch(
        settings,
        [("one.pdf", pdf), ("two.pdf", pdf)],
    )
    get_settings.cache_clear()
    assert len(results) == 2
    assert all(r["ok"] for r in results)
    assert all(r["chunks_written"] >= 1 for r in results)

    chunks = search_keyword_chunks(
        settings,
        "page",
        top_k=4,
    )
    assert len(chunks) >= 1
