# [bolt-004 PDF] アップロードバッチ（service）と multipart 空検証
import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.core.config import Settings, get_settings
from app.main import app
from app.rag.ingest_pipeline.processors.upload_multipart import collect_mixed_ordered_items
from app.rag.ingest_pipeline.service import run_queued_upload_batch


def _tiny_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    w.write(buf)
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    with patch("app.db.dispose_app_database", new_callable=AsyncMock):
        return TestClient(app)


def test_collect_mixed_ordered_items_empty_422() -> None:
    async def _run() -> None:
        await collect_mixed_ordered_items([], get_settings())

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_run())
    assert ei.value.status_code == 422


def test_run_queued_upload_batch_pdf_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
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
    results = run_queued_upload_batch(
        settings,
        [("one.pdf", pdf), ("two.pdf", pdf)],
    )
    get_settings.cache_clear()
    assert len(results) == 2
    assert all(r["ok"] for r in results)
    assert all(r["chunks_written"] >= 1 for r in results)
    sr = client.post(
        "/api/v1/rag/search",
        json={"q": "page", "k": 4, "rag_search_mode": "keyword_search"},
    )
    assert sr.status_code == 200
    assert len(sr.json()["chunks"]) >= 1
