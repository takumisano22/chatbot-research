# RAG 取り込みジョブ API（PDF / txt/md の DB キュー + ワーカー 1 回処理）
import asyncio
import io
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_db_session
from app.main import app
from app.models import Base
from app.core.config import Settings, get_settings
from app.db import reset_engine_for_tests
from app.rag.ingest_pipeline.jobs import process_next_job_once


def _tiny_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    w.write(buf)
    return buf.getvalue()


@pytest.fixture
def client_no_db() -> TestClient:
    with patch("app.db.dispose_app_database", new_callable=AsyncMock):
        return TestClient(app)


@pytest.fixture
def client_with_db() -> tuple[TestClient, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    async def init_schema() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_schema())
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    with patch("app.db.dispose_app_database", new_callable=AsyncMock):
        yield TestClient(app), factory
    app.dependency_overrides.clear()

    async def shutdown() -> None:
        await engine.dispose()

    asyncio.run(shutdown())


def test_ingest_jobs_requires_database(
    client_no_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    get_settings.cache_clear()
    reset_engine_for_tests()
    try:
        r = client_no_db.post(
            "/api/v1/rag/ingest/jobs",
            files=[("files", ("a.pdf", _tiny_pdf_bytes(), "application/pdf"))],
        )
        assert r.status_code == 501
    finally:
        get_settings.cache_clear()
        reset_engine_for_tests()


def test_ingest_jobs_get_unknown_404(
    client_with_db: tuple[TestClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _factory = client_with_db
    r = client.get(f"/api/v1/rag/ingest/jobs/{uuid4()}")
    assert r.status_code == 404


def test_ingest_job_enqueue_worker_succeeds(
    client_with_db: tuple[TestClient, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_SERVER_HOST", "")
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("VECTOR_STORE_PERSIST_DIR", str(tmp / "vs_data"))
    monkeypatch.setenv("RAG_COLLECTION_NAME", "rag_ingest_job_test")
    from app.core.config import get_settings

    get_settings.cache_clear()

    def _fake_convert_pdf_bytes(data: bytes, *, settings: Settings) -> str:
        _ = (data, settings)
        return "## Page 1\n\npage\n"

    monkeypatch.setattr(
        "app.rag.ingest_pipeline.registry.convert_markit_pdf_bytes",
        _fake_convert_pdf_bytes,
    )

    client, factory = client_with_db
    pdf = _tiny_pdf_bytes()
    r = client.post(
        "/api/v1/rag/ingest/jobs",
        files=[("files", ("queued.pdf", pdf, "application/pdf"))],
    )
    get_settings.cache_clear()
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    async def run_worker() -> None:
        ok = await process_next_job_once(factory)
        assert ok is True

    asyncio.run(run_worker())

    gr = client.get(f"/api/v1/rag/ingest/jobs/{job_id}")
    assert gr.status_code == 200
    body = gr.json()
    assert body["status"] == "succeeded"
    assert body["results"] is not None
    assert len(body["results"]) == 1
    assert body["results"][0]["ok"] is True

    sr = client.post(
        "/api/v1/rag/search",
        json={"q": "page", "k": 4, "rag_search_mode": "keyword_search"},
    )
    assert sr.status_code == 200
    assert len(sr.json()["chunks"]) >= 1


def test_text_md_ingest_job_enqueue_worker_succeeds(
    client_with_db: tuple[TestClient, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_SERVER_HOST", "")
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("VECTOR_STORE_PERSIST_DIR", str(tmp / "vs_text_md"))
    monkeypatch.setenv("RAG_COLLECTION_NAME", "rag_ingest_txt_md_test")
    from app.core.config import get_settings

    get_settings.cache_clear()

    client, factory = client_with_db
    body = "hello queue md ingest unique phrase xyz".encode("utf-8")
    r = client.post(
        "/api/v1/rag/ingest/jobs",
        files=[("files", ("queued.md", body, "text/markdown"))],
    )
    get_settings.cache_clear()
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    async def run_worker() -> None:
        ok = await process_next_job_once(factory)
        assert ok is True

    asyncio.run(run_worker())

    gr = client.get(f"/api/v1/rag/ingest/jobs/{job_id}")
    assert gr.status_code == 200
    j = gr.json()
    assert j["status"] == "succeeded"
    assert j["results"] is not None
    assert len(j["results"]) == 1
    assert j["results"][0]["ok"] is True

    sr = client.post(
        "/api/v1/rag/search",
        json={"q": "unique phrase xyz", "k": 4, "rag_search_mode": "keyword_search"},
    )
    assert sr.status_code == 200
    assert len(sr.json()["chunks"]) >= 1
