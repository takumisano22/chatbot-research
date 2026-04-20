# MarkItDown 経由の PDF コンバータ（markitdown 未導入環境でも import 解決できるよう sys.modules で差し替え）
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from app.core.config import get_settings
from app.rag.ingest_pipeline.converters.markit_pdf_converter import convert_markit_pdf_bytes


def test_convert_markit_pdf_bytes_delegates_to_markitdown(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    try:
        s = get_settings()
        fake = MagicMock()
        fake.convert_stream.return_value = MagicMock(markdown="  # Hello  \n")
        stub = ModuleType("markitdown")
        stub.MarkItDown = MagicMock(return_value=fake)
        monkeypatch.setitem(sys.modules, "markitdown", stub)
        out = convert_markit_pdf_bytes(b"%PDF-1.4", settings=s)
        assert out == "# Hello"
        fake.convert_stream.assert_called_once()
    finally:
        get_settings.cache_clear()
