from __future__ import annotations

import io

from app.core.config import Settings

# -----------------------------------------------------------------------------
# 役割: PDF バイト列を MarkItDown で Markdown 相当テキストへ変換する。
# 主な呼び出し元: ingest_pipeline.registry。
# 流れ: BytesIO → convert_stream(file_extension=.pdf) → 文字列を返す。
# -----------------------------------------------------------------------------



def convert_markit_pdf_bytes(data: bytes, *, settings: Settings) -> str:
    _ = settings
    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise RuntimeError("markitdown が未インストールです。") from e
    md = MarkItDown()
    stream = io.BytesIO(data)
    try:
        result = md.convert_stream(stream, file_extension=".pdf")
    except Exception as e:
        if _is_markitdown_pdf_dependency_error(e):
            raise RuntimeError("MarkItDown の PDF 用依存が不足です（markitdown[pdf]）。") from e
        raise
    return (result.markdown or "").strip()


def _is_markitdown_pdf_dependency_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return "MissingDependencyException" in s or "markitdown[pdf]" in s
