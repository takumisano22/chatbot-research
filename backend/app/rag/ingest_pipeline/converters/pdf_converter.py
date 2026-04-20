# アップロード／ローカル PDF バイト列を全文テキストへ（抽出は pdf_extract、結合は本ファイル）。
from __future__ import annotations

from app.core.config import Settings

from app.rag.ingest_pipeline.converters.pdf_convert_process.pdf_extract import (
    extract_pdf_page_texts,
)

# -----------------------------------------------------------------------------
# 役割: registry から呼ばれる PDF 専用の変換。CLI 用のパス列挙もここに集約する。
# 流れ: extract_pdf_page_texts でページ単位抽出 → 空ページを除外し全文を連結。
# -----------------------------------------------------------------------------

"""
現在非使用
"""

def convert_pdf_bytes(data: bytes, *, settings: Settings) -> str:
    pages = extract_pdf_page_texts(data, settings=settings)
    return "\n\n".join(page.strip() for page in pages if page.strip())

