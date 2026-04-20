# [bolt-004 PDF] pypdf または Tesseract OCR でページ単位のテキストを抽出する（Markdown 結合は processors 側）。
from __future__ import annotations

import io

from pypdf import PdfReader

from app.core.config import Settings

from app.rag.ingest_pipeline.converters.pdf_convert_process.ocr_preprocess import (
    preprocess_for_tesseract,
)

# -----------------------------------------------------------------------------
# 役割: PDF バイト列からページごとの生テキスト列を返す（native / OCR / auto）。Markdown 化・正規化は processors。
# 流れ: extract_pdf_page_texts がモード分岐 → native は pypdf、OCR は pdf2image + Tesseract（前処理は ocr_preprocess）。
# -----------------------------------------------------------------------------


def extract_pdf_page_texts(pdf_bytes: bytes, *, settings: Settings) -> list[str]:
    mode = settings.pdf_extraction_mode
    if mode == "native":
        _, pages = extract_pages_native(pdf_bytes)
        return pages
    if mode == "ocr":
        return extract_pages_ocr(pdf_bytes, settings=settings)
    n, pages = extract_pages_native(pdf_bytes)
    if n == 0:
        return []
    if _auto_use_ocr(n, pages, settings.pdf_ocr_auto_min_chars_per_page):
        return extract_pages_ocr(pdf_bytes, settings=settings)
    return pages


def extract_pages_native(pdf_bytes: bytes) -> tuple[int, list[str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        raw = page.extract_text()
        pages.append(raw if raw else "")
    return len(reader.pages), pages


def extract_pages_ocr(pdf_bytes: bytes, *, settings: Settings) -> list[str]:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "OCR 用パッケージ（pdf2image / pytesseract）の読み込みに失敗しました。"
        ) from e
    lang = settings.pdf_ocr_lang.strip() or "eng"
    cfg = f"--oem {settings.pdf_ocr_oem} --psm {settings.pdf_ocr_psm}"
    try:
        images = convert_from_bytes(pdf_bytes, dpi=settings.pdf_ocr_dpi)
    except Exception as e:
        raise RuntimeError(
            "PDF を画像に変換できませんでした。poppler-utils（pdftoppm）がインストールされているか確認してください。"
        ) from e
    texts: list[str] = []
    try:
        for img in images:
            ocr_in = preprocess_for_tesseract(img) if settings.pdf_ocr_preprocess else img
            t = pytesseract.image_to_string(ocr_in, lang=lang, config=cfg)
            texts.append(t if t else "")
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract が見つかりません。tesseract-ocr をインストールし PATH を通してください。"
        ) from e
    return texts


# -----------------------------------------------------------------------------
# 補助: auto モードで OCR に切り替えるかどうか
# -----------------------------------------------------------------------------


def _auto_use_ocr(num_pages: int, per_page_text: list[str], min_chars_per_page: int) -> bool:
    if num_pages <= 0:
        return False
    total = sum(len(p.strip()) for p in per_page_text)
    threshold = max(1, num_pages) * max(0, min_chars_per_page)
    return total < threshold
