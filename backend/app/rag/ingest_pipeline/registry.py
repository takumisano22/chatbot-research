from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.rag.ingest_pipeline.converters.markit_pdf_converter import convert_markit_pdf_bytes
from app.rag.ingest_pipeline.converters.text_converter import convert_text_bytes

# -----------------------------------------------------------------------------
# 役割: アップロードファイルの拡張子に応じてバイト列をプレーンテキストへ変換する（.md は .txt と同様に UTF-8 本文）。
# 主な呼び出し元: ingest_pipeline.service（アップロード 1 件の取り込み）。
# 流れ: convert_upload_bytes_to_text → 拡張子に対応するコンバータ関数を呼ぶ。
# -----------------------------------------------------------------------------


def convert_upload_bytes_to_text(filename: str, data: bytes, settings: Settings) -> str:
    suf = Path(filename).suffix.lower()
    fn = _BY_SUFFIX.get(suf)
    if fn is None:
        raise ValueError(f"未対応の拡張子です: {filename!r}")
    return fn(data, settings)


_BY_SUFFIX: dict[str, Callable[[bytes, Settings], str]] = {
    ".pdf": lambda data, s: convert_markit_pdf_bytes(data, settings=s),
    ".txt": lambda data, _: convert_text_bytes(data),
    ".md": lambda data, _: convert_text_bytes(data),
}
