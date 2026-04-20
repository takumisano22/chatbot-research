from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: .txt / .md のバイト列を UTF-8 でデコードし、プレーンテキストとして返す。
# 主な呼び出し元: ingest_pipeline.registry。
# 流れ: data.decode("utf-8")。
# -----------------------------------------------------------------------------


def convert_text_bytes(data: bytes) -> str:
    return data.decode("utf-8")
