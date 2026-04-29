from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# 役割: CHUNKING logic_01 — 固定長チャンク（Settings の chunk_size / chunk_overlap を ingest 側から渡される）。
# 固定長分割のためロジック固有 metadata は無し。空 dict を返して共通 I/F に準拠する。
# -----------------------------------------------------------------------------


def split_for_rag_with_metadata(
    *, text: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [{"text": t, "metadata": {}} for t in splitter.split_text(text)]
