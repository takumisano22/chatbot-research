from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# 役割: CHUNKING logic_01 — 固定長チャンク（Settings の chunk_size / chunk_overlap を ingest 側から渡される）。
# -----------------------------------------------------------------------------


def split_for_rag(*, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)
