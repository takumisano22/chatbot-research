from __future__ import annotations
import re
from typing import Final


from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# 役割: RAG 取り込み時の「テキスト分割ロジック」を独立させ、差し替えしやすくする。
# 主な呼び出し元: rag.vectorstore.chunker.build_chunks_for_source。
# 流れ: split_for_rag を呼ぶと分割器を作成し、入力テキストをチャンク配列へ変換する。
# -----------------------------------------------------------------------------


def split_for_rag(*, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    ここでロジック差し替え可能。
    """
    #splitter = build_chunks(text=text, max_chars=chunk_size, overlap_chars=chunk_overlap)
    splitter = _build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# -----------------------------------------------------------------------------
# 開発者向け追記ポイント:
# - 分割単位（文字数/トークン数）を変えたい場合は _build_splitter を差し替える。
# - 前処理を追加したい場合は split_for_rag の入口で text を整形する。
# -----------------------------------------------------------------------------
def _build_splitter(
    *, chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

SENTENCE_ENDINGS: Final[tuple[str, ...]] = ("。", "！", "？", ".", "!", "?")
SUB_SPLIT_MARKERS: Final[tuple[str, ...]] = ("、", ",", "；", ";", "：", ":")


def split_into_semantic_units(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    units: list[str] = []

    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block:
            continue

        lines = [line.strip() for line in stripped_block.split("\n") if line.strip()]
        merged = " ".join(lines)

        sentences = split_into_sentences(merged)
        if sentences:
            units.extend(sentences)
        else:
            units.append(merged)

    return units


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def split_long_unit(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]

    chunks: list[str] = []
    current = ""

    parts = split_by_sub_markers(unit)

    for part in parts:
        if not current:
            current = part
            continue

        candidate = current + part
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current.strip())
        current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def split_by_sub_markers(text: str) -> list[str]:
    pattern = r"(?<=[、,；;：:])"
    parts = re.split(pattern, text)
    return [part for part in parts if part.strip()]


def build_chunks(
    *,
    text: str,
    max_chars: int = 300,
    overlap_chars: int = 50,
) -> list[str]:
    units = split_into_semantic_units(text)

    expanded_units: list[str] = []
    for unit in units:
        expanded_units.extend(split_long_unit(unit, max_chars))

    chunks: list[str] = []
    current = ""

    for unit in expanded_units:
        candidate = f"{current}{unit}" if current else unit

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if overlap_chars > 0 and chunks:
            overlap = current[-overlap_chars:]
            current = f"{overlap}{unit}"
            if len(current) > max_chars:
                chunks.append(unit[:max_chars].strip())
                current = unit[max_chars:].strip()
        else:
            current = unit

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]