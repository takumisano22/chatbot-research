from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.rag.logic.chunking import split_for_rag

# -----------------------------------------------------------------------------
# 役割: 全文をチャンクに分割し、ストア投入用メタデータ（doc_id / chunk_id / document_lower）を付与する。
# 主な呼び出し元: ingest_pipeline.runner、取り込み処理全般。
# 流れ: normalize 済み全文 → logic.chunking.split_for_rag → ChunkForStore 列。
# -----------------------------------------------------------------------------


def build_chunks_for_source(
    *,
    repo_relative_source: str,
    full_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkForStore]:
    doc_id = _stable_doc_id(repo_relative_source)
    pieces = split_for_rag(
        text=full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    out: list[ChunkForStore] = []
    for i, text in enumerate(pieces):
        chunk_id = f"{doc_id}:{i}"
        document_lower = text.lower()
        out.append(
            ChunkForStore(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=repo_relative_source,
                chunk_text=text,
                document_lower=document_lower,
            )
        )
    return out


def to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


# -----------------------------------------------------------------------------
# データ型・補助
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkForStore:
    chunk_id: str
    doc_id: str
    source: str
    chunk_text: str
    document_lower: str


def _stable_doc_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
