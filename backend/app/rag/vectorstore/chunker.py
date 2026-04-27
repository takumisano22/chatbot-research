from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.logic.experiment_context import (
    get_split_for_rag,
    get_split_for_rag_with_metadata,
)

# -----------------------------------------------------------------------------
# 役割: 全文をチャンクに分割し、ストア投入用メタデータ（doc_id / chunk_id / document_lower）を付与する。
# 主な呼び出し元: ingest_pipeline.runner、取り込み処理全般。
# 流れ:
#   1) experiment_context.get_split_for_rag_with_metadata() があればそれを使い、
#      ロジック側 metadata を取り込む（[{"text","metadata"}, ...]）。
#   2) 無ければ get_split_for_rag() を使い、テキストのみのリストを得る。
#   3) いずれの経路でも backend 側で chunk_id / doc_id / source / document_lower の
#      最低限デフォルトを必ず付与する（ロジック側で metadata を作らなくても整合する）。
# -----------------------------------------------------------------------------


def build_chunks_for_source(
    *,
    repo_relative_source: str,
    full_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkForStore]:
    doc_id = _stable_doc_id(repo_relative_source)
    items = _resolve_chunk_items(
        text=full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    out: list[ChunkForStore] = []
    for i, item in enumerate(items):
        text = item["text"]
        chunk_id = f"{doc_id}:{i}"
        document_lower = text.lower()
        out.append(
            ChunkForStore(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=repo_relative_source,
                chunk_text=text,
                document_lower=document_lower,
                metadata=dict(item.get("metadata") or {}),
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
    # ロジック固有のメタデータ。chunk_id / doc_id / source / document_lower は
    # backend 側のデフォルトで生成・上書きされるため、metadata 側にこれらと
    # 同名キーを入れても vectordb の必須フィールドには影響しない。
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_chunk_items(
    *, text: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    """metadata 付きスプリッタを優先利用し、無ければテキストのみで補完する。

    返り値は常に [{"text": str, "metadata": dict}, ...] 形式に統一する。
    """
    split_with_meta = get_split_for_rag_with_metadata()
    if split_with_meta is not None:
        raw = split_with_meta(
            text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return [
            {"text": r.get("text", ""), "metadata": r.get("metadata") or {}}
            for r in raw
            if r.get("text")
        ]
    pieces = get_split_for_rag()(
        text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [{"text": p, "metadata": {}} for p in pieces if p]


def _stable_doc_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
