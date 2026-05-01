from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.logic.experiment_context import get_split_for_rag_with_metadata

# -----------------------------------------------------------------------------
# 役割: 全文をチャンクに分割し、ストア投入用メタデータ（doc_id / chunk_id / document_lower）を付与する。
# 主な呼び出し元: ingest_pipeline.runner、取り込み処理全般。
# 流れ:
#   1) experiment_context.get_split_for_rag_with_metadata() でロジック側 metadata 付き
#      チャンク（[{"text": str, "metadata": dict}, ...]）を取得する。
#   2) backend 側で chunk_id / doc_id / source / document_lower の
#      最低限デフォルトを必ず付与する（ロジック側で metadata が空でも整合する）。
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
        vector_texts = _normalize_vector_texts(item.get("vector_texts"))
        document_lower = _first_vector_text(vector_texts, fallback=text).lower()
        out.append(
            ChunkForStore(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=repo_relative_source,
                chunk_text=text,
                document_lower=document_lower,
                vector_texts=vector_texts,
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
    # logic_06 など、1 論理チャンクから複数の検索用テキストを持つ場合だけ利用する。
    # 空なら document_lower を 1 本だけ保存する従来互換の経路になる。
    vector_texts: dict[str, str] = field(default_factory=dict)


def _resolve_chunk_items(
    *, text: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    raw = get_split_for_rag_with_metadata()(
        text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        {
            "text": r.get("text", ""),
            "metadata": r.get("metadata") or {},
            "vector_texts": _normalize_vector_texts(r.get("vector_texts")),
        }
        for r in raw
        if r.get("text")
    ]


def _stable_doc_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]


def _normalize_vector_texts(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        text = str(value).strip() if value is not None else ""
        if name and text:
            out[name] = text
    return out


def _first_vector_text(vector_texts: dict[str, str], *, fallback: str) -> str:
    if vector_texts:
        return next(iter(vector_texts.values()))
    return fallback
