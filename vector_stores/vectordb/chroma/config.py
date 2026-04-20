# Chroma 用の設定型と、backend / ingest が共有する行型（ChunkRecord / VectorSearchHit）。
# VectorStoreConfig は Settings から写像され、client / store の入口で使う。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorStoreConfig:
    """Chroma 接続・永続化先。backend 側では実装非依存の共通名として扱う。"""

    # HTTP モード時は server_host を設定。埋め込みクライアント時は空で persist_dir を使う（pytest 等）。
    server_host: str
    server_port: int
    persist_dir: Path
    collection_name: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    chunk_text: str
    document_lower: str


@dataclass(frozen=True)
class VectorSearchHit:
    """query() の 1 行。backend 側で RetrievedChunk へ写像する。"""

    doc_id: str
    chunk_id: str
    source: str
    chunk_text: str
    distance: float
