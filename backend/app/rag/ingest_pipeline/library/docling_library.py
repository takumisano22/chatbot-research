from __future__ import annotations

import hashlib
import io

from app.core.config import Settings
from app.rag.vectorstore.chunker import ChunkForStore
from app.rag.vectorstore.vector_db import RagWriteSession

# -----------------------------------------------------------------------------
# 役割: docling ライブラリで「変換 → 正規化 → チャンク化」を一括実行する取り込みパイプライン。
# 主な呼び出し元: app.experiment.batch_runner（research_pair で ingest_pipeline_id 指定時）。
# 流れ: bytes → DocumentConverter で DoclingDocument → HybridChunker で chunk 列 →
#       ChunkForStore に詰め直して session.delete_by_source + session.add_chunks。
# 既存の converters / normalizer / logic.chunking は全て docling 内部で代替するため
# SUPERSEDES = ("convert", "normalize", "chunking") を宣言する。
# -----------------------------------------------------------------------------


# 上位 (app.experiment.ingest_pipeline_registry) が読み取るステージ宣言。
SUPERSEDES: tuple[str, ...] = ("convert", "normalize", "chunking")

# 既存 _policy_for_file と整合させるため、扱う拡張子は pdf/txt/md のみ。
_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt", ".md"})


def ingest(
    settings: Settings,
    session: RagWriteSession,
    *,
    filename: str,
    data: bytes,
    source: str,
) -> int:
    _ = settings  # docling 側の設定は固定（YAGNI）。Settings はシグネチャ整合のため受け取るのみ。
    suffix = _suffix_lower(filename)
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(
            f"docling_library が対応していない拡張子です: {filename!r}"
            f"（対応: {sorted(_ALLOWED_SUFFIXES)})"
        )

    chunks = _build_chunks_via_docling(filename=filename, data=data, source=source)

    session.delete_by_source(source)
    session.add_chunks(chunks)
    return len(chunks)


# -----------------------------------------------------------------------------
# 内部: docling 呼び出しと ChunkForStore 化
# -----------------------------------------------------------------------------


def _build_chunks_via_docling(
    *, filename: str, data: bytes, source: str
) -> list[ChunkForStore]:
    # docling は実験的経路でしか使わないため import は遅延する（依存欠落時の影響を最小化）。
    try:
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise RuntimeError(
            "docling が未インストールです（pip install docling）。"
        ) from e

    converter = DocumentConverter()
    stream = DocumentStream(name=filename, stream=io.BytesIO(data))
    result = converter.convert(stream)
    document = result.document

    chunker = HybridChunker()
    iter_chunks = chunker.chunk(dl_doc=document)

    doc_id = _stable_doc_id(source)
    out: list[ChunkForStore] = []
    for i, ch in enumerate(iter_chunks):
        # contextualize で見出しパスを含めた本文を取り出す（HybridChunker 推奨手順）。
        text = chunker.contextualize(chunk=ch).strip()
        if not text:
            continue
        chunk_id = f"{doc_id}:{i}"
        out.append(
            ChunkForStore(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                chunk_text=text,
                document_lower=text.lower(),
                metadata={
                    "chunking_strategy": "docling_hybrid",
                    "docling_index": i,
                },
            )
        )
    return out


def _suffix_lower(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def _stable_doc_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
