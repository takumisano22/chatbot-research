from __future__ import annotations

import re
from typing import Any

from app.core.adapters import VectorDbAdapterModule, load_vectordb_adapter
from app.core.config import Settings
from app.langfuse.tracer import observe_vector_store_query
from app.rag.logic.embedding import EmbeddingService, build_embedding_service
from app.rag.schemas import RetrievedChunk
from app.rag.vectorstore.chunker import ChunkForStore

# -----------------------------------------------------------------------------
# 役割: Settings を vectordb アダプタの設定型へ写し、読み書き API を束ねる。
# 主な呼び出し元: retrieval_service（検索）、ingest_pipeline（チャンク書き込み）。
# 流れ: rag_write_session / rag_load_keyword_rows / rag_search_by_vector → アダプタの実装。
# -----------------------------------------------------------------------------


class RagWriteSession:
    """vector DB への書き込みセッション（ChunkForStore 用）。内部実装は動的に読んだ vectordb アダプタ。"""

    __slots__ = ("_inner", "_vs", "_embedding_service", "_config")

    def __init__(
        self,
        inner: Any,
        vs: VectorDbAdapterModule,
        embedding_service: EmbeddingService,
        config: Any,
    ) -> None:
        self._inner = inner
        self._vs = vs
        self._embedding_service = embedding_service
        self._config = config

    def add_chunks(self, chunks: list[ChunkForStore]) -> None:
        if not chunks:
            return
        expanded = _expand_chunks_for_vector_records(chunks)
        documents = [record["document_lower"] for record in expanded]
        records = [self._chunk_record_from_expanded(record) for record in expanded]
        embeddings = self._embedding_service.embed_texts(documents)
        try:
            self._inner.add_chunks(records, embeddings)
        except Exception as exc:
            if not self._vs.is_embedding_dimension_mismatch_error(exc):
                raise
            # 実装依存の次元不一致を backend 側で薄く吸収する。
            self._vs.reset_rag_collection(self._config)
            self._inner = self._vs.RagWriteSession(self._config)
            self._inner.add_chunks(records, embeddings)

    def delete_by_source(self, source: str) -> None:
        self._inner.delete_by_source(source)

    def _chunk_record_from_expanded(self, record: dict[str, Any]) -> Any:
        return self._vs.ChunkRecord(
            chunk_id=record["vector_record_id"],
            doc_id=record["doc_id"],
            source=record["source"],
            chunk_text=record["chunk_text"],
            document_lower=record["document_lower"],
            metadata=record["metadata"],
        )


def rag_write_session(settings: Settings) -> RagWriteSession:
    vs = _vector_store(settings)
    config = _to_vector_store_config(settings)
    inner = vs.RagWriteSession(config)
    embedding_service = build_embedding_service(settings)
    return RagWriteSession(inner, vs, embedding_service, config)


def rag_load_keyword_rows(
    settings: Settings, tokens: list[str]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    vs = _vector_store(settings)
    return vs.rag_load_keyword_rows(_to_vector_store_config(settings), tokens)


def rag_reset_collection(settings: Settings) -> None:
    """実験バッチ等でコレクションを空にする。次回 get_or_create で再作成される。"""
    vs = _vector_store(settings)
    vs.reset_rag_collection(_to_vector_store_config(settings))


def rag_search_by_vector(
    settings: Settings,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    def _run() -> list[RetrievedChunk]:
        vs = _vector_store(settings)
        config = _to_vector_store_config(settings)
        hits: list[Any] = vs.rag_search_by_vector(config, query_vector, top_k)
        return _vector_hits_to_retrieved_chunks(hits)

    return observe_vector_store_query(settings, top_k, _run)


## 公開API一覧 呼び出し先：
# retrieval_service.py
# ingest_pipeline/service.py
#
__all__ = [
    "RagWriteSession",
    "rag_load_keyword_rows",
    "rag_reset_collection",
    "rag_search_by_vector",
    "rag_write_session",
]


def _vector_store(settings: Settings) -> VectorDbAdapterModule:
    return load_vectordb_adapter(settings.vector_db_adapter_subpackage)


def _to_vector_store_config(settings: Settings) -> Any:
    vs = _vector_store(settings)
    return vs.VectorStoreConfig(
        server_host=(settings.vector_store_server_host or "").strip(),
        server_port=settings.vector_store_server_port,
        persist_dir=settings.resolve_vector_store_persist_dir(),
        collection_name=settings.rag_collection_name,
    )


def _min_max_normalize(raws: list[float]) -> list[float]:
    if not raws:
        return []
    lo = min(raws)
    hi = max(raws)
    if hi <= lo:
        return [1.0 for _ in raws]
    return [(raw - lo) / (hi - lo) for raw in raws]


def _expand_chunks_for_vector_records(chunks: list[ChunkForStore]) -> list[dict[str, Any]]:
    """1 論理チャンクを、保存する検索用テキスト数だけ Chroma レコードへ展開する。"""
    expanded: list[dict[str, Any]] = []
    for chunk in chunks:
        variants = _vector_text_variants(chunk)
        multi_variant = bool(chunk.vector_texts)
        for variant_name, vector_text in variants:
            vector_record_id = (
                _vector_record_id(chunk.chunk_id, variant_name)
                if multi_variant
                else chunk.chunk_id
            )
            metadata = dict(chunk.metadata)
            if multi_variant:
                metadata["logical_chunk_id"] = chunk.chunk_id
                metadata["vector_record_id"] = vector_record_id
                metadata["vector_text_variant"] = variant_name
                metadata["vector_text_variant_count"] = len(variants)
            expanded.append(
                {
                    "vector_record_id": vector_record_id,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "chunk_text": chunk.chunk_text,
                    "document_lower": vector_text.lower(),
                    "metadata": metadata,
                }
            )
    return expanded


def _vector_text_variants(chunk: ChunkForStore) -> list[tuple[str, str]]:
    if chunk.vector_texts:
        variants = [
            (str(name), text)
            for name, text in chunk.vector_texts.items()
            if str(name).strip() and text.strip()
        ]
        if variants:
            return variants
    return [("default", chunk.document_lower)]


def _vector_record_id(chunk_id: str, variant_name: str) -> str:
    safe_variant = re.sub(r"[^0-9A-Za-z_.-]+", "_", variant_name).strip("_")
    if not safe_variant:
        safe_variant = "vector"
    return f"{chunk_id}::vector::{safe_variant}"


def _vector_hits_to_retrieved_chunks(hits: list[Any]) -> list[RetrievedChunk]:
    # 距離は小さいほど近い想定。類似度に変換してから Min-Max（keyword_search と同様の並び意図）。
    if not hits:
        return []
    sims = [
        1.0 / (1.0 + min(100.0, float(getattr(h, "distance", 0.0)))) for h in hits
    ]
    norms = _min_max_normalize(sims)
    out: list[RetrievedChunk] = []
    for h, norm in zip(hits, norms, strict=True):
        # adapter 側 (VectorSearchHit) が metadata を持たない実装でも壊れないように getattr で取り出す。
        meta_raw = getattr(h, "metadata", None) or {}
        out.append(
            RetrievedChunk(
                doc_id=str(getattr(h, "doc_id", "")),
                chunk_id=str(getattr(h, "chunk_id", "")),
                source=str(getattr(h, "source", "")),
                chunk_text=str(getattr(h, "chunk_text", "")),
                keyword_score_raw=0.0,
                keyword_score_norm=0.0,
                vector_score_raw=float(getattr(h, "distance", 0.0)),
                vector_score_norm=norm,
                final_score=norm,
                retrieval_type="vector",
                metadata=dict(meta_raw),
            )
        )
    return out
