# チャンクの追加・削除、ベクトル / キーワード読み取り、リセット、RagWriteSession。
# LangChain Chroma・chromadb のクエリ API はこのモジュールに閉じる。
from __future__ import annotations

from typing import Any

from chromadb.api.models.Collection import Collection

from vectordb.chroma.config import ChunkRecord, VectorSearchHit, VectorStoreConfig
from vectordb.chroma.client import get_rag_collection, get_vector_store_client


# -----------------------------------------------------------------------------
# 取り込み・削除
# -----------------------------------------------------------------------------


def delete_chunks_by_source(collection: Collection, source: str) -> None:
    collection.delete(where={"source": source})


def add_chunks(
    collection: Collection,
    chunks: list[ChunkRecord],
    embeddings: list[list[float]] | None = None,
) -> None:
    if not chunks:
        return
    ids, documents, metadatas = _ids_documents_metadatas(chunks)
    if embeddings is not None:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def add_chunks_for_config(
    config: VectorStoreConfig,
    chunks: list[ChunkRecord],
    embeddings: list[list[float]] | None = None,
) -> None:
    add_chunks(get_rag_collection(config), chunks, embeddings)


def delete_chunks_by_source_for_config(config: VectorStoreConfig, source: str) -> None:
    delete_chunks_by_source(get_rag_collection(config), source)


class RagWriteSession:
    """同一 VectorStoreConfig で複数ファイルを取り込むとき、コレクション取得を 1 回に抑える。"""

    __slots__ = ("_config", "_collection")

    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._collection: Collection | None = None

    def _col(self) -> Collection:
        if self._collection is None:
            self._collection = get_rag_collection(self._config)
        return self._collection

    def add_chunks(
        self, chunks: list[ChunkRecord], embeddings: list[list[float]] | None = None
    ) -> None:
        add_chunks(self._col(), chunks, embeddings)

    def delete_by_source(self, source: str) -> None:
        delete_chunks_by_source(self._col(), source)


# -----------------------------------------------------------------------------
# 読み取り・リセット
# -----------------------------------------------------------------------------


def rag_search_by_vector(
    config: VectorStoreConfig,
    query_vector: list[float],
    top_k: int,
) -> list[VectorSearchHit]:
    if top_k <= 0 or not query_vector:
        return []
    rows = get_rag_collection(config).query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    return _vector_hits_from_query_rows(rows)


def rag_load_keyword_rows(
    config: VectorStoreConfig, tokens: list[str]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    collection = get_rag_collection(config)
    ids = _gather_keyword_candidate_ids(collection, tokens)
    if not ids:
        return [], [], []
    rows = collection.get(ids=list(ids), include=["metadatas", "documents"])
    id_list = rows["ids"]
    metas_raw = rows["metadatas"] or []
    docs_lower = rows["documents"] or []
    metas: list[dict[str, Any]] = []
    for m in metas_raw:
        metas.append(m if isinstance(m, dict) else {})
    return list(id_list), metas, list(docs_lower)


def reset_rag_collection(config: VectorStoreConfig) -> None:
    client = get_vector_store_client(config)
    client.delete_collection(name=config.collection_name)


def is_embedding_dimension_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "dimension" in message and "embedding" in message


# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------


def _ids_documents_metadatas(
    chunks: list[ChunkRecord],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    ids = [c.chunk_id for c in chunks]
    documents = [c.document_lower for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "chunk_text": c.chunk_text,
        }
        for c in chunks
    ]
    return ids, documents, metadatas


def _vector_hits_from_query_rows(rows: dict[str, Any]) -> list[VectorSearchHit]:
    metadatas_rows = rows.get("metadatas") or []
    documents_rows = rows.get("documents") or []
    distances_rows = rows.get("distances") or []
    if not metadatas_rows or not documents_rows or not distances_rows:
        return []
    metadatas = metadatas_rows[0] if metadatas_rows[0] else []
    documents = documents_rows[0] if documents_rows[0] else []
    distances = distances_rows[0] if distances_rows[0] else []
    out: list[VectorSearchHit] = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        m = metadata if isinstance(metadata, dict) else {}
        out.append(
            VectorSearchHit(
                doc_id=str(m.get("doc_id", "")),
                chunk_id=str(m.get("chunk_id", "")),
                source=str(m.get("source", "")),
                chunk_text=str(m.get("chunk_text") or (document or "")),
                distance=float(distance),
            )
        )
    return out


def _gather_keyword_candidate_ids(collection: Collection, tokens: list[str]) -> set[str]:
    acc: set[str] = set()
    for t in tokens:
        got = collection.get(where_document={"$contains": t})
        acc.update(got["ids"])
    if acc or not tokens:
        return acc
    head = tokens[0][:32]
    if head:
        got = collection.get(where_document={"$contains": head})
        acc.update(got["ids"])
    return acc
