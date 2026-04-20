# Chroma 実装。adapter（vectordb.chroma）向けの公開 API のみを束ねる。
from vectordb.chroma.client import get_rag_collection, get_vector_store_client
from vectordb.chroma.config import ChunkRecord, VectorSearchHit, VectorStoreConfig
from vectordb.chroma.store import (
    RagWriteSession,
    add_chunks,
    add_chunks_for_config,
    delete_chunks_by_source,
    delete_chunks_by_source_for_config,
    is_embedding_dimension_mismatch_error,
    rag_load_keyword_rows,
    rag_search_by_vector,
    reset_rag_collection,
)

__all__ = [
    "ChunkRecord",
    "RagWriteSession",
    "VectorSearchHit",
    "VectorStoreConfig",
    "add_chunks",
    "add_chunks_for_config",
    "delete_chunks_by_source",
    "delete_chunks_by_source_for_config",
    "get_rag_collection",
    "get_vector_store_client",
    "is_embedding_dimension_mismatch_error",
    "rag_load_keyword_rows",
    "rag_search_by_vector",
    "reset_rag_collection",
]
