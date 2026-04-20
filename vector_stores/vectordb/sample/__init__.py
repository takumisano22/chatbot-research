# 互換: `vectordb.sample` 名前空間（旧 remote）。実装は vectordb.chroma へ委譲。
from vectordb.chroma.config import VectorStoreConfig as RemoteVectorStoreConfig
from vectordb.chroma import (
    RagWriteSession,
    add_chunks,
    add_chunks_for_config,
    delete_chunks_by_source,
    delete_chunks_by_source_for_config,
    get_rag_collection,
    get_vector_store_client,
    rag_load_keyword_rows,
    rag_search_by_vector,
)
from vectordb.chroma.config import ChunkRecord

__all__ = [
    "ChunkRecord",
    "RagWriteSession",
    "RemoteVectorStoreConfig",
    "add_chunks",
    "add_chunks_for_config",
    "delete_chunks_by_source",
    "delete_chunks_by_source_for_config",
    "get_rag_collection",
    "get_vector_store_client",
    "rag_load_keyword_rows",
    "rag_search_by_vector",
]
