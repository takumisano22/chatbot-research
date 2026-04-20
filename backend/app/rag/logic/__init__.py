from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: RAG の差し替え可能ロジック層（normalizer/chunking/tokenizer/embedding/search）を束ねる。
# 主な呼び出し元: retrieval_service、ingest_pipeline、vectorstore.chunker、将来の各種 RAG 拡張実装。
# 流れ: API/サービス層から本パッケージの公開関数を呼び、具体ロジックは各モジュールに委譲する。
# -----------------------------------------------------------------------------

__all__ = [
    "chunking",
    "normalizer",
    "embedding",
    "hybrid_search",
    "keyword_search",
    "tokenizer",
    "vector_search",
]
