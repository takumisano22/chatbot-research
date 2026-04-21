from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: RAG の差し替え可能ロジック層（normalizer/chunking/tokenizer/embedding/search）を束ねる。
# 検索窓口は search パッケージ（__init__.py の search_documents）。retrieval_service はそれを再エクスポート。
# -----------------------------------------------------------------------------

__all__ = [
    "chunking",
    "normalizer",
    "embedding",
    "keyword_search",
    "search",
    "tokenizer",
    "vector_search",
]
