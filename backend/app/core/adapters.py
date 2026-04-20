from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Callable, Protocol, cast

from langchain_core.language_models.chat_models import BaseChatModel

# -----------------------------------------------------------------------------
# 役割: 設定に応じて llm_bridges.llm_provider.* / vectordb.* / llm_bridges.embedding_provider.* を importlib で読み込み、型付け用に cast する。
# 主な呼び出し元: llm_factory（Chat モデル）、vector_db、embedding（埋め込み）。
# 流れ: load_*_adapter → 各サービスが build_* を呼ぶ。
# -----------------------------------------------------------------------------


@lru_cache(maxsize=8)
def load_llm_provider_adapter(subpackage: str) -> LlmProviderAdapterModule:
    mod = importlib.import_module(f"llm_bridges.llm_provider.{subpackage}")
    return cast(LlmProviderAdapterModule, mod)


@lru_cache(maxsize=8)
def load_vectordb_adapter(subpackage: str) -> VectorDbAdapterModule:
    mod = importlib.import_module(f"vectordb.{subpackage}")
    return cast(VectorDbAdapterModule, mod)


@lru_cache(maxsize=8)
def load_embedding_provider_adapter(provider: str) -> EmbeddingProviderAdapterModule:
    name = (provider or "").strip().lower()
    if not name:
        raise ValueError(
            "EMBEDDING_PROVIDER が空です。'ollama' または 'ruri_http' を指定してください。"
        )
    module_path = f"llm_bridges.embedding_provider.{name}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"埋め込みプロバイダ {name!r} を読み込めませんでした。"
            f" モジュール {module_path} が存在するか、依存関係が揃っているか確認してください。"
        ) from e
    return cast(EmbeddingProviderAdapterModule, mod)


# -----------------------------------------------------------------------------
# 補助: importlib で得たモジュールの最小 API（型チェッカ向け）
# -----------------------------------------------------------------------------


class LlmProviderAdapterModule(Protocol):
    """llm_bridges.llm_provider.<subpackage> が満たすべき汎用 API。"""

    LlmHttpChatParams: type[Any]
    build_compatible_chat_model: Callable[[Any], BaseChatModel]


class VectorDbAdapterModule(Protocol):
    """vectordb.<subpackage> が満たすべき汎用 API。"""

    VectorStoreConfig: type[Any]
    ChunkRecord: type[Any]
    RagWriteSession: type[Any]
    add_chunks_for_config: Callable[..., None]
    delete_chunks_by_source_for_config: Callable[..., None]
    reset_rag_collection: Callable[[Any], None]
    is_embedding_dimension_mismatch_error: Callable[[Exception], bool]
    rag_load_keyword_rows: Callable[
        [Any, list[str]], tuple[list[str], list[dict[str, Any]], list[str]]
    ]
    # クエリ埋め込みによる近傍検索。戻りはストア実装の行型（backend の vector_db で RetrievedChunk へ写像）。
    rag_search_by_vector: Callable[[Any, list[float], int], list[Any]]


class EmbeddingProviderAdapterModule(Protocol):
    """llm_bridges.embedding_provider.<provider> が満たすべき汎用 API。"""

    EmbeddingParams: type[Any]
    build_embedding_service: Callable[[Any], Any]
