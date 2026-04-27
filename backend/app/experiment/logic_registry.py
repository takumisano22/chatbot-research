from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable, Final, Literal

from app.core.config import Settings
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: research_pair の logic_id から app.rag.logic.<pkg>.<pkg>_<id> を import（例: chunking_logic_01）。
# -----------------------------------------------------------------------------

LogicCategory = Literal["chunking", "tokenizer", "search", "reranking", "prompt"]

_CATEGORY_MODULE: Final[dict[LogicCategory, str]] = {
    "chunking": "app.rag.logic.chunking",
    "tokenizer": "app.rag.logic.tokenizer",
    "search": "app.rag.logic.search",
    "reranking": "app.rag.logic.reranking",
    "prompt": "app.rag.logic.prompt",
}


def normalize_logic_id(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    if not s:
        raise ValueError("logic_id が空です")
    return s


def import_logic_module(category: LogicCategory, logic_id: str) -> ModuleType:
    lid = normalize_logic_id(logic_id)
    base = _CATEGORY_MODULE[category]
    name = f"{base}.{category}_{lid}"
    return importlib.import_module(name)


def load_split_for_rag(category: LogicCategory, logic_id: str) -> Callable[..., list[str]]:
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "split_for_rag", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__} に split_for_rag がありません")
    return fn  # type: ignore[return-value]


def load_split_for_rag_with_metadata(
    category: LogicCategory, logic_id: str
) -> Callable[..., list[dict[str, Any]]] | None:
    """metadata 付きスプリッタが提供されていれば返す。無ければ None。

    任意の拡張点であり、未提供のロジック (例: chunking_logic_01) でも
    エラーにはせず None を返す。chunker.py 側はこの場合 split_for_rag
    経由でテキストのみ取得する。
    """
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "split_for_rag_with_metadata", None)
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(
            f"{mod.__name__}.split_for_rag_with_metadata が呼び出し不可です"
        )
    return fn  # type: ignore[return-value]


def load_tokenize_query(category: LogicCategory, logic_id: str) -> Callable[[str], list[str]]:
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "tokenize_query", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__} に tokenize_query がありません")
    return fn  # type: ignore[return-value]


def load_retrieve_fn(
    category: LogicCategory, logic_id: str
) -> Callable[..., Any]:
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "retrieve", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__} に retrieve がありません")
    return fn  # type: ignore[return-value]


def load_rerank_fn(
    category: LogicCategory, logic_id: str
) -> Callable[..., Any]:
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "rerank", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__} に rerank がありません")
    return fn  # type: ignore[return-value]


def call_retrieve(
    category: LogicCategory,
    logic_id: str,
    settings: Settings,
    query: str,
    *,
    top_k: int | None,
) -> list[RetrievedChunk]:
    fn = load_retrieve_fn(category, logic_id)
    out = fn(settings, query, top_k=top_k)
    if not isinstance(out, list):
        raise TypeError("retrieve は list[RetrievedChunk] を返す必要があります")
    return out


def call_rerank(
    category: LogicCategory,
    logic_id: str,
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    fn = load_rerank_fn(category, logic_id)
    out = fn(settings, query, chunks)
    if not isinstance(out, list):
        raise TypeError("rerank は list[RetrievedChunk] を返す必要があります")
    return out


def load_rag_system_message(logic_id: str) -> str:
    mod = import_logic_module("prompt", logic_id)
    msg = getattr(mod, "RAG_SYSTEM_MESSAGE", None)
    if not isinstance(msg, str) or not msg.strip():
        raise TypeError(f"{mod.__name__} に有効な RAG_SYSTEM_MESSAGE（非空 str）がありません")
    return msg
