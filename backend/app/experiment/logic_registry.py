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


def load_split_for_rag_with_metadata(
    category: LogicCategory, logic_id: str
) -> Callable[..., list[dict[str, Any]]]:
    mod = import_logic_module(category, logic_id)
    fn = getattr(mod, "split_for_rag_with_metadata", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__} に split_for_rag_with_metadata がありません")
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
    *,
    top_k: int,
) -> tuple[list[RetrievedChunk], int]:
    # rerank 関数契約: (list[RetrievedChunk], effective_top_k)。後段のプロンプト・CSV 列本数は effective_top_k に合わせる。
    fn = load_rerank_fn(category, logic_id)
    out = fn(settings, query, chunks, top_k=top_k)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("rerank は (list[RetrievedChunk], int) のタプルを返す必要があります")
    ranked, k_eff = out[0], out[1]
    if not isinstance(ranked, list):
        raise TypeError("rerank の第1要素は list である必要があります")
    if not isinstance(k_eff, int) or k_eff < 0:
        raise TypeError("rerank の第2要素は 0 以上の int である必要があります")
    return ranked, k_eff


def load_rag_system_message(logic_id: str) -> str:
    mod = import_logic_module("prompt", logic_id)
    msg = getattr(mod, "RAG_SYSTEM_MESSAGE", None)
    if not isinstance(msg, str) or not msg.strip():
        raise TypeError(f"{mod.__name__} に有効な RAG_SYSTEM_MESSAGE（非空 str）がありません")
    return msg
