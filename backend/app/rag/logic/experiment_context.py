from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

# -----------------------------------------------------------------------------
# 役割: 実験時のみ chunking / tokenizer を ContextVar で差し替える薄い窓口。
# search / reranking は logic_registry → 各 logic モジュールの retrieve / rerank で解決。
# -----------------------------------------------------------------------------

_split_for_rag_var: ContextVar[Callable[..., list[str]] | None] = ContextVar(
    "experiment_split_for_rag", default=None
)
_tokenize_query_var: ContextVar[Callable[[str], list[str]] | None] = ContextVar(
    "experiment_tokenize_query", default=None
)


def get_split_for_rag() -> Callable[..., list[str]]:
    custom = _split_for_rag_var.get()
    if custom is not None:
        return custom
    from app.rag.logic.chunking import split_for_rag

    return split_for_rag


def get_tokenize_query() -> Callable[[str], list[str]]:
    custom = _tokenize_query_var.get()
    if custom is not None:
        return custom
    from app.rag.logic.tokenizer import tokenize_query

    return tokenize_query


@contextmanager
def active_chunking_split(fn: Callable[..., list[str]]) -> Iterator[None]:
    tok: Token = _split_for_rag_var.set(fn)
    try:
        yield
    finally:
        _split_for_rag_var.reset(tok)


@contextmanager
def active_tokenizer(fn: Callable[[str], list[str]]) -> Iterator[None]:
    tok: Token = _tokenize_query_var.set(fn)
    try:
        yield
    finally:
        _tokenize_query_var.reset(tok)
