from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

# -----------------------------------------------------------------------------
# 役割: 実験時のみ chunking / tokenizer を ContextVar で差し替える薄い窓口。
# search / reranking は logic_registry → 各 logic モジュールの retrieve / rerank で解決。
# 補足: chunking は 2 系統の差し替え口を持つ。
#   - split_for_rag(text, chunk_size, chunk_overlap) -> list[str] : 既存・必須経路
#   - split_for_rag_with_metadata(...) -> list[{"text","metadata"}] : 任意経路
#     （metadata をロジック側で生成したい場合のみ提供。chunker.py 側で優先利用される）
# -----------------------------------------------------------------------------

_split_for_rag_var: ContextVar[Callable[..., list[str]] | None] = ContextVar(
    "experiment_split_for_rag", default=None
)
# metadata 付きスプリッタ。提供されていれば chunker.py が優先利用する。
# 戻り値は [{"text": str, "metadata": dict}, ...] 形式を想定。
_split_for_rag_with_metadata_var: ContextVar[
    Callable[..., list[dict[str, Any]]] | None
] = ContextVar("experiment_split_for_rag_with_metadata", default=None)
_tokenize_query_var: ContextVar[Callable[[str], list[str]] | None] = ContextVar(
    "experiment_tokenize_query", default=None
)


def get_split_for_rag() -> Callable[..., list[str]]:
    custom = _split_for_rag_var.get()
    if custom is not None:
        return custom
    from app.rag.logic.chunking import split_for_rag

    return split_for_rag


def get_split_for_rag_with_metadata() -> Callable[..., list[dict[str, Any]]] | None:
    """metadata 付きスプリッタを返す。設定されていなければ None。

    None の場合は呼び出し側 (chunker.py) で get_split_for_rag() への
    フォールバックを行う想定。
    """
    return _split_for_rag_with_metadata_var.get()


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
def active_chunking_split_with_metadata(
    fn: Callable[..., list[dict[str, Any]]] | None,
) -> Iterator[None]:
    """metadata 付きスプリッタを差し替える。fn=None で「未提供」状態を表現できる。

    研究ペアごとに切り替える際、当該ロジックが metadata 経路を持たない場合は
    None を渡しておくことで chunker.py 側のフォールバック動作 (テキストのみ経路)
    に確実に戻せる。
    """
    tok: Token = _split_for_rag_with_metadata_var.set(fn)
    try:
        yield
    finally:
        _split_for_rag_with_metadata_var.reset(tok)


@contextmanager
def active_tokenizer(fn: Callable[[str], list[str]]) -> Iterator[None]:
    tok: Token = _tokenize_query_var.set(fn)
    try:
        yield
    finally:
        _tokenize_query_var.reset(tok)
