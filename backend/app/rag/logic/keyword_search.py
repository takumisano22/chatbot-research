from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import Settings
from app.langfuse.tracer import observe_keyword_retrieval
from app.rag.logic.experiment_context import get_tokenize_query
from app.rag.schemas import RetrievedChunk
from app.rag.vectorstore.vector_db import rag_load_keyword_rows

# -----------------------------------------------------------------------------
# 役割: キーワード検索処理を集約し、ドキュメント保持型ベクトル DB へ被せる薄いラッパを提供する。
# 主な呼び出し元: rag.logic.search（keyword / hybrid の keyword 側）。
# 流れ: tokenize → キーワード行ロード → スコア計算/正規化 → RetrievedChunk に詰め替え。
# -----------------------------------------------------------------------------


class KeywordRowsGateway(Protocol):
    def load_keyword_rows(
        self, settings: Settings, tokens: list[str]
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        ...


@dataclass(frozen=True)
class VectorStoreKeywordRowsGateway:
    """
    vector_db.rag_load_keyword_rows を KeywordRowsGateway に合わせる薄いラッパ。
    Chroma 固有実装は vector_db 側に閉じ込める。
    """

    def load_keyword_rows(
        self, settings: Settings, tokens: list[str]
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        return rag_load_keyword_rows(settings, tokens)


def search_keyword_chunks(
    settings: Settings,
    query: str,
    top_k: int | None = None,
    gateway: KeywordRowsGateway | None = None,
) -> list[RetrievedChunk]:
    k = settings.rag_top_k if top_k is None else top_k

    def _run() -> list[RetrievedChunk]:
        tokens = get_tokenize_query()(query)
        if not tokens:
            return []

        active_gateway = gateway or VectorStoreKeywordRowsGateway()
        id_list, metas, docs_lower = active_gateway.load_keyword_rows(settings, tokens)
        if not id_list:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for index, _ in enumerate(id_list):
            meta = metas[index] if index < len(metas) and metas[index] else {}
            doc_l = docs_lower[index] if index < len(docs_lower) and docs_lower[index] else ""
            scored.append((meta, _keyword_raw_score(doc_l, tokens)))

        scored.sort(key=lambda row: row[1], reverse=True)
        scored = scored[:k]
        raw_scores = [raw for _, raw in scored]
        normalized = _min_max_normalize(raw_scores)
        keyword_weight = float(settings.rag_keyword_weight)

        out: list[RetrievedChunk] = []
        for (meta, raw), norm in zip(scored, normalized, strict=True):
            out.append(
                RetrievedChunk(
                    doc_id=str(meta.get("doc_id", "")),
                    chunk_id=str(meta.get("chunk_id", "")),
                    source=str(meta.get("source", "")),
                    chunk_text=str(meta.get("chunk_text", "")),
                    keyword_score_raw=raw,
                    keyword_score_norm=norm,
                    vector_score_raw=None,
                    vector_score_norm=None,
                    final_score=min(1.0, max(0.0, norm * keyword_weight)),
                    retrieval_type="keyword",
                )
            )
        return out

    return observe_keyword_retrieval(settings, query, k, _run)


# -----------------------------------------------------------------------------
# 開発者向け追記ポイント:
# - BM25 などへ差し替える場合は _keyword_raw_score を置き換える。
# - DB 差し替え時は KeywordRowsGateway 実装を追加し search_keyword_chunks に渡す。
# -----------------------------------------------------------------------------
def _keyword_raw_score(chunk_lower: str, tokens: Iterable[str]) -> float:
    return sum(float(chunk_lower.count(token)) for token in tokens if token)


def _min_max_normalize(raws: list[float]) -> list[float]:
    if not raws:
        return []
    lo = min(raws)
    hi = max(raws)
    if hi <= lo:
        return [1.0 for _ in raws]
    return [(raw - lo) / (hi - lo) for raw in raws]
