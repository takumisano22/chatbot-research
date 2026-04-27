from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 役割: RAG 検索ヒットの Pydantic スキーマ。
# 主な呼び出し元: app.rag.logic.search、logic プラグイン、実験バッチ。
# -----------------------------------------------------------------------------

RetrievalType = Literal["keyword", "vector"]


class RetrievedChunk(BaseModel):
    doc_id: str = Field(..., description="同一ソースファイルを表す安定 ID")
    chunk_id: str = Field(..., description="ストア内のチャンク行に相当する一意キー")
    source: str = Field(..., description="元ファイルの相対パス文字列")
    chunk_text: str = Field(..., description="チャンク本文（表示・プロンプト用）")
    keyword_score_raw: float = Field(
        ..., description="キーワード由来の生スコア（大きいほど関連が高い想定）"
    )
    keyword_score_norm: float = Field(
        ..., ge=0.0, le=1.0, description="同一クエリ内 Min-Max 正規化"
    )
    vector_score_raw: float | None = Field(
        None, description="ベクトル検索時はストアが返す距離などの生値（キーワードのみのときは null）"
    )
    vector_score_norm: float | None = Field(
        None, description="ベクトル類似度の Min-Max 正規化（キーワードのみのときは null）"
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="キーワード: keyword_weight * keyword_score_norm。ベクトル: vector_score_norm",
    )
    retrieval_type: RetrievalType = "keyword"
    # ロジック由来の追加 metadata（予約キーを除く）。Chroma 書き込み時に dict/list は
    # JSON 文字列化されているため、必要に応じて呼び出し側で json.loads する。
    metadata: dict[str, Any] = Field(default_factory=dict)
