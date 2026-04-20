from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.rag.schemas import RagSearchMode

# -----------------------------------------------------------------------------
# 役割: POST /api/v1/experiment/batch の manifest JSON 形を定義する。
# 流れ: multipart の manifest 文字列を ExperimentBatchManifest.model_validate_json で復元。
# 要点: rag_search_mode で検索経路を切替え。hybrid 時は rag_hybrid_delegate で窓口の委譲先を上書き可能。
# -----------------------------------------------------------------------------


class ExperimentLlmSelection(BaseModel):
    llm_model: str | None = None
    llm_api_base_url: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_request_timeout_seconds: float | None = Field(default=None, ge=1.0)


class ExperimentEmbeddingSelection(BaseModel):
    embedding_provider: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None


class ExperimentBatchManifest(BaseModel):
    chunking_logic_id: str = Field(..., min_length=1)
    tokenizer_logic_id: str = Field(..., min_length=1)
    search_logic_id: str = Field(..., min_length=1)
    questions: list[str] = Field(..., min_length=1, max_length=200)
    rag_search_mode: RagSearchMode = Field(
        default="vector_search",
        description="vector_search / keyword_search / hybrid_search（hybrid は窓口→rag_hybrid_delegate へ）。",
    )
    rag_hybrid_delegate: Literal["vector_search", "keyword_search"] | None = Field(
        default=None,
        description="rag_search_mode=hybrid_search のときのみ有効。未指定なら runtime Settings の既定を使う。",
    )
    run_ragas_metrics: bool = Field(
        default=False,
        description="true のとき Faithfulness を RAGAS で計算し Langfuse に experiment.ragas を送る。",
    )
    llm: ExperimentLlmSelection = Field(default_factory=ExperimentLlmSelection)
    embedding: ExperimentEmbeddingSelection = Field(default_factory=ExperimentEmbeddingSelection)
