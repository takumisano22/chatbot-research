from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings
from app.rag.schemas import RagSearchMode

# -----------------------------------------------------------------------------
# 役割: research_pair YAML/JSON のスキーマとファイル解決（実験 1 条件 = 1 ファイル）。
# 流れ: load_research_pair_file → ResearchPair 検証 → runner が参照。
# -----------------------------------------------------------------------------

_LOGIC_ID_RE = re.compile(r"^logic_[a-z0-9_]+$")


class ResearchPair(BaseModel):
    research_pair_id: str = Field(..., min_length=1, max_length=128)
    llm_model: str | None = None
    llm_api_base_url: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_request_timeout_seconds: float | None = Field(default=None, ge=1.0)

    embedding_provider: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None

    search_logic_id: str = Field(..., min_length=1)
    chunking_logic_id: str = Field(..., min_length=1)
    reranking_logic_id: str = Field(..., min_length=1)
    tokenizer_logic_id: str = Field(..., min_length=1)
    prompt_logic_id: str = Field(..., min_length=1)

    top_k: int = Field(..., ge=1, le=500)
    qa_dataset: str = Field(
        ...,
        min_length=1,
        description="qa_datasets ディレクトリ内のファイル名（例: my_set.json）",
    )
    document_set_id: str = Field(..., min_length=1)
    ragas_enabled: bool = False
    rag_search_mode: RagSearchMode = "vector_search"
    rag_hybrid_delegate: Literal["vector_search", "keyword_search"] | None = None

    @field_validator(
        "search_logic_id",
        "chunking_logic_id",
        "reranking_logic_id",
        "tokenizer_logic_id",
        "prompt_logic_id",
    )
    @classmethod
    def _logic_id_fmt(cls, v: str) -> str:
        s = (v or "").strip().lower().replace("-", "_")
        if not _LOGIC_ID_RE.match(s):
            raise ValueError(f"logic_id は logic_01 形式にしてください（現在: {v!r}）")
        return s

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def _emb_prov(cls, v: object) -> str | None:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip().lower()

    def qa_dataset_path(self, settings: Settings) -> Path:
        base = settings.resolve_experiment_qa_datasets_dir().resolve()
        p = (base / self.qa_dataset).resolve()
        try:
            p.relative_to(base)
        except ValueError as e:
            raise ValueError("qa_dataset パスが qa_datasets ディレクトリ外を指しています") from e
        return p

    def research_pair_path_for_display(self, settings: Settings) -> Path:
        return settings.resolve_experiment_research_pairs_dir() / f"{self.research_pair_id}.yaml"

    def spec_json_for_csv(self) -> str:
        """CSV 比較用: ロジック ID・モデル等の主要項目（.env 既定との差が分かる形）。"""
        d: dict[str, Any] = {
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_base_url": self.embedding_base_url,
            "embedding_model": self.embedding_model,
            "search_logic_id": self.search_logic_id,
            "chunking_logic_id": self.chunking_logic_id,
            "reranking_logic_id": self.reranking_logic_id,
            "tokenizer_logic_id": self.tokenizer_logic_id,
            "prompt_logic_id": self.prompt_logic_id,
            "top_k": self.top_k,
            "qa_dataset": self.qa_dataset,
            "document_set_id": self.document_set_id,
            "rag_search_mode": self.rag_search_mode,
            "ragas_enabled": self.ragas_enabled,
        }
        return json.dumps(d, ensure_ascii=False, sort_keys=True)


def load_research_pair_file(path: Path) -> ResearchPair:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"未対応の拡張子です: {path.suffix}")
    if not isinstance(raw, dict):
        raise ValueError("research_pair のルートはオブジェクトである必要があります")
    return ResearchPair.model_validate(raw)


def load_research_pair_by_id(settings: Settings, research_pair_id: str) -> ResearchPair:
    rid = research_pair_id.strip()
    if not rid:
        raise ValueError("research_pair_id が空です")
    if rid.endswith((".yaml", ".yml", ".json")):
        p = Path(rid)
        if not p.is_absolute():
            p = (settings.resolve_experiment_research_pairs_dir() / p.name).resolve()
    else:
        for ext in (".yaml", ".yml", ".json"):
            p = settings.resolve_experiment_research_pairs_dir() / f"{rid}{ext}"
            if p.is_file():
                return load_research_pair_file(p)
        raise FileNotFoundError(
            f"research_pair が見つかりません: {rid}（探索先: {settings.resolve_experiment_research_pairs_dir()}）"
        )
    if not p.is_file():
        raise FileNotFoundError(f"research_pair ファイルがありません: {p}")
    return load_research_pair_file(p)
