from __future__ import annotations

from app.core.config import Settings
from app.experiment.schemas import ExperimentBatchManifest

# -----------------------------------------------------------------------------
# 役割: プロセス起動時の .env 由来 Settings を基に、実験 1 回分だけ上書きした Settings を作る（グローバルは不変）。
# 流れ: get_settings → model_copy(update=...) → バッチ終了後は参照を捨てる。
# -----------------------------------------------------------------------------


def build_runtime_settings(base: Settings, manifest: ExperimentBatchManifest) -> Settings:
    patch: dict[str, object] = {}
    llm = manifest.llm
    if llm.llm_model is not None:
        patch["llm_model"] = llm.llm_model
    if llm.llm_api_base_url is not None:
        patch["llm_api_base_url"] = llm.llm_api_base_url
    if llm.llm_temperature is not None:
        patch["llm_temperature"] = llm.llm_temperature
    if llm.llm_request_timeout_seconds is not None:
        patch["llm_request_timeout_seconds"] = llm.llm_request_timeout_seconds

    emb = manifest.embedding
    if emb.embedding_provider is not None:
        patch["embedding_provider"] = emb.embedding_provider.strip().lower()
    if emb.embedding_base_url is not None:
        patch["embedding_base_url"] = emb.embedding_base_url
    if emb.embedding_model is not None:
        patch["embedding_model"] = emb.embedding_model

    if manifest.rag_hybrid_delegate is not None:
        patch["rag_hybrid_delegate"] = manifest.rag_hybrid_delegate

    if not patch:
        return base
    return base.model_copy(update=patch)
