# -----------------------------------------------------------------------------
# 役割: プロバイダ別サブパッケージ（ollama 等）の名前空間。
# 流れ: backend は load_llm_provider_adapter 経由で llm_bridges.llm_provider.<name> を参照する。
# -----------------------------------------------------------------------------

from __future__ import annotations

from . import ollama

__all__ = ["ollama"]
