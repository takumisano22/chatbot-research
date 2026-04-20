# LangChain の ChatOllama 等の import は llm_bridges.llm_provider.ollama にのみ置く（backend は llm_factory 経由）。
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


@dataclass(frozen=True)
class LlmHttpChatParams:
    """HTTP で到達するローカル互換チャット API 向けの最小パラメータ。"""

    api_base_url: str
    model: str
    temperature: float
    request_timeout_seconds: float


def build_compatible_chat_model(params: LlmHttpChatParams) -> BaseChatModel:
    """Ollama 互換 HTTP API 向けの Chat モデル（ChatOllama）。"""
    return ChatOllama(
        base_url=params.api_base_url.rstrip("/"),
        model=params.model,
        temperature=params.temperature,
        client_kwargs={"timeout": params.request_timeout_seconds},
    )
