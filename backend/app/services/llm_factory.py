from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.core.adapters import load_llm_provider_adapter

# -----------------------------------------------------------------------------
# 役割: Settings に基づき LangChain の Chat モデル（BaseChatModel）を組み立てる。
# 主な呼び出し元: experiment batch_runner（get_chat_model）。
# 流れ: プロバイダ分岐 → load_llm_provider_adapter → LlmHttpChatParams → build_compatible_chat_model。
# -----------------------------------------------------------------------------


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        llm = load_llm_provider_adapter(settings.llm_adapter_subpackage)
        params = llm.LlmHttpChatParams(
            api_base_url=settings.llm_api_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
        )
        return llm.build_compatible_chat_model(params)
    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
        "現状は 'ollama' のみ。別プロバイダは llm_factory と llm_provider 配下の実装を拡張し、"
        "LLM_ADAPTER_SUBPACKAGE の許可値を増やしてください。"
    )
