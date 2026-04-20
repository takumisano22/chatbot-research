from __future__ import annotations

from app.core.config import Settings

# -----------------------------------------------------------------------------
# 役割: Langfuse を実際に送るかどうかの判定（設定・キー不足時は観測を無効化）。
# -----------------------------------------------------------------------------


def is_langfuse_configured(settings: Settings) -> bool:
    if not settings.langfuse_enabled:
        return False
    pk = (settings.langfuse_public_key or "").strip()
    sk = (settings.langfuse_secret_key or "").strip()
    host = (settings.langfuse_host or "").strip()
    return bool(pk and sk and host)
