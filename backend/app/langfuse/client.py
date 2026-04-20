from __future__ import annotations

import logging

from langfuse import Langfuse

from app.core.config import Settings
from app.langfuse.config import is_langfuse_configured

# -----------------------------------------------------------------------------
# 役割: Langfuse クライアント生成と安全な flush（送信失敗で業務処理を落とさない）。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def get_langfuse_client(settings: Settings) -> Langfuse | None:
    if not is_langfuse_configured(settings):
        return None
    pk = (settings.langfuse_public_key or "").strip()
    sk = (settings.langfuse_secret_key or "").strip()
    host = (settings.langfuse_host or "").strip()
    env = (settings.langfuse_environment or "").strip() or None
    try:
        return Langfuse(
            public_key=pk,
            secret_key=sk,
            host=host,
            environment=env,
            tracing_enabled=True,
        )
    except Exception:
        logger.exception("Langfuse クライアント初期化に失敗しました（観測をスキップします）")
        return None


def safe_flush(client: Langfuse | None) -> None:
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse flush に失敗しました（無視します）")
