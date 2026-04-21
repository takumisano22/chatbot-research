from __future__ import annotations

from fastapi import FastAPI

from app.core.config import get_settings

# -----------------------------------------------------------------------------
# 役割: オプションのヘルスチェック用のみ（実験バッチは python -m app.experiment.runner）。
# -----------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="RAG experiment helpers", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        _ = get_settings()
        return {"status": "ok"}

    return app


app = create_app()
