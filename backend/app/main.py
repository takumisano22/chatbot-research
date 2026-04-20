from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.experiment import router as experiment_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.rag import router as rag_router
from app.core.config import get_settings

# -----------------------------------------------------------------------------
# 役割: FastAPI アプリを組み立て、CORS・API ルーター・lifespan（DB dispose / 実験アイドル）を登録する。
# 流れ: create_app → 設定取得 → CORS/ルーター登録 → lifespan で終了時に DB プール破棄。
# -----------------------------------------------------------------------------


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> object:
    from app.db import dispose_app_database
    from app.experiment.activity import reset_activity_clock
    from app.experiment.idle_shutdown import idle_watch_loop

    settings = get_settings()
    task: asyncio.Task[None] | None = None
    if settings.experiment_batch_enabled and settings.experiment_idle_shutdown_seconds > 0:
        reset_activity_clock()
        task = asyncio.create_task(idle_watch_loop(settings))
    yield
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await dispose_app_database()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Local Chatbot API",
        version="0.1.0",
        lifespan=_app_lifespan,
    )
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.experiment_batch_enabled and settings.experiment_idle_shutdown_seconds > 0:
        from starlette.requests import Request
        from starlette.responses import Response

        from app.experiment.activity import mark_activity

        @app.middleware("http")
        async def _touch_activity(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            mark_activity()
            return await call_next(request)

    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(ingest_router)
    if settings.experiment_batch_enabled:
        app.include_router(experiment_router)
    return app


app = create_app()
