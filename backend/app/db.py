from __future__ import annotations

import logging
import threading
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

# -----------------------------------------------------------------------------
# 役割: AsyncEngine / sessionmaker の遅延初期化と FastAPI Depends(get_db_session) の提供。
# 主な呼び出し元: ingest ルート、ingest ジョブ以外の将来の RDB ルート。jobs は独自 engine を持つ。
# 流れ: 初回 DB アクセスで create_async_engine → async_sessionmaker → yield session → shutdown で dispose。
# 要点: テストは dependency_overrides で差し替え可能。DATABASE_URL 未設定時は get_db_session が失敗する。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine_lock = threading.Lock()


def engine_kwargs_for_url(database_url: str) -> dict[str, Any]:
    if database_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


def _build_engine() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = get_settings()
    url = (settings.database_url or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL が未設定です。ingest API を使う場合は .env を設定してください。")
    eng = create_async_engine(
        url,
        **engine_kwargs_for_url(url),
        pool_pre_ping=True,
    )
    factory = async_sessionmaker(eng, expire_on_commit=False)
    return eng, factory


def _ensure_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    with _engine_lock:
        if _session_factory is not None:
            return _session_factory
        eng, factory = _build_engine()
        _engine = eng
        _session_factory = factory
        logger.info("AsyncEngine を初期化しました（非同期 SQLAlchemy）")
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    try:
        factory = _ensure_factory()
    except RuntimeError as exc:
        msg = str(exc)
        if "DATABASE_URL" in msg:
            raise HTTPException(status_code=501, detail=msg) from exc
        raise
    async with factory() as session:
        yield session


async def dispose_app_database() -> None:
    """アプリ終了時にプールを閉じる（未初期化なら何もしない）。"""
    global _engine, _session_factory
    with _engine_lock:
        if _engine is None:
            return
        try:
            await _engine.dispose()
        finally:
            _engine = None
            _session_factory = None


def reset_engine_for_tests() -> None:
    """テスト用: グローバル engine を同期で破棄（非同期 dispose は別途呼ぶ想定）。"""
    global _engine, _session_factory
    with _engine_lock:
        _engine = None
        _session_factory = None


__all__ = [
    "dispose_app_database",
    "engine_kwargs_for_url",
    "get_db_session",
    "reset_engine_for_tests",
]
