from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db import engine_kwargs_for_url
from app.models.rag_ingest_job import RagIngestJobRow
from app.repositories.ingestion_job_repository import (
    claim_next_queued_job_id,
    load_job_upload_items,
    mark_job_failed,
    mark_job_succeeded,
)
from app.rag.ingest_pipeline.enums import JOB_KIND_RAG_UPLOAD
from app.rag.ingest_pipeline.service import run_queued_upload_batch


# ---------------------------------------------------------------------------
# rag_ingest_jobs キューをポーリングする常駐ワーカー用モジュール。
# 主な呼び出し元: docker起動時に、compose.yamlの宣言で `python -m app.rag.ingest_pipeline.jobs` 
# として起動（`__main__` から `main`）。
# 大まかな流れ:
#   - キューから次ジョブを claim する
#   - claimed job の job_kind を解釈し、対応する取り込み処理へディスパッチする
#   - 成功時は `mark_job_succeeded`、失敗時は `mark_job_failed` で DB に反映する
# ---------------------------------------------------------------------------


_log = logging.getLogger(__name__)

_LEGACY_JOB_KINDS = frozenset({"pdf_upload", "txt_md_upload"})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="RAG ingest queue worker")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="キュー空き時のポーリング間隔（秒）",
    )
    args = parser.parse_args()
    settings = get_settings()
    db_url = (settings.database_url or "").strip()
    if not db_url:
        _log.error("DATABASE_URL が未設定のためワーカーを起動できません")
        sys.exit(1)
    engine = create_async_engine(
        db_url,
        **engine_kwargs_for_url(db_url),
        pool_pre_ping=True,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        asyncio.run(_poll_loop(factory, args.interval))
    except KeyboardInterrupt:
        _log.info("停止します")
    finally:
        asyncio.run(_dispose_engine(engine))


async def _poll_loop(factory: async_sessionmaker[AsyncSession], interval_sec: float) -> None:
    # 停止条件のない無限ループ。常駐ワーカーとして動き、キューが空のときだけ interval（デフォルトで 1.0秒）で待つ。
    while True:
        processed = await process_next_job_once(factory)
        if not processed:
            await asyncio.sleep(interval_sec)


async def process_next_job_once(factory: async_sessionmaker[AsyncSession]) -> bool:
    # 1 件分の入口。claim だけ短い DB session に閉じ、成功の確定は dispatch 内。dispatch が例外のときだけここで mark_job_failed。
    async with factory() as session:
        job_id = await claim_next_queued_job_id(session)
    if job_id is None:
        return False
    settings = get_settings()
    try:
        await dispatch_claimed_job(factory, settings, job_id)
    except Exception as e:
        _log.exception("ingest job failed: %s", job_id)
        async with factory() as session:
            await mark_job_failed(session, job_id, str(e))
    return True


async def dispatch_claimed_job(
    factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    job_id: UUID,
) -> None:
    # 行の読み取り・job_kind 検証・items 取得までを同一 DB session にまとめる。不整合はこの session 内で mark_job_failed して return。
    async with factory() as session:
        job = await session.get(RagIngestJobRow, job_id)
        if job is None:
            return
        kind_raw = (job.job_kind or "").strip()
        if not kind_raw:
            await mark_job_failed(session, job_id, "job_kind が空です")
            return
        kind = _normalize_job_kind(kind_raw)
        if kind != JOB_KIND_RAG_UPLOAD:
            await mark_job_failed(session, job_id, f"未対応の job_kind: {kind_raw!r}")
            return
        items = await load_job_upload_items(session, job_id)

    # 重いバッチは session を閉じてから実行し、正常完了後に別 session で mark_job_succeeded。バッチ例外は呼び出し側が mark_job_failed。
    results = run_queued_upload_batch(settings, items)

    async with factory() as session:
        await mark_job_succeeded(session, job_id, results)


def _normalize_job_kind(raw: str) -> str:
    # 旧ストレージ上の job_kind 表記を現行の単一 kind に寄せ、ディスパッチ側の分岐を一本化する。
    if raw in _LEGACY_JOB_KINDS:
        return JOB_KIND_RAG_UPLOAD
    return raw


async def _dispose_engine(engine: AsyncEngine) -> None:
    await engine.dispose()


if __name__ == "__main__":
    main()
