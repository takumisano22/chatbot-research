from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess

from app.core.config import Settings
from app.experiment.activity import seconds_since_activity

# -----------------------------------------------------------------------------
# 役割: 設定秒数 HTTP が無いときにシェルコマンド（任意）実行後、SIGTERM でプロセスを終了する。
# 流れ: 60 秒毎に seconds_since_activity を見る → 閾値超えで stop_shell → kill。
# 要点: compose 全体停止はホスト側 docker 等を experiment_idle_stop_shell に渡す（コンテナ単体では限界がある）。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


async def idle_watch_loop(settings: Settings) -> None:
    interval = 60.0
    limit = float(settings.experiment_idle_shutdown_seconds)
    if limit <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        if seconds_since_activity() < limit:
            continue
        await asyncio.to_thread(_run_stop_shell, settings)
        logger.warning("実験アイドル %s 秒超過のためプロセスを終了します。", int(limit))
        os.kill(os.getpid(), signal.SIGTERM)
        return


def _run_stop_shell(settings: Settings) -> None:
    cmd = (settings.experiment_idle_stop_shell or "").strip()
    if not cmd:
        return
    try:
        subprocess.run(
            ["/bin/sh", "-c", cmd],
            check=False,
            timeout=120,
        )
    except Exception:
        logger.exception("experiment_idle_stop_shell の実行に失敗しました（続行します）")
