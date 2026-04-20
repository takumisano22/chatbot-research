from __future__ import annotations

import time

# -----------------------------------------------------------------------------
# 役割: 直近の HTTP アクティビティ時刻（monotonic）を記録し、アイドル停止判定に使う。
# -----------------------------------------------------------------------------

_last_activity_mono: float = time.monotonic()


def mark_activity() -> None:
    global _last_activity_mono
    _last_activity_mono = time.monotonic()


def seconds_since_activity() -> float:
    return time.monotonic() - _last_activity_mono


def reset_activity_clock() -> None:
    """起動直後にアイドル誤検知しないよう lifespan で呼ぶ。"""
    mark_activity()
