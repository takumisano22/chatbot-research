from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# 役割: Langfuse 無効時の型互換用スタブ（クライアントは None を返し、呼び出し側は分岐する）。
# 実際の no-op は get_langfuse_client が None を返すことで表現する。
# -----------------------------------------------------------------------------


class NoOpSpan:
    """観測無効時に渡す可能性のあるプレースホルダ（属性アクセスで落ちないようにする）。"""

    id: str | None = None
    trace_id: str | None = None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def end(self, *_args: Any, **_kwargs: Any) -> None:
        return None
