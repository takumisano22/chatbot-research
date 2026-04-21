from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: 既定チャンク分割は logic_01 と同一（experiment_context のデフォルト参照先）。
# -----------------------------------------------------------------------------

from app.rag.logic.chunking.chunking_logic_01 import split_for_rag

__all__ = ["split_for_rag"]
