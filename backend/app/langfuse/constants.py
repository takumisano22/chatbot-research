from __future__ import annotations

from typing import Final

# -----------------------------------------------------------------------------
# 役割: 観測メタデータ用の固定識別子（chunking 実装と照合しやすい文字列を1箇所に集約）。
# 取り込みは rag.vectorstore.chunker → experiment_context.get_split_for_rag_with_metadata（既定は chunking_logic_01）。
# -----------------------------------------------------------------------------

CHUNKING_STRATEGY_RECURSIVE_CHARACTER_TEXT_SPLITTER: Final[str] = (
    "recursive_character_text_splitter"
)
CHUNKING_STRATEGY_CUSTOM_BUILD_CHUNKS: Final[str] = "custom_build_chunks"
