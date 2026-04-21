from __future__ import annotations

# -----------------------------------------------------------------------------
# 役割: TOKENIZER logic_01 — 分割はせず 1 トークンにまとめる（no-op 相当）。
# keyword_search は doc 側が lower なので、トークンも lower で揃える。
# -----------------------------------------------------------------------------


def tokenize_query(query: str) -> list[str]:
    s = query.strip().lower()
    if not s:
        return []
    return [s]
