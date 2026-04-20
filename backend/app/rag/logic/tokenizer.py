from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# 役割: 検索クエリのトークン分割ロジックを 1 箇所に集約する。
# 主な呼び出し元: rag.logic.keyword_search.search_keyword_chunks。
# 流れ: tokenize_query が入力を正規化し、英数/CJK を考慮したトークン配列を返す。
# -----------------------------------------------------------------------------


def tokenize_query(query: str) -> list[str]:
    normalized = query.strip().lower()
    if not normalized:
        return []

    parts = re.findall(
        r"[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",
        normalized,
        re.UNICODE,
    )
    if not parts:
        return [normalized]
    if len(parts) == 1 and _contains_cjk(parts[0]) and len(parts[0]) >= 2:
        return _cjk_bigrams(parts[0])
    return parts


# -----------------------------------------------------------------------------
# 開発者向け追記ポイント:
# - 形態素解析器や独自辞書を使う場合は tokenize_query を差し替える。
# - CJK 向けの分割規則を変えたい場合は _cjk_bigrams を置き換える。
# -----------------------------------------------------------------------------
def _contains_cjk(text: str) -> bool:
    return re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text) is not None


def _cjk_bigrams(text: str) -> list[str]:
    normalized = text.strip().lower()
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return [normalized[i : i + 2] for i in range(len(normalized) - 1)]
