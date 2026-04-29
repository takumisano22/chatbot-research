from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# 役割: コンバータが返した全文を、チャンク分割前に正規化する。
# 構造文書(条文等)・日記・図式入りテキストなど多様な入力に共通で効く処理のみを置く。
# 構造文書特有の見出し改行整備（例: 見出し行と本文を密着させる詰め直し等）は
# chunking ロジック側 (例: chunking_logic_03.normalize_text) に委譲する。
# 主な呼び出し元: ingest_pipeline.runner、取り込み処理全般。
# 流れ: normalize_document_text が各補助処理を順に適用して返す。
# -----------------------------------------------------------------------------

def normalize_document_text(text: str) -> str:

    normalized = text
    normalized = _unicode_spaces_strict(normalized) ## 空白文字をすべて半角スペースに置換（ゼロ幅系は除去）
    normalized = re.sub(r"[ \t]+", " ", normalized) ## 連続する空白文字またはタブ文字を半角スペースに置換
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n") ## 改行コードを\nに統一
    normalized = re.sub(r"\n{3,}", "\n\n", normalized) ## 3つ以上の連続する改行を2つの改行に変換
    normalized = _collapse_intraword_spaces(normalized) ## 「日本語文の内部」とみなせる位置の水平空白を詰める。
    normalized = _merge_intrusive_linebreaks(normalized) ## 和文の語中に紛れた単独改行を除去（句読点直後は段落境界として残す）。

    return normalized.strip() ## 全文の前後の空白を除去して完了


_JA = r"぀-ゟ゠-ヿ㐀-䶿一-鿿"
_JA_GLUE = ( ## 日本語と判定する文字のunicodeパターンの全て
    _JA
    + r"0-9"
    + r"０-９"
    + r"ｦ-ﾟ"
    + r"ㇰ-ㇿ"
    + r"、-〃"
    + r"〈-】《》【】"
    + r"（）［］｛｝"
    + r"・·"
)


_HSPACE = r"[^\S\n]+"
_ALLOWED_WHITESPACE = {" ", "　", "\n", "\r", "\t"}
_EXTRA_INVISIBLE_TO_SPACE = {
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "⁠",  # WORD JOINER
    "﻿",  # BOM / ZERO WIDTH NO-BREAK SPACE
}

## 句読点・括弧閉じ直後の改行は「段落区切り」として残す（和文の語中改行除去と両立させる）。
_KEEP_NEWLINE_AFTER_PUNCT = frozenset("。.．,、!?！？）」』…")


def _unicode_spaces_strict(text: str) -> str:

    ## 半角スペース・全角スペース・\\n・\\r・\\t はそのまま。ゼロ幅系は除去。その他の空白類は半角スペースに寄せる。

    out = []

    for ch in text:
        if ch in _ALLOWED_WHITESPACE:
            out.append(ch)
        elif ch in _EXTRA_INVISIBLE_TO_SPACE:
            ## ゼロ幅・BOM は語間の「見えない隙間」として扱い、空白に置かず詰める（キーワード一致のブレを抑える）。
            continue
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)

    return "".join(out)


def _merge_intrusive_linebreaks(text: str) -> str:

    ## 和文の「語の途中」に挟まった単独改行を除去する汎用処理。
    ## 構造文書特有の見出しパターン判定は持たず、句読点・括弧閉じ直後のみを段落境界として改行を残す。
    ## 見出し直後で誤って詰まる単独改行は、後段の chunking ロジック側で行頭整備により復元する想定。
    ## 二重改行は段落区切りとしてそのまま保持し、構造的な詰め直しは chunking 側に任せる。

    pat = re.compile(
        rf"(?<=[{_JA_GLUE}])" ## 直前の文字が日本語文字
        r"(?<!\n)" ## 直前の文字が改行でない
        r"\n" ## 消したい改行（単独改行のみ。直後が \n の場合は除外したいので別ガード不要）
        r"(?!\n)" ## 直後の文字が改行でない（= 二重改行は対象外）
        rf"(?=[{_JA_GLUE}])", ## 直後の文字が日本語文字
    )

    out = text
    old = ""

    while out != old:
        old = out

        def repl(m: re.Match[str]) -> str: ## 句読点・括弧閉じ直後だけ改行を残し、それ以外は除去する。
            i = m.start()
            if i > 0 and out[i - 1] in _KEEP_NEWLINE_AFTER_PUNCT:
                return "\n"
            return ""

        out = pat.sub(repl, out)

    return out


def _collapse_intraword_spaces(text: str) -> str:

    ## 和数字まわり・「語内部」と改行前後とみなせる位置の水平空白を詰める。変化がなくなるまで繰り返す。
    g = rf"{_JA_GLUE}\n"
    h = _HSPACE
    out = text
    old = ""

    while out != old:
        old = out
        out = re.sub(rf"([{g}]){h}(\d+){h}([{g}])", r"\1\2\3", out) ## 第 9 条　みたいなやつの空白をつめる
        out = re.sub(rf"(?<=[{g}]){h}(?=[{g}])","", out) ## 「"日本語"の内部」とみなせる位置の水平空白を詰める。
    return out
