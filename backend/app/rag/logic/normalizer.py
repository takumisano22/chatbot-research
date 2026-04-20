from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# 役割: コンバータが返した全文を、チャンク分割前に正規化する。
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
    normalized = _merge_intrusive_linebreaks(normalized) ## 和文と段落タイトルを判別して改行を除去。
    normalized = _split_heading_linebreaks(normalized) ## 残った二重改行のうち、上が見出し行かつ、下が見出し行でない日本語文のところだけ詰める。

    return normalized.strip() ## 全文の前後の空白を除去して完了


_JA = r"\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"
_JA_GLUE = ( ## 日本語と判定する文字のunicodeパターンの全て
    _JA
    + r"0-9"
    + r"\uFF10-\uFF19"
    + r"\uFF66-\uFF9F"
    + r"\u31F0-\u31FF"
    + r"\u3001-\u3003"
    + r"\u3008-\u3011\u300a\u300b\u3010\u3011"
    + r"\uff08\uff09\uff3b\uff3d\uff5b\uff5d"
    + r"\u30fb\u00b7"
)


_HSPACE = r"[^\S\n]+"
_ALLOWED_WHITESPACE = {" ", "\u3000", "\n", "\r", "\t"}
_EXTRA_INVISIBLE_TO_SPACE = {
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\uFEFF",  # BOM / ZERO WIDTH NO-BREAK SPACE
}

## 句読点・括弧閉じ直後の改行は「段落区切り」として残す（和文の語中改行除去と両立させる）。
_KEEP_NEWLINE_AFTER_PUNCT = frozenset("。.．,、!?！？）」』…")

HEADING_PATTERNS = (
    r"(?:"
    r"第[0-9一二三四五六七八九十百千〇]+(?:条|章|項|説|目)"
    r"|[0-9]+[.)．]"
    r"|[（(][0-9一二三四五六七八九十百千〇]+[)）]"
    r"|[•・\-]"
    r")"
    )


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

    # 和文の「語の途中」に挟まった単独改行を除去。安定するまで繰り返す。

    heading_before_newline = re.compile(rf"^{HEADING_PATTERNS}")

    ## 正規表現メモ：　?< 直前の文字　OR　? 直後の文字　＋　! 否定　OR　= 肯定
    ## rがraw文字列でバックスラッシュを認識する。fがpythonで組み込み変数文字列を示す。
    pat = re.compile(
        rf"(?<=[{_JA_GLUE}])" ## 直前の文字が日本語文字
        r"(?<!\n)" ## 直前の文字が改行でない。
        r"\n" ## 消したい改行
        rf"(?!{HEADING_PATTERNS})" ## 直後の文字が見出しやリストアイテムでない。
        rf"(?=[{_JA_GLUE}])", ## 直後の文字が日本語文字
    )

    patdouble = re.compile(
        rf"(?<=[{_JA_GLUE}])"
        r"\n"
        rf"(?=\n[{_JA_GLUE}])"
        rf"(?!\n{HEADING_PATTERNS})"
    )

    out = text
    old = ""
    
    while out != old:
        old = out
        ## 正規表現のマッチ結果を引数でもらえる。
        def repl(m: re.Match[str]) -> str: ## 正規表現では、"直前の文字列"を可変長で参照できないので、別途行ごと判定をかませてからpat判定する。
            i = m.start()
            if i > 0 and out[i - 1] in _KEEP_NEWLINE_AFTER_PUNCT:
                return "\n"
            line_start = out.rfind("\n", 0, i) + 1 ## outの先頭位置からiまでの範囲で最後に出てくる/nの位置を取得
            prev_line = out[line_start:i]

            if heading_before_newline.search(prev_line) and len(prev_line) <= 30: ##固定値が気に食わないので、空白削除した時点での最大行長とかをページ幅で見立てて、それ-10くらいがよいかも。
                return "\n"   # 見出し行の末尾なら改行を残す
            return ""  # それ以外は削除

        out = pat.sub(repl, out) ## pad判定のものをreplに置換してoutに入れる。（つまり直前行が見出しなら改行を残す）
        out = patdouble.sub(repl, out) ## 二重改行部分を除去

    return out

def _split_heading_linebreaks(text: str) -> str:

    ## 残った二重改行のうち、上が見出し行かつ、下が見出し行でない日本語文のところだけ詰める。

    heading_before_newline = re.compile(rf"^{HEADING_PATTERNS}")

    pat = re.compile(
        rf"(?<=[{_JA_GLUE}])"
        r"\n"
        rf"(?=\n[{_JA_GLUE}])"
        rf"(?!\n{HEADING_PATTERNS})"
    )


    out = text
    old = ""
    while out != old:
        old = out
        def repl(m: re.Match[str]) -> str:
            i = m.start()
            line_start = out.rfind("\n", 0, i) + 1
            prev_line = out[line_start:i]
            if heading_before_newline.search(prev_line):
                return ""
            return "\n"
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
