from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Metadata builder のシグネチャ。
# default_metadata_builder と互換のキーワード引数を受け取り dict を返す。
# chunker.py 等で独自フィールドを足したい場合は、このシグネチャに準拠した
# 関数を flatten_chunks() に渡す。
MetadataBuilder = Callable[..., dict[str, Any]]

# ============================================================
# 主要処理フロー（呼び出し関係）
# ============================================================
# split_for_rag_with_metadata()  ← 公開 API（logic_registry から動的解決）
#   -> normalize_text()
#        -> _split_embedded_headings()
#        -> _ensure_blank_before_headings()
#        -> _split_heading_linebreaks()
#   -> extract_heading_candidates()
#        -> _extract_marker_value()
#        -> _award_sequence_bonus()
#   -> analyze_heading_groups()
#        -> _sequence_score()
#        -> _compute_containment_and_reset()
#   -> infer_heading_levels()
#   -> build_section_tree()
#        -> _rebuild_tree_by_recursive_scoring()
#             -> _assign_levels_recursive()
#                  -> _score_type_in_scope()
#                       -> _interval_containment()
#                       -> _sequence_score()
#        -> _add_paragraph_fallback()  ← 見出しゼロ時のみ
#   -> flatten_chunks()
#        -> default_metadata_builder()  ← 差し替え可
#        -> _meaningful_body_lines()
#        -> _format_chunk_md()
#        -> _group_items_balanced()
#        -> _split_text_evenly()


# ============================================================
# スコアリング・パラメータ（調整対象を集約）
# ============================================================
# 数値の意味は隣接コメント参照。文書傾向に合わせて調整したい時はここを編集する。

# --- 見出し行スコア (extract_heading_candidates) ---
# 正規化後は indent≒0 のため、旧 LINE_BONUS_INDENT_ZERO(1.0) を BASE に吸収済み。
LINE_SCORE_BASE = 4.0              # 行頭でルールに一致した時点の基礎点
LINE_BONUS_PREV_BLANK = 1.0        # 直前が空行
LINE_BONUS_SHORT_LINE = 1.0        # 行長 <= LINE_SHORT_MAX
LINE_BONUS_CHAPTER_LIKE = 4.0      # japanese_chapter / japanese_article / appendix
LINE_BONUS_SECTION = 2.0           # japanese_section
LINE_BONUS_PAREN_TITLE = 2.0       # 行内の "（題名）" 形式
LINE_BONUS_NO_PERIOD_END = 1.0     # 句点で終わらない

LINE_PENALTY_LONG_LINE = 2.0       # 行長 > LINE_LONG_MIN
LINE_PENALTY_MULTI_PERIOD = 2.0    # 句点が 2 個以上
LINE_PENALTY_URL_LIKE = 3.0        # URL / メールっぽい記号を含む
LINE_SHORT_MAX = 40
LINE_LONG_MIN = 60

# --- 連番ボーナス（候補スコアへの直接加点） ---
HEADING_SEQ_BONUS = 2.0            # 同タイプで marker_value が +1 連続
HEADING_RESET_BONUS = 0.5          # 同タイプで 1 にリセット
# --- 連番一貫性比率（_sequence_score 内の重み） ---
RESET_RATIO_WEIGHT = 0.5           # +1 連続 1 に対するリセットの相対重み

# --- グループ統計ベースの外側スコア (infer_heading_levels) ---
INFER_W_CONTAINMENT = 4.0
INFER_W_RESET = 3.0
INFER_W_SEQUENCE = 1.0
INFER_W_PRIORITY = 1.5
INFER_GAP_SCALE = 10.0             # avg_gap / text_len を 0-10 にスケール（文書長に依存しない）
INFER_GAP_CAP = 2.0                # 上限

# --- スコープ再帰スコア (_score_type_in_scope) ---
SCOPE_W_CONTAINMENT = 5.0
SCOPE_W_SEQUENCE = 2.5
SCOPE_W_COVERAGE = 1.5
SCOPE_W_FREQUENCY = 0.7
SCOPE_W_PRIORITY = 0.5
# 上位スコープで「次点（=採用 hint より内側で最高スコア）」と認定されたタイプに、
# 直下スコープ評価でのみ与える bias。少数派の条が numeric_paren 多数派に競り負け
# て章直下に出てこないケースで、章 → 条 → 列挙の階層を保つために使う。
SCOPE_W_PREFERRED = 2.0
SCOPE_HEAD_ZONE = 0.15             # 「スコープ先頭付近」判定の共通比率
SCOPE_COVERAGE_SOLO_HEAD = 1.0     # 単独で先頭付近にあるときの coverage
SCOPE_COVERAGE_SOLO_OTHER = 0.3    # 単独で先頭付近以外にあるときの coverage

# --- 共通閾値 ---
GAP_OUTER_RATIO = 0.8              # A_gap > B_gap * 0.8 で A は B の外側候補
CONTAINMENT_CAP = 4.0              # avg_contained / 4.0 で内包率の上限
DEFAULT_LEVEL_HINT = 4             # ルール不在時の level_hint フォールバック


# ============================================================
# dataclasses
# ============================================================


@dataclass
class ChunkingConfig:
    # 見出し階層の上限。推論でこれより深い level はこの値に丸める。
    max_depth: int = 4
    # 見出し候補として採用する最低スコア。低いほど候補が増え、誤検出も増えやすい。
    min_heading_score: float = 3.0
    # 見出しタイプを「有効なグループ」とみなす最小出現数。
    min_group_count: int = 2
    # 見出し1行の最大文字数。長文行を見出しとして誤検出しにくくする閾値。
    max_heading_line_length: int = 80
    # 子セクションとして残す最小本文長。短すぎるノードの乱立を抑える。
    min_child_text_length: int = 10
    # グループ統計から見出し level を推論するか。False ならルール既定寄りになる。
    enable_level_inference: bool = True
    # 行内に埋め込まれた強見出し ("...する。第10条..." 等) を行分割するか。
    enable_inline_heading_repair: bool = True
    # 見出し抽出に失敗した場合、段落ベースのフォールバックチャンクを作るか。
    fallback_to_paragraph: bool = True
    # チャンク本文をマークダウン見出し記法 (### / ## / #) で整形して出力するか。
    output_markdown: bool = True
    # max_chunk_chars: この文字数を超えるセクションは子ノードへ分割委譲する。
    # 収まる場合は子孫テキストを含めて1チャンクにまとめ、重複を防ぐ。
    max_chunk_chars: int = 500


@dataclass
class HeadingRule:
    name: str
    regex: str
    default_level_hint: int  # フォールバック時のlevel (1=最外側)


@dataclass
class HeadingCandidate:
    text: str
    start_char: int
    marker_type: str
    marker_value: int | str | None
    score: float
    inferred_level: int | None = None


@dataclass
class SectionNode:
    id: str
    heading: str
    heading_type: str
    level: int
    text: str
    start_char: int
    parent_id: str | None
    children: list["SectionNode"] = field(default_factory=list)


# ============================================================
# ルール定義 / 共通正規表現 / 漢数字テーブル
# ============================================================

# リスト先頭ほど外側階層らしいデフォルト扱い。
# priority はリスト定義順から自動算出する（先頭=最大）。
# 利用先:
#   - default_level_hint: infer_heading_levels の代替 level / fallback_types の level
#   - priority (定義順): analyze_heading_groups の fallback_priority、スコア補助加算
DEFAULT_HEADING_RULES: list[HeadingRule] = [
    #                  name               regex（行頭マッチ例）           level_hint
    HeadingRule("appendix",        r"^附則", 1),                        # 附則
    HeadingRule("japanese_chapter", r"^第[0-9一二三四五六七八九十百千〇]+章", 1),  # 第一章, 第3章
    HeadingRule("japanese_article", r"^第[0-9一二三四五六七八九十百千〇]+条", 2),  # 第1条, 第十二条
    HeadingRule("japanese_section", r"^第[0-9一二三四五六七八九十百千〇]+項", 3),  # 第1項, 第三項
    HeadingRule("decimal_number",  r"^\d+\.\d+", 3),                   # 1.1, 3.2.1
    HeadingRule("numeric_dot",     r"^\d{1,2}[.)．](?!\d)", 4),        # 1. , 2．配偶者は…（空白なし可、最大 2 桁・小数/日付除外）
    HeadingRule("numeric_paren",   r"^[（(]\d+[)）]", 4),              # (1), （3）
    HeadingRule("japanese_paren",  r"^[（(][一二三四五六七八九十〇]+[)）]", 4),  # (一), （三）
    HeadingRule("roman",           r"^[IVX]{1,5}[.)]\s|^[Ⅰ-Ⅻ]\s", 4),  # III. , Ⅱ
    HeadingRule("alpha",           r"^[A-Z][.)]\s", 4),                # A. , B)
    HeadingRule("circle_bullet",   r"^○\s*\S", 5),                    # ○ 概要
    HeadingRule("bullet",          r"^[•・\-]\s", 5),                  # ・項目, - 注意
]

# ルール名 → HeadingRule の辞書。`next((r for r in ... if r.name == ...))` の重複検索を回避。
_RULE_BY_NAME: dict[str, HeadingRule] = {r.name: r for r in DEFAULT_HEADING_RULES}

# リスト定義順から priority を自動算出（先頭=最大, 末尾=1）。
_RULE_PRIORITY: dict[str, int] = {
    r.name: len(DEFAULT_HEADING_RULES) - i
    for i, r in enumerate(DEFAULT_HEADING_RULES)
}
_PRIORITY_DIVISOR: float = float(len(DEFAULT_HEADING_RULES) + 1)

# 行ごとの見出し判定で毎回 re.compile しないようにあらかじめコンパイル。
_COMPILED_RULES: list[tuple[HeadingRule, re.Pattern[str]]] = [
    (rule, re.compile(rule.regex)) for rule in DEFAULT_HEADING_RULES if rule.regex
]

# 全ヘッダールールの行頭プレフィックスを OR で連結。^ アンカーは除去して結合する。
# 利用先: _split_heading_linebreaks() / _classify_heading_line()
_HEADING_LINE_PREFIX_PATTERN = (
    r"(?:" + r"|".join(re.sub(r"\^", "", rule.regex) for rule in DEFAULT_HEADING_RULES) + r")"
)
_HEADING_LINE_PREFIX_RE = re.compile(_HEADING_LINE_PREFIX_PATTERN)

# 「日本語文字」とみなす Unicode 範囲。_split_heading_linebreaks の境界判定で使用。
_JA_GLUE = (
    r"\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"  # ひらがな・カタカナ・漢字の範囲
    r"0-9０-９"             # 半角/全角数字
    r"\uFF66-\uFF9F\u31F0-\u31FF"  # 半角カナとカタカナ拡張
    r"、-〃・·"             # 和文句読点・中点
    r"〈-】《》【】"         # 括弧/引用符（法令文で頻出）
    r"（）［］｛｝"         # 全角の丸/角/波括弧
)

# 強い見出し判定。_ensure_blank_before_headings(match) と _split_embedded_headings(search)
# の両用のため ^ アンカーは付けない。
_STRONG_HEADING_LINE_RE = re.compile(r"(?:附則|第[0-9一二三四五六七八九十百千〇]+(?:章|条))")

# 強見出しマーカー直後の文中参照接尾辞（"労働基準法第89条に基づき" 等）。
# _split_embedded_headings で「見出しではなく参照」と判定し分割を抑止する。
_INLINE_REF_RE = re.compile(
    r"(?:"
    r"各号"                                   # 第N条各号
    r"|第[0-9一二三四五六七八九十百千〇]+項"   # 第N条第M項
    r"|の[0-9一二三四五六七八九十百千〇]+"     # 第N条の2 (枝番)
    r"|及び|並びに|又は|若しくは"              # 接続詞による列挙参照
    r"|[にをはがとや等のへもで]"              # 助詞・並列で続く参照
    r")"
)

# 例外的ノード（附則・総則・前文）の判定。_format_chunk_md で "**...**" 装飾に使用。
_EXCEPTIONAL_HEADING_RE = re.compile(r"^(?:附則|総則|前文)(?:\s|$)")

# 漢数字 → 整数の変換テーブル
_KANJI_NUM: dict[str, int] = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


# ============================================================
# 共通ヘルパー
# ============================================================


# 漢数字混じり文字列を整数に変換。失敗時は None。
def _kanji_to_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        pass
    result, current = 0, 0
    for ch in s:
        v = _KANJI_NUM.get(ch)
        if v is None:
            return None
        if v >= 10:
            result += (current or 1) * v
            current = 0
        else:
            current = v
    return result + current or None


# 見出し文字列と開始位置から安定したセクション ID を生成する。
def _section_id(heading: str, start_char: int) -> str:
    key = f"{start_char}|{heading}"
    return "sec_" + hashlib.sha256(key.encode()).hexdigest()[:8]


# 見出しマーカー文字列から比較可能な番号値（int）を抽出する。
# 比較不能な場合は文字列または None を返す。
def _extract_marker_value(marker_type: str, raw_marker: str) -> int | str | None:
    if marker_type in ("japanese_chapter", "japanese_article", "japanese_section", "appendix"):
        m = re.search(r"[0-9一二三四五六七八九十百千〇]+", raw_marker)
        if m:
            return _kanji_to_int(m.group()) or m.group()
        return "附則" if marker_type == "appendix" else None
    if marker_type in ("numeric_dot", "decimal_number"):
        m = re.match(r"(\d+)", raw_marker)
        return int(m.group(1)) if m else None
    if marker_type == "numeric_paren":
        m = re.search(r"[（(](\d+)[)）]", raw_marker)
        return int(m.group(1)) if m else None
    if marker_type == "japanese_paren":
        m = re.search(r"[（(]([一二三四五六七八九十〇]+)[)）]", raw_marker)
        return _kanji_to_int(m.group(1)) if m else None
    return None


# marker_value 列の連続性比率 (0.0-1.0) を返す。
# +1 連続を 1.0、リセット (1 への戻り) を RESET_RATIO_WEIGHT として加点。
# n<2 では判断不能のため呼び出し側の意図に応じた short_default を返す。
def _sequence_score(values: list[int], *, short_default: float = 0.0) -> float:
    if len(values) < 2:
        return short_default
    hits = 0.0
    for i in range(len(values) - 1):
        if values[i + 1] == values[i] + 1:
            hits += 1.0
        elif values[i + 1] == 1 and values[i] >= 1:
            hits += RESET_RATIO_WEIGHT
    return hits / (len(values) - 1)


# heading_type に対応するルールの default_level_hint を返す（不在時は DEFAULT_LEVEL_HINT）。
def _hint_for_type(mtype: str) -> int:
    rule = _RULE_BY_NAME.get(mtype)
    return rule.default_level_hint if rule else DEFAULT_LEVEL_HINT


# ============================================================
# 1. normalize_text
# ============================================================


# チャンク抽出前の正規化。
# - 行内に埋め込まれた強見出しを行分割（"...する。第10条..." → 2 行へ分離）
# - 強見出し前の空行挿入（後段の after_blank 判定を効かせる）
# - 見出し直後の二重改行詰め直し（構造文書専用）
# - 連番列挙アイテム間の空行のみのギャップ詰め（章チャンク経路にも効かせる）
# 改行コード統一・全/半角・行末空白・連続空行圧縮は別工程で済んでいる前提。
def normalize_text(text: str, config: ChunkingConfig | None = None) -> str:
    if not text:
        return ""

    lines = text.split("\n")
    if config is None or config.enable_inline_heading_repair:
        lines = _split_embedded_headings(lines)
    lines = _ensure_blank_before_headings(lines)
    normalized = "\n".join(lines)
    normalized = _split_heading_linebreaks(normalized)
    # 章チャンク（_emit_parent 経路）でも列挙の空行が詰まるよう、ここで一括適用する。
    # 子チャンク経路（_emit_child）でも保険として再呼び出しするが操作は冪等。
    normalized = _collapse_enumeration_blanks(normalized)
    return normalized


# 行内に埋め込まれた強い見出し ("...する。第10条（服務）...") を行分割する。
# 文中参照 ("労働基準法第89条に基づき") は対象外（_INLINE_REF_RE で識別）。
def _split_embedded_headings(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        # 行内のどこかに強見出しがあるかを探す（search）。
        m = _STRONG_HEADING_LINE_RE.search(line)
        if not m or m.start() == 0:
            # 見出しが無い、または既に行頭にある → 何もしない。
            result.append(line)
            continue
        # 直後が参照接尾辞なら文中参照とみなして分割しない。
        if _INLINE_REF_RE.match(line, m.end()):
            result.append(line)
            continue
        pre = line[: m.start()].rstrip()
        if pre:
            result.append(pre)
        result.append(line[m.start():])
    return result


# 強い見出し行の直前が非空行なら空行を挿入する。
def _ensure_blank_before_headings(lines: list[str]) -> list[str]:
    result: list[str] = []
    for i, line in enumerate(lines):
        if _STRONG_HEADING_LINE_RE.match(line.strip()) and i > 0 and lines[i - 1].strip():
            result.append("")
        result.append(line)
    return result


# 二重改行のうち、上が見出し行・下が見出しでない日本語文の箇所だけ詰める。
# 構造文書専用の整備（汎用 normalizer 側ではなく本工程で実施）。
def _split_heading_linebreaks(text: str) -> str:
    pat = re.compile(
        rf"(?<=[{_JA_GLUE}])"
        r"\n"
        rf"(?=\n[{_JA_GLUE}])"
        rf"(?!\n{_HEADING_LINE_PREFIX_PATTERN})"
    )

    out = text
    old = ""
    while out != old:
        old = out

        def repl(m: re.Match[str]) -> str:
            i = m.start()
            line_start = out.rfind("\n", 0, i) + 1
            prev_line = out[line_start:i]
            if _HEADING_LINE_PREFIX_RE.match(prev_line):
                return ""
            return "\n"

        out = pat.sub(repl, out)
    return out


# 行頭ルールマッチを軽量に判定する分類器。スコアリングは行わない。
def _classify_heading_line(line: str) -> tuple[str, int | str | None] | None:
    stripped = line.lstrip()
    if not stripped:
        return None
    for rule, pattern in _COMPILED_RULES:
        m = pattern.match(stripped)
        if m:
            return (rule.name, _extract_marker_value(rule.name, m.group(0)))
    return None


# 列挙項目（数字+点 / 括弧数字 / 括弧漢数字）の種類と番号を返す。
# DEFAULT_HEADING_RULES の numeric_dot と同じ「最大 2 桁・小数/日付を除外」条件で、
# 空行詰めの判定を見出し抽出と一貫させる。章・条等の強見出しは対象外（章/条同士の
# 空行は段落区切りとして残したい）。
_ENUM_NUMERIC_DOT_RE = re.compile(r"^(\d{1,2})[.)．](?!\d)")
_ENUM_NUMERIC_PAREN_RE = re.compile(r"^[（(](\d+)[)）]")
_ENUM_JAPANESE_PAREN_RE = re.compile(r"^[（(]([一二三四五六七八九十〇]+)[)）]")


def _classify_enum_item(line: str) -> tuple[str, int] | None:
    stripped = line.lstrip()
    if not stripped:
        return None
    m = _ENUM_NUMERIC_DOT_RE.match(stripped)
    if m:
        return ("numeric_dot", int(m.group(1)))
    m = _ENUM_NUMERIC_PAREN_RE.match(stripped)
    if m:
        return ("numeric_paren", int(m.group(1)))
    m = _ENUM_JAPANESE_PAREN_RE.match(stripped)
    if m:
        n = _kanji_to_int(m.group(1))
        if n is not None:
            return ("japanese_paren", n)
    return None


# 連番列挙アイテム（1., 2., (1), (2) 等）間の「空行のみ」隙間を詰める。
# OCR 等で混入する余計な空行を補正する目的。本文段落が挟まる場合は段落区切りとして残す。
# 詰める条件: 同タイプ + 番号 +1 連続。章・条等の強見出しは対象外（誤って詰めると
# 別ノードの境界を飲み込んでしまうため）。
def _collapse_enumeration_blanks(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        result.append(lines[i])
        cur = _classify_enum_item(lines[i])

        if cur is None:
            i += 1
            continue

        # 直後の空行をスキップした位置にあるのが「同タイプ + 連番 +1」かを確認。
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j > i + 1 and j < len(lines):
            nxt = _classify_enum_item(lines[j])
            if (
                nxt is not None
                and nxt[0] == cur[0]
                and nxt[1] == cur[1] + 1
            ):
                # i+1 〜 j-1 (空行のみ) を捨て、j を次ループで処理する。
                i = j
                continue

        i += 1

    return "\n".join(result)


# ============================================================
# 2. extract_heading_candidates
# ============================================================


# 各行から見出し候補を抽出し、特徴量ベースでスコアリングする。
def extract_heading_candidates(
    text: str, config: ChunkingConfig
) -> list[HeadingCandidate]:
    if not text:
        return []

    lines = text.split("\n")
    candidates: list[HeadingCandidate] = []
    char_pos = 0

    for line_idx, line in enumerate(lines):
        line_start = char_pos
        char_pos += len(line) + 1  # +1 for \n

        stripped = line.lstrip()
        line_len = len(stripped)

        # 強見出し（附則 / 第N章 / 第N条）は OCR で条文本文が同行に連結された長文行でも
        # 見出しとして扱う。親チャンクの _format_chunk_md 側は _classify_heading_line で
        # 行長に関係なく構造認識しているため、候補抽出もこれに揃え、子チャンクの分割が
        # 親チャンクの MD 化判断と一致するようにする。スコア下限 (min_heading_score) は
        # 引き続き効くため、誤検出は score 段階で抑止される。
        is_strong_heading = bool(_STRONG_HEADING_LINE_RE.match(stripped))

        if not stripped or (
            line_len > config.max_heading_line_length and not is_strong_heading
        ):
            continue

        # 直前が空行か（after_blank の判定材料）。
        prev_blank = line_idx > 0 and not lines[line_idx - 1].strip()

        # ルールマッチ。最初に当たったルールを採用する（リスト順 = 優先順位）。
        matched_rule: HeadingRule | None = None
        raw_marker = ""
        for rule, pattern in _COMPILED_RULES:
            m = pattern.match(stripped)
            if m:
                matched_rule = rule
                raw_marker = m.group(0)
                break
        if matched_rule is None:
            continue

        marker_type = matched_rule.name
        marker_value = _extract_marker_value(marker_type, raw_marker)

        # スコア加算（強見出し / 短行 / 括弧題名 / etc.）
        score = LINE_SCORE_BASE
        if prev_blank:
            score += LINE_BONUS_PREV_BLANK
        if line_len <= LINE_SHORT_MAX:
            score += LINE_BONUS_SHORT_LINE
        if marker_type in ("japanese_chapter", "japanese_article", "appendix"):
            score += LINE_BONUS_CHAPTER_LIKE
        elif marker_type == "japanese_section":
            score += LINE_BONUS_SECTION
        if re.search(r"[（(][^）)\n]{1,30}[）)]", stripped):
            score += LINE_BONUS_PAREN_TITLE
        if not stripped.endswith(("。", ".", "．")):
            score += LINE_BONUS_NO_PERIOD_END

        # スコア減算（長文 / 多句点 / URL）
        if line_len > LINE_LONG_MIN:
            score -= LINE_PENALTY_LONG_LINE
        if stripped.count("。") + stripped.count("．") >= 2:
            score -= LINE_PENALTY_MULTI_PERIOD
        if re.search(r"https?://|@\w+\.", stripped):
            score -= LINE_PENALTY_URL_LIKE

        if score < config.min_heading_score:
            continue

        candidates.append(
            HeadingCandidate(
                text=stripped,
                start_char=line_start,
                marker_type=marker_type,
                marker_value=marker_value,
                score=score,
            )
        )

    # 候補列順に同タイプの番号連続を加点（リセットも弱く加点）。
    _award_sequence_bonus(candidates)
    return candidates


# 同 marker_type で番号が +1 連続またはリセット (→1) なら候補スコアを加点する。
def _award_sequence_bonus(candidates: list[HeadingCandidate]) -> None:
    last: dict[str, int | None] = {}
    for c in candidates:
        if not isinstance(c.marker_value, int):
            continue
        prev = last.get(c.marker_type)
        if prev is not None:
            if c.marker_value == prev + 1:
                c.score += HEADING_SEQ_BONUS
            elif c.marker_value == 1 and prev >= 1:
                # 別親セクションへの遷移を示すリセットは弱めに加点。
                c.score += HEADING_RESET_BONUS
        last[c.marker_type] = c.marker_value


# ============================================================
# 3. analyze_heading_groups
# ============================================================


# 見出しタイプごとに統計値（出現数 / 平均間隔 / 連番性 / 包含 / リセット 等）を計算する。
def analyze_heading_groups(
    candidates: list[HeadingCandidate],
    text: str,
    config: ChunkingConfig,
) -> dict[str, Any]:
    if not candidates:
        return {}

    groups: dict[str, list[HeadingCandidate]] = defaultdict(list)
    for c in candidates:
        groups[c.marker_type].append(c)

    text_len = len(text) or 1
    stats: dict[str, Any] = {}

    for mtype, group in groups.items():
        positions = [c.start_char for c in group]
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        avg_gap = statistics.mean(gaps) if gaps else float(text_len)

        # 連番一貫性（短列時は判断保留として 0.5 を返す）
        seq_values = [c.marker_value for c in group if isinstance(c.marker_value, int)]
        seq_score = _sequence_score(seq_values, short_default=0.5)

        fallback_priority = _RULE_PRIORITY.get(mtype, 1)

        stats[mtype] = {
            "count": len(group),
            "positions": positions,
            "average_gap": avg_gap,
            "sequence_score": seq_score,
            "containment_score": 0.0,
            "reset_score": 0.0,
            "fallback_priority": fallback_priority,
        }

    _compute_containment_and_reset(stats, groups, text_len)
    return stats


# タイプ A の各区間にタイプ B が何件入るかを集計し、内包傾向 / リセット傾向を計算する。
# A_avg_gap が B_avg_gap より十分大きい（GAP_OUTER_RATIO 倍超）場合のみ「A は B の外側候補」とみなす。
def _compute_containment_and_reset(
    stats: dict[str, Any],
    groups: dict[str, list[HeadingCandidate]],
    text_len: int,
) -> None:
    types = list(stats.keys())
    sentinel = text_len + 1

    for a_type in types:
        a_positions = stats[a_type]["positions"]
        a_avg_gap = stats[a_type]["average_gap"]
        # A の (start, end) 区間境界
        boundaries = list(zip(a_positions, a_positions[1:] + [sentinel]))

        total_containment = 0.0
        total_reset = 0.0
        compared = 0

        for b_type in types:
            if b_type == a_type:
                continue
            b_avg_gap = stats[b_type]["average_gap"]
            if a_avg_gap <= b_avg_gap * GAP_OUTER_RATIO:
                continue

            b_cands = groups[b_type]
            counts_in_ranges: list[int] = []
            for a_start, a_end in boundaries:
                contained = [c for c in b_cands if a_start <= c.start_char < a_end]
                counts_in_ranges.append(len(contained))
                # この区間の最初の B 番号が 1 ならリセットとみなす。
                b_vals = [c.marker_value for c in contained if isinstance(c.marker_value, int)]
                if b_vals and b_vals[0] == 1:
                    total_reset += 1.0

            avg_contained = statistics.mean(counts_in_ranges) if counts_in_ranges else 0.0
            if avg_contained >= 1.0:
                total_containment += min(1.0, avg_contained / CONTAINMENT_CAP)
                compared += 1

        stats[a_type]["containment_score"] = total_containment / max(compared, 1)
        stats[a_type]["reset_score"] = total_reset / max(len(a_positions), 1)


# ============================================================
# 4. infer_heading_levels
# ============================================================


# 候補の見出しタイプから階層 level を推定して付与する。
# - 出現数が min_group_count 未満のタイプはランキング除外し default_level_hint を使う。
# - 残りのタイプを「外側らしさスコア」で降順ソートし、level=1, 2, 3 ... を割り当てる。
def infer_heading_levels(
    candidates: list[HeadingCandidate],
    group_stats: dict[str, Any],
    config: ChunkingConfig,
    *,
    text_len: int = 1,
) -> list[HeadingCandidate]:
    if not candidates:
        return candidates

    # 統計が無い / 推論無効ならルール既定で埋めて終了。
    if not group_stats or not config.enable_level_inference:
        for c in candidates:
            c.inferred_level = _hint_for_type(c.marker_type)
        return candidates

    infer_types = {
        mtype for mtype, st in group_stats.items()
        if st["count"] >= config.min_group_count
    }
    fallback_types = set(group_stats.keys()) - infer_types

    # 推定対象タイプの「外側らしさスコア」を計算する。
    type_outer_score: dict[str, float] = {}
    for mtype in infer_types:
        st = group_stats[mtype]
        safe_len = max(text_len, 1)
        gap_score = min(st["average_gap"] / safe_len * INFER_GAP_SCALE, INFER_GAP_CAP)
        type_outer_score[mtype] = (
            st.get("containment_score", 0.0) * INFER_W_CONTAINMENT
            + st.get("reset_score", 0.0) * INFER_W_RESET
            + st.get("sequence_score", 0.0) * INFER_W_SEQUENCE
            + (st["fallback_priority"] / _PRIORITY_DIVISOR) * INFER_W_PRIORITY
            + gap_score
        )

    # スコア降順で level=1..N を割り当てる。
    sorted_types = sorted(type_outer_score, key=lambda t: type_outer_score[t], reverse=True)
    type_to_level: dict[str, int] = {
        t: min(i + 1, config.max_depth) for i, t in enumerate(sorted_types)
    }

    # フォールバックタイプは default_level_hint を使う。
    for mtype in fallback_types:
        type_to_level[mtype] = min(_hint_for_type(mtype), config.max_depth)

    for c in candidates:
        lv = type_to_level.get(c.marker_type, _hint_for_type(c.marker_type))
        c.inferred_level = min(lv, config.max_depth)

    return candidates


# ============================================================
# 5. build_section_tree
# ============================================================


# 推定済み候補から親子構造のセクションツリーを構築する。
# 初期構築後に _rebuild_tree_by_recursive_scoring() を呼び、各スコープ内で
# 連番性 / 内部包含 / カバレッジ等のスコアによる level 再決定を行う。
def build_section_tree(
    text: str,
    candidates: list[HeadingCandidate],
    config: ChunkingConfig | None = None,
) -> SectionNode:
    if config is None:
        config = ChunkingConfig()

    first_line = next((ln.strip() for ln in text.split("\n") if ln.strip()), "document")
    text_len = len(text)

    root = SectionNode(
        id=_section_id(first_line, 0),
        heading=first_line,
        heading_type="document_root",
        level=0,
        text="",
        start_char=0,
        parent_id=None,
    )

    valid = [c for c in candidates if c.inferred_level is not None]

    # 見出し候補ゼロ → 段落フォールバックでツリーを埋める。
    if not valid:
        root.text = text
        if config.fallback_to_paragraph:
            _add_paragraph_fallback(root, text)
        return root

    # 各候補のテキスト終端 = 次の「同階層以上の見出し」の start_char。
    def _section_end(idx: int, level: int) -> int:
        for j in range(idx + 1, len(valid)):
            if valid[j].inferred_level <= level:
                return valid[j].start_char
        return text_len

    # スタックベースで一次ツリーを構築する。
    stack: list[tuple[int, SectionNode]] = [(0, root)]
    for i, cand in enumerate(valid):
        end = _section_end(i, cand.inferred_level)
        node = SectionNode(
            id=_section_id(cand.text, cand.start_char),
            heading=cand.text,
            heading_type=cand.marker_type,
            level=cand.inferred_level,
            text=text[cand.start_char:end],
            start_char=cand.start_char,
            parent_id=None,
        )
        while len(stack) > 1 and stack[-1][0] >= cand.inferred_level:
            stack.pop()
        parent_node = stack[-1][1]
        node.parent_id = parent_node.id
        parent_node.children.append(node)
        stack.append((cand.inferred_level, node))

    # root 本文 = 最初の見出し直前のテキスト（前文）
    root.text = text[: valid[0].start_char]

    # 各スコープ内で再帰的スコアリングを行い、親子関係を再決定する。
    # 章が 1 件のみで条と numeric_dot が混在する小規模規程など、固定優先度では
    # 拾えない構造に対しても、内部連続パターンの包含関係を主軸に level を振り直す。
    _rebuild_tree_by_recursive_scoring(root, text, config)
    return root


# ツリー配下を再帰的スコアリングで再構築する。
# 手順:
#   1) 既存ツリーの全ノードを document order で flat 化
#   2) 各ノードの marker_value を heading 文字列から再抽出
#   3) ルートスコープで _assign_levels_recursive() を呼び level を振る
#   4) 採用ノード (level>=1) のみでスタックベースに親子関係を再構築
#   5) text 範囲（[start_char, 次 sibling の start_char or 親 end)）を再計算
def _rebuild_tree_by_recursive_scoring(
    root: SectionNode,
    full_text: str,
    config: ChunkingConfig,
) -> None:
    nodes: list[SectionNode] = []

    def _collect(n: SectionNode) -> None:
        for c in n.children:
            nodes.append(c)
            _collect(c)

    _collect(root)
    if not nodes:
        return
    nodes.sort(key=lambda n: n.start_char)

    # SectionNode は marker_value を持たないため heading 文字列から再抽出する。
    node_values: dict[str, int | str | None] = {
        n.id: _extract_marker_value(n.heading_type, n.heading) for n in nodes
    }

    # 既存の親子関係をクリアしてから level を振り直す。
    root.children = []
    for n in nodes:
        n.children = []
        n.level = 0  # 採用外の初期値

    text_len = len(full_text)
    _assign_levels_recursive(
        nodes, node_values,
        scope_start=0, scope_end=text_len,
        current_level=1, max_depth=config.max_depth,
    )

    # 採用ノード (level >= 1) のみツリーへ組み込む。
    accepted = [n for n in nodes if n.level >= 1]
    stack: list[tuple[int, SectionNode]] = [(root.level, root)]
    for n in accepted:
        n.level = min(n.level, config.max_depth)
        while len(stack) > 1 and stack[-1][0] >= n.level:
            stack.pop()
        parent_node = stack[-1][1]
        n.parent_id = parent_node.id
        parent_node.children.append(n)
        stack.append((n.level, n))

    # children の再 attach に伴い text 範囲を再計算する。
    def _recompute_text(n: SectionNode, end_char: int) -> None:
        if n.heading_type != "document_root":
            n.text = full_text[n.start_char:end_char]
        children = sorted(n.children, key=lambda c: c.start_char)
        for i, c in enumerate(children):
            child_end = children[i + 1].start_char if i + 1 < len(children) else end_char
            _recompute_text(c, child_end)

    _recompute_text(root, text_len)


# スコープ内のノード群を再帰的にスコアリングして level を割り当てる。
# 手順:
#   1) heading_type ごとにグルーピング
#   2) 各 type をスコープ内で _score_type_in_scope() でスコア化
#      （preferred_subtypes に該当する type には SCOPE_W_PREFERRED の bias を付与）
#   3) 最高スコア type を採用。同 default_level_hint かつ正スコアの他 type も同 level に昇格
#      （附則と章を level=1 へ並列配置するケースなど）
#   4) 採用 type より「内側らしい」(default_level_hint がより大きい) 正スコア type の中から
#      最高スコアの 1 つを次の sub_scope 用の preferred_subtypes として選出
#   5) 採用ノードに current_level を付与し、隣接採用ノード間の sub_scope を再帰評価
def _assign_levels_recursive(
    nodes: list[SectionNode],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
    current_level: int,
    max_depth: int,
    *,
    preferred_subtypes: frozenset[str] = frozenset(),
) -> None:
    if not nodes or current_level > max_depth:
        return

    by_type: dict[str, list[SectionNode]] = defaultdict(list)
    for n in nodes:
        by_type[n.heading_type].append(n)

    scores: dict[str, float] = {
        t: _score_type_in_scope(
            g, by_type, node_values, scope_start, scope_end,
            preferred_types=preferred_subtypes,
        )
        for t, g in by_type.items()
    }

    if not scores or max(scores.values()) <= 0.0:
        return

    best_type = max(scores, key=lambda t: scores[t])
    best_hint = _hint_for_type(best_type)

    # ベストと同 default_level_hint で正スコアの type も同 level へ昇格。
    # 別 hint のノイズ（hint=4 の numeric_dot 等）が hint=2 の article と並列に
    # 繰り上がることはない（同 hint 同士のみ昇格）。
    chosen_types = {
        t for t, s in scores.items()
        if _hint_for_type(t) == best_hint and s > 0.0
    }

    # 採用 type より「内側らしい」(default_level_hint がより大きい) 正スコア type の中で
    # 最高スコアの 1 つを次階層の予約として伝搬する。
    # 章 (hint=1) が選ばれたら条 (hint=2) を、条が選ばれたら numeric_paren (hint=4) を
    # 予約することで、少数派が多数派の列挙に競り負けて階層から落ちるのを防ぐ。
    next_preferred: frozenset[str] = frozenset()
    inner_candidates = [
        (t, s) for t, s in scores.items()
        if t not in chosen_types
        and _hint_for_type(t) > best_hint
        and s > 0.0
    ]
    if inner_candidates:
        inner_candidates.sort(key=lambda x: x[1], reverse=True)
        next_preferred = frozenset({inner_candidates[0][0]})

    chosen_nodes = sorted(
        [n for n in nodes if n.heading_type in chosen_types],
        key=lambda n: n.start_char,
    )
    for n in chosen_nodes:
        n.level = current_level

    # 未採用 type は隣接採用ノード間の sub_scope ごとに再帰評価。
    others = [n for n in nodes if n.heading_type not in chosen_types]
    for i, c in enumerate(chosen_nodes):
        sub_start = c.start_char
        sub_end = chosen_nodes[i + 1].start_char if i + 1 < len(chosen_nodes) else scope_end
        sub_others = [x for x in others if sub_start < x.start_char < sub_end]
        _assign_levels_recursive(
            sub_others, node_values,
            sub_start, sub_end,
            current_level + 1, max_depth,
            preferred_subtypes=next_preferred,
        )


# 1 つの heading_type が、与えられたスコープ内で「その階層らしい」度合いをスコア化する。
# 指標 (重み):
#   - containment (SCOPE_W_CONTAINMENT): 連続ノード間に別 type の連続パターンを内包する区間の比率
#   - sequence    (SCOPE_W_SEQUENCE)   : 自身の marker_value の +1 連続性（リセット 0.5 で正評価）
#   - coverage    (SCOPE_W_COVERAGE)   : スコープ内の位置範囲（広いほど外側らしい）
#   - frequency   (SCOPE_W_FREQUENCY)  : 出現数の log スケール（頻度の偏り抑制）
#   - priority    (SCOPE_W_PRIORITY)   : ヘッダー語ヒント（tie-breaker）
#   - preferred   (SCOPE_W_PREFERRED)  : 上位スコープで予約された次点 type への bias
def _score_type_in_scope(
    group: list[SectionNode],
    all_groups: dict[str, list[SectionNode]],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
    *,
    preferred_types: frozenset[str] = frozenset(),
) -> float:
    if not group:
        return 0.0

    positions = sorted(c.start_char for c in group)
    values = [v for v in (node_values.get(n.id) for n in group) if isinstance(v, int)]

    # 1) 内部連続パターンの包含率
    containment = _interval_containment(group, all_groups, node_values, scope_start, scope_end)

    # 2) 自身の連番一貫性
    seq = _sequence_score(values)

    # 3) スコープ内カバレッジ
    scope_size = max(scope_end - scope_start, 1)
    if len(positions) >= 2:
        coverage = (positions[-1] - positions[0]) / scope_size
    else:
        # 単独配置: 先頭付近にあれば外側を覆う wrapper とみなして高評価。
        head_zone = scope_start + scope_size * SCOPE_HEAD_ZONE
        coverage = (
            SCOPE_COVERAGE_SOLO_HEAD
            if positions and positions[0] <= head_zone
            else SCOPE_COVERAGE_SOLO_OTHER
        )

    # 4) 出現頻度（スコープ内全候補数で正規化。スコープが大きいほど基準が上がる）
    total_in_scope = sum(len(g) for g in all_groups.values())
    freq = math.log(1 + len(group)) / math.log(max(total_in_scope, 2))

    # 5) ヘッダー語ヒント（tie-breaker のみ）
    priority_hint = _RULE_PRIORITY.get(group[0].heading_type, 0) / _PRIORITY_DIVISOR

    # 6) 上位スコープからの「次点予約」bias。
    # 章 → 条 → 列挙の階層構造を保つため、上位スコープで「内側で最も有力」と
    # 認定された type に対し、直下スコープでのみ加点する（非伝搬）。
    preferred_bonus = 1.0 if group[0].heading_type in preferred_types else 0.0

    return (
        containment * SCOPE_W_CONTAINMENT
        + seq * SCOPE_W_SEQUENCE
        + coverage * SCOPE_W_COVERAGE
        + freq * SCOPE_W_FREQUENCY
        + priority_hint * SCOPE_W_PRIORITY
        + preferred_bonus * SCOPE_W_PREFERRED
    )


# group の各区間に、別 type の「連続パターン (>=2 ノードで marker_value が +1 連続)」を
# 含む区間の比率を返す。最外側階層を見抜くための主信号。
#
# 区間の取り方（"連番かつ閉じられている" の評価）:
#   - count >= 2: 連続する同 type ノード間（閉区間）のみを計上。最後のノードから scope_end
#                 までの open な末尾は計上しない（外側 type を内包と誤評価しないため）。
#   - count == 1: スコープ先頭付近にある場合のみ wrapper 候補として (node, scope_end) を 1 区間扱い。
#                 中ほどの単独出現は子要素である可能性が高く wrapper 扱いしない。
def _interval_containment(
    group: list[SectionNode],
    all_groups: dict[str, list[SectionNode]],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
) -> float:
    positions = sorted(c.start_char for c in group)
    if not positions:
        return 0.0

    if len(positions) == 1:
        scope_size = max(scope_end - scope_start, 1)
        head_zone = scope_start + scope_size * SCOPE_HEAD_ZONE
        if positions[0] > head_zone:
            return 0.0
        boundaries: list[tuple[int, int]] = [(positions[0], scope_end)]
    else:
        boundaries = list(zip(positions, positions[1:]))

    own_type = group[0].heading_type
    n_with_seq = 0
    for a, b in boundaries:
        for inner_type, inner_group in all_groups.items():
            if inner_type == own_type:
                continue
            inner_in = sorted(
                [c for c in inner_group if a < c.start_char < b],
                key=lambda c: c.start_char,
            )
            if len(inner_in) < 2:
                continue
            inner_vals = [
                v for v in (node_values.get(n.id) for n in inner_in) if isinstance(v, int)
            ]
            if len(inner_vals) < 2:
                continue
            consecutive = any(
                inner_vals[i + 1] == inner_vals[i] + 1
                for i in range(len(inner_vals) - 1)
            )
            if consecutive:
                n_with_seq += 1
                break

    return n_with_seq / len(boundaries)


# 見出し抽出ゼロ時の最終フォールバック。段落単位で children を作る。
def _add_paragraph_fallback(root: SectionNode, text: str) -> None:
    char_pos = 0
    for para in re.split(r"\n{2,}", text):
        para_len = len(para)
        if para.strip():
            root.children.append(
                SectionNode(
                    id=_section_id(para[:40], char_pos),
                    heading="",
                    heading_type="paragraph",
                    level=1,
                    text=para,
                    start_char=char_pos,
                    parent_id=root.id,
                )
            )
        char_pos += para_len + 2  # +2 for \n\n


# ============================================================
# 6. flatten_chunks
# ============================================================


# テキストを max_chars 以下でなるべく均等な長さに分割する。
def _split_text_evenly(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []
    n = math.ceil(len(text) / max_chars)
    target = math.ceil(len(text) / n)
    parts = [text[i * target:(i + 1) * target] for i in range(n)]
    return [p for p in parts if p.strip()] or [text]


# サイズ配列を max_per_group 以下に均等分割し、各グループのインデックス配列を返す。
# 合計が max 以下なら 1 グループにまとめる。
# それ以外は n=ceil(total/max) を最小分割数とし、target=ceil(total/n) を均等ターゲットとする。
def _group_items_balanced(
    item_sizes: list[int], max_per_group: int
) -> list[list[int]]:
    if not item_sizes:
        return []
    total = sum(item_sizes)
    if total <= max_per_group:
        return [list(range(len(item_sizes)))]

    n = math.ceil(total / max_per_group)
    target = math.ceil(total / n)

    groups: list[list[int]] = []
    current: list[int] = []
    current_size = 0

    for i, size in enumerate(item_sizes):
        hard_overflow = current and current_size + size > max_per_group
        soft_overflow = current and current_size >= target and len(groups) < n - 1
        if hard_overflow or soft_overflow:
            groups.append(current)
            current = []
            current_size = 0
        current.append(i)
        current_size += size

    if current:
        groups.append(current)
    return groups


def default_metadata_builder(
    *,
    chunk_id: str,
    node: SectionNode,
    path: list[str],
    section_parent_id: str | None,
    ancestor_chain: list[dict[str, Any]],
    doc_root_id: str,
    split_index: int = 0,
    split_total: int = 1,
) -> dict[str, Any]:
    """既定の metadata builder。最低限の検索/系統情報のみ生成する。

    差し替え方法:
      chunker.py 等で独自フィールド (doc_id, source 等) を含めたい場合は、
      本関数と同じシグネチャ (キーワード引数のみ) を持つ関数を作成し、
      flatten_chunks() の metadata_builder 引数に渡す。
    """
    # heading_type=="paragraph" は見出し抽出失敗時のフォールバック扱い。
    # level>=3 のノードは通常独立チャンク化しないが safety net として child へ寄せる。
    if node.heading_type == "paragraph":
        chunk_role = "fallback"
    elif node.level == 1:
        chunk_role = "parent"
    else:
        chunk_role = "child"

    return {
        "chunk_id": chunk_id,
        "parent_id": section_parent_id,
        "root_id": doc_root_id,
        "level": node.level,
        "path_text": " > ".join(path),
        "chunk_role": chunk_role,
        "chunking_strategy": "structure_aware_v4",
    }


# ノード text から先頭の見出し行を除いた本文の、空白行を除いた行数。
# 子チャンク採否（>=2）/ 昇格判定（>=1 採用、==0 除外）に使用。
def _meaningful_body_lines(text: str, heading: str) -> int:
    body = text.strip()
    if heading and body.startswith(heading):
        body = body[len(heading):]
    return sum(1 for ln in body.split("\n") if ln.strip())


# チャンク本文にマークダウン見出し記法を付与する。
# 適用ルール:
#   - 親ノードの見出し行  → "### " (例外的ノードは "**...**")
#   - 子ノードの見出し行  → "## "
#   - 列挙構造の見出し行  → "# "
#   - それ以外            → そのまま
# 加えて、同 heading_type で構造ツリーから漏れた行（候補スコア閾値で落ちた長文項目等）
# は、子/孫として採用された type 集合と一致する場合に同じ記法を補完付与する。
# 文字列完全一致では拾えないバラバラ付与（"1．要約 / 2．長文 / 3．要約" のような混在）
# を防ぎつつ、構造ツリー外の行頭パターンを丸ごと拾うような強制マッピングは行わない。
def _format_chunk_md(
    text: str,
    node: SectionNode,
    *,
    is_parent: bool,
    config: ChunkingConfig,
) -> str:
    if not config.output_markdown:
        return text

    node_heading = node.heading.strip()
    child_headings = {c.heading.strip() for c in node.children if c.heading.strip()}
    grandchild_headings: set[str] = {
        gc.heading.strip()
        for child in node.children
        for gc in child.children
        if gc.heading.strip()
    }
    # 構造ツリーで child / grandchild として認識された heading_type の集合。
    # 同 type なら、文字列マッチで漏れた行にも同じ記法を補完する根拠とする。
    child_types: set[str] = {c.heading_type for c in node.children}
    grandchild_types: set[str] = {
        gc.heading_type
        for child in node.children
        for gc in child.children
    }

    is_exceptional = (
        node.heading_type == "appendix"
        or bool(_EXCEPTIONAL_HEADING_RE.match(node_heading))
    )

    result: list[str] = []
    heading_found = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # 1) ノード自身の見出し行
        if not heading_found and stripped == node_heading:
            heading_found = True
            if is_exceptional and is_parent:
                result.append(f"**{stripped}**")
            elif is_parent:
                result.append(f"### {stripped}")
            else:
                result.append(f"## {stripped}")
            continue

        # 2) 構造ツリーで採用された child / grandchild との文字列完全一致
        if stripped in child_headings:
            result.append(f"## {stripped}" if is_parent else f"# {stripped}")
            continue
        if stripped in grandchild_headings:
            result.append(f"# {stripped}")
            continue

        # 3) 採用された type と同じ heading_type の行は補完する。
        # 章チャンクで child=条 が認識されていれば本文中の条文行（文字列ぶれや候補漏れ）
        # を拾い、孫=列挙が認識されていれば長文の "2．..." も同じ記法に揃える。
        line_class = _classify_heading_line(stripped)
        if line_class is not None:
            line_type = line_class[0]
            if line_type in child_types:
                result.append(f"## {stripped}" if is_parent else f"# {stripped}")
                continue
            if line_type in grandchild_types:
                result.append(f"# {stripped}")
                continue

        result.append(line)

    return "\n".join(result)


# セクションツリーを RAG 投入用チャンク配列へ平坦化する。
#
# 分割戦略 (structure_aware_v4):
#   - level==1 (章相当)  → 親チャンク。max_chunk_chars 超過時は子(条)境界を尊重した
#                         均等分割を行い、各分割の先頭に章見出し prefix を付与する。
#   - level==2 (条相当)  → 子チャンク。本文（見出し除く）が空行を除き 2 行以上で採用。
#                         親章内に基準を満たす同 heading_type が 1 つでもあれば本文 1 行も
#                         昇格採用する（見出しのみ＝本文 0 行は除外）。
#                         単独で max_chunk_chars 超過時のみ見出し付きで均等分割する。
#   - level>=3           → 独立チャンク化せず親/子チャンク本文に含める。
#   - 0 件時             → root を 1 チャンク補償する。
def flatten_chunks(
    root: SectionNode,
    config: ChunkingConfig,
    metadata_builder: MetadataBuilder | None = None,
) -> list[dict]:
    if metadata_builder is None:
        metadata_builder = default_metadata_builder

    doc_root_id = "doc_" + hashlib.sha256(root.heading.encode()).hexdigest()[:8]
    chunks: list[dict] = []

    # チャンク 1 件分の metadata を組み立てる。
    def _build_meta(
        chunk_id: str,
        node: SectionNode,
        path: list[str],
        section_parent_id: str | None,
        ancestor_chain: list[dict[str, Any]],
        *,
        split_index: int = 0,
        split_total: int = 1,
    ) -> dict[str, Any]:
        return metadata_builder(
            chunk_id=chunk_id,
            node=node,
            path=path,
            section_parent_id=section_parent_id,
            ancestor_chain=ancestor_chain,
            doc_root_id=doc_root_id,
            split_index=split_index,
            split_total=split_total,
        )

    # default_builder の metadata には載らないが、差し替え builder が利用する可能性に備えて
    # 渡し続ける ancestor 情報。
    def _ancestor_entry(node: SectionNode) -> dict[str, Any]:
        return {
            "id": f"chunk_{node.id}",
            "heading": node.heading,
            "level": node.level,
            "heading_type": node.heading_type,
        }

    # 共通: path / ancestor_chain / base_id / section_parent_id を一括で組み立てる。
    def _common_meta(node: SectionNode, ancestors: list[SectionNode]):
        ancestor_chain = [_ancestor_entry(a) for a in ancestors]
        path = [root.heading] + [a.heading for a in ancestors] + [node.heading]
        base_id = f"chunk_{node.id}"
        section_parent_id = f"chunk_{node.parent_id}" if node.parent_id else None
        return path, ancestor_chain, base_id, section_parent_id

    # チャンク本文の配列 parts をまとめて出力する。
    # parts が 1 件なら id をそのまま、2 件以上なら "_p{i}" を付ける。
    def _emit_parts(
        parts: list[str],
        *,
        node: SectionNode,
        is_parent: bool,
        base_id: str,
        path: list[str],
        section_parent_id: str | None,
        ancestor_chain: list[dict[str, Any]],
    ) -> None:
        n = len(parts)
        for pi, part in enumerate(parts):
            cid = f"{base_id}_p{pi}" if n > 1 else base_id
            chunks.append({
                "id": cid,
                "text": _format_chunk_md(part, node, is_parent=is_parent, config=config),
                "metadata": _build_meta(
                    cid, node, path, section_parent_id, ancestor_chain,
                    split_index=pi, split_total=n,
                ),
            })

    # level==1（章）→ 親チャンク
    def _emit_parent(node: SectionNode, ancestors: list[SectionNode]) -> None:
        text = node.text.strip()
        if not text or len(text) < config.min_child_text_length:
            return
        path, ancestor_chain, base_id, section_parent_id = _common_meta(node, ancestors)
        kw = dict(
            node=node, is_parent=True, base_id=base_id,
            path=path, section_parent_id=section_parent_id, ancestor_chain=ancestor_chain,
        )

        # 収まるならそのまま 1 チャンク。
        if len(text) <= config.max_chunk_chars:
            _emit_parts([text], **kw)
            return

        # 章サイズ超過: 子(条)境界で均等分割を試みる。
        children_with_text = [c for c in node.children if c.text.strip()]
        if children_with_text:
            first_child = children_with_text[0]
            prefix_text = node.text[: first_child.start_char - node.start_char].rstrip()
        else:
            prefix_text = ""
        if not prefix_text:
            prefix_text = node.heading
        prefix_line = prefix_text + "\n"
        available = config.max_chunk_chars - len(prefix_line)

        # prefix だけで超過 or 子無し → 文字数ベースの単純均等分割。
        if available <= 0 or not children_with_text:
            _emit_parts(_split_text_evenly(text, config.max_chunk_chars), **kw)
            return

        # 子境界で均等グルーピング → 各グループを 1 チャンクに連結（+1 は \n 区切り分）。
        child_texts = [c.text.strip() for c in children_with_text]
        child_sizes = [len(t) + 1 for t in child_texts]
        groups = _group_items_balanced(child_sizes, available)
        parts = [prefix_line + "\n".join(child_texts[i] for i in g) for g in groups]
        _emit_parts(parts, **kw)

    # level==2（条）→ 子チャンク
    def _emit_child(node: SectionNode, ancestors: list[SectionNode]) -> None:
        text = node.text.strip()
        if not text or len(text) < config.min_child_text_length:
            return

        # 連番列挙に紛れた余計な空行を詰めて見栄えを整える。
        text = _collapse_enumeration_blanks(text)

        body_lines = _meaningful_body_lines(text, node.heading)
        if body_lines < 2:
            # 親章内に複数行本文を持つ同 heading_type が 1 つでもあれば 1 行本文も昇格採用。
            # 法規類で 2 行以上の条と 1 行の条が混在する章で後者の漏れを防ぐ救済。
            # ただし本文 0 行（見出しのみ）は検索ノイズ抑制のため除外。
            parent = ancestors[-1] if ancestors else None
            promoted = bool(parent) and any(
                _meaningful_body_lines(c.text.strip(), c.heading) >= 2
                for c in parent.children
                if c.heading_type == node.heading_type
            )
            if not promoted or body_lines == 0:
                return

        path, ancestor_chain, base_id, section_parent_id = _common_meta(node, ancestors)
        kw = dict(
            node=node, is_parent=False, base_id=base_id,
            path=path, section_parent_id=section_parent_id, ancestor_chain=ancestor_chain,
        )

        if len(text) <= config.max_chunk_chars:
            _emit_parts([text], **kw)
            return

        # 条単独で超過 → 見出しを各チャンク先頭に付与しつつ本文を均等分割。
        heading_line = node.heading
        body = text[len(heading_line):].lstrip("\n") if text.startswith(heading_line) else text
        prefix = f"{heading_line}\n"
        available = config.max_chunk_chars - len(prefix)

        if available <= 0:
            # 異常ケース（見出し自体が max_chunk_chars 超）。条文保全のため全文を均等分割。
            _emit_parts(_split_text_evenly(text, config.max_chunk_chars), **kw)
            return

        body_parts = _split_text_evenly(body, available)
        _emit_parts([f"{prefix}{p}" for p in body_parts], **kw)

    # ツリーを再帰的にたどり level に応じて出力する。
    def _walk(node: SectionNode, ancestors: list[SectionNode]) -> None:
        if node.heading_type == "document_root":
            for child in node.children:
                _walk(child, ancestors)
            return
        if node.level == 1:
            _emit_parent(node, ancestors)
        elif node.level == 2:
            _emit_child(node, ancestors)
        # level>=3 は独立チャンク化しない（親/子チャンク本文に含まれる）。
        for child in node.children:
            _walk(child, ancestors + [node])

    _walk(root, [])

    # 0 件時の最終フォールバック（最低 1 チャンクを補償）。
    if not chunks:
        chunks.append({
            "id": f"chunk_{root.id}",
            "text": root.text.strip() or "(empty)",
            "metadata": _build_meta(
                f"chunk_{root.id}", root, [root.heading], None, [],
            ),
        })
    return chunks


# ============================================================
# Public API
# ============================================================


# metadata 付き構造認識チャンク化。logic_registry から動的解決される主経路。
# chunk_overlap は構造認識の性質上未使用（互換のため受け取るのみ）。
def split_for_rag_with_metadata(
    *, text: str, chunk_size: int = 800, chunk_overlap: int = 0
) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    config = ChunkingConfig(max_chunk_chars=chunk_size)
    normalized = normalize_text(text, config)
    candidates = extract_heading_candidates(normalized, config)
    group_stats = analyze_heading_groups(candidates, normalized, config)
    candidates = infer_heading_levels(
        candidates, group_stats, config, text_len=len(normalized)
    )
    root = build_section_tree(normalized, candidates, config)
    return [
        {"text": c["text"], "metadata": c.get("metadata", {})}
        for c in flatten_chunks(root, config)
        if c["text"]
    ]
