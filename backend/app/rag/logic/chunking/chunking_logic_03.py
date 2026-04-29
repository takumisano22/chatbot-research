from __future__ import annotations

import argparse
import hashlib
import json
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
# 関数を flatten_chunks() / split_for_rag_structure_aware() に渡す。
MetadataBuilder = Callable[..., dict[str, Any]]

# ============================================================
# 主要処理フロー（呼び出し関係）
# ============================================================
# 1) split_for_rag_structure_aware()
#    -> normalize_text()
#       -> _ensure_blank_before_headings()
#    -> extract_heading_candidates()
#       -> _extract_marker_value()
#       -> _award_sequence_bonus()
#    -> analyze_heading_groups()
#       -> _compute_sequence_score()
#       -> _compute_containment_and_reset()
#    -> infer_heading_levels()
#    -> build_section_tree()
#       -> _rebuild_tree_by_recursive_scoring()  # スコープごとに再帰スコアで level 決定
#           -> _assign_levels_recursive()
#               -> _score_type_in_scope()
#                   -> _interval_containment()        # 内部連続パターンの包含率
#                   -> _sequence_consistency_values() # +1 連続率
#       -> _add_paragraph_fallback()  # 見出し無し時のみ
#    -> flatten_chunks(metadata_builder=...)  # 親(章)/子(条) チャンク生成 (structure_aware_v4)
#       -> default_metadata_builder() # 既定の metadata 生成 (差し替え可)
#       -> _meaningful_body_lines()   # 条本文の有意行数 (子チャンク採否/昇格判定)
#       -> _format_chunk_md()         # マークダウン見出し記法の付与 (output_markdown=True 時)
#       -> _group_items_balanced()    # 親チャンクの子境界均等分割
#       -> _split_text_evenly()       # 子チャンク超過時のフォールバック均等分割
#
# 2) split_for_rag(chunk_size=800)
#    -> split_for_rag_structure_aware() をラップ
#
# 3) main() (CLI)
#    -> split_for_rag_structure_aware() と同等の各ステップを順に実行


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
    max_chunk_chars: int = 1500


@dataclass
class HeadingRule:
    name: str
    regex: str
    default_level_hint: int  # フォールバック時のlevel (1=最外側)
    priority: int             # 高いほど外側に推定されやすい


@dataclass
class HeadingCandidate:
    text: str
    start_char: int
    marker_type: str
    marker_value: int | str | None
    indent: int
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
# ルール定義
# ============================================================

DEFAULT_HEADING_RULES: list[HeadingRule] = [
    # priority が高いほど「外側階層らしい」デフォルト扱い
    # 第3引数は default_level_hint（フォールバック時の推定階層）を表す。
    # 例: HeadingRule("japanese_article", ..., 2, 9) の 2 が default_level_hint。
    # 利用箇所:
    #   - infer_heading_levels() で「統計不足/推定無効時」の代替 level に利用
    #   - infer_heading_levels() で fallback_types の level 決定に利用
    # priority は HeadingRule の第4引数（末尾の数値）に定義している。
    # 例: HeadingRule("appendix", ..., 1, 11) の 11 が priority。
    # 利用箇所:
    #   - analyze_heading_groups() で fallback_priority として取り込み
    #   - infer_heading_levels() で外側スコア計算の補助要素として加算
    HeadingRule("appendix", r"^附則", 1, 11),
    HeadingRule("japanese_chapter", r"^第[0-9一二三四五六七八九十百千〇]+章", 1, 10),
    HeadingRule("japanese_article", r"^第[0-9一二三四五六七八九十百千〇]+条", 2, 9),
    HeadingRule("japanese_section", r"^第[0-9一二三四五六七八九十百千〇]+項", 3, 8),
    HeadingRule("decimal_number", r"^\d+\.\d+", 3, 7),
    HeadingRule("numeric_dot", r"^\d+[.)．]\s", 4, 6),
    HeadingRule("numeric_paren", r"^[（(]\d+[)）]", 4, 5),
    HeadingRule("japanese_paren", r"^[（(][一二三四五六七八九十〇]+[)）]", 4, 5),
    HeadingRule("roman", r"^[IVX]{1,5}[.)]\s|^[Ⅰ-Ⅻ]\s", 4, 4),
    HeadingRule("alpha", r"^[A-Z][.)]\s", 4,  3),
    HeadingRule("circle_bullet", r"^○\s*\S", 5, 3),
    HeadingRule("bullet", r"^[•・\-]\s", 5, 2),
]

# コンパイル済みルール (モジュール読込時に1回だけ生成)
# - 目的: 行ごとの見出し判定で毎回 re.compile しないようにして高速化する
# - 利用箇所: extract_heading_candidates() の「ルールマッチ」ループ
_COMPILED_RULES: list[tuple[HeadingRule, re.Pattern[str]]] = [
    (rule, re.compile(rule.regex)) for rule in DEFAULT_HEADING_RULES if rule.regex
]

# 強い見出し判定用の正規表現。
# - _ensure_blank_before_headings: match() で行頭判定 (match() 自体が暗黙の先頭アンカー)
# - _split_embedded_headings: search() で行内のどこかにあるかを判定
# 上記両方で使えるよう ^ アンカーは敢えて付けない。
_STRONG_HEADING_LINE_RE = re.compile(r"(?:附則|第[0-9一二三四五六七八九十百千〇]+(?:章|条))")


# 強い見出しマーカー (第N条 等) の直後に続いた場合に、それが文中参照
# (例: "労働基準法第89条に基づき") であることを示す接尾辞パターン。
# _split_embedded_headings で「見出しではなく参照だから分割しない」判定に使う。
# 見出し直後位置を起点に re.Pattern.match() で当てることを前提とし、^ アンカーは付けない。
_INLINE_REF_RE = re.compile(
    r"(?:"
    r"各号"                                       # 第N条各号
    r"|第[0-9一二三四五六七八九十百千〇]+項"       # 第N条第M項
    r"|の[0-9一二三四五六七八九十百千〇]+"         # 第N条の2 (枝番)
    r"|及び|並びに|又は|若しくは"                  # 接続詞による列挙参照
    r"|[にをはがとや等のへもで]"                   # 助詞・並列で続く参照
    r")"
)

# 例外的ノード (附則・総則等) の見出し判定用正規表現。
# _format_chunk_md() で親チャンクの見出しを "**...**" で囲む判定に使用。
_EXCEPTIONAL_HEADING_RE = re.compile(r"^(?:附則|総則|前文)(?:\s|$)")

# 漢数字変換テーブル
_KANJI_NUM: dict[str, int] = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


# ============================================================
# ヘルパー
# ============================================================


def _kanji_to_int(s: str) -> int | None:
    """漢数字混じり文字列を整数に変換。失敗時はNone。"""
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


def _section_id(heading: str, start_char: int) -> str:
    """見出し文字列と開始位置から安定したセクションIDを生成する。"""
    key = f"{start_char}|{heading}"
    return "sec_" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _extract_marker_value(marker_type: str, raw_marker: str) -> int | str | None:
    """見出しマーカー文字列から比較可能な番号値を抽出する。"""
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


# ============================================================
# 1. normalize_text
# ============================================================


def normalize_text(text: str, config: ChunkingConfig | None = None) -> str:
    """チャンク抽出前の正規化を行う。

    実施内容:
      - 行内に埋め込まれた強見出しを行分割
        ("...する。第10条..." を 2 行に分けて見出しを行頭へ揃える)
      - 強見出し前の空行挿入

    改行コード統一・全角/半角・行末空白・連続空行の圧縮は別工程で
    正規化済みである前提のため、ここでは扱わない。
    """
    if not text:
        return ""

    lines = text.split("\n")

    # 行内に埋め込まれた強い見出しを行分割する
    if config is None or config.enable_inline_heading_repair:
        lines = _split_embedded_headings(lines)

    # 強い見出し行の直前に空行が無ければ挿入し、後段の after_blank 判定を効かせる
    lines = _ensure_blank_before_headings(lines)
    return "\n".join(lines)


def _split_embedded_headings(lines: list[str]) -> list[str]:
    """"...する。第10条（服務）..." のような行内埋め込み見出しを分割する。

    文中参照 (例: "労働基準法第89条に基づき") は対象外で、見出し直後の文字が
    _INLINE_REF_RE に当てはまる場合は分割しない。

    判定条件:
      - 行内に強見出しパターン (附則|第N章|第N条) が存在
      - そのマッチ位置が行頭ではない (行頭なら既に整っているので何もしない)
      - 直後が参照接尾辞 (に・各号・第M項 等) ではない
    """
    result: list[str] = []
    for line in lines:
        # 行内のどこかに強見出しがあるかを探す (search; 行頭限定の match ではない)
        m = _STRONG_HEADING_LINE_RE.search(line)
        if not m or m.start() == 0:
            # 見出しが無い or 既に行頭にある → 分割不要
            result.append(line)
            continue
        # 見出し直後が参照接尾辞なら文中参照とみなして分割しない
        if _INLINE_REF_RE.match(line, m.end()):
            result.append(line)
            continue
        pre = line[: m.start()].rstrip()
        heading_and_rest = line[m.start():]
        if pre:
            result.append(pre)
        result.append(heading_and_rest)
    return result


def _ensure_blank_before_headings(lines: list[str]) -> list[str]:
    """強い見出し行 (第n章/第n条/附則) の直前が非空行なら空行を挿入する。"""
    result: list[str] = []
    for i, line in enumerate(lines):
        if _STRONG_HEADING_LINE_RE.match(line.strip()) and i > 0 and lines[i - 1].strip():
            result.append("")
        result.append(line)
    return result


# ============================================================
# 2. extract_heading_candidates
# ============================================================


def extract_heading_candidates(
    text: str, config: ChunkingConfig
) -> list[HeadingCandidate]:
    """各行から見出し候補を抽出し、特徴量ベースでスコアリングする。"""
    if not text:
        return []

    lines = text.split("\n")
    candidates: list[HeadingCandidate] = []
    char_pos = 0

    for line_idx, line in enumerate(lines):
        line_start = char_pos
        char_pos += len(line) + 1  # +1 for \n

        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        line_len = len(stripped)

        if not stripped:
            continue

        # 長すぎる行はスキップ
        if line_len > config.max_heading_line_length:
            continue

        # 直前が空行かどうか
        prev_blank = line_idx > 0 and not lines[line_idx - 1].strip()

        # ルールマッチ
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

        # --- スコアリング ---
        score = 3.0  # 行頭でパターン一致

        if prev_blank:
            score += 1.0
        if line_len <= 40:
            score += 1.0
        if marker_type in ("japanese_chapter", "japanese_article", "appendix"):
            score += 4.0
        elif marker_type == "japanese_section":
            score += 2.0
        if re.search(r"[（(][^）)\n]{1,30}[）)]", stripped):
            score += 2.0
        if not stripped.endswith(("。", ".", "．")):
            score += 1.0
        if indent == 0:
            score += 1.0
        elif indent <= 2:
            score += 0.5

        # 減点
        if line_len > 60:
            score -= 2.0
        if stripped.count("。") + stripped.count("．") >= 2:
            score -= 2.0
        if re.search(r"https?://|@\w+\.", stripped):
            score -= 3.0

        if score < config.min_heading_score:
            continue

        candidates.append(
            HeadingCandidate(
                text=stripped,
                start_char=line_start,
                marker_type=marker_type,
                marker_value=marker_value,
                indent=indent,
                score=score,
            )
        )

    # 連番ボーナス: 同タイプの候補が+1連続なら加点
    _award_sequence_bonus(candidates)

    return candidates


def _award_sequence_bonus(candidates: list[HeadingCandidate]) -> None:
    """同marker_typeで番号が+1連続またはリセット(→1)なら加点する。"""
    last: dict[str, int | None] = {}
    for c in candidates:
        if not isinstance(c.marker_value, int):
            continue
        prev = last.get(c.marker_type)
        if prev is not None:
            if c.marker_value == prev + 1:
                c.score += 2.0
            elif c.marker_value == 1 and prev >= 1:
                # リセット (別親セクションへの遷移) - 弱いボーナス
                c.score += 0.5
        last[c.marker_type] = c.marker_value


# ============================================================
# 3. analyze_heading_groups
# ============================================================


def analyze_heading_groups(
    candidates: list[HeadingCandidate],
    text: str,
    config: ChunkingConfig,
) -> dict[str, Any]:
    """見出しタイプごとに統計を計算し、階層推定の材料を作る。"""
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
        avg_indent = statistics.mean(c.indent for c in group)
        seq_score = _compute_sequence_score(group)
        shallow_indent_score = max(0.0, 1.0 - avg_indent / 10.0)

        rule = next((r for r in DEFAULT_HEADING_RULES if r.name == mtype), None)
        fallback_priority = rule.priority if rule else 1

        stats[mtype] = {
            "count": len(group),
            "positions": positions,
            "average_gap": avg_gap,
            "sequence_score": seq_score,
            "containment_score": 0.0,
            "reset_score": 0.0,
            "shallow_indent_score": shallow_indent_score,
            "fallback_priority": fallback_priority,
        }

    _compute_containment_and_reset(stats, groups, text_len)
    return stats


def _compute_sequence_score(group: list[HeadingCandidate]) -> float:
    """候補列の番号連続性を 0.0-1.0 のスコアで返す。"""
    values = [c.marker_value for c in group if isinstance(c.marker_value, int)]
    if len(values) < 2:
        return 0.5
    hits = 0.0
    for i in range(len(values) - 1):
        diff = values[i + 1] - values[i]
        if diff == 1:
            hits += 1.0
        elif values[i + 1] == 1 and values[i] >= 1:
            hits += 0.5  # リセットも正の連続シグナル
    return hits / (len(values) - 1)


def _compute_containment_and_reset(
    stats: dict[str, Any],
    groups: dict[str, list[HeadingCandidate]],
    text_len: int,
) -> None:
    """タイプA区間にタイプBが何件入るかを使って内包/リセット傾向を計算する。"""
    types = list(stats.keys())
    sentinel = text_len + 1

    for a_type in types:
        a_positions = stats[a_type]["positions"]
        a_avg_gap = stats[a_type]["average_gap"]
        # 区間境界リスト (start, end)
        boundaries = list(zip(a_positions, a_positions[1:] + [sentinel]))

        total_containment = 0.0
        total_reset = 0.0
        compared = 0

        for b_type in types:
            if b_type == a_type:
                continue
            b_avg_gap = stats[b_type]["average_gap"]
            # A の間隔 > B の間隔 でないなら A は B の外側ではない
            if a_avg_gap <= b_avg_gap * 0.8:
                continue

            b_cands = groups[b_type]
            counts_in_ranges: list[int] = []
            for a_start, a_end in boundaries:
                contained = [c for c in b_cands if a_start <= c.start_char < a_end]
                counts_in_ranges.append(len(contained))
                # リセット判定: この区間の最初のB番号が1
                b_vals = [c.marker_value for c in contained if isinstance(c.marker_value, int)]
                if b_vals and b_vals[0] == 1:
                    total_reset += 1.0

            avg_contained = statistics.mean(counts_in_ranges) if counts_in_ranges else 0.0
            if avg_contained >= 1.0:
                total_containment += min(1.0, avg_contained / 4.0)
                compared += 1

        stats[a_type]["containment_score"] = total_containment / max(compared, 1)
        stats[a_type]["reset_score"] = total_reset / max(len(a_positions), 1)


# ============================================================
# 4. infer_heading_levels
# ============================================================


def infer_heading_levels(
    candidates: list[HeadingCandidate],
    group_stats: dict[str, Any],
    config: ChunkingConfig,
) -> list[HeadingCandidate]:
    """候補の見出しタイプから階層レベル(level)を推定して付与する。"""
    if not candidates:
        return candidates

    if not group_stats or not config.enable_level_inference:
        for c in candidates:
            rule = next((r for r in DEFAULT_HEADING_RULES if r.name == c.marker_type), None)
            c.inferred_level = rule.default_level_hint if rule else 4
        return candidates

    # min_group_count 未満のタイプはランキングから除外しデフォルト level_hint を使う
    # (1件のみの appendix 等が全体のランキングを歪めないようにする)
    infer_types = {
        mtype for mtype, st in group_stats.items()
        if st["count"] >= config.min_group_count
    }
    fallback_types = set(group_stats.keys()) - infer_types

    # 各タイプの「外側らしさスコア」を計算 (推定対象のみ)
    type_outer_score: dict[str, float] = {}
    for mtype in infer_types:
        st = group_stats[mtype]
        # 出現間隔が広い = 外側 (ただし上限あり)
        gap_score = min(st["average_gap"] / 3000.0, 2.0)
        score = (
            st.get("containment_score", 0.0) * 4.0   # 他タイプを内包している
            + st.get("reset_score", 0.0) * 3.0        # 子タイプ番号がリセットされる
            + st.get("sequence_score", 0.0) * 1.0     # 連続番号
            + st.get("shallow_indent_score", 0.0) * 1.5
            + (st["fallback_priority"] / 12.0) * 1.5  # フォールバック優先度は補助
            + gap_score
        )
        type_outer_score[mtype] = score

    # スコア降順でソート → level 1, 2, 3... を割り当て
    sorted_types = sorted(type_outer_score, key=lambda t: type_outer_score[t], reverse=True)
    type_to_level: dict[str, int] = {
        t: min(i + 1, config.max_depth) for i, t in enumerate(sorted_types)
    }

    # フォールバックタイプには default_level_hint を使う
    for mtype in fallback_types:
        rule = next((r for r in DEFAULT_HEADING_RULES if r.name == mtype), None)
        hint = rule.default_level_hint if rule else config.max_depth
        type_to_level[mtype] = min(hint, config.max_depth)

    # 候補にlevel付与
    for c in candidates:
        lv = type_to_level.get(c.marker_type)
        if lv is None:
            rule = next((r for r in DEFAULT_HEADING_RULES if r.name == c.marker_type), None)
            lv = rule.default_level_hint if rule else 4
        c.inferred_level = min(lv, config.max_depth)

    return candidates


# ============================================================
# 5. build_section_tree
# ============================================================


def build_section_tree(
    text: str,
    candidates: list[HeadingCandidate],
    config: ChunkingConfig | None = None,
) -> SectionNode:
    """推定済み見出し候補から親子構造のセクションツリーを構築する。

    初期ツリー構築後、_rebuild_tree_by_recursive_scoring() を呼び、
    各スコープ内で連番性・内部包含・カバレッジ等のスコアを再帰的に
    評価して親子関係を再決定する。これにより固定優先度では拾い切れない
    文書ごとの階層構造に対しても、章/条/節の境界を保ったまま分割できる。
    """
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

    if not valid:
        root.text = text
        if config.fallback_to_paragraph:
            _add_paragraph_fallback(root, text)
        return root

    # 各候補のテキスト終端を計算: 次の「同階層以上の見出し」の start_char
    def _section_end(idx: int, level: int) -> int:
        for j in range(idx + 1, len(valid)):
            if valid[j].inferred_level <= level:
                return valid[j].start_char
        return text_len

    # スタックベースでツリー構築
    # stack: [(level, node)]
    stack: list[tuple[int, SectionNode]] = [(0, root)]

    for i, cand in enumerate(valid):
        end = _section_end(i, cand.inferred_level)
        sec_text = text[cand.start_char:end]

        node = SectionNode(
            id=_section_id(cand.text, cand.start_char),
            heading=cand.text,
            heading_type=cand.marker_type,
            level=cand.inferred_level,
            text=sec_text,
            start_char=cand.start_char,
            parent_id=None,
        )

        # スタックを現在level以上の要素をpopする
        while len(stack) > 1 and stack[-1][0] >= cand.inferred_level:
            stack.pop()

        parent_node = stack[-1][1]
        node.parent_id = parent_node.id
        parent_node.children.append(node)
        stack.append((cand.inferred_level, node))

    # rootのtextは最初の見出し直前のテキスト(前文)
    root.text = text[: valid[0].start_char]

    # 後処理: 各スコープ内で再帰的スコアリングを行い、親子関係を再決定する。
    # 固定優先度では拾い切れない文書固有の階層 (例: 出現頻度が低い章と
    # 高頻度の条が並ぶケース) でも、内部連続パターンの包含関係を主軸に
    # 適切な親子へ振り直す。
    _rebuild_tree_by_recursive_scoring(root, text, config)

    return root


def _rebuild_tree_by_recursive_scoring(
    root: SectionNode,
    full_text: str,
    config: ChunkingConfig,
) -> None:
    """ツリー配下を再帰的スコアリングで再構築する。

    狙い:
      固定優先度 (default_level_hint) ではなく、各スコープ内での連番性 /
      内部包含 / カバレッジ等を加味したスコアで「最外側らしい type」を
      動的に決定する。再帰的に sub_scope へ降りて level=2..N を割り当てる
      ことで、文書ごとに異なる構造 (例: 章が 1 件のみ、条と numeric_dot が
      混在する小規模規程) にも汎用に対応する。

    手順:
      1. 既存ツリーの全ノードを document order で flat 化
      2. 各ノードの marker_value を heading 文字列から再抽出
      3. ルートスコープで _assign_levels_recursive() を呼び、各ノードに
         level を割り当てる (採用外は level=0 となり後段で破棄)
      4. 採用ノードでスタックベースに親子を再構築
      5. text 範囲を再計算
    """
    nodes: list[SectionNode] = []

    def _collect(n: SectionNode) -> None:
        for c in n.children:
            nodes.append(c)
            _collect(c)

    _collect(root)
    if not nodes:
        return
    nodes.sort(key=lambda n: n.start_char)

    # SectionNode は marker_value を保持していないので heading 文字列から再抽出
    node_values: dict[str, int | str | None] = {
        n.id: _extract_marker_value(n.heading_type, n.heading) for n in nodes
    }

    # 既存の親子関係をクリア
    root.children = []
    for n in nodes:
        n.children = []
        n.level = 0  # 採用外を表す初期値

    text_len = len(full_text)
    _assign_levels_recursive(
        nodes,
        node_values,
        scope_start=0,
        scope_end=text_len,
        current_level=1,
        max_depth=config.max_depth,
    )

    # 採用ノード (level >= 1) のみツリーへ組み込む
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

    # children の再 attach に伴い text 範囲を再計算する
    # (各ノードの text は [start_char, 次 sibling の start_char or 親 end))
    def _recompute_text(n: SectionNode, end_char: int) -> None:
        if n.heading_type != "document_root":
            n.text = full_text[n.start_char:end_char]
        children = sorted(n.children, key=lambda c: c.start_char)
        for i, c in enumerate(children):
            child_end = children[i + 1].start_char if i + 1 < len(children) else end_char
            _recompute_text(c, child_end)

    _recompute_text(root, text_len)


def _assign_levels_recursive(
    nodes: list[SectionNode],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
    current_level: int,
    max_depth: int,
) -> None:
    """スコープ内のノード群を再帰的にスコアリングして level を割り当てる。

    手順:
      1. nodes を heading_type ごとにグルーピング
      2. 各 type をスコープ内で _score_type_in_scope() でスコア化
      3. 最高スコアの type を採用し、同じ default_level_hint を持つ
         他 type も同 level に昇格させる (附則と章の並列など)
      4. 採用ノードに current_level を付与
      5. 隣接する採用ノード間の sub_scope ごとに、未採用 type を再帰評価
    """
    if not nodes or current_level > max_depth:
        return

    by_type: dict[str, list[SectionNode]] = defaultdict(list)
    for n in nodes:
        by_type[n.heading_type].append(n)

    scores: dict[str, float] = {
        t: _score_type_in_scope(g, by_type, node_values, scope_start, scope_end)
        for t, g in by_type.items()
    }

    if not scores or max(scores.values()) <= 0.0:
        return

    best_type = max(scores, key=lambda t: scores[t])
    best_hint = _hint_for_type(best_type)

    # ベスト type と同じ default_level_hint を持ち、かつ正のスコアの type も
    # 同 level として昇格させる。
    # 用途: 附則 (単独・スコア低) と章 (複数・スコア高) の両方を level=1 に並列配置。
    # 同 hint 同士でしか昇格しないため、別 hint のノイズ (例: hint=4 の numeric_dot)
    # が hint=2 の article と並列に L2 へ繰り上がることはない。
    chosen_types = {
        t for t, s in scores.items()
        if _hint_for_type(t) == best_hint and s > 0.0
    }

    chosen_nodes = sorted(
        [n for n in nodes if n.heading_type in chosen_types],
        key=lambda n: n.start_char,
    )
    for n in chosen_nodes:
        n.level = current_level

    # 採用されなかった type のノードは sub_scope で再評価
    others = [n for n in nodes if n.heading_type not in chosen_types]

    for i, c in enumerate(chosen_nodes):
        sub_start = c.start_char
        sub_end = chosen_nodes[i + 1].start_char if i + 1 < len(chosen_nodes) else scope_end
        sub_others = [x for x in others if sub_start < x.start_char < sub_end]
        _assign_levels_recursive(
            sub_others,
            node_values,
            sub_start,
            sub_end,
            current_level + 1,
            max_depth,
        )


def _hint_for_type(mtype: str) -> int:
    """heading_type に対応するルールの default_level_hint を返す。"""
    rule = next((r for r in DEFAULT_HEADING_RULES if r.name == mtype), None)
    return rule.default_level_hint if rule else 4


def _score_type_in_scope(
    group: list[SectionNode],
    all_groups: dict[str, list[SectionNode]],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
) -> float:
    """1 つの heading_type が、与えられたスコープ内で「その階層らしい」度合いをスコア化する。

    指標 (weight):
      - containment (5.0): 連続ノード間に別 type の連続パターンを内包する区間の比率
      - sequence (2.5):    自身の marker_value の +1 連続性 (リセットも 0.5 で正評価)
      - coverage (1.5):    スコープ内での位置範囲 (広いほど外側らしい)
      - frequency (0.7):   出現数の log スケール (頻度の偏りを抑制)
      - priority_hint (0.5): ヘッダー語ヒント (tie-breaker)
    """
    if not group:
        return 0.0

    positions = sorted(c.start_char for c in group)
    values = [v for v in (node_values.get(n.id) for n in group) if isinstance(v, int)]

    # 1) 内部連続パターンの包含率 (最重視)
    containment = _interval_containment(group, all_groups, node_values, scope_start, scope_end)

    # 2) 自身の連番一貫性
    seq = _sequence_consistency_values(values)

    # 3) スコープ内カバレッジ
    scope_size = max(scope_end - scope_start, 1)
    if len(positions) >= 2:
        coverage = (positions[-1] - positions[0]) / scope_size
    else:
        # 単独の場合: スコープの先頭付近にあれば「外側を覆う見出し」とみなして高評価
        head_zone = scope_start + scope_size * 0.2
        coverage = 1.0 if positions and positions[0] <= head_zone else 0.3

    # 4) 出現頻度 (log スケールで頭打ち)
    freq = math.log(1 + len(group)) / math.log(20)

    # 5) ヘッダー語ヒント (tie-breaker のみ)
    rule = next(
        (r for r in DEFAULT_HEADING_RULES if r.name == group[0].heading_type),
        None,
    )
    priority_hint = (rule.priority / 12.0) if rule else 0.0

    score = (
        containment * 5.0
        + seq * 2.5
        + coverage * 1.5
        + freq * 0.7
        + priority_hint * 0.5
    )
    return score


def _interval_containment(
    group: list[SectionNode],
    all_groups: dict[str, list[SectionNode]],
    node_values: dict[str, int | str | None],
    scope_start: int,
    scope_end: int,
) -> float:
    """group の各区間に、別 type の連続パターン (>=2 ノードで marker_value が +1 連続)
    を含む区間の比率を返す。

    "ノードの番号と番号の内側" を直接的に評価する指標で、最外側の階層を見抜くための主信号。

    区間の取り方 ("連番かつ閉じられている" の評価):
      - count >= 2: 連続する同 type ノード間 (閉区間) のみを計上。最後のノードから
        scope_end までの "open な末尾" は計上しない。
        例) numeric_dot [2., 3.] が 第10条 内で閉じ、その後に 第11条, 第12条 が外側で続く
        ケース。末尾 (3., scope_end) を除外することで、numeric_dot が 第11条/第12条 を
        内包していると誤評価するのを防ぐ。
      - count == 1: ノードが「スコープの先頭付近」にある場合のみ wrapper 候補として
        (node, scope_end) を 1 区間扱い。中ほどに単独で出現する type は別 type の内側に
        ぶら下がっている子要素である可能性が高く、wrapper 扱いしない (containment=0)。
        例) 章が 1 件のみ・附則が 1 件のみ等の正当な wrapper のみを救済する。
    """
    positions = sorted(c.start_char for c in group)
    if not positions:
        return 0.0

    if len(positions) == 1:
        scope_size = max(scope_end - scope_start, 1)
        head_zone = scope_start + scope_size * 0.1
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


def _sequence_consistency_values(values: list[int]) -> float:
    """marker_value 列の +1 連続またはリセット (1 へ戻る) の比率を返す。"""
    if len(values) < 2:
        return 0.0
    hits = 0.0
    for i in range(len(values) - 1):
        if values[i + 1] == values[i] + 1:
            hits += 1.0
        elif values[i + 1] == 1 and values[i] >= 1:
            hits += 0.5  # 別親への遷移を示すリセットも弱めに正評価
    return hits / (len(values) - 1)


def _add_paragraph_fallback(root: SectionNode, text: str) -> None:
    """見出しが存在しない場合、段落単位でchildrenを作る。"""
    char_pos = 0
    for para in re.split(r"\n{2,}", text):
        para_len = len(para)
        if para.strip():
            node = SectionNode(
                id=_section_id(para[:40], char_pos),
                heading="",
                heading_type="paragraph",
                level=1,
                text=para,
                start_char=char_pos,
                parent_id=root.id,
            )
            root.children.append(node)
        char_pos += para_len + 2  # +2 for \n\n


# ============================================================
# 6. flatten_chunks
# ============================================================


def _split_text_evenly(text: str, max_chars: int) -> list[str]:
    """テキストを max_chars 以下でなるべく均等な長さに分割する。"""
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []
    n = math.ceil(len(text) / max_chars)
    target = math.ceil(len(text) / n)
    parts: list[str] = []
    for i in range(n):
        part = text[i * target : (i + 1) * target]
        if part.strip():
            parts.append(part)
    return parts or [text]


def _group_items_balanced(
    item_sizes: list[int], max_per_group: int
) -> list[list[int]]:
    """サイズ配列を max_per_group 以下のグループに均等分割し、
    各グループに属するアイテムのインデックスリストを返す。

    分割方針:
      - 合計 total が max 以下なら 1 グループにまとめる
      - そうでなければ n = ceil(total / max) を最小分割数とし、
        target = ceil(total / n) を均等ターゲットにする
      - 子のサイズを順に積み、target 到達 or max 到達で次のグループへ
    """
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
      flatten_chunks() / split_for_rag_structure_aware() の
      metadata_builder 引数に渡す。
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


def _meaningful_body_lines(text: str, heading: str) -> int:
    """ノードのテキストから先頭の見出し行を除いた本文の、空白行を除いた行数。

    - >= 2 → 「複数行本文あり」(従来の _has_multiline_body 相当)
    - >= 1 → 本文に何らかの内容がある (昇格採用時の最低条件)
    - == 0 → 見出しのみで本文が空 (昇格対象でも子チャンク化しない)
    """
    body = text.strip()
    if heading and body.startswith(heading):
        body = body[len(heading):]
    return sum(1 for ln in body.split("\n") if ln.strip())


def _format_chunk_md(
    text: str,
    node: SectionNode,
    *,
    is_parent: bool,
    config: ChunkingConfig,
) -> str:
    """チャンクテキストにマークダウン見出し記法を付与する。

    ツリー構造から各見出し行を特定し、階層に応じた記法を行頭に挿入する。
    適用ルール:
      - 親ノードの見出し行 → "### " (例外的ノードは "**...**")
      - 子ノードの見出し行 → "## "
      - 列挙構造の見出し行 → "# "
      - 上記以外の行 → そのまま
    """
    if not config.output_markdown:
        return text

    node_heading = node.heading.strip()

    # ツリーから子・孫の見出し文字列を収集し、行単位の照合に利用する
    child_headings = {c.heading.strip() for c in node.children if c.heading.strip()}
    grandchild_headings: set[str] = set()
    for child in node.children:
        for gc in child.children:
            if gc.heading.strip():
                grandchild_headings.add(gc.heading.strip())

    is_exceptional = (
        node.heading_type == "appendix"
        or bool(_EXCEPTIONAL_HEADING_RE.match(node_heading))
    )

    lines = text.split("\n")
    result: list[str] = []
    heading_found = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        if not heading_found and stripped == node_heading:
            heading_found = True
            if is_exceptional and is_parent:
                result.append(f"**{stripped}**")
            elif is_parent:
                result.append(f"### {stripped}")
            else:
                result.append(f"## {stripped}")
        elif stripped in child_headings:
            if is_parent:
                result.append(f"## {stripped}")
            else:
                result.append(f"# {stripped}")
        elif stripped in grandchild_headings:
            result.append(f"# {stripped}")
        else:
            result.append(line)

    return "\n".join(result)


def flatten_chunks(
    root: SectionNode,
    config: ChunkingConfig,
    metadata_builder: MetadataBuilder | None = None,
) -> list[dict]:
    """セクションツリーをRAG投入用チャンク配列へ平坦化する。

    分割戦略 (structure_aware_v4):
      - level==1 (章相当) を親チャンクとして出力。max_chunk_chars 超過時は
        子(条)境界を尊重した均等分割を行い、各分割の先頭に章見出し prefix
        を付与する。
      - level==2 (条相当) を子チャンクとして出力。基準は本文 (見出しを除いた
        部分) が空行を除いて 2 行以上あること。
        加えて、親章内に基準を満たす同種ノード (同 heading_type) が
        1 つでも存在する場合は、本文 1 行のみの条も昇格採用する。
        ただし見出しのみ (本文 0 行) のノードは昇格対象でも除外する。
      - 条単独で max_chunk_chars 超過時のみ見出し付きで均等分割する。
      - level>=3 のノードは独立チャンク化せず親/子チャンク本文に含める。
      - 0 件時は最低 1 チャンクを補償する。
    """
    if metadata_builder is None:
        metadata_builder = default_metadata_builder

    doc_root_id = "doc_" + hashlib.sha256(root.heading.encode()).hexdigest()[:8]
    chunks: list[dict] = []

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

    def _ancestor_entry(node: SectionNode) -> dict[str, Any]:
        # default builder の metadata には含めないが、差し替え builder が
        # 利用する可能性に備えて引数として渡し続ける。
        return {
            "id": f"chunk_{node.id}",
            "heading": node.heading,
            "level": node.level,
            "heading_type": node.heading_type,
        }

    def _emit_parent(node: SectionNode, ancestors: list[SectionNode]) -> None:
        # level==1 (章 / 附則) を親チャンクとして出力
        text = node.text.strip()
        if not text or len(text) < config.min_child_text_length:
            return
        ancestor_chain = [_ancestor_entry(a) for a in ancestors]
        path = [root.heading] + [a.heading for a in ancestors] + [node.heading]
        base_id = f"chunk_{node.id}"
        section_parent_id = f"chunk_{node.parent_id}" if node.parent_id else None

        if len(text) <= config.max_chunk_chars:
            chunks.append({
                "id": base_id,
                "text": _format_chunk_md(text, node, is_parent=True, config=config),
                "metadata": _build_meta(
                    base_id, node, path, section_parent_id, ancestor_chain,
                ),
            })
            return

        # --- 章サイズ超過: 子(条)境界で均等分割 ---
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

        if available <= 0 or not children_with_text:
            # prefix だけで超過 or 子が無い → 文字数ベースの単純均等分割
            parts = _split_text_evenly(text, config.max_chunk_chars)
            for pi, part in enumerate(parts):
                cid = f"{base_id}_p{pi}" if len(parts) > 1 else base_id
                chunks.append({
                    "id": cid,
                    "text": _format_chunk_md(part, node, is_parent=True, config=config),
                    "metadata": _build_meta(
                        cid, node, path, section_parent_id, ancestor_chain,
                        split_index=pi, split_total=len(parts),
                    ),
                })
            return

        # 子(条)テキスト群と各サイズ (+1 は結合時の \n 区切り分)
        child_texts = [c.text.strip() for c in children_with_text]
        child_sizes = [len(t) + 1 for t in child_texts]
        groups = _group_items_balanced(child_sizes, available)

        for gi, indices in enumerate(groups):
            body = "\n".join(child_texts[i] for i in indices)
            chunk_text = prefix_line + body
            cid = f"{base_id}_p{gi}" if len(groups) > 1 else base_id
            chunks.append({
                "id": cid,
                "text": _format_chunk_md(chunk_text, node, is_parent=True, config=config),
                "metadata": _build_meta(
                    cid, node, path, section_parent_id, ancestor_chain,
                    split_index=gi, split_total=len(groups),
                ),
            })

    def _emit_child(node: SectionNode, ancestors: list[SectionNode]) -> None:
        # level==2 (条) を子チャンクとして出力
        text = node.text.strip()
        if not text or len(text) < config.min_child_text_length:
            return

        body_lines = _meaningful_body_lines(text, node.heading)
        if body_lines < 2:
            # 親章内に「複数行本文を持つ同種ノード」が 1 つでもあれば本文 1 行の条も
            # 昇格して子チャンク化する。法規類で 2 行以上の条と 1 行の条が混在する
            # 章で、後者だけ漏れるのを防ぐための救済措置。
            # ただし見出しのみ (本文 0 行) のノードは検索ノイズ抑制のため除外。
            parent = ancestors[-1] if ancestors else None
            promoted = bool(parent) and any(
                _meaningful_body_lines(c.text.strip(), c.heading) >= 2
                for c in parent.children
                if c.heading_type == node.heading_type
            )
            if not promoted or body_lines == 0:
                return

        ancestor_chain = [_ancestor_entry(a) for a in ancestors]
        path = [root.heading] + [a.heading for a in ancestors] + [node.heading]
        base_id = f"chunk_{node.id}"
        section_parent_id = f"chunk_{node.parent_id}" if node.parent_id else None

        if len(text) <= config.max_chunk_chars:
            chunks.append({
                "id": base_id,
                "text": _format_chunk_md(text, node, is_parent=False, config=config),
                "metadata": _build_meta(
                    base_id, node, path, section_parent_id, ancestor_chain,
                ),
            })
            return

        # 条単独で超過 → 見出しを各チャンク先頭に付与しつつ均等分割
        heading_line = node.heading
        if text.startswith(heading_line):
            body = text[len(heading_line):].lstrip("\n")
        else:
            body = text
        prefix = f"{heading_line}\n"
        available = config.max_chunk_chars - len(prefix)

        if available <= 0:
            # 見出し自体が max_chunk_chars を超える異常ケース。
            # 条文の保全を最優先し全文をそのまま均等分割する。
            parts = _split_text_evenly(text, config.max_chunk_chars)
            for pi, part in enumerate(parts):
                cid = f"{base_id}_p{pi}" if len(parts) > 1 else base_id
                chunks.append({
                    "id": cid,
                    "text": _format_chunk_md(part, node, is_parent=False, config=config),
                    "metadata": _build_meta(
                        cid, node, path, section_parent_id, ancestor_chain,
                        split_index=pi, split_total=len(parts),
                    ),
                })
            return

        parts = _split_text_evenly(body, available)
        for pi, part in enumerate(parts):
            cid = f"{base_id}_p{pi}" if len(parts) > 1 else base_id
            chunk_text = f"{prefix}{part}"
            chunks.append({
                "id": cid,
                "text": _format_chunk_md(chunk_text, node, is_parent=False, config=config),
                "metadata": _build_meta(
                    cid, node, path, section_parent_id, ancestor_chain,
                    split_index=pi, split_total=len(parts),
                ),
            })

    def _walk(node: SectionNode, ancestors: list[SectionNode]) -> None:
        if node.heading_type == "document_root":
            for child in node.children:
                _walk(child, ancestors)
            return

        # 親チャンク (level==1) と 子チャンク (level==2) を適切に出力
        if node.level == 1:
            _emit_parent(node, ancestors)
        elif node.level == 2:
            _emit_child(node, ancestors)
        # level>=3 のノードは独立チャンク化しない (親/子チャンク本文に含まれる)

        # 自身を ancestors に積みつつ子孫を再帰探索 (level==2 を見つけるため)
        for child in node.children:
            _walk(child, ancestors + [node])

    _walk(root, [])

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


def split_for_rag_structure_aware(
    text: str,
    config: ChunkingConfig | None = None,
    metadata_builder: MetadataBuilder | None = None,
) -> list[dict]:
    """構造認識チャンク化のエントリーポイント。"""
    if config is None:
        config = ChunkingConfig()

    if not text or not text.strip():
        return [{
            "id": "chunk_empty",
            "text": "",
            "metadata": {
                "chunk_id": "chunk_empty",
                "parent_id": None,
                "root_id": "doc_empty",
                "level": 0,
                "path_text": "",
                "chunk_role": "fallback",
                "chunking_strategy": "structure_aware_v4",
            },
        }]

    normalized = normalize_text(text, config)
    candidates = extract_heading_candidates(normalized, config)
    group_stats = analyze_heading_groups(candidates, normalized, config)
    candidates = infer_heading_levels(candidates, group_stats, config)
    root = build_section_tree(normalized, candidates, config)
    return flatten_chunks(root, config, metadata_builder=metadata_builder)


def split_for_rag(
    *, text: str, chunk_size: int = 800, chunk_overlap: int = 0
) -> list[str]:
    """既存 split_for_rag との互換インタフェース。chunk_size を max_chunk_chars として使用。"""
    config = ChunkingConfig(max_chunk_chars=chunk_size)
    return [c["text"] for c in split_for_rag_structure_aware(text, config) if c["text"]]


# ============================================================
# CLI
# ============================================================


def _print_tree(node: SectionNode, indent: int = 0) -> None:
    """デバッグ表示用にセクションツリーを再帰出力する。"""
    prefix = "  " * indent
    label = node.heading[:60] if node.heading else "(no heading)"
    print(f"{prefix}[L{node.level}:{node.heading_type}] {label}")
    for child in node.children:
        _print_tree(child, indent + 1)


def main() -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(description="Structure-aware chunker")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--out", help="Output JSON file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-tree", action="store_true", dest="print_tree")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    config = ChunkingConfig()

    with open(args.input, encoding=args.encoding) as f:
        raw_text = f.read()

    normalized = normalize_text(raw_text, config)
    candidates = extract_heading_candidates(normalized, config)
    group_stats = analyze_heading_groups(candidates, normalized, config)
    candidates = infer_heading_levels(candidates, group_stats, config)
    root = build_section_tree(normalized, candidates, config)
    chunks = flatten_chunks(root, config)

    print(f"\n=== Structure Aware Chunker ===")
    print(f"total chunks: {len(chunks)}")

    level_counts: dict[int, int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        m = c["metadata"]
        level_counts[m["level"]] += 1
        role_counts[m["chunk_role"]] += 1

    print("\n--- level ---")
    for lv, cnt in sorted(level_counts.items()):
        print(f"  level {lv}: {cnt}")

    print("\n--- chunk_role ---")
    for r, cnt in sorted(role_counts.items()):
        print(f"  {r}: {cnt}")

    print("\n--- first 5 chunks path_text ---")
    for c in chunks[:5]:
        print(f"  {c['metadata']['path_text']}")

    if args.debug:
        print(f"\n=== heading candidates ({len(candidates)}) ===")
        for c in candidates:
            print(
                f"  L{c.inferred_level} score={c.score:5.1f} "
                f"type={c.marker_type:<20} | {c.text[:60]}"
            )

        print("\n=== group stats ===")
        for t, st in sorted(group_stats.items(), key=lambda x: -x[1].get("containment_score", 0)):
            print(
                f"  {t:<22} cnt={st['count']:3d} "
                f"gap={st['average_gap']:7.0f} "
                f"seq={st['sequence_score']:.2f} "
                f"contain={st.get('containment_score', 0):.2f} "
                f"reset={st.get('reset_score', 0):.2f} "
                f"indent={st['shallow_indent_score']:.2f} "
                f"pri={st['fallback_priority']}"
            )

    if args.print_tree:
        print("\n=== section tree ===")
        _print_tree(root)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\nchunks saved to: {args.out}")


if __name__ == "__main__":
    main()
