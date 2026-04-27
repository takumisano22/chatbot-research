"""
chunking_logic_02.py — Structure Aware Chunker

文書全体の見出し分布・包含関係・連続性を解析し、
親子構造付きチャンクを生成する。

CLI: python chunking_logic_02.py input.txt --out chunks.json [--debug] [--print-tree]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# dataclasses
# ============================================================


@dataclass
class ChunkingConfig:
    max_depth: int = 4
    min_heading_score: float = 3.0
    min_group_count: int = 2
    max_heading_line_length: int = 80
    min_child_text_length: int = 10
    enable_inline_heading_repair: bool = True
    enable_level_inference: bool = True
    fallback_to_paragraph: bool = True
    include_parent_heading_in_child_text: bool = True
    create_parent_chunks: bool = True
    create_main_chunks: bool = True
    create_child_chunks: bool = True
    debug: bool = False


@dataclass
class HeadingRule:
    name: str
    regex: str
    default_level_hint: int  # フォールバック時のlevel (1=最外側)
    priority: int             # 高いほど外側に推定されやすい


@dataclass
class HeadingCandidate:
    text: str
    line_index: int
    start_char: int
    end_char: int
    raw_marker: str
    marker_type: str
    marker_value: int | str | None
    normalized_marker: str
    indent: int
    line_length: int
    score: float
    features: dict[str, Any]
    inferred_level: int | None = None
    inference_reason: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionNode:
    id: str
    heading: str
    heading_type: str
    level: int
    ordinal: int | None
    text: str
    start_char: int
    end_char: int
    parent_id: str | None
    children: list["SectionNode"] = field(default_factory=list)
    confidence: float = 0.0
    inference_reason: dict[str, Any] = field(default_factory=dict)


# ============================================================
# ルール定義
# ============================================================

DEFAULT_HEADING_RULES: list[HeadingRule] = [
    # priority が高いほど「外側階層らしい」デフォルト扱い
    HeadingRule("appendix",        r"^附則",                                                       1, 11),
    HeadingRule("japanese_chapter", r"^第[0-9一二三四五六七八九十百千〇]+章",                      1, 10),
    HeadingRule("japanese_article", r"^第[0-9一二三四五六七八九十百千〇]+条",                      2,  9),
    HeadingRule("japanese_section", r"^第[0-9一二三四五六七八九十百千〇]+項",                      3,  8),
    HeadingRule("decimal_number",  r"^\d+\.\d+",                                                   3,  7),
    HeadingRule("numeric_dot",     r"^\d+[.)．]\s",                                                4,  6),
    HeadingRule("numeric_paren",   r"^[（(]\d+[)）]",                                             4,  5),
    HeadingRule("japanese_paren",  r"^[（(][一二三四五六七八九十〇]+[)）]",                        4,  5),
    HeadingRule("roman",           r"^[IVX]{1,5}[.)]\s|^[Ⅰ-Ⅻ]\s",                               4,  4),
    HeadingRule("alpha",           r"^[A-Z][.)]\s",                                                4,  3),
    HeadingRule("circle_bullet",   r"^○\s*\S",                                                    5,  3),
    HeadingRule("bullet",          r"^[•・\-]\s",                                                  5,  2),
]

# コンパイル済みルール (起動時に1回だけ生成)
_COMPILED_RULES: list[tuple[HeadingRule, re.Pattern[str]]] = [
    (rule, re.compile(rule.regex)) for rule in DEFAULT_HEADING_RULES if rule.regex
]

# 文中参照を見出しと誤認しないためのパターン
_INLINE_REF_RE = re.compile(
    r"(?:"
    # XX法/規則等: 他の法令・規程への条文参照
    r"[^\s。、]{1,20}(?:法|規則|条例|規程)第[0-9一二三四五六七八九十百千〇]+[条章項号]?"
    # 前条・本条・次項 etc.: 同一文書内の相対参照
    r"|(?:前|本|次|各|当該|同)(?:条|項|号|章)"
    # 第n条 + 接尾辞: 文章内で参照として使われており見出し行ではない
    r"|第[0-9一二三四五六七八九十百千〇]+条"
    r"(?:各号|第[0-9一二三四五六七八九十百千〇]+項"  # 下位番号参照
    r"|に基づ|の規定|を準用|の定め"                  # 規定を援用する表現
    r"|により|について|に従|に違)"                    # 条文を修飾する助詞・動詞
    r")"
)

# 強い見出しパターン (文中への埋め込み検出用)
_STRONG_HEADING_EMBEDDED_RE = re.compile(
    r"(。|．)(第[0-9一二三四五六七八九十百千〇]+(?:章|条)(?:[（(][^）)\n]{0,20}[）)])?)"
)

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


def _is_inline_ref(text: str) -> bool:
    return bool(_INLINE_REF_RE.search(text))


def _is_strong_heading_line(stripped: str) -> bool:
    """第n章/第n条/附則 で始まる行かどうか (文中参照を除く)。"""
    if _is_inline_ref(stripped):
        return False
    return bool(re.match(r"^(?:附則|第[0-9一二三四五六七八九十百千〇]+(?:章|条))", stripped))


def _section_id(heading: str, start_char: int) -> str:
    key = f"{start_char}|{heading}"
    return "sec_" + hashlib.sha256(key.encode()).hexdigest()[:8]


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


# ============================================================
# 1. normalize_text
# ============================================================


def normalize_text(text: str, config: ChunkingConfig | None = None) -> str:
    if not text:
        return ""

    # 改行コード統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 全角スペース→半角スペース
    text = text.replace("　", " ")

    # 行末空白削除
    lines = [line.rstrip() for line in text.split("\n")]

    # 行内に埋め込まれた強い見出しを行分割する
    if config is None or config.enable_inline_heading_repair:
        lines = _split_embedded_headings(lines)

    # 強い見出し行の直前に空行がなければ挿入する
    if config is None or config.enable_inline_heading_repair:
        lines = _ensure_blank_before_headings(lines)

    # 3つ以上の連続空行を2つに圧縮
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _split_embedded_headings(lines: list[str]) -> list[str]:
    """
    "...する。第10条（服務）..." のような行内埋め込み見出しを分割する。
    文中参照（労働基準法第89条等）は対象外。
    """
    result: list[str] = []
    for line in lines:
        m = _STRONG_HEADING_EMBEDDED_RE.search(line)
        if not m:
            result.append(line)
            continue
        # 見出し候補の後続コンテキストが参照表現なら分割しない
        after = line[m.end():]
        if _INLINE_REF_RE.search(after[:40]):
            result.append(line)
            continue
        pre = line[: m.start(2)]   # 句点まで
        heading_and_rest = line[m.start(2):]
        result.append(pre)
        result.append(heading_and_rest)
    return result


def _ensure_blank_before_headings(lines: list[str]) -> list[str]:
    """強い見出し行 (第n章/第n条/附則) の直前が非空行なら空行を挿入する。"""
    result: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_strong_heading_line(stripped) and i > 0 and lines[i - 1].strip():
            result.append("")
        result.append(line)
    return result


# ============================================================
# 2. extract_heading_candidates
# ============================================================


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
        normalized_marker = raw_marker.strip()

        # --- スコアリング ---
        features: dict[str, Any] = {}
        score = 3.0  # 行頭でパターン一致
        features["line_start_match"] = True

        if prev_blank:
            score += 1.0
            features["after_blank"] = True

        if line_len <= 40:
            score += 1.0
            features["short_line"] = True

        if marker_type in ("japanese_chapter", "japanese_article", "appendix"):
            score += 4.0
            features["strong_japanese"] = True
        elif marker_type == "japanese_section":
            score += 2.0

        if re.search(r"[（(][^）)\n]{1,30}[）)]", stripped):
            score += 2.0
            features["paren_title"] = True

        if not stripped.endswith(("。", ".", "．")):
            score += 1.0
            features["no_period"] = True

        if indent == 0:
            score += 1.0
            features["zero_indent"] = True
        elif indent <= 2:
            score += 0.5

        # 減点
        if _is_inline_ref(stripped):
            score -= 4.0
            features["inline_ref"] = True

        if line_len > 60:
            score -= 2.0
            features["long_line"] = True

        if stripped.count("。") + stripped.count("．") >= 2:
            score -= 2.0
            features["multiple_periods"] = True

        if re.search(r"https?://|@\w+\.", stripped):
            score -= 3.0
            features["url_like"] = True

        if score < config.min_heading_score:
            continue

        candidates.append(
            HeadingCandidate(
                text=stripped,
                line_index=line_idx,
                start_char=line_start,
                end_char=line_start + len(line),
                raw_marker=raw_marker,
                marker_type=marker_type,
                marker_value=marker_value,
                normalized_marker=normalized_marker,
                indent=indent,
                line_length=line_len,
                score=score,
                features=features,
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
                c.features["seq_bonus"] = True
            elif c.marker_value == 1 and prev >= 1:
                # リセット (別親セクションへの遷移) - 弱いボーナス
                c.score += 0.5
                c.features["seq_reset"] = True
        last[c.marker_type] = c.marker_value


# ============================================================
# 3. analyze_heading_groups
# ============================================================


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
        count = len(group)
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        avg_gap = statistics.mean(gaps) if gaps else float(text_len)
        median_gap = statistics.median(gaps) if gaps else float(text_len)
        avg_indent = statistics.mean(c.indent for c in group)
        avg_len = statistics.mean(c.line_length for c in group)
        seq_score = _compute_sequence_score(group)
        shallow_indent_score = max(0.0, 1.0 - avg_indent / 10.0)

        rule = next((r for r in DEFAULT_HEADING_RULES if r.name == mtype), None)
        fallback_priority = rule.priority if rule else 1

        stats[mtype] = {
            "count": count,
            "average_line_length": avg_len,
            "average_indent": avg_indent,
            "positions": positions,
            "average_gap": avg_gap,
            "median_gap": median_gap,
            "sequence_score": seq_score,
            "containment_score": 0.0,
            "reset_score": 0.0,
            "shallow_indent_score": shallow_indent_score,
            "fallback_priority": fallback_priority,
            "confidence": 0.0,
        }

    _compute_containment_and_reset(stats, groups, text_len)
    return stats


def _compute_sequence_score(group: list[HeadingCandidate]) -> float:
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
    if not candidates:
        return candidates

    if not group_stats or not config.enable_level_inference:
        for c in candidates:
            rule = next((r for r in DEFAULT_HEADING_RULES if r.name == c.marker_type), None)
            c.inferred_level = rule.default_level_hint if rule else 4
            c.inference_reason = {"fallback": "no_stats_or_disabled"}
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
    # ただし infer_types のどのlevelとも被らないよう調整する
    used_levels = set(type_to_level.values())
    for mtype in fallback_types:
        rule = next((r for r in DEFAULT_HEADING_RULES if r.name == mtype), None)
        hint = rule.default_level_hint if rule else config.max_depth
        type_to_level[mtype] = min(hint, config.max_depth)

    # スコアの広がりからconfidenceを推定
    scores = list(type_outer_score.values())
    score_range = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0
    confidence_base = min(1.0, score_range / 5.0)
    for mtype in group_stats:
        group_stats[mtype]["confidence"] = confidence_base

    # 候補にlevel付与
    for c in candidates:
        lv = type_to_level.get(c.marker_type)
        if lv is None:
            rule = next((r for r in DEFAULT_HEADING_RULES if r.name == c.marker_type), None)
            lv = rule.default_level_hint if rule else 4
        c.inferred_level = min(lv, config.max_depth)
        st = group_stats.get(c.marker_type, {})
        c.inference_reason = {
            "type_outer_score": round(type_outer_score.get(c.marker_type, 0.0), 4),
            "containment_score": round(st.get("containment_score", 0.0), 4),
            "reset_score": round(st.get("reset_score", 0.0), 4),
            "sequence_score": round(st.get("sequence_score", 0.0), 4),
            "shallow_indent_score": round(st.get("shallow_indent_score", 0.0), 4),
            "fallback_priority": st.get("fallback_priority", 0),
            "fallback_used": c.marker_type in fallback_types,
        }

    return candidates


# ============================================================
# 5. build_section_tree
# ============================================================


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
        ordinal=None,
        text="",
        start_char=0,
        end_char=text_len,
        parent_id=None,
        confidence=1.0,
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
            ordinal=cand.marker_value if isinstance(cand.marker_value, int) else None,
            text=sec_text,
            start_char=cand.start_char,
            end_char=end,
            parent_id=None,
            confidence=_candidate_confidence(cand),
            inference_reason=cand.inference_reason,
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
    root.end_char = text_len

    return root


def _candidate_confidence(cand: HeadingCandidate) -> float:
    ir = cand.inference_reason
    if not ir:
        return 0.5
    return round(
        min(1.0,
            ir.get("containment_score", 0.0) * 0.4
            + ir.get("sequence_score", 0.0) * 0.3
            + ir.get("reset_score", 0.0) * 0.2
            + ir.get("shallow_indent_score", 0.0) * 0.1
        ),
        4,
    )


def _add_paragraph_fallback(root: SectionNode, text: str) -> None:
    """見出しが存在しない場合、段落単位でchildrenを作る。"""
    char_pos = 0
    for i, para in enumerate(re.split(r"\n{2,}", text)):
        para_len = len(para)
        if para.strip():
            node = SectionNode(
                id=_section_id(para[:40], char_pos),
                heading="",
                heading_type="paragraph",
                level=1,
                ordinal=i,
                text=para,
                start_char=char_pos,
                end_char=char_pos + para_len,
                parent_id=root.id,
                confidence=0.3,
                inference_reason={"fallback": "paragraph"},
            )
            root.children.append(node)
        char_pos += para_len + 2  # +2 for \n\n


# ============================================================
# 6. flatten_chunks
# ============================================================


def _will_create_chunk(node: SectionNode, config: ChunkingConfig) -> bool:
    """このノードがチャンクとして生成されるかどうかを判定する。"""
    if node.heading_type == "document_root":
        return False
    if node.heading_type == "paragraph":
        return True
    if node.level == 1:
        return config.create_parent_chunks
    if node.level == 2:
        return config.create_main_chunks
    # level 3+
    if not config.create_child_chunks:
        return False
    return len(node.text.strip()) >= config.min_child_text_length


def flatten_chunks(root: SectionNode, config: ChunkingConfig) -> list[dict]:
    doc_root_id = "doc_" + hashlib.sha256(root.heading.encode()).hexdigest()[:8]
    chunks: list[dict] = []

    def _walk(node: SectionNode, path: list[str]) -> None:
        if node.heading_type == "document_root":
            for child in node.children:
                _walk(child, [node.heading])
            return

        # chunk_role 決定
        if node.heading_type == "paragraph":
            chunk_role = "fallback"
        elif node.level == 1:
            chunk_role = "parent"
            if not config.create_parent_chunks:
                for child in node.children:
                    _walk(child, path + [node.heading])
                return
        elif node.level == 2:
            chunk_role = "main"
            if not config.create_main_chunks:
                for child in node.children:
                    _walk(child, path + [node.heading])
                return
        else:
            chunk_role = "child"
            if not config.create_child_chunks:
                return
            # 短すぎる子チャンクはスキップ
            if len(node.text.strip()) < config.min_child_text_length:
                return

        current_path = path + [node.heading]
        chunk_id = f"chunk_{node.id}"
        parent_chunk_id = f"chunk_{node.parent_id}" if node.parent_id else None
        # document_root は chunk として存在しないので、level 1 の parent_id は None にする
        if node.level == 1:
            parent_chunk_id = None

        # 実際にchunkとして生成される子だけを children_ids に含める
        children_ids = [
            f"chunk_{c.id}" for c in node.children
            if _will_create_chunk(c, config)
        ]

        # 子チャンクのテキストには親見出しを付与
        text = node.text
        if config.include_parent_heading_in_child_text and chunk_role == "child" and path:
            parent_heading = path[-1]
            text = f"{parent_heading}\n{text}" if parent_heading else text

        chunks.append({
            "id": chunk_id,
            "text": text.strip(),
            "metadata": {
                "chunk_id": chunk_id,
                "parent_id": parent_chunk_id,
                "root_id": doc_root_id,
                "children_ids": children_ids,
                "level": node.level,
                "chunk_role": chunk_role,
                "heading": node.heading,
                "heading_type": node.heading_type,
                "path": current_path,
                "path_text": " > ".join(current_path),
                "ordinal": node.ordinal,
                "start_char": node.start_char,
                "end_char": node.end_char,
                "source_type": "txt",
                "chunking_strategy": "structure_aware_v1",
                "structure_confidence": node.confidence,
                "inference_reason": node.inference_reason,
            },
        })

        for child in node.children:
            _walk(child, current_path)

    for child in root.children:
        _walk(child, [root.heading])

    if not chunks:
        # 最低1チャンク保証
        chunks.append({
            "id": f"chunk_{root.id}",
            "text": root.text.strip() or "(empty)",
            "metadata": {
                "chunk_id": f"chunk_{root.id}",
                "parent_id": None,
                "root_id": doc_root_id,
                "children_ids": [],
                "level": 0,
                "chunk_role": "fallback",
                "heading": root.heading,
                "heading_type": "document_root",
                "path": [root.heading],
                "path_text": root.heading,
                "ordinal": None,
                "start_char": 0,
                "end_char": len(root.text),
                "source_type": "txt",
                "chunking_strategy": "structure_aware_v1",
                "structure_confidence": 0.0,
                "inference_reason": {},
            },
        })

    return chunks


# ============================================================
# Public API
# ============================================================


def split_for_rag_structure_aware(
    text: str,
    config: ChunkingConfig | None = None,
) -> list[dict]:
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
                "children_ids": [],
                "level": 0,
                "chunk_role": "fallback",
                "heading": "",
                "heading_type": "empty",
                "path": [],
                "path_text": "",
                "ordinal": None,
                "start_char": 0,
                "end_char": 0,
                "source_type": "txt",
                "chunking_strategy": "structure_aware_v1",
                "structure_confidence": 0.0,
                "inference_reason": {},
            },
        }]

    normalized = normalize_text(text, config)
    candidates = extract_heading_candidates(normalized, config)
    group_stats = analyze_heading_groups(candidates, normalized, config)
    candidates = infer_heading_levels(candidates, group_stats, config)
    root = build_section_tree(normalized, candidates, config)
    return flatten_chunks(root, config)


def split_for_rag_texts_only(text: str) -> list[str]:
    return [c["text"] for c in split_for_rag_structure_aware(text) if c["text"]]


def split_for_rag(
    *, text: str, chunk_size: int = 0, chunk_overlap: int = 0
) -> list[str]:
    """既存 split_for_rag との互換インタフェース。"""
    return split_for_rag_texts_only(text)


# ============================================================
# CLI
# ============================================================


def _print_tree(node: SectionNode, indent: int = 0) -> None:
    prefix = "  " * indent
    label = node.heading[:60] if node.heading else "(no heading)"
    print(f"{prefix}[L{node.level}:{node.heading_type}] {label}")
    for child in node.children:
        _print_tree(child, indent + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-aware chunker")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("--out", help="Output JSON file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print-tree", action="store_true", dest="print_tree")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    config = ChunkingConfig(debug=args.debug)

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

    type_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[int, int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        m = c["metadata"]
        type_counts[m["heading_type"]] += 1
        level_counts[m["level"]] += 1
        role_counts[m["chunk_role"]] += 1

    print("\n--- heading_type ---")
    for t, cnt in sorted(type_counts.items()):
        print(f"  {t}: {cnt}")

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

        fallback_used = any(
            c["metadata"]["inference_reason"].get("fallback_used") for c in chunks
        )
        print(f"\nfallback_used: {fallback_used}")

    if args.print_tree:
        print("\n=== section tree ===")
        _print_tree(root)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\nchunks saved to: {args.out}")


if __name__ == "__main__":
    main()
