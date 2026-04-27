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
#       -> _split_embedded_headings()
#       -> _ensure_blank_before_headings()
#    -> extract_heading_candidates()
#       -> _extract_marker_value()
#       -> _award_sequence_bonus()
#    -> analyze_heading_groups()
#       -> _compute_sequence_score()
#       -> _compute_containment_and_reset()
#    -> infer_heading_levels()
#    -> build_section_tree()
#       -> _candidate_confidence()
#       -> _add_paragraph_fallback()  # 見出し無し時のみ
#    -> flatten_chunks(metadata_builder=...)  # 親(章)/子(条) チャンク生成 (structure_aware_v4)
#       -> default_metadata_builder() # 既定の metadata 生成 (差し替え可)
#       -> _has_multiline_body()      # 条本文が複数行か判定 (子チャンク採否)
#       -> _group_items_balanced()    # 親チャンクの子境界均等分割
#       -> _split_text_evenly()       # 子チャンク超過時のフォールバック均等分割
#
# 2) split_for_rag_texts_only() / split_for_rag(chunk_size=800)
#    -> split_for_rag_structure_aware() をラップ
#
# 3) main() (CLI)
#    -> split_for_rag_structure_aware() と同等の各ステップを順に実行
#    -> _print_tree() を必要時に実行


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
    # 行内に埋め込まれた強い見出し（例: 「。第10条...」）を分割・補正するか。
    enable_inline_heading_repair: bool = True
    # グループ統計から見出し level を推論するか。False ならルール既定寄りになる。
    enable_level_inference: bool = True
    # 見出し抽出に失敗した場合、段落ベースのフォールバックチャンクを作るか。
    fallback_to_paragraph: bool = True
    # max_chunk_chars: この文字数を超えるセクションは子ノードへ分割委譲する。
    # 収まる場合は子孫テキストを含めて1チャンクにまとめ、重複を防ぐ。
    max_chunk_chars: int = 1500
    # True の場合、候補抽出・推論の診断情報を出力する（CLIデバッグ向け）。
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
# - 内容: (HeadingRule, コンパイル済み正規表現) のタプル配列
# - 元データ: DEFAULT_HEADING_RULES
# - 利用箇所: extract_heading_candidates() の「ルールマッチ」ループ
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
    """漢数字混じり文字列を整数に変換。失敗時はNone。

    呼び出し元:
      - _extract_marker_value()
    """
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
    """文中参照らしい表現を含むか判定する補助関数。

    呼び出し元:
      - _is_strong_heading_line()
      - _split_embedded_headings()
      - extract_heading_candidates()
    """
    return bool(_INLINE_REF_RE.search(text))


def _is_strong_heading_line(stripped: str) -> bool:
    """第n章/第n条/附則 で始まる行かどうか (文中参照を除く)。"""
    if _is_inline_ref(stripped):
        return False
    return bool(re.match(r"^(?:附則|第[0-9一二三四五六七八九十百千〇]+(?:章|条))", stripped))


def _section_id(heading: str, start_char: int) -> str:
    """見出し文字列と開始位置から安定したセクションIDを生成する。

    呼び出し元:
      - build_section_tree()
      - _add_paragraph_fallback()
    """
    key = f"{start_char}|{heading}"
    return "sec_" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _extract_marker_value(marker_type: str, raw_marker: str) -> int | str | None:
    """見出しマーカー文字列から比較可能な番号値を抽出する。

    呼び出し元:
      - extract_heading_candidates()
    """
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
      - 改行コード統一
      - 全角空白の半角化
      - 行末空白除去
      - 行内埋め込み見出しの分割
      - 強見出し前の空行挿入
      - 過剰な連続空行の圧縮

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    呼び出し先:
      - _split_embedded_headings()
      - _ensure_blank_before_headings()
    """
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
    """強い見出し行 (第n章/第n条/附則) の直前が非空行なら空行を挿入する。

    呼び出し元:
      - normalize_text()
    """
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
    """各行から見出し候補を抽出し、特徴量ベースでスコアリングする。

    実施内容:
      - 見出しルールとの先頭マッチ判定
      - 文字長・空行後・インデント・句点有無などで加点/減点
      - 文中参照らしさやURLらしさなどを減点
      - 閾値未満を除外後、候補を返却
      - 最後に連番ボーナスを適用

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    呼び出し先:
      - _extract_marker_value()
      - _is_inline_ref()
      - _award_sequence_bonus()
    """
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
    """見出しタイプごとに統計を計算し、階層推定の材料を作る。

    実施内容:
      - タイプ別グルーピング
      - 出現位置ギャップ、平均インデント、行長、連番性を算出
      - その後、タイプ間の内包関係・番号リセット傾向を追加算出

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    呼び出し先:
      - _compute_sequence_score()
      - _compute_containment_and_reset()
    """
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
    """候補列の番号連続性を 0.0-1.0 のスコアで返す。

    呼び出し元:
      - analyze_heading_groups()
    """
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
    """タイプA区間にタイプBが何件入るかを使って内包/リセット傾向を計算する。

    呼び出し元:
      - analyze_heading_groups()
    """
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
    """候補の見出しタイプから階層レベル(level)を推定して付与する。

    実施内容:
      - 統計が無い/推定無効時はデフォルトlevelを適用
      - 出現間隔・内包・リセット・連番・インデント等を統合して外側スコア化
      - タイプごとに level を割当て、各候補へ反映
      - 推定根拠を inference_reason に保存

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    """
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
    """推定済み見出し候補から親子構造のセクションツリーを構築する。

    実施内容:
      - 文書ルートノード生成
      - 各候補のテキスト範囲(end)計算
      - スタックで親子関係を解決しながらノード追加
      - 見出しが無ければ段落フォールバックを適用

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    呼び出し先:
      - _section_id()
      - _candidate_confidence()
      - _add_paragraph_fallback()
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
    """推定根拠スコアからノード信頼度を算出する。

    呼び出し元:
      - build_section_tree()
    """
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


def _split_text_evenly(text: str, max_chars: int) -> list[str]:
    """テキストを max_chars 以下でなるべく均等な長さに分割する。

    呼び出し元:
      - flatten_chunks() 内の _emit_parent() / _emit_child()
    """
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

    呼び出し元:
      - flatten_chunks() 内の _emit_parent()

    分割方針:
      - 合計 total が max 以下なら 1 グループにまとめる
      - そうでなければ n = ceil(total / max) を最小分割数とし、
        target = ceil(total / n) を均等ターゲットにする
      - 子のサイズを順に積み、target 到達 or max 到達で次のグループへ
      - これにより「子境界を尊重」「max_per_group 以下」「均等に近い」
        の3条件を素直に両立できる
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


def _chunk_role_for_node(node: SectionNode) -> str:
    """ノードの heading_type / level からチャンクの役割を判定する。

    呼び出し元:
      - default_metadata_builder()
    """
    if node.heading_type == "paragraph":
        return "fallback"
    if node.level == 1:
        return "parent"
    if node.level == 2:
        return "child"
    # level>=3 は通常独立チャンクにしないが、safety net として child 扱い
    return "child"


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
    """既定の metadata builder。チャンクの親子構造情報を生成する。

    呼び出し元:
      - flatten_chunks() (metadata_builder 未指定時)

    差し替え方法:
      chunker.py 等で独自フィールド (doc_id, source 等) を含めたい場合は、
      本関数と同じシグネチャ (キーワード引数のみ) を持つ関数を作成し、
      flatten_chunks() / split_for_rag_structure_aware() の
      metadata_builder 引数に渡す。差し替え時は本関数を内部で呼び出して
      ベース dict を取得し、独自フィールドをマージする実装を推奨する。
    """
    return {
        "chunk_id": chunk_id,
        "parent_id": section_parent_id,
        # v3 互換のため残置 (v4 では使用しない)
        "grandparent_id": None,
        "root_id": doc_root_id,
        "children_ids": [],
        "level": node.level,
        "chunk_role": _chunk_role_for_node(node),
        "heading": node.heading,
        "heading_type": node.heading_type,
        "path": path,
        "path_text": " > ".join(path),
        "ordinal": node.ordinal,
        "start_char": node.start_char,
        "end_char": node.end_char,
        "source_type": "txt",
        "chunking_strategy": "structure_aware_v4",
        "structure_confidence": node.confidence,
        "inference_reason": node.inference_reason,
        "split_index": split_index,
        "split_total": split_total,
        "ancestor_chain": ancestor_chain,
    }


def _has_multiline_body(text: str, heading: str) -> bool:
    """条のテキストから先頭の見出しを除いた本文が、空行を除いて2行以上あるかを判定。

    呼び出し元:
      - flatten_chunks() 内の _emit_child()

    判定ルール:
      - text の先頭が heading に一致するなら、その分を除外して本文を取り出す
      - 本文を行分割して、空白のみの行を除外した残りが 2 行以上なら True
      - 1 行しかない条 (例: 第22条 のような単一段落) は子チャンク化対象外となる
      - 「不要な空行」が混じっていても、空行は除外されるので影響しない
    """
    body = text.strip()
    if heading and body.startswith(heading):
        body = body[len(heading):]
    meaningful = [ln for ln in body.split("\n") if ln.strip()]
    return len(meaningful) >= 2


def flatten_chunks(
    root: SectionNode,
    config: ChunkingConfig,
    metadata_builder: MetadataBuilder | None = None,
) -> list[dict]:
    """セクションツリーをRAG投入用チャンク配列へ平坦化する。

    分割戦略 (structure_aware_v4):
      - level==1 のノード (章相当) を「親チャンク」として出力する。
        サイズが max_chunk_chars 以内なら 1 チャンク、超過時は子(条)境界を
        尊重した均等分割を行う (子は分割されない)。各分割チャンクの先頭に
        は親見出し (章ヘッダ + 章先頭の本文) を必ず付与する。
        附則のような少数構造でも level 推論で 1 と判定されたものは
        同様に親チャンクとして扱う (「1の層」共通認識)。
      - level==2 のノード (条相当) を「子チャンク」として出力する。
        ただし、見出し行を除いた本文に空行を除いて 2 行以上残るもののみ
        子チャンク化する (単一段落で完結する条はノイズ抑制のため除外)。
      - 子チャンクのサイズが max_chunk_chars を超える場合は、条の見出しを
        各分割チャンクの先頭に付与しつつ均等分割する (条は通常分割しない
        が、超過時のみフォールバックとして文字数ベースで均等分割)。
      - level>=3 のノード (列挙等) は独立したチャンクにしない。
        親/子チャンクの本文に内包されるかたちで保持される。
      - 親-子の階層情報は ancestor_chain / parent_id で表現する。
      - 0 件時は最低 1 チャンクを補償する。

    引数:
      metadata_builder: チャンク metadata の生成関数。None なら
        default_metadata_builder を使用。chunker.py 等から差し替えて
        独自フィールド (doc_id 等) をマージできる。

    呼び出し元:
      - split_for_rag_structure_aware()
      - main()
    呼び出し先:
      - default_metadata_builder()
      - _has_multiline_body()
      - _group_items_balanced()
      - _split_text_evenly()
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
        # metadata_builder へキーワード引数で委譲。差し替え可能設計。
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
                "text": text,
                "metadata": _build_meta(
                    base_id, node, path, section_parent_id, ancestor_chain,
                ),
            })
            return

        # --- 章サイズ超過: 子(条)境界で均等分割 ---
        # 章 prefix = 「章見出し + 最初の条までの本文」。各分割の先頭に付与する。
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
                    "text": part,
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
                "text": chunk_text,
                "metadata": _build_meta(
                    cid, node, path, section_parent_id, ancestor_chain,
                    split_index=gi, split_total=len(groups),
                ),
            })

    def _emit_child(node: SectionNode, ancestors: list[SectionNode]) -> None:
        # level==2 (条) を子チャンクとして出力 (本文が複数行のもののみ)
        text = node.text.strip()
        if not text or len(text) < config.min_child_text_length:
            return
        if not _has_multiline_body(text, node.heading):
            # 単一段落の条 (例: 第22条) は子チャンク化しない
            return

        ancestor_chain = [_ancestor_entry(a) for a in ancestors]
        path = [root.heading] + [a.heading for a in ancestors] + [node.heading]
        base_id = f"chunk_{node.id}"
        section_parent_id = f"chunk_{node.parent_id}" if node.parent_id else None

        if len(text) <= config.max_chunk_chars:
            chunks.append({
                "id": base_id,
                "text": text,
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
                    "text": part,
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
                "text": chunk_text,
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
    """構造認識チャンク化のエントリーポイント。

    引数:
      metadata_builder: チャンク metadata の生成関数。None なら
        default_metadata_builder を使用。chunker.py 等から差し替え可能。

    呼び出し順:
      normalize_text()
      -> extract_heading_candidates()
      -> analyze_heading_groups()
      -> infer_heading_levels()
      -> build_section_tree()
      -> flatten_chunks(metadata_builder=...)
    """
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
                "chunking_strategy": "structure_aware_v4",
                "structure_confidence": 0.0,
                "inference_reason": {},
                "ancestor_chain": [],
                "grandparent_id": None,
            },
        }]

    normalized = normalize_text(text, config)
    candidates = extract_heading_candidates(normalized, config)
    group_stats = analyze_heading_groups(candidates, normalized, config)
    candidates = infer_heading_levels(candidates, group_stats, config)
    root = build_section_tree(normalized, candidates, config)
    return flatten_chunks(root, config, metadata_builder=metadata_builder)


def split_for_rag_texts_only(text: str) -> list[str]:
    """text 本文のみを返す薄いラッパー。

    呼び出し元:
      - split_for_rag()
    呼び出し先:
      - split_for_rag_structure_aware()
    """
    return [c["text"] for c in split_for_rag_structure_aware(text) if c["text"]]


def split_for_rag(
    *, text: str, chunk_size: int = 800, chunk_overlap: int = 0
) -> list[str]:
    """既存 split_for_rag との互換インタフェース。chunk_size を max_chunk_chars として使用。"""
    config = ChunkingConfig(max_chunk_chars=chunk_size)
    return [c["text"] for c in split_for_rag_structure_aware(text, config) if c["text"]]


def split_for_rag_with_metadata(
    *, text: str, chunk_size: int = 800, chunk_overlap: int = 0
) -> list[dict[str, Any]]:
    """metadata 付きスプリッタ（chunker.py から experiment_context 経由で利用される）。

    返り値は [{"text": str, "metadata": dict}, ...] 形式。
    default_metadata_builder が生成する structure_aware_v4 の親子チャンク情報
    (chunk_role, ancestor_chain, path, parent_id 等) を metadata として保持する。
    chunker.py 側で chunk_id / doc_id / source / document_lower の最低限デフォルト
    を後付けするので、ここではロジック固有情報のみを返せばよい。

    chunk_overlap は構造認識の性質上未使用 (引数互換のため受け取るのみ)。
    """
    config = ChunkingConfig(max_chunk_chars=chunk_size)
    chunks = split_for_rag_structure_aware(text, config)
    items: list[dict[str, Any]] = []
    for c in chunks:
        ctext = c.get("text", "")
        if not ctext:
            continue
        items.append(
            {
                "text": ctext,
                "metadata": dict(c.get("metadata") or {}),
            }
        )
    return items


# ============================================================
# CLI
# ============================================================


def _print_tree(node: SectionNode, indent: int = 0) -> None:
    """デバッグ表示用にセクションツリーを再帰出力する。

    呼び出し元:
      - main()  # --print-tree 指定時
    """
    prefix = "  " * indent
    label = node.heading[:60] if node.heading else "(no heading)"
    print(f"{prefix}[L{node.level}:{node.heading_type}] {label}")
    for child in node.children:
        _print_tree(child, indent + 1)


def main() -> None:
    """CLI エントリーポイント。

    処理フロー:
      - 入力読み込み
      - normalize -> candidate抽出 -> group解析 -> level推定
      - ツリー構築 -> チャンク平坦化
      - 集計・デバッグ表示・ファイル出力
    """
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
