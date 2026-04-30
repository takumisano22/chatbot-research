# chunking_logic_03 チャンク化戦略

## 概要

`chunking_logic_03.py` は、法規・規程類など **見出し階層のある構造化文書** を対象とした **構造認識チャンク化** を実装する。メタデータ上の戦略名は **`structure_aware_v4`**。

- 章・条・列挙などの **見出しタイプを行頭パターン＋スコアで候補化**し、**グループ統計**と **ツリー構築後のスコープ内再帰スコア**で level（外側／内側）を決める。
- **level 1＝親チャンク（章相当）、level 2＝子チャンク（条相当）** を RAG の主単位とし、**level 3 以上は独立チャンク化しない**（装飾済み本文として親／子に内包）。
- 公開 API は **`split_for_rag_with_metadata`**。内部で `ChunkingConfig(max_chunk_chars=chunk_size)` を組み立て、**引数 `chunk_size` の既定は 800**（`ChunkingConfig` 単体の `max_chunk_chars` 既定は **500**）。返却時は **`text` が空の要素を除外**。`chunk_overlap` は互換のため受け取るのみで **未使用**。

---

## 方針

「章 → 条 → 項 → 列挙」のような階層を、**固定ルールのみ**ではなく **文中パターンと統計から動的に推定**する。

1. **見出し候補を行単位の特徴量スコアで抽出する**  
   行頭の `DEFAULT_HEADING_RULES` 一致を起点に、直前空行・行長・章／条／附則・項・括弧題名・句点・URL 風文字列などを加減点し、`min_heading_score` 未満を捨てる。**附則／第N章／第N条** は、OCR で本文が同行した **長行でも** `max_heading_line_length` を超えていれば候補に残せる（その他のタイプは超過行はスキップ）。

2. **見出しタイプの「外側らしさ」を統計と再帰スコアで決める**  
   `analyze_heading_groups` で間隔・連番・内包・リセット等を集計し `infer_heading_levels` で大枠の level を付与したうえで、`build_section_tree` 後に **`_rebuild_tree_by_recursive_scoring`** によりスコープごとに level を振り直す。**章が 1 件しかない小規模規程**など、グローバル優先だけでは崩れる構造を拾う。

3. **RAG 投入単位は主に親／子の 2 層**  
   level 1 を親、level 2 を子とし、level 3 以上は **別チャンクとして出さない**（検索粒度を条に寄せつつ、章まとまりは親で保持）。

4. **`max_chunk_chars` 超過時は装飾後テキストの「ブロック境界」で分割する**  
   ツリー構築後、**先に `_format_subtree` でマークダウン装飾**した文字列に対し、親（level 1）は **`### ` で始まる行（条の見出し）**、子（level 2）は **`#### ` で始まる行（孫＝列挙見出し）** を境界に **`_split_at_block_boundary`** でグルーピングし、`_group_items_balanced` で長さを抑える。**ブロックが取れない／prefix だけで上限を食い潰す**場合は **`_split_text_evenly`**。境界より前の行は **分割された各パート先頭に共通 prefix** として付く（旧来の「章見出しだけを機械的に付けて均等切り」ではない）。

5. **見出しのみ・1 行本文の条は除外と昇格を分ける**  
   採否は **`_meaningful_body_lines`**（装飾済みテキストの **先頭 1 行を見出しとして除いた**非空行数）。**2 行以上は採用**。**1 行**は、**同一親（章）配下に同 `heading_type` で 2 行以上本文の兄弟が 1 件でもあれば昇格**。**0 行（見出しのみ）は除外**（昇格もしない）。子チャンク生成前に **`_collapse_enumeration_blanks`** をもう一度当てる（冪等）。

---

## 処理フロー

```
split_for_rag_with_metadata(text, chunk_size, …)
  └─ normalize_text()
  └─ extract_heading_candidates()
  └─ analyze_heading_groups()
  └─ infer_heading_levels()
  └─ build_section_tree()  → _rebuild_tree_by_recursive_scoring()
  └─ _identify_document_title_ids()   … 総則／前文が最浅 level でユニークなとき # 用
  └─ _format_subtree()                 … node.id → 装飾済み全文（子内は _classify_grandchild_levels）
  └─ flatten_chunks()                  … 長さ超過時のみ _split_at_block_boundary(### / ####)
```

---

## 各ステップの詳細

### 1. normalize_text

`enable_inline_heading_repair=True`（既定）のとき **1→2** を実行。`False` のとき **両方スキップ**。

1. **`_split_embedded_headings`** — 行内の **附則／第N章／第N条** を分割（`...する。第10条（服務）...`）。**文中参照**は `_INLINE_REF_RE` で判別し分割しない（例: 「労働基準法第89条に基づき」）。
2. **`_split_embedded_enumerations`** — 行内の **弱い列挙**（`1.`、`（1）`、漢数字括弧など）を分割。直前文字が境界パターンで、かつ参照接尾でない場合のみ。
3. **`_ensure_blank_before_headings`** — 強見出し行の直前が非空なら空行挿入（直前空行スコアの安定化）。
4. **`_split_heading_linebreaks`** — 見出し行のあとの **余計な二重改行** だけ詰める（構造文書向け。全行ペアを無差別に詰めない）。
5. **`_join_soft_linebreaks`** — 和文途中の **単独改行**（OCR 等）を、次行が見出しパターンでなければ結合。
6. **`_collapse_enumeration_blanks`** — `_classify_enum_item` が **同タイプかつ番号 +1 連続** と認める列挙行のあいだの **空行だけ** 除去。章・条境界は壊さない。

---

### 2. extract_heading_candidates

`DEFAULT_HEADING_RULES` は **12 種**（リスト先頭ほど priority が高い）。最初にマッチしたルールを採用。

判別するパターンの例: 附則、第N章／条／項、`1.1`、`1.`／`2．`（2 桁まで・小数除外）、`（1）`、`（一）`、ローマ数字・英字番号、`○`、`-` / `・` 等。

| 要素 | 加点 / 減点（実装定数） |
| --- | --- |
| ルール一致（基礎点） | +4.0 (`LINE_SCORE_BASE`) |
| 直前が空行 | +1.0 |
| 行長 ≤ 40 | +1.0 |
| 第n章 / 第n条 / 附則 | +4.0 |
| 第n項 | +2.0 |
| 括弧付き補足（`（…）` 1〜30 文字） | +2.0 |
| `。` `.` `．` で終わらない | +1.0 |
| 行長 > 60 | −2.0 |
| `。` と `．` の合計が 2 以上 | −2.0 |
| URL / メール風 | −3.0 |

`min_heading_score`（既定 3.0）未満は除外。最後に **`_award_sequence_bonus`**: 同 `marker_type` で int 値が **+1 連続なら +2.0**、**1 へリセットなら +0.5**。

> 注: 旧版の「インデント 0 の加点」は `LINE_SCORE_BASE` に吸収済み。

---

### 3. analyze_heading_groups

タイプ別に階層推定用の統計を計算する。

- **average_gap**: 出現間隔の平均（外側ほど広い傾向の材料）。
- **sequence_score**: `_sequence_score(..., short_default=0.5)` — 番号列の **+1 連続** と **1 リセット**（重み 0.5）。**n < 2** は **0.5**（判断保留）。
- **containment_score** / **reset_score**: `GAP_OUTER_RATIO`（0.8）で「A が B より外側候補」とみなせる場合のみ比較し、A の区間内の B の内包・先頭 1 リセットを集計。
- **fallback_priority**: ルール定義順から算出（先頭ほど大）。

---

### 4. infer_heading_levels

**`min_group_count` 未満**のタイプはランキングから外し **`default_level_hint`**（`max_depth` で切り詰め）を使用。

残りは「外側らしさ」でソートし level 1, 2, … を割当:

```
スコア =
  containment_score × 4.0
  + reset_score       × 3.0
  + sequence_score    × 1.0
  + (fallback_priority / (ルール数 + 1)) × 1.5   … 現状 12 ルールなので÷13
  + min(avg_gap / text_len × 10, 2.0)
```

`enable_level_inference=False` のときは **全候補を hint のみ**で埋めて終了。

---

### 5. build_section_tree

1. `inferred_level` でスタックにより親子ツリー化。各区間の `text` は次の **同階層以上の見出し** まで。
2. **`_rebuild_tree_by_recursive_scoring`**: ノードを扁平化し、`_assign_levels_recursive` でスコープ内の **`_score_type_in_scope`** により type を選び level を付け直し、再スタックして `text` を再計算。

**`_score_type_in_scope` の重み**（実装どおり）:

- **containment** ×5.0: 別 type の **+1 連続（≥2 ノード）** を内包する区間の比率（`_interval_containment`）。
- **sequence** ×2.5: `marker_value` の連番性（スコープ内は short 時 **デフォルト 0.0**）。
- **coverage** ×1.5: スコープ内の位置範囲。単独ノードは先頭 15% ゾーンかどうかで別スコア。
- **frequency** ×0.7: log 正規化した出現数。
- **priority** ×0.5: ルール順由来の tie-breaker。
- **preferred** ×2.0: **ひとつ上のスコープ**で「採用された hint より **内側**の type のうち **最高スコアの 1 つ**」として **`next_preferred` に入ったタイプ**にだけ、**直下スコープの評価で**加算（章→条→列挙の階層維持用）。

採用 type と **同一 `default_level_hint` かつ正スコア**の他 type も **同じ level に昇格**（例: 附則と章の並列）。

見出し候補が **0 件**で `fallback_to_paragraph` なら **`_add_paragraph_fallback`**。

---

### 6. `_format_subtree` と flatten_chunks

**`output_markdown=True`（既定）** のとき、装飾ルールは次のとおり（`_format_subtree`）。

| 条件 | 行頭 |
| --- | --- |
| `_identify_document_title_ids` に該当（**総則／前文** が最浅 level で **1 ノードだけ**） | `# ` |
| `appendix` または **level 1** | `## ` |
| **level 2**（条） | `### ` |
| **level 3** | `#### ` |
| **level ≥ 4**（ツリー上のより深い見出し） | `- ` |

**level 2 ノード**の本文行について、**自分より内側の `default_level_hint`** を持つ行を列挙し **`_classify_grandchild_levels`**（type 横断スタック）で **孫＝`####`、ひ孫＝`-`** に振り分ける（ツリー上は level 3 にまとまっていても **表示だけ**階層化）。

**`flatten_chunks`**:

| ノード | 出力 |
| --- | --- |
| level 1 | 親チャンク。超過時 `### ` 境界で分割 |
| level 2 | 子チャンク。超過時 `#### ` 境界で分割 |
| level ≥ 3 | 独立チャンクにしない |
| `paragraph` | フォールバックツリー由来の葉 |

整形結果が **`min_child_text_length` 未満**は出力しない。チャンク **0 件**のときはルートを 1 件で補償。複数パート時は chunk id に **`_p0`, `_p1`, …**。

---

## 主要な設定パラメータ（ChunkingConfig）

| パラメータ | デフォルト | 意味 |
| --- | --- | --- |
| `max_depth` | 4 | level の上限 |
| `min_heading_score` | 3.0 | 見出し候補の下限スコア |
| `min_group_count` | 2 | 統計推論対象とする最小件数 |
| `max_heading_line_length` | 80 | 通常行の最大長（**強見出し行は例外**） |
| `min_child_text_length` | 10 | チャンクとして出す最小文字数 |
| `enable_level_inference` | True | グループ統計で level を推論 |
| `enable_inline_heading_repair` | True | 行内強／弱見出し分割を行う |
| `fallback_to_paragraph` | True | 見出し 0 件時に段落フォールバック |
| `output_markdown` | True | 上記マークダウン装飾 |
| `max_chunk_chars` | 500 | 1 チャンクの上限（API では `chunk_size` で上書き） |

---

## チャンクのメタデータ（`default_metadata_builder`）

```json
{
  "chunk_id": "chunk_sec_xxxxxxxx",
  "parent_id": "chunk_sec_yyyyyyyy | null",
  "root_id": "doc_zzzzzzzz",
  "level": 1,
  "path_text": "文書名 > 第1章 > 第1条",
  "chunk_role": "parent | child | fallback",
  "chunking_strategy": "structure_aware_v4"
}
```

- **`chunk_role`**: `heading_type == "paragraph"` → `fallback`、**level == 1** → `parent`、**それ以外** → `child`（level ≥ 3 が独立化した場合も実装上は `child`）。
- 分割時は **`split_index` / `split_total`** が builder に渡る。
- **`flatten_chunks(metadata_builder=...)`** は `default_metadata_builder` と同じ **キーワード専用引数**（`chunk_id`, `node`, `path`, `section_parent_id`, `ancestor_chain`, `doc_root_id`, `split_index`, `split_total`）を取る関数を差し替え可能。
