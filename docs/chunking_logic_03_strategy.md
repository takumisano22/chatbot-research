# chunking_logic_03 チャンク化戦略

## 概要

`chunking_logic_03.py` は、法規・規程類などの構造化文書を対象とした **構造認識チャンク化 (structure_aware_v4)** を実装する。  
文書の章・条階層を自動推定し、RAG 投入用のチャンク配列を生成する。

---

## 処理フロー

```
入力テキスト
  └─ 1. normalize_text()         — 強見出し前への空行挿入
  └─ 2. extract_heading_candidates() — 行単位でスコアリングし見出し候補を抽出
  └─ 3. analyze_heading_groups()    — 見出しタイプごとに統計を計算
  └─ 4. infer_heading_levels()      — タイプ別「外側らしさスコア」で階層を推定
  └─ 5. build_section_tree()        — スタックベースで親子ツリーを構築
  └─ 6. flatten_chunks()            — ツリーをチャンク配列へ平坦化
```

---

## 各ステップの詳細

### 1. normalize_text
`第n章 / 第n条 / 附則` など強い見出し行の直前に空行がなければ挿入する。  
後段の「直前が空行か」判定（スコア加点）を確実に効かせるための前処理。

---

### 2. extract_heading_candidates
各行を `DEFAULT_HEADING_RULES`（正規表現 14 種）でマッチし、以下の特徴量でスコアリング。

| 要素 | 加点/減点 |
|------|----------|
| ルールに一致 | +3.0 |
| 直前が空行 | +1.0 |
| 行長 ≤ 40 文字 | +1.0 |
| `第n章 / 第n条 / 附則` | +4.0 |
| 括弧付き補足あり | +2.0 |
| 句点で終わらない | +1.0 |
| インデント 0 | +1.0 |
| 行長 > 60 文字 | −2.0 |
| 句点が 2 つ以上 | −2.0 |
| URL・メール含む | −3.0 |

`min_heading_score`（デフォルト 3.0）未満は除外。  
最後に **連番ボーナス**（同タイプで番号が +1 連続なら +2.0）を付与。

---

### 3. analyze_heading_groups
タイプ別に以下の統計を算出し、階層推定の材料にする。

- **average_gap**: 出現間隔の平均（外側ほど広い）
- **sequence_score**: 番号連続性 (0.0–1.0)
- **containment_score**: 他タイプを区間内に内包する度合い
- **reset_score**: 内包タイプの番号が区間先頭で 1 にリセットされる割合

---

### 4. infer_heading_levels
各タイプの「外側らしさスコア」を計算し、降順ソートで level 1, 2, 3... を割り当てる。

```
外側らしさスコア =
  containment_score × 4.0
  + reset_score       × 3.0
  + sequence_score    × 1.0
  + shallow_indent    × 1.5
  + fallback_priority × 1.5 / 12.0
  + gap_score（上限 2.0）
```

出現数が `min_group_count`（デフォルト 2）未満のタイプ（例: 附則）は推定対象外とし、ルール定義の `default_level_hint` をそのまま使う。

---

### 5. build_section_tree
スタックベースで見出し候補を親子ツリーへ組み立て、後処理として `_rebuild_tree_by_hints()` を実行する。

**_rebuild_tree_by_hints の役割**  
統計推論で生じうる level 衝突（例: `numeric_dot` が条と同 level になる）を解消するため、各 `heading_type` のルール上の `default_level_hint` を真値としてツリーを再構築する。これにより **親チャンクの分割境界が常に条レベルで揃う**。

見出しが 1 つも検出されない場合は `_add_paragraph_fallback()` で段落単位のノードを生成する。

---

### 6. flatten_chunks (structure_aware_v4)

| ノード条件 | 出力チャンク種別 |
|------------|----------------|
| `level == 1`（章・附則相当） | **親チャンク** |
| `level == 2`（条相当） | **子チャンク** |
| `level >= 3` | 独立チャンク化しない（親/子本文に含める） |

#### 親チャンク (level == 1)
- `max_chunk_chars`（デフォルト 1500）以内 → 子孫テキストを含む 1 チャンク
- 超過時 → 子(条)境界を尊重した均等分割。各分割の先頭に章見出し prefix を付与

#### 子チャンク (level == 2)
採用基準（本文の空行を除いた行数）：
- **2 行以上** → 無条件採用
- **1 行** → 同章内に「2 行以上の同種条」が 1 つでも存在すれば昇格採用
- **0 行（見出しのみ）** → 除外（昇格対象でも除外）

`max_chunk_chars` を超える条は見出しを先頭に付与しつつ均等分割。

---

## 主要な設定パラメータ (ChunkingConfig)

| パラメータ | デフォルト | 意味 |
|-----------|-----------|------|
| `max_depth` | 4 | 階層の上限 |
| `min_heading_score` | 3.0 | 見出し採用の最低スコア |
| `min_group_count` | 2 | 統計推論に使う最小出現数 |
| `max_heading_line_length` | 80 | 見出しとみなす最大文字数 |
| `min_child_text_length` | 10 | 子チャンクとして残す最小本文長 |
| `max_chunk_chars` | 1500 | チャンク最大文字数（超過時に分割） |
| `fallback_to_paragraph` | True | 見出し未検出時に段落単位フォールバック |

---

## チャンクのメタデータ

各チャンクには以下のフィールドが付与される。

```json
{
  "chunk_id": "chunk_xxxxxxxx",
  "parent_id": "chunk_yyyyyyyy | null",
  "root_id": "doc_zzzzzzzz",
  "level": 1,
  "path_text": "文書名 > 第1章 > 第1条",
  "chunk_role": "parent | child | fallback",
  "chunking_strategy": "structure_aware_v4"
}
```

`metadata_builder` 引数に独自関数を渡すことで、`doc_id` や `source` などのフィールドを追加できる。
