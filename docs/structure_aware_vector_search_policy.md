# Structure Aware Chunking Vector Search Policy

## 1. このchunking方式の概要

`chunking_logic_02.py` は、正規化済みtxt文書を入力として、文書全体の見出し分布・包含関係・連続性を解析し、親子構造付きのチャンクを生成する。

固定長分割と異なり、文書構造（章・条・番号・箇条書きなど）を動的に推定してlevelを割り当てるため、異なる種類の文書でも同一ロジックが使える。

処理フロー:
1. `normalize_text` — 改行統一、見出し直前の空行補正
2. `extract_heading_candidates` — スコアリングによる見出し候補抽出
3. `analyze_heading_groups` — marker_typeごとの包含関係・連続性集計
4. `infer_heading_levels` — 文書全体の分布から階層推定
5. `build_section_tree` — 親子ツリー構築
6. `flatten_chunks` — `list[dict]` 生成（metadata付き）

---

## 2. metadata項目の説明

| フィールド | 型 | 説明 |
|---|---|---|
| `chunk_id` | str | このチャンクの一意ID (`chunk_<hash>`) |
| `parent_id` | str \| None | 親チャンクのchunk_id。level 1はNone |
| `root_id` | str | ドキュメントルートID (`doc_<hash>`) |
| `children_ids` | list[str] | 直接の子チャンクIDリスト |
| `level` | int | ツリー上の深さ（1=最外側）|
| `chunk_role` | str | `parent` / `main` / `child` / `fallback` |
| `heading` | str | このチャンクの見出しテキスト |
| `heading_type` | str | 見出しパターン種別（下表参照）|
| `path` | list[str] | ルートからこのチャンクへのパス（見出しのリスト）|
| `path_text` | str | pathを ` > ` で結合した文字列 |
| `ordinal` | int \| None | 見出しの番号（第5条なら5）|
| `start_char` | int | 元テキスト内の開始文字位置 |
| `end_char` | int | 元テキスト内の終了文字位置 |
| `source_type` | str | `"txt"` 固定 |
| `chunking_strategy` | str | `"structure_aware_v1"` 固定 |
| `structure_confidence` | float | 階層推定の信頼度（0.0〜1.0）|
| `inference_reason` | dict | 推定根拠スコア詳細 |

### heading_type の種類

| 値 | 対象パターン |
|---|---|
| `japanese_chapter` | 第n章 |
| `japanese_article` | 第n条 |
| `japanese_section` | 第n項 |
| `appendix` | 附則 |
| `numeric_dot` | 1. / 2. |
| `numeric_paren` | (1) / （1）|
| `japanese_paren` | （一）/ (一)|
| `decimal_number` | 1.1 / 1.2.3 |
| `roman` | I. / Ⅰ |
| `alpha` | A. |
| `circle_bullet` | ○ xxx |
| `bullet` | ・ / • / - |
| `paragraph` | fallback段落 |

---

## 3. ベクトル検索時の基本方針

### 主検索対象

`chunk_role == "main"` を主検索対象にする。就業規則であれば第n条レベルが該当し、質問に対して最も適切な粒度の回答が得られる。

### child チャンクがhitした場合

`parent_id` を使って `main` または `parent` チャンクを展開する。  
例: 「第9条 > 1.」がhitした場合、`parent_id` で第9条本文を取得して文脈補完する。

### parent チャンクの用途

`chunk_role == "parent"` は章単位の広い文脈補完に使う。単独では検索対象にせず、childやmainがhitしたときの補完として参照する。

---

## 4. chunk_role の使い方

| chunk_role | 対象 | 用途 |
|---|---|---|
| `parent` | level 1（章等）| 広い文脈補完。章全体の概要把握に使う |
| `main` | level 2（条等）| 主検索対象。RAGの回答根拠として使う |
| `child` | level 3以降 | 詳細検索。hitしたらparent_idで親を補完 |
| `fallback` | 段落 | 構造推定が失敗したチャンク。信頼度が低い |

---

## 5. levelだけに依存しない理由

文書ごとに推定levelが変わる可能性がある。例えば：
- 就業規則: level 1=章, level 2=条, level 3=番号
- 契約書: level 1=条, level 2=項, level 3=号
- マニュアル: level 1=章, level 2=節, level 3=項

そのため検索フィルタには `level` ではなく `chunk_role` と `heading_type` を組み合わせて使うことを推奨する。

---

## 6. metadata filterの例

```python
# 主検索: 条文レベルのみ
filter = {"chunk_role": "main"}

# 就業規則特化: 条文のみ
filter = {"chunk_role": "main", "heading_type": "japanese_article"}

# 信頼度が高いチャンクのみ
filter = {"structure_confidence": {"$gte": 0.6}}

# 章全体を取得
filter = {"chunk_role": "parent", "heading_type": "japanese_chapter"}
```

---

## 7. rerank時の方針

1. **同一parent_idの重複をまとめる**: 同じ親の複数childがhitしたら、parent_idのmainを1件として代表させる
2. **同一path配下のchildが複数hitしたら親を優先**: `children_ids` を持つmainチャンクを代表にする
3. **mainとchildの両方がhitしたらmainを代表にする**: childは文脈補完に使い、スコアはmainに集約する

---

## 8. 回答生成時の方針

- `path_text` をプロンプトのソース表示に含める  
  例: `「株式会社エックス就業規則 > 第2章採用、異動等 > 第5条（採用時の提出書類）」`
- 複数チャンクを使う場合は `path_text` でソートし、文書の論理的な順序で並べる
- `heading_type == "japanese_article"` のチャンクは「第n条」として引用根拠を明示する
- `structure_confidence` が低いチャンク（< 0.4）を使う場合は「参考情報」として注記する

---

## 9. structure_confidenceが低い場合の扱い

`structure_confidence < 0.4` のチャンク（`chunk_role == "fallback"` を含む）は：

- 構造推定が不確かなため、回答根拠として単独では使わない
- 必要に応じて `parent_id` で親チャンクを取得し、広めの文脈を添える
- 検索時に `structure_confidence >= 0.4` でフィルタして弾くことも検討する

---

## 10. 将来的なhybrid search

### BM25 + vector

- vectorはセマンティクス検索（意味的な類似）に使う
- BM25はキーワード検索（条文番号、固有名詞）に使う
- 両スコアをRRF（Reciprocal Rank Fusion）などで統合する

### heading / path_text へのキーワード重み付け

- `heading` と `path_text` フィールドに対してBM25スコアを高く設定する
- 「第5条」「採用手続」などの見出し語が質問に含まれる場合、headingマッチを優先する

### metadata filterとrerankの組み合わせ

```
1. chunk_role == "main" でvector検索 (top-k)
2. 同一parent_idで重複除去
3. BM25スコアでrerank
4. path_textを根拠としてプロンプトに付与
```
