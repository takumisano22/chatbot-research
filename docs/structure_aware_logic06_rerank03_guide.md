# chunking_logic_06 / reranking_logic_03 説明書

## 概要

`chunking_logic_06.py` は、就業規則や規程類のような「章・条・番号列挙」を持つ文書を、構造を保ったまま RAG 用チャンクへ変換するロジックです。固定長で機械的に切るのではなく、見出し候補を抽出し、文書内の出現間隔・連番・内包関係から階層を推定します。

`reranking_logic_03.py` は、`chunking_logic_06.py` が付与した階層 metadata を使い、ベクトル検索で取得した候補を LLM に渡しやすい件数・粒度へ圧縮するロジックです。`reranking_logic_02.py` との違いは、**孫→子への集約は行うが、子→親への集約は行わない**点です。親チャンクはスコア閾値を超えれば子と並列に残ります。

関連ファイル:

| ファイル | 役割 |
| --- | --- |
| `backend/app/rag/logic/chunking/chunking_logic_06.py` | 構造認識チャンク化本体 |
| `backend/app/rag/logic/reranking/reranking_logic_03.py` | 階層 metadata ベースの候補圧縮（子→親集約なし） |
| `backend/app/rag/vectorstore/chunker.py` | chunking 出力に backend 側の `doc_id` / `chunk_id` を付与 |
| `backend/app/rag/vectorstore/vector_db.py` | `vector_texts` を物理ベクトルレコードへ展開 |

---

## チャンク分割過程

### 方針

`chunking_logic_06.py` の目的は、検索粒度を細かくしすぎず、回答時に必要な上位文脈を落とさないことです。

主な考え方は次の通りです。

1. **文書構造を先に復元する**  
   OCR や PDF 変換後のテキストでは、見出しが本文中に埋まったり、列挙の間に不要な空行が入ったりします。先にこれを補正してから見出し判定します。

2. **固定ルールだけで階層を決めない**  
   `第N章` は外側、`第N条` は内側、というヒントは持ちますが、実際の level は出現間隔・連番・内包関係から再評価します。章が 1 件だけの文書や、条と番号列挙が混在する文書でも崩れにくくするためです。

3. **親・子・孫の複数粒度を持つ**  
   level 1 を `parent`、level 2 を `child`、level 3 相当を `grandchild` として扱います。大きな章だけでなく、条や列挙単位も検索に乗せられるようにします。

4. **重複は出力時に抑える**  
   子が 1 件しかない親は子チャンクを省略し、親が子の metadata を兼ねます。孫が 1 件しかない子も同様に、子が孫の metadata を兼ねます。これにより、同じ本文を複数粒度で過剰に LLM へ渡すことを抑えます。

### 実装

公開 API は `split_for_rag_with_metadata(text, chunk_size=800, chunk_overlap=0)` です。現実装では `chunk_size` / `chunk_overlap` は互換用に受け取るだけで、実際の上限は `ChunkingConfig.max_chunk_chars = 1000` です。

処理の全体像:

```text
split_for_rag_with_metadata
  -> normalize_text
  -> extract_heading_candidates
  -> analyze_heading_groups
  -> infer_heading_levels
  -> build_section_tree
       -> _rebuild_tree_by_recursive_scoring
  -> flatten_chunks
       -> _format_subtree
       -> _emit_parent / _emit_child / _emit_grandchild
       -> _build_vector_texts_for_chunk
```

#### 1. テキスト正規化

`normalize_text()` で、見出し抽出前の形を整えます。

| 処理 | 内容 |
| --- | --- |
| `_split_embedded_headings` | `...する。第10条...` のように本文中へ埋まった強見出しを行分割する |
| `_split_embedded_enumerations` | `...とき2. ...3. ...` のような弱い列挙を行分割する |
| `_ensure_blank_before_headings` | `附則` / `第N章` / `第N条` の直前へ空行を補う |
| `_join_soft_linebreaks` | OCR 由来の文中単独改行を結合する |
| `_collapse_enumeration_blanks` | `1.` `2.` など連番列挙の間にある空行だけを詰める |

文中参照、例えば「労働基準法第89条に基づき」のような文字列は見出しとして分割しないよう、接尾語パターンで除外します。

#### 2. 見出し候補抽出

`DEFAULT_HEADING_RULES` で、行頭パターンを見出し候補として拾います。対象は `附則`、`第N章`、`第N条`、`第N項`、`1.`、`(1)`、`（一）`、ローマ数字、英字、箇条書きなどです。

`extract_heading_candidates()` は、単に正規表現に一致した行を全部採用するのではなく、次のような特徴でスコアを加減点します。

| 評価対象 | 意味 |
| --- | --- |
| 直前が空行 | 見出しらしさを加点 |
| 短い行 | 見出しらしさを加点 |
| `第N章` / `第N条` / `附則` | 強い見出しとして大きく加点 |
| 括弧付き題名 | `第5条（休職）` のような条文見出しを加点 |
| 長文・句点多数・URL 風 | 本文やノイズの可能性として減点 |
| 同 type の連番 | `第1条` -> `第2条` などを追加加点 |

最低スコア `min_heading_score = 3.0` 未満は捨てます。ただし `附則` / `第N章` / `第N条` は、OCR で本文が同じ行に連結された長い行でも候補に残りやすい扱いです。

#### 3. 階層推定

`analyze_heading_groups()` は、見出し type ごとに統計を取ります。

- 出現数
- 出現位置の平均間隔
- 番号の連続性
- ある type の区間に別 type が内包される傾向
- 内側番号が 1 に戻るリセット傾向

`infer_heading_levels()` は、これらの統計から「外側らしさ」を計算し、level 1, 2, 3... を割り当てます。出現数が少なすぎる type は統計推定せず、ルール定義の `default_level_hint` を使います。

その後、`build_section_tree()` で一度ツリー化し、さらに `_rebuild_tree_by_recursive_scoring()` でスコープごとに level を振り直します。ここが logic_06 の重要点です。文書全体では少数派に見える `第N条` でも、章の内側スコープでは主要な区切りとして再評価されます。

#### 4. ツリーの Markdown 化

`_format_subtree()` は、推定したツリーを RAG 用に読みやすい Markdown へ整形します。

| ノード | 出力例 |
| --- | --- |
| 文書タイトル | `# 就業規則` |
| level 1 / 附則 | `## 第1章 総則` |
| level 2 | `### 第1条（目的）` |
| level 3 | `#### 1. 対象者` |
| level 4 以降 | `- (1) 詳細条件` |

level 2 の本文内では、列挙行を `_classify_grandchild_levels()` で再分類します。`1.` の下に `(1)` `(2)` が続くような type 横断の列挙も、孫とひ孫に分けて表現します。

#### 5. チャンク出力

`flatten_chunks()` は、ツリーを実際の保存単位へ平坦化します。

| 出力関数 | 対象 | 主な動き |
| --- | --- | --- |
| `_emit_parent` | level 1 | 親チャンクを出す。長すぎる場合は `### ` 境界で分割 |
| `_emit_child` | level 2 | 子チャンクを出す。長すぎる場合は `#### ` 境界で分割 |
| `_emit_grandchild` | level 3 相当 | 子内の `#### ` ブロックから孫チャンクを合成して出す |

子チャンクは、本文が 2 行以上なら採用します。本文が 1 行だけの場合は、同じ親配下に複数行本文を持つ同 type の兄弟があれば採用します。本文 0 行の見出しだけのノードは検索ノイズになるため除外します。

重複抑制は次のルールです。

| ケース | 動き |
| --- | --- |
| 親配下の有効な子が 1 件以下 | 子を出さず、親が `child_chunk_id` も兼ねる |
| 子配下の孫ブロックが 1 件以下 | 孫を出さず、子が `grandchild_chunk_id` も兼ねる |
| 子を省略しても孫ブロックが複数ある | 孫は独立チャンクとして出す |

#### 6. metadata と検索用テキスト

`default_metadata_builder()` は、最低限の階層情報を返します。`flatten_chunks()` がさらに階層 ID を追記します。

主要 metadata:

| キー | 内容 |
| --- | --- |
| `chunk_id` | section-based の論理 ID。例: `chunk_sec_xxxxxxxx` |
| `root_id` | 文書ルート ID |
| `level` | 推定階層 |
| `path_text` | ルートから対象ノードまでの見出し列 |
| `chunk_role` | `parent` / `child` / `grandchild` / `fallback` |
| `parent_chunk_id` | 所属する親チャンクの section-based ID |
| `child_chunk_id` | 所属する子チャンクの section-based ID |
| `grandchild_chunk_id` | 所属する孫チャンクの section-based ID |
| `chunking_strategy` | `structure_aware_v4` |

ここでの `metadata.chunk_id` は chunking 出力時点の section-based ID です。Chroma 保存時には backend 側の予約キー `chunk_id` が `{doc_id}:{i}` 系の論理チャンク ID として上書きされ、検索後の `RetrievedChunk.metadata` からは予約キーが除外されます。そのため、再ランキングで階層をたどるキーは `metadata.chunk_id` ではなく、`parent_chunk_id` / `child_chunk_id` / `grandchild_chunk_id` です。

また、logic_06 は `vector_texts` を返します。

| variant | 内容 |
| --- | --- |
| `full_context_plain` | Markdown 記号を外した全文脈。親見出しを含む |
| `local_context_plain` | `child` / `grandchild` のみ生成。文書タイトルと親章を外した局所文脈 |

`vector_db.py` は `vector_texts` がある場合、1 論理チャンクを複数の物理ベクトルレコードへ展開します。これにより、上位文脈付きの検索と局所文脈寄りの検索を同じ Chroma TopK に混ぜられます。

---

## 再ランキング過程

### 方針

`reranking_logic_03.py` の目的は、ベクトル検索で拾った候補を「LLM に渡す文脈」として適切な粒度へ寄せることです。

logic_06 では、同じ本文が次の理由で複数候補として出やすくなります。

- `full_context_plain` と `local_context_plain` の両方が同じ論理チャンクを指す
- 親・子・孫の粒度違いが同時にヒットする
- 長い親や子が `_p0` `_p1` のように分割される

`reranking_logic_02` との設計上の違いは、**複数の子が同じ親配下にヒットしても、子を親に集約しない**点です。長文の親チャンクを LLM に渡すと推論精度が下がる場合があるため、子粒度を維持したまま渡します。親チャンクはスコア閾値を超えれば子と並列に残ります。

### 実装

公開 API は `rerank(settings, query, chunks, top_k)` です。`settings` と `query` はインターフェース互換のため受け取りますが、現実装ではスコア計算に使いません。

処理の流れ:

```text
rerank
  -> 件数が 5 未満ならそのまま返す
  -> new_top_k = max(5, top_k // 6)
  -> final_score を作業スコアとして保持
  -> 同一 chunk_id を最高スコアへ集約
  -> parent / child / grandchild の重複を段階的に削除（孫→子への集約のみ）
  -> 最高スコアの 60% 未満を削除
  -> スコア順で new_top_k 件を返す
```

`top_k=30` の research_pair では `new_top_k = max(5, 30 // 6) = 5` になります。

#### 1. 少数候補はそのまま返す

候補数が `_MIN_TOP_K = 5` 未満なら、再ランキングをスキップします。少数候補に対して平均カットや階層集約をかけると、LLM に渡る文脈が薄くなりすぎるためです。

#### 2. 同一論理チャンクの重複排除

`_dedupe_by_chunk_id()` は、同じ `RetrievedChunk.chunk_id` の候補を最高スコア 1 件にまとめます。

ここでの `RetrievedChunk.chunk_id` は backend / Chroma 経由の論理チャンク ID です。`vector_texts` により物理レコードが複数あっても、Chroma 保存時に `logical_chunk_id` が予約キー `chunk_id` として復元されるため、同一論理チャンクを束ねられます。

#### 3. 親があるなら低スコアの子孫を落とす

`_drop_lower_descendants(group_key="parent_chunk_id", keep_role="parent")` で、同じ親配下に `parent` が存在する場合、その親よりスコアが低い `child` / `grandchild` を削除します。

親の方が関連度として十分高いなら、下位粒度を重複して渡さないという判断です。

#### 4. 子があるなら低スコアの孫を落とす

同じ `child_chunk_id` 内に `child` が存在する場合、その子よりスコアが低い `grandchild` を削除します。

#### 4b. 孫スコアを子へ引き継ぐ

残った `grandchild` は、同じ `child_chunk_id` の代表 `child` へ最高スコアを引き継ぎ、`grandchild` 自体は削除します。

代表 `child` が複数ある場合は、元スコアが最も高い `child` だけが受け取ります。スコア引き継ぎは `max` で、下位スコアが上位スコアを下げることはありません。

#### 4c. 孤立孫を親へ引き継ぐ

logic_06 では、子が 1 件だけの場合に `merge_child=True` となり、子チャンクを emit しないことがあります。このとき孫は独立して出る場合がありますが、孫の `child_chunk_id` は「emit されていない子」を指します。

この残った孫を「孤立孫」として扱い、`parent_chunk_id` をキーに親へスコアを引き継いで削除します。これは merge_child ケースで親が子を兼ねているため、孫→子（=親）の集約として扱います。

注意点として、`RetrievedChunk.chunk_id` は `{doc_id}:{i}` 系の backend 側 ID、`parent_chunk_id` / `child_chunk_id` は `chunk_sec_xxxxxxxx` 系の section-based ID です。名前空間が違うため、孤立孫の処理では `chunk_id` 集合による照合ではなく、親チャンク自身の `meta("parent_chunk_id")` が自己参照になる性質を使って逆引きしています。

#### 5. スコア閾値で足切りし、上位件数へ圧縮

最高スコアの `_SCORE_RETAIN_RATIO`（= 0.6）倍未満の候補を削除します。その後、スコア降順に並べ、`new_top_k` 件までに絞ります。残数が `new_top_k` 未満なら、返却する `new_top_k` も残数に合わせます。

`reranking_logic_02` にあった「複数子を親へ集約する」ステップはありません。複数の子が同一親配下でヒットしていても、子は個別に残ります。

### 互換性

`chunk_role`、`parent_chunk_id`、`child_chunk_id` など必要 metadata がない候補は、階層グループ化の対象外として通します。非構造化チャンクや他の chunking logic 由来の候補でも、同一 `chunk_id` の集約、スコア閾値カット、TopK 圧縮だけが適用される設計です。

---

## まとめ

`chunking_logic_06.py` は「構造を推定して、親・子・孫の検索粒度を作る」ロジックです。`reranking_logic_03.py` は「検索結果に混ざった複数粒度の候補を、孫→子の粒度に寄せつつ親への集約は行わずに整理する」ロジックです。

`reranking_logic_02` と比べ、親チャンクへの集約を省くことで LLM に渡るチャンクの粒度を細かく保ちます。長文の親チャンクが推論精度を下げるケースに適しています。
