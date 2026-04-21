このリポジトリの最新実装を前提に、`rag/logic` 周りを **より薄い窓口 + logic差し替え中心** の構成へ整理してください。  
まず既存構成を読んで設計方針を簡潔に整理してから実装してください。  
**KISS / YAGNI / 最小変更** を強く意識し、不要な抽象化や過剰な共通化はしないでください。

---

# 今回の目的

`rag/logic` 周りを整理して、  
**chunking / tokenizer / search / reranking の4系統を、できるだけ同じ思想で差し替え可能にする** ことが目的です。

特に今回やりたいことは以下です。

- `search.py` を logic 側の窓口にする
- `hybrid_search.py` は削除する
- `chunking_core.py` / `tokenizer.py` / `chunking_core.py` のような中間層を減らす
- `experiment_context.py` には **最小限のラッパだけ** を残す
- `logic/.../logic_01.py` に最小構成の実装を置く
- `search` と `reranking` にも **何もしない実装** を追加し、RAG の有無まで logic 側で制御できるようにする
- 全体として **コード量・ファイル数を減らしつつ整合性を取る**

---

# 必須要件

## 1. search の整理
### やること
- `rag/logic/search.py` を新設し、**検索ロジックの窓口** にしてください
- `search.py` から、選択された検索ロジックを呼び出す構成にしてください
- `keyword_search.py` と `vector_search.py` は **基本機能として最小限の構成のまま残す** こと
- `hybrid_search.py` は削除してください
- `logic/search/search_logic_01.py` には  
  **ただ `vector_search` を呼び出すだけ** の最小コードを書いてください

### 意図
- 今後 search ロジックを増やすときに、`search.py` が窓口になるようにしたい
- `retrieval_service.py` が vector/keyword/hybrid を直接分岐する形をやめたい
- 「どの検索ロジックを使うか」は logic 側で決まるように寄せたい

## 2. tokenizer の整理
### やること
- tokenizer は、`experiment_context.py` に **最小限のラッパだけ** 残してください
- `tokenizer.py` は削除してください  
  ※現状 `tokenizer_core.py` があるなら、今回の要件ではそれも含めて整理対象として扱ってください
- `logic/tokenizer/tokenizer_logic_01.py` には  
  **何もしないでそのまま通す logic** を作ってください

### 想定
- たとえば `tokenize_query(query: str) -> list[str]` なら、
  「加工せずそのまま 1 要素の list にする」など、最小で一貫した no-op にしてください
- 既存 keyword_search 側の期待インターフェースに合わせて、破綻しない実装にしてください

## 3. search / reranking の no-op 実装追加
### やること
- `search` にも **何もしない実装** を追加してください
- `reranking` にも **何もしない実装** を追加してください
- これにより、**RAG の有無まで logic 側で制御できる** ようにしてください

### 想定
- search no-op は `[]` を返す形でよいです
- reranking no-op は入力された chunk をそのまま返す形でよいです
- research_pair の logic 指定だけで、
  - 通常の RAG
  - RAG なし
  - 検索だけ無し
  のような切り替えができる土台にしてください

## 4. chunking の整理
### やること
- chunking も `experiment_context.py` に **薄いラッパだけ** 残してください
- `chunking_core.py` は削除してください
- `logic/chunking/chunking_logic_01.py` に  
  **固定長だけで分割するロジック** を作ってください

### 条件
- 実装はシンプルにしてください
- 今の ingest フローが壊れない最小限でよいです
- 既存 settings の `chunk_size`, `chunk_overlap` 相当が使えるなら使ってください
- 使えないなら、既存フローとの整合性を保てる最小案で実装してください

## 5. experiment_context.py の役割整理
### やること
`experiment_context.py` は、今回の基準では **差し替え用の薄いラッパ / ContextVar 窓口** に徹してください。

少なくとも以下の思想に寄せてください。

- chunking: active / get の薄いラッパ
- tokenizer: active / get の薄いラッパ
- 必要なら search / reranking も同じ思想で寄せることを検討してよい

ただし、**過剰な仕組みは作らない** でください。  
すでに `experiment/logic_registry.py` があるので、それと責務が衝突しないように整理してください。

## 6. retrieval_service.py の整理
### やること
- `retrieval_service.py` は、可能な限り薄くしてください
- `vector / keyword / hybrid` を直接分岐する現在の構成はやめてください
- `search.py` を窓口として使う形に寄せてください

### 注意
- API 側や chat_service 側の既存呼び出しと整合性を取ってください
- 必要なら `rag_search_mode` の扱いは縮小して構いませんが、修正範囲は最小限にしてください
- `hybrid_search` 前提のコードや設定が残る場合は、不要なら削除・簡略化してください

## 7. search_logic_01 / no-op search の設計
### 必須
- `logic/search/search_logic_01.py`
  - `vector_search` を呼ぶだけ
- `logic/search/search_logic_02.py` など適切な番号で
  - no-op search を追加
  - 空配列を返すだけ

### 方針
- 既存の `logic_registry.py` に合うファイル命名規則を維持してください
- `research_pair` から切り替えられるようにしてください

## 8. reranking の整理
### 必須
- 既存 reranking は最小構成で整理
- `logic/reranking/reranking_logic_01.py`
  - no-op rerank を置く
- 必要なら別番号で同等の no-op を作る必要はありません
- 既存実装が no-op なら、そのまま最小化・整理だけでよいです

## 9. 不要ファイル削除
今回の要件に照らして不要になったファイルは削除してください。  
少なくとも対象候補は以下です。

- `rag/logic/hybrid_search.py`
- `rag/logic/tokenizer.py`（存在するなら）
- `rag/logic/chunking_core.py`
- `rag/logic/tokenizer_core.py`（今回の整理方針に沿って不要なら削除）

ただし、削除前に必ず参照元を解消し、全体整合性を取ってください。

## 10. 全体整合性の修正
上記の変更を基準として、周辺コードも整合するように修正してください。

対象候補:
- `experiment/logic_registry.py`
- `experiment/batch_runner.py`
- `rag/retrieval_service.py`
- `rag/vectorstore/chunker.py`
- `tests/*`
- `research_pair_schema.py`
- `runtime_settings.py`
- `README / docs` の必要箇所

---

# 実装方針

## 重要方針
- **なるべくコードの記述とファイル数が少なくなるようにすること**
- **簡潔かつ正確**
- **今必要なことだけ**
- 新しい抽象化は、本当に必要なものだけにすること

## 設計の寄せ方
最終的な思想は以下です。

- 基本機能
  - `vector_search.py`
  - `keyword_search.py`
- 差し替え窓口
  - `search.py`
  - `experiment_context.py`（chunking/tokenizer は薄いラッパ）
- 実験条件側
  - `research_pair` で logic_id を指定
- logic 実体
  - `logic/search/search_logic_01.py`
  - `logic/search/search_logic_02.py`
  - `logic/chunking/chunking_logic_01.py`
  - `logic/tokenizer/tokenizer_logic_01.py`
  - `logic/reranking/reranking_logic_01.py`

---

# テスト対応

既存テストも要件に合わせて整理してください。

## やること
- `hybrid_search` 前提のテストは削除または置換してください
- 新構成に合わせて、必要最小限のテストへ直してください

最低限確認したい観点:
- search_logic_01 が vector_search を呼ぶ
- search no-op が空配列を返す
- tokenizer_logic_01 が no-op として期待どおり動く
- chunking_logic_01 が固定長分割する
- reranking_logic_01 が入力をそのまま返す
- experiment/batch 実行時に logic 選択が壊れていない

---

# 実装後に必ずやること

1. 変更ファイル一覧を出す
2. 削除ファイル一覧を出す
3. 新しい責務分担を簡潔に説明する
4. `research_pair` でどう指定すれば
   - 通常RAG
   - RAGなし
   を切り替えられるか例を示す
5. テストと lint を実行する

利用可能なら以下を実行してください。
- `ruff check backend`
- `pytest backend/tests`

失敗した場合は、失敗内容を踏まえて修正してください。

---

# 期待する最終形のイメージ

- `search.py` が search の窓口
- `vector_search.py` / `keyword_search.py` は基本機能
- `hybrid_search.py` は存在しない
- `chunking_core.py` / `tokenizer_core.py` のような中間層は削減
- `experiment_context.py` は薄いラッパ
- logic_01 群は最小構成
- no-op search / no-op reranking がある
- RAG の有無を logic 指定で切り替えられる
- 全体として今より簡潔

まずはこの方針で設計を要約してから実装してください。