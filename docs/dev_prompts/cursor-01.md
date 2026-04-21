このリポジトリを、**QA50〜100件のRAG実験を1ワークフローでバッチ実行するための実験専用プロジェクト**として整理・実装してください。  
まず **既存構成を十分に読んで設計方針を決めてから** 実装を開始してください。  
実装は **KISS / YAGNI / 最小変更** を強く意識し、不要な機能追加はしないでください。

## 前提
このリポジトリ内で実装するのは **Docker にあたる部分だけ** です。  
運用全体は以下です。

- GitHub Actions
  - 実験開始のトリガー
  - ジョブ実行の制御
  - ログと成果物の回収
- self-hosted runner（Windows デスクトップ上）
  - GitHub Actions の job を受け取って実行する主体
- Docker
  - QA バッチ処理の固定実行環境
- Windowsデスクトップ ローカルホスト上の LLM サーバー
  - 推論リクエストの実行先
- Tailscale + ノートPCからデスクPCへリモートデスクトップ接続
  - 設定変更、障害調査、GUI確認時のみ使用

したがって、このリポジトリでは **GitHub Actions や Tailscale の実装そのものは行わず**、  
**Actions から self-hosted runner 上で `docker compose run ...` などで起動できる固定実験環境** を作ってください。

---

# ゴール

現在の実装を踏まえて、  
**research_pair を 1 つ指定して実行すると、その条件で PDF 取り込み → RAG 推論 → RAGAS/Langfuse 観測 → CSV 保存までを一気通貫で実行できる構成**  
に変更してください。

基本運用は以下です。

- 実行ごとに変更するのは **research_pair の指定のみ**
- 指定を変えて push する、または同等のコマンドを叩くと実験が走る想定
- CSV は **self-hosted runner の Windows デスクトップのローカルに保存** されればよい
- 1回の実験で使う PDF 群・QA データセット・各ロジック選択は research_pair で決まる

---

# 必須要件

## 1. 実行方式
- 実験は **GitHub Actions から self-hosted runner 上で Docker コンテナを起動して実行できる** 構成にすること
- この repo では **Actions workflow 自体の本実装は不要**
- ただし、Actions から呼びやすいように、**CLI エントリポイント** または **docker compose run で叩ける単一コマンド** を用意すること
- 例:
  - `docker compose run --rm experiment_runner python -m app.experiment.runner --research-pair RP-0001`
  - のような運用にしやすい設計にすること

## 2. 実験ごとの vectorDB 初期化
- **データセット1セットの実行のたびに、vectorDB のコレクションは空にしてから再度 PDF を取り込む**
- その後に QA 全件の推論を実行すること
- 途中で QA ごとに再 ingest はしないこと
- 単位は **「1 research_pair × 1 QA dataset」実行ごとに初期化 → ingest → 全問推論」** とすること

## 3. ingest_document ディレクトリ
- `backend` の外側に `ingest_document/` ディレクトリを新設すること
- その中に、**id付きサブディレクトリ** を複数作れる構成にすること
- 実行時に **どの document セットを使うか research_pair で指定** できるようにすること

想定例:
- `ingest_document/DOCSET-0001/*.pdf`
- `ingest_document/DOCSET-0002/*.pdf`

## 4. .env と research_pair の責務分離
### .env 側
- API接続設定
- ホスト名、ポート
- Langfuse 接続設定
- Docker 内外の URL 実態
- 既定のベース設定

### research_pair 側
その実験ワークフローで使う条件だけを、簡潔に書けるようにすること。

research_pair で持つ項目は現時点で次の 8 つ:
1. 推論LLM
2. 埋め込みLLM
3. 検索ロジック
4. チャンキングロジック
5. 再ランキングロジック
6. トークナイザーロジック
7. トップK
8. 使用QAデータセット

加えて、実運用上必要なので以下も research_pair に含めてよい:
- 使用する ingest document セットID
- RAGAS ON/OFF

## 5. 実行当たりで変更するのは research_pair 指定のみ
- 基本的に実行ごとに変えるのは **research_pair の指定のみ**
- `.env` は環境の実態定義として固定寄り
- 実験条件は research_pair に寄せる

## 6. 最終成果 CSV
各実験結果は CSV で出力すること。  
最低限、以下を含めること。

- 選択 research_pair
- 入力
- 出力
- RAG検索結果ドキュメント × topK 個
- distance × topK 個
- RAGAS 観測結果

必要に応じて、吟味のうえ以下のような列を追加してよい:
- question_index
- document_set_id
- dataset_name
- llm_model
- embedding_model
- search_logic_id
- chunking_logic_id
- reranking_logic_id
- tokenizer_logic_id
- rag_search_mode
- top_k
- rag_latency_ms
- total_latency_ms
- prompt
- retrieved_chunk_ids
- retrieved_sources
- ragas_faithfulness
- ragas_answer_relevancy
- langfuse_trace_id 相当の参照情報（安全に取得できるなら）

ただし **過剰実装はしないこと**。  
必要十分で、後から集計しやすい列構成にすること。

## 7. CSV 保存先
- CSV は **実行している Windows デスクトップのローカルに保存** されればよい
- Docker から見えるよう、`outputs/` のようなディレクトリを backend 外に切って、volume mount する設計でよい

## 8. RAGAS ON/OFF 切替
- RAGAS は ON/OFF できること
- OFF の場合は CSV の列は維持し、値は `None` あるいは空文字で埋めること
- 列数は必ず揃えること

## 9. research_pair の選択項目
research_pair の項目は現時点で以下 8 つを必須とする:
- 推論LLM
- 埋め込みLLM
- 検索ロジック
- チャンキングロジック
- 再ランキングロジック
- トークナイザーロジック
- トップK
- 使用QAデータセット

補足:
- `topK` は research_pair のテキスト変更で直接指定
- 使用QAデータセットは backend の外に事前配置したファイル名指定
- 推論LLM・埋め込みLLMは現時点の実装の切替ができれば十分

## 10. ロジック差し替えディレクトリ
以下の4つについては、各ロジックごとにディレクトリを作成し、
その中に番号付きファイルを並列配置できるようにしてください。

- 検索ロジック
- チャンキングロジック
- 再ランキングロジック
- トークナイザーロジック

例:
- `backend/app/rag/logic/search/search_LOGIC-01.py`
- `backend/app/rag/logic/search/search_LOGIC-02.py`
- `backend/app/rag/logic/chunking/chunking_LOGIC-01.py`
- `backend/app/rag/logic/reranking/reranking_LOGIC-01.py`
- `backend/app/rag/logic/tokenizer/tokenizer_LOGIC-01.py`

research_pair では **どの logic ID を使うか** を決めるだけにしてください。  
ロジック本体はファイルに閉じ込め、選択ローダーで解決する構成にしてください。

## 11. QA データセット
- QA データセットは `json` 形式などで記述し、backend の外に特定ディレクトリを設けて事前配置すること
- research_pair でファイル名または dataset ID を指定して読み込めること
- 想定規模は 50〜100 件程度
- 不要に DB 化しないこと

---

# 設計方針（重要）

## A. APIアップロード型より CLI / ファイル参照型を優先
現状の `POST /api/v1/experiment/batch` は multipart で質問とファイルを投げる形だが、
今回の運用は **GitHub Actions → self-hosted runner → Docker の固定バッチ実行** なので、
**アップロード API 中心ではなく、research_pair + 外部配置ファイルを読む CLI 実行中心** に寄せてください。

- experiment API は不要なら削減・整理してよい
- ただし既存コードが流用できるなら、**内部ロジックは活かしつつ、入口を CLI に置き換える** 方向を優先してください

## B. 実験条件の表現はシンプルに
research_pair は **人間が読みやすく、差分管理しやすい** 形式にしてください。  
JSON か YAML で、過度な抽象化は不要です。  
個人的には YAML or JSON の単純な1ファイル1条件でよいです。

例:
- `research_pairs/RP-0001.yaml`
- `research_pairs/RP-0002.yaml`

## C. 現時点では必要十分の差し替えだけ
- 推論LLMと埋め込みLLMは、今ある切替の延長で十分
- 検索ロジック/チャンキング/再ランキング/トークナイザーは、**ファイル選択できる土台** を作ればよい
- 高度なプラグイン機構や自動発見を過剰に作らないこと
- まずは **logic ID → 対応ファイルを import して使う** で十分

## D. Docker 側に閉じた実装
- Windows ローカルの LLM サーバーは `.env` の `host.docker.internal` 経由など、今の設計を踏襲してよい
- GitHub Actions から実行しやすいように、**compose service を追加する** 方向でよい
- 例:
  - `experiment_runner` サービス追加
  - `volumes` で `research_pairs/`, `datasets/`, `ingest_document/`, `outputs/` をマウント
- フロントエンドは今回の目的では不要なら触らなくてよい

---

# 実装してほしい内容

## 1. ディレクトリ整理
backend 外に最低限以下を作成してください。

- `ingest_document/`
- `qa_datasets/`
- `research_pairs/`
- `outputs/`

必要なら README やサンプルファイルも置いてください。

## 2. research_pair スキーマ定義
research_pair のスキーマを定義してください。  
型付きで読み込めるようにし、値検証も最低限入れてください。

含める項目の例:
- `research_pair_id`
- `llm_model`
- `embedding_provider`
- `embedding_model`
- `search_logic_id`
- `chunking_logic_id`
- `reranking_logic_id`
- `tokenizer_logic_id`
- `top_k`
- `qa_dataset`
- `document_set_id`
- `ragas_enabled`
- 必要なら `rag_search_mode`

## 3. QA dataset loader
`qa_datasets/` から JSON を読み、質問配列を取得するローダーを作成してください。  
必要なら正解 answer や metadata を含められる構成にしてよいですが、
現時点では **最低限 user input を回せる構造** で十分です。

## 4. document set loader
`ingest_document/<document_set_id>/` 配下の PDF 群を集めるローダーを作成してください。

## 5. logic selector
以下4種の selector / registry を作成してください。
- search logic
- chunking logic
- reranking logic
- tokenizer logic

research_pair の logic ID から対象ファイルを解決し、実装を呼べるようにしてください。

## 6. reranking 層の追加
現状に再ランキングが薄い/無い場合は、**最小構成で差し込みポイント** を追加してください。  
最低限、
- `RERANKING_LOGIC-01 = no-op`
のような実装でよいです。  
今は土台だけで十分です。

## 7. 実験実行 CLI
最重要です。  
以下のような単一 CLI を作成してください。

例:
- `python -m app.experiment.runner --research-pair RP-0001`
- または同等の明快な形

処理の流れ:
1. research_pair を読む
2. QA dataset を読む
3. document set を読む
4. runtime settings を構築
5. vectorDB collection を reset
6. document set の PDF を ingest
7. QA 全件を順に推論
8. 必要なら RAGAS 実行
9. CSV を `outputs/` に保存
10. 終了コードを返す

## 8. CSV writer
CSV は topK を扱いやすい形にしてください。  
例えば以下のように **固定列で展開** してください。

- `retrieved_source_1 ... retrieved_source_k`
- `retrieved_chunk_id_1 ... retrieved_chunk_id_k`
- `retrieved_distance_1 ... retrieved_distance_k`
- `retrieved_text_1 ... retrieved_text_k`（長すぎるなら省略可）

`topK` は research_pair で変わるので、
**CSV の列は topK に応じて動的生成** して構いません。  
ただし1回の出力内では必ず列数を固定してください。

## 9. compose.yaml 整理
- experiment 実行専用の service を追加してください
- self-hosted runner から叩きやすい構成にしてください
- `outputs/`, `ingest_document/`, `qa_datasets/`, `research_pairs/` を volume mount してください
- backend API サーバーが必須でないなら、experiment_runner 単体で完結できる構成を優先してください
- ただし既存コード再利用のため backend サービス経由が必要なら、その理由が明確なら許容します
- 不要なサービスは削減を検討してください
- ただし現行機能を壊さない範囲で最小にしてください

## 10. .env.example 更新
今回の運用に必要な項目がわかるように `.env.example` を整理してください。
特に以下が伝わるようにしてください。
- Docker 内から Windows ローカル LLM に向ける URL
- Langfuse
- vector store
- 実験 runner が参照するベース設定

## 11. ドキュメント整備
最低限 README か docs を追加/更新し、以下を明記してください。
- ディレクトリの責務
- research_pair の書き方
- dataset の置き方
- document set の置き方
- 実行コマンド
- GitHub Actions / self-hosted runner からどう叩く想定か
- CSV の保存先

---

# 既存実装を踏まえた具体指示

現状、以下のような experiment 系の土台があるはずです。
- `backend/app/experiment/*`
- `run_experiment_batch_sync`
- `runtime_settings`
- `rag_reset_collection`
- `run_queued_upload_batch`
- `RAGAS / Langfuse` 関連

これらは **使えるものは活かしてください**。  
ただし今回の主目的は **アップロード API ではなく research_pair 中心の固定バッチ実行** です。  
そのため、必要なら以下の方向で整理してください。

- `ExperimentBatchManifest` 依存を薄くする
- 代わりに `ResearchPair` スキーマを主役にする
- `questions` や `files` を API から受けるのではなく、外部ディレクトリから読む
- logic fingerprint ベースの検証は、今回の用途では過剰なら削除または簡略化してよい
- 重要なのは **「どの logic file を選んだか」を CSV に残すこと**

---

# 実装上の注意

- **修正範囲は必要最小限**
- **import は必要なものだけ**
- **型ヒントを必ず付ける**
- コメントは必要な箇所だけ簡潔に
- 1ファイル100行目安を意識し、長くなりすぎるなら分割を検討
- 共通化はやりすぎない
- 1回しか使わない抽象化は避ける
- バッチ実行前提なので、**軽量に動くことを優先**
- 不要なAPI、不要なUI、不要な複雑性は削除してよい

---

# 期待する成果物

実装後は、以下を提示してください。

1. 変更したファイル一覧
2. ディレクトリ構成の要約
3. 実行方法
4. research_pair サンプル
5. QA dataset サンプル
6. CSV 出力列の一覧
7. GitHub Actions / self-hosted runner からの想定起動コマンド
8. 不要と判断して削除したものの説明
9. 今回は見送った拡張ポイント（必要なら簡潔に）

---

# 最後に
まずは **設計を先に要約してから** 実装してください。  
その後、実装は **小さな責務に分けて** 進めてください。  
必要なら既存 experiment API は縮小・整理して構いませんが、  
**最終的に「research_pair を指定して 1 コマンドでバッチ実験できる」ことを最優先** にしてください。