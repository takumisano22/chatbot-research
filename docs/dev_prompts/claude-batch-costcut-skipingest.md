# バッチ処理効率化の修正依頼プロンプト

## 役割
あなたは、RAG検索を用いたLLM推論のバッチ処理のコード改善をサポートするアシスタントです。

## 目的
現在の実装では、.github/workflows/all-run-batch-store.yaml を実行すると、毎回RAG検索用ドキュメントの取り込みを行うことで時間がかかってしまいます。これを改善するため、複数のresearch_pairを実行するときに、一つ前のpairとchunking_logicとdocument_setがどちらも一致する場合、取り込みをスキップして推論のバッチ処理のみできるようにしたいです。

## 設計案
- 既存のbatch_runner.pyを参考に、batch_runner_skip_ingest.pyを作成し、ドキュメント取り込みを省いてバッチ処理するコードを作成する。
- compose.yamlに新規のskip_ingest用のrunnerを作成する。
- all-run-batch-store.yamlを実行したら、初めの一ペアは必ず取り込み付きの既存runnerを実行する。その際にresearch_pairからchuking_logic_idとdocument_set_idを保持する。
- 2回目以降のペアからは、保持している一つ前のペアのidを確認し、取り込みドキュメントを更新する必要がなければskip_ingestのrunnerを実行する。
- どちらのrunnerを実行したとしても、直前のpairを保持できるようにして毎回判定する。
  
## 注意点
- KISS・YAGNI原則に則り、必要最小の実装を行ってください。
- バッチ処理の中身そのものは、既存のロジックを使い、既存の実装に影響がでないようにしてください。
- 適宜コメントを修正してください。
- 判断に迷う箇所は対話的に実装してください。