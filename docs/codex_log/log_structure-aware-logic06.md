# structure-aware logic06 作業ログ

## 2026-05-01: 検索用テキスト分岐と search_logic_03

- `chunking_logic_06.py` の出力に `vector_texts` を追加した。
  - `full_context_plain`: LLM 推論用 Markdown から見出し記号や箇条書き記号を外した全文脈検索用テキスト。
  - `local_context_plain`: child / grandchild のみ、`#` 文書タイトル行と `##` 親章行を除去し、`###` 以降は残した検索用テキスト。
- `ChunkForStore` に `vector_texts` を追加し、未指定の既存ロジックは従来どおり `document_lower` 1 本で保存する互換経路にした。
- Chroma 書き込み前に、1 論理チャンクを検索用テキスト数だけ物理レコードへ展開するようにした。
  - `chunk_id`: 検索結果では論理チャンク ID のまま返す。
  - `logical_chunk_id`: 同一チャンク由来の候補を束ねるための ID。
  - `vector_record_id`: Chroma に保存する物理レコード ID。
  - `vector_text_variant`: `full_context_plain` / `local_context_plain` など、ヒットした検索用テキスト種別。
- `search_logic_03.py` を追加した。保存時に展開された各ベクトルを通常の Chroma TopK 候補として扱うため、同一チャンク由来の複数候補も現時点ではそのまま返す。

## 2026-05-01: level2 判定の containment 補強

- `chunking_logic_06.py` の `_interval_containment()` に、候補化されなかった本文中の列挙連番を補助信号として加点する処理を追加した。
- 長い `3．` / `4．` などが見出し候補から落ちても、条ノード本文内の raw な連番構造を wrapper 判定へ反映する。
- 最後の兄弟から `scope_end` までの開区間も、内側連番が見つかった場合だけ低重みで評価するようにした。
- raw 由来の加点は控えめにし、条が章より強くなりすぎないよう `RAW_INNER_ENUM_CONTAINMENT` と `OPEN_TAIL_CONTAINMENT_WEIGHT` を低めに設定した。
- 参考の失敗例で `第7章 -> 第17条/第18条 -> 列挙` になること、既存成功例相当の `normalized_txt.txt` 第4章チャンクが崩れないことを手元確認した。
