# 構造化チャンキング用再ランキングロジック作成プロンプト

## 役割と指示
- あなたは、再ランキングのロジックに詳しく、作業者のコード実装をサポートするエージェントです。
- backend/app/rag/logic/reranking/reranking_logic_02.pyに現在の実装をコピーしたので、このファイルを編集し、下記の内容を実装してください。

### 目的
- 多数に分割したチャンクのベクトルデータから、距離検索によって取得した数10件のデータを、LLMが読みやすいように数を絞って渡せるロジックを作成することが目的です。

### 設計
- .github/workflowsから処理の流れを追って、既存のコードとの互換性があるように、設計を行ってください。
- また、今回のロジックは、chunking_logic_06.pyを前提としています。ただし、他のチャンクロジックとの組み合わせでもエラーがでないように互換性は保ってください。

#### topKの再構築
既存の渡されたtopKから再ランキング後のtopKを決定します。
- 30件くらいの元データ取得を考えているので、1/6した数を超えない整数と、"5"のうちどちらか大きい方を再ランキング後のtopKとしてください。つまり、最低でも5件は取ってくるということです。ただし、元データが5件を下回る場合は、その件数のままで渡してください。

#### metadataによる再ランキング
- 現在のsearchロジックがchunking_logic_06.pyを基準として、metadataを適切に取得できているか、確認してください。
  - 取得できていない場合は、search_logic_03.pyに適切なロジックを実装してください。ただし、既存の実装との互換性は維持し、必要に応じて他のファイルも編集して構いません。
- 下記のロジックを順に行って件数を絞るやり方を考えています。chunking_logic_06.pyも確認した上で、妥当性を検証し、適宜対話的に提案をしながら実装してください。
- 
  1. 全チャンクのスコアの平均をとって保持してください。
  2. chunk_idが同一のものは、同じchunk_textを持つので、より検索スコアが高い方を一つだけ残してください。
  3. parent_chunk_idが同一のもののうちに、roleが"parent"のものがあった場合、そのチャンクよりもスコアの低い同一の"parent_chunk_id"をもつかつ、roleが"child"または"grandchild"のものだけをすべて削除してください。
  4. child_chunk_idが同一のもののうちに、roleが"child"のものがあった場合、そのチャンクよりもスコアの低い同一の"child_chunk_id"をもつかつ、roleが"grandchild"のものだけをすべて削除してください。
  5. roleが"grandchild"のものが残っていて、かつそれぞれの"grandchild"のチャンクと同一の"child_chunk_id"を持つroleが"child"のチャンクが残っている場合、同一の"child_chunk_id"をもつ"grandchild"のうちスコアが最も高い物の検索スコアを該当する"child"に引き継いで、同一"child_chunk_id"をもつ"grandchild"チャンクを全て削除してください。もし、同一の"child_chunk_id"を持つ"child"が複数あった場合、元のスコアが最も高いものに引き継いでください（ほかのchildは削除はしなくていいです）。
  6. parent_chunck_idが同一でroleが"child"なチャンクが2つ以上残っていて、かつroleが"parent"でparent_chunk_idが一致するものが残っていた場合、同一の"parent_chunk_id"をもつ"child"のうちスコアが最も高い物の検索スコアを該当する"parent"に引き継いで、同一"parent_chunk_id"をもつ"child"チャンクを全て削除してください。もし、同一の"parent_chunk_id"を持つ"parent"が複数あった場合、元のスコアが最も高いものに引き継いでください（ほかのparentは削除はしなくていいです）。
  7. 1.で取ったスコア平均を下回るチャンクを全て削除してください。
  8. 残ったチャンク数が再構築したTopKを下回っていた場合、topKを残った数に再修正して出力してください。そうでなければ、スコア上位のtopK個を出力してください。

####参考情報
- 下記に現在のchunking_logic_06.pyによって作成されたgrandchildチャンクのmetadataを記載します。参考にしてください。
- vector_text_variantが"full_context_plain"のものと"local_context_plain"のものがあるので、同一のchunk_idがありえます。
- 最大文字数を設定しているので、同一のparent_chunk_idのparentや、同一のchild_chunk_idのchildがありえます。

***
--参考metadata--
{
  "chunk_id": "163a6bd2d4506b0f:15",
  "chunking_strategy": "structure_aware_v4",
  "vector_text_variant": "full_context_plain",
  "chunk_role": "grandchild",
  "vector_text_variant_count": 2,
  "vector_record_id": "163a6bd2d4506b0f:15::vector::full_context_plain",
  "root_id": "doc_b66569f0",
  "chunk_text": "# 株式会社エックス国内出張旅費規程\n## 第3章旅費の種類と支給基準\n### 第9条交通費は、鉄道、船舶、航空機、バス等の公共交通機関を利用した場合の実費を支給する。\n- 3. 航空機を利用する場合は、エコノミークラスの料金を原則とする。ただし、業務上の必要性があり、会社が事前に承認した場合は、ビジネスクラスの料金を支給することができる。",
  "path_text": "株式会社エックス国内出張旅費規程 > 第3章旅費の種類と支給基準 > 第9条交通費は、鉄道、船舶、航空機、バス等の公共交通機関を利用した場合の実費を支給する。 > 3. 航空機を利用する場合は、エコノミークラスの料金を原則とする。ただし、業務上の必要性があり、会社が事前に承認した場合は、ビジネスクラスの料金を支給することができる。",
  "parent_chunk_id": "chunk_sec_3192045c",
  "level": 3,
  "child_chunk_id": "chunk_sec_8a75b890",
  "source": "uploaded/国内出張旅費規程.pdf",
  "logical_chunk_id": "163a6bd2d4506b0f:15",
  "grandchild_chunk_id": "chunk_sec_b9f8b1f2",
  "doc_id": "163a6bd2d4506b0f"
}
***

## 注意点
- KISS・YAGNI原則に則り必要十分な実装を行ってください。
- 私の提案を思考して、改善・訂正した方がよい箇所があれば、対話的に提案してください。
- 判断に迷う箇所は対話的に実装を行ってください。
- コメントについても適宜修正してください。
- 実装内容は、docs/claude_logs/log_structure-aware-rerank-logic02.md に簡潔に記録してください。（同名ファイルがあれば作業単位を分けて追記、なければ新規作成）