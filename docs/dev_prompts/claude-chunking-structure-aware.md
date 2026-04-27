あなたはPython/RAG/文書構造解析に詳しいエンジニアです。

現在、正規化済みtxt文書をRAG用にチャンキングしています。
既存実装は、以下のような単純なRecursiveCharacterTextSplitterです。

def split_for_rag(*, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

これを、structure aware chunking に置き換えたロジックを作成してください。

今回は chunk_size / chunk_overlap による固定長分割は一旦使わず、文書構造を優先して分割してください。

# ゴール

正規化済みtxtを入力として、文書全体から見出し候補を抽出し、見出しパターンの出現分布・連続性・包含関係・インデント・空行などを使って階層を推定し、親子構造付きのチャンクを作ってください。
文書は、@docs/dev_prompts/normalized_txt.txtを参考にしてください。

今回の文書では、

- 第n章
- 第n条
- 1. / 2. / 3.
- 箇条書き

の構造が多く、結果的に、

- 章 = 親チャンク
- 条 = 主チャンク
- 条文内の番号・箇条書き = 子チャンク

になるのが自然です。

ただし、別文書でも同じロジックを使う可能性があります。
そのため、「第n章なら必ずlevel 1」「第n条なら必ずlevel 2」と完全固定しないでください。

基本方針は以下です。

1. 強い見出しルールで候補を拾う
2. 文書全体の出現分布で階層を推定する
3. 推定に失敗した場合だけ、chapter/article/numeric/bullet の既定優先度にフォールバックする

完全な自然言語理解を目指す必要はありません。
KISS原則に従い、実装は1つの.pyファイルで完結させてください。

# 作成・修正するファイル

以下を作成・修正してください。

1. backend/rag/logic/chunking/chunking_logic_02.py
   - 文書全体解析型の structure aware chunker
   - 1ファイルで完結
   - 標準ライブラリ中心
   - CLI対応
   - --debug対応

2. docs/structure_aware_vector_search_policy.md
   - 今回のmetadataを使ったベクトル検索方針をまとめる

3. 既存の split_for_rag 呼び出し
   - 現在の実装通り、research_pairで置き換えられるようにしてください。

# 重要な設計思想

見出し候補検出と階層推定を分離してください。

悪い例:
- 第n章ならlevel 1
- 第n条ならlevel 2
- 1.ならlevel 3

良い例:
- 第n章は強い見出し候補として抽出する
- 第n条も強い見出し候補として抽出する
- その後、文書全体の分布を見て、第n章が第n条を内包しているなら、第n章をlevel 1、第n条をlevel 2に推定する
- 推定が曖昧なときだけ、既定優先度として japanese_chapter > japanese_article > numeric > paren_numbered > bullet を使う

# 入力文書の想定

対象は正規化済みtxtです。

ただし、OCRや正規化のズレにより、以下のような乱れがあります。

- 本文途中に余計な空行が入る
- 本来改行されるべきでない文章が改行される
- 逆に表や列挙が1行に潰れる
- 「1.」のような番号が、項番号なのか、箇条書きなのか、前後文脈で見ないと分からない
- 文書によっては「第n章」「第n条」という構造とは限らない
- 文中に「労働基準法第89条」「第12条各号」のような参照表現が出る

そのため、単純なsplitではなく、以下を実装してください。

- 見出し候補の検出
- 見出し候補のスコアリング
- 文書全体での見出しパターン解析
- 階層推定
- SectionNodeツリー構築
- 親子関係metadata付きチャンク生成

# HEADING_PATTERNS

以下のような見出しパターンを利用してください。

HEADING_PATTERNS = (
    r"(?:"
    r"第[0-9一二三四五六七八九十百千〇]+(?:条|章|項|説|目)"
    r"|[0-9]+[.)．]"
    r"|[（(][0-9一二三四五六七八九十百千〇]+[)）]"
    r"|[•・\\-]"
    r")"
)

ただし、これをそのまま機械的に使うのではなく、以下も考慮してください。

- 行頭一致を強く評価する
- 文書先頭または空行直後にある場合を強く評価する
- 行の短さ、句点の有無、括弧付きタイトルの有無を評価する
- 本文中の参照表現は見出し扱いしない
- OCRズレで見出し前の改行が欠けている場合、強い見出しだけ改行補正する

# 実装する主なdataclass

## ChunkingConfig

@dataclass
class ChunkingConfig:
    max_depth: int = 4
    min_heading_score: float = 3.0
    min_group_count: int = 2
    max_heading_line_length: int = 80
    enable_inline_heading_repair: bool = True
    enable_level_inference: bool = True
    fallback_to_paragraph: bool = True
    include_parent_heading_in_child_text: bool = True
    create_parent_chunks: bool = True
    create_main_chunks: bool = True
    create_child_chunks: bool = True
    debug: bool = False

## HeadingRule

@dataclass
class HeadingRule:
    name: str
    regex: str
    default_level_hint: int
    priority: int

default_level_hint は固定levelではなく、推定失敗時のフォールバック優先度として使ってください。

デフォルトルールとして以下を用意してください。

- japanese_chapter: 第n章
- japanese_article: 第n条
- japanese_section: 第n項
- decimal_number: 1.1 / 1.2.3
- numeric_dot: 1. / 1．
- numeric_paren: (1) / （1）
- japanese_paren: （一） / (一)
- roman: I. / Ⅰ
- alpha: A.
- bullet: ・ / • / -
- circle_bullet: ○ xxx
- appendix: 附則
- unknown_heading: 見出しっぽい短い行

## HeadingCandidate

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

## SectionNode

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

# 処理フロー

以下の流れで実装してください。

1. normalize_text(text)
2. extract_heading_candidates(text, config)
3. analyze_heading_groups(candidates, text, config)
4. infer_heading_levels(candidates, group_stats, config)
5. build_section_tree(text, candidates, config)
6. flatten_chunks(root, config)
7. split_for_rag_structure_aware(text) で list[dict] を返す

# 1. normalize_text

normalize_text(text: str, config: ChunkingConfig | None = None) -> str

以下を行ってください。

- 改行コード統一
- 全角スペースを半角スペースへ
- 行末空白削除
- 3つ以上の連続改行は2つに圧縮
- 見出し直前に改行がない場合でも、強い見出しパターンを検出して改行を補う

例:
"...する。第10条（服務）"
↓
"...する。\\n\\n第10条（服務）"

ただし、以下のようなものは改行しないでください。

- 労働基準法第89条
- 第12条各号
- 前条
- 本条
- 第n条に基づき
- 第n条各号
- 法第n条

つまり、文中参照っぽいものは見出し化しないでください。

また、行結合として以下を行ってください。

- 句点で終わっていない短い本文行は次行と結合する候補にする
- ただし見出し行、番号行、箇条書き行は結合しない
- 「ただし、」のような文途中改行は自然に結合する

# 2. extract_heading_candidates

extract_heading_candidates(text: str, config: ChunkingConfig) -> list[HeadingCandidate]

行単位で見出し候補を抽出してください。

候補判定では以下を見てください。

加点:
- 行頭で見出しパターン一致: +3
- 空行直後: +1
- 行が短い: +1
- 「第n章」「第n条」形式: +4
- 括弧付きタイトルがある: +2
- 行末が句点でない: +1
- インデントが浅い: +1
- 番号が連続している: +2

減点:
- 行の途中に出てくる: -3
- 参照表現っぽい: -4
- 行が長すぎる: -2
- 句点を複数含む本文っぽい行: -2
- URLやメールアドレスっぽい: -3

score >= config.min_heading_score の候補だけ採用してください。

# 3. analyze_heading_groups

analyze_heading_groups(candidates, text, config) -> dict[str, Any]

marker_typeごとに以下を集計してください。

- count
- average_line_length
- average_indent
- positions
- average_gap
- median_gap
- sequence_score
- containment_score
- reset_score
- shallow_indent_score
- fallback_priority
- confidence

sequence_score:
- 1,2,3... のように連続しているほど高い
- 同じ親範囲内で番号が自然にリセットされる場合も評価する

containment_score:
- あるmarker_type Aの範囲内に、別marker_type Bが複数含まれる場合、Aは外側階層らしい
- Aの候補間隔がBより広く、Bを複数内包するほどAを上位にしやすい

reset_score:
- Bの番号がAの範囲ごとにリセットされるなら、A > B の親子関係を強める

shallow_indent_score:
- インデントが浅いほど外側階層らしい

# 4. infer_heading_levels

infer_heading_levels(
    candidates: list[HeadingCandidate],
    group_stats: dict[str, Any],
    config: ChunkingConfig
) -> list[HeadingCandidate]

候補ごとに inferred_level を付与してください。

基本方針:

- 文書全体で外側階層らしいmarker_typeを level 1 にする
- その内側で繰り返されるmarker_typeを level 2 にする
- さらに内側を level 3 以降にする
- max_depthを超えない
- ただし推定が弱い場合は default_level_hint / priority にフォールバックする

外側階層らしさは以下で評価してください。

- 文書全体で少数だが複数回出現する
- 出現間隔が広い
- 別marker_typeを内包している
- 行が短い
- 空行に囲まれている
- 番号が連続している
- インデントが浅い
- fallback_priorityが高い

内側階層らしさは以下です。

- 頻繁に出現する
- 出現間隔が短い
- 特定の上位候補範囲内で番号がリセットされる
- インデントが深い
- 箇条書き記号である
- 本文に近い

重要:
- japanese_chapter > japanese_article は強いシグナルとして優先してよい
- ただし最終的には文書全体の分布・包含関係を見て判断する
- marker_type名だけで完全固定しない

# 5. build_section_tree

build_section_tree(
    text: str,
    candidates: list[HeadingCandidate],
    config: ChunkingConfig | None = None
) -> SectionNode

SectionNodeツリーを構築してください。

実装方針:

- root node を作る
- inferred_level順に候補を処理する
- 現在候補の親は、直前に出現した自分より低いlevelの候補にする
- 同levelまたは上位levelが来たら、スタックを戻す
- 各nodeの start_char / end_char を正しく設定する
- 各nodeのtextは、自分のheadingから次の同階層以上のheading直前まで
- rootのheadingは、文書先頭の最初の非空行を使う
- 見出しが存在しない場合はroot直下にfallback paragraph nodeを作る

注意:
- ここは再帰でもスタックでもよいですが、結果として親子ツリーになるようにしてください。
- 特定の「章→条→項」固定処理にはしないでください。

# 6. flatten_chunks

flatten_chunks(root: SectionNode, config: ChunkingConfig) -> list[dict]

ツリーから list[dict] を生成してください。

各チャンク形式:

{
  "id": "chunk_xxxxxxxx",
  "text": "...",
  "metadata": {
    "chunk_id": "chunk_xxxxxxxx",
    "parent_id": "chunk_yyyyyyyy",
    "root_id": "doc_xxxxxxxx",
    "children_ids": ["chunk_aaaaaaa", "chunk_bbbbbbb"],
    "level": 2,
    "chunk_role": "main",
    "heading": "第5条（採用時の提出書類）",
    "heading_type": "japanese_article",
    "path": ["株式会社エックス就業規則", "第2章採用、異動等", "第5条（採用時の提出書類）"],
    "path_text": "株式会社エックス就業規則 > 第2章採用、異動等 > 第5条（採用時の提出書類）",
    "ordinal": 5,
    "start_char": 1234,
    "end_char": 1567,
    "source_type": "txt",
    "chunking_strategy": "structure_aware_v1",
    "structure_confidence": 0.82,
    "inference_reason": {
      "sequence_score": 0.9,
      "containment_score": 0.8,
      "indent_score": 0.7,
      "fallback_used": false
    }
  }
}

必須metadata:

- chunk_id
- parent_id
- root_id
- children_ids
- level
- chunk_role
- heading
- heading_type
- path
- path_text
- ordinal
- start_char
- end_char
- source_type
- chunking_strategy
- structure_confidence
- inference_reason

chunk_roleは以下のルールにしてください。

- level 1: parent
- level 2: main
- level 3以降: child
- fallback: fallback

ただし、levelの推定が文書ごとに変わるため、検索時には level だけでなく chunk_role と heading_type も併用できるmetadataにしてください。

# 7. チャンク粒度

基本方針:

- level 1は親チャンク
- level 2は主チャンク
- level 3以降は子チャンク
- 短いlevel 3は無理に独立させず、親level 2の本文に含めてもよい
- ただし、children_ids / parent_id で親子関係は保持する

細かすぎるチャンクを避けるため、以下の調整を入れてください。

- min_child_text_length をconfigに追加してもよい
- 短すぎるbulletは単独chunkにせず、親本文に含める
- ただし、metadata上はSectionNodeとして保持できるなら保持する

子チャンクのtextには、親見出しを軽く含めてください。

例:
第9条（休職） > 1.
労働者が次の各号のいずれかに該当するときは...

# 8. フォールバック

構造推定が弱い場合のフォールバックを入れてください。

- 明確なchapter/article構造があればそれを使う
- なければ decimal_number / numeric_dot / numeric_paren / bullet の順で仮置き
- 見出し候補が少なすぎる場合は、段落単位で分割
- 見出しがまったくない場合は、空行段落単位で分割
- それでも巨大な段落がある場合のみ、文単位で安全に分割

ただし、固定長chunk_size / overlap分割には戻さないでください。
構造優先の最後の逃げ道として段落分割するだけにしてください。

# 9. API

以下の関数を提供してください。

- split_for_rag_structure_aware(text: str) -> list[dict]
- split_for_rag_texts_only(text: str) -> list[str]
- normalize_text(text: str, config: ChunkingConfig | None = None) -> str
- extract_heading_candidates(text: str, config: ChunkingConfig) -> list[HeadingCandidate]
- analyze_heading_groups(candidates: list[HeadingCandidate], text: str, config: ChunkingConfig) -> dict[str, Any]
- infer_heading_levels(candidates: list[HeadingCandidate], group_stats: dict[str, Any], config: ChunkingConfig) -> list[HeadingCandidate]
- build_section_tree(text: str, candidates: list[HeadingCandidate], config: ChunkingConfig | None = None) -> SectionNode
- flatten_chunks(root: SectionNode, config: ChunkingConfig) -> list[dict]

既存の split_for_rag 互換が必要なら、以下のようにしてもよいです。

def split_for_rag(*, text: str, chunk_size: int = 0, chunk_overlap: int = 0) -> list[str]:
    return split_for_rag_texts_only(text)

# 10. CLI対応

structure_aware_chunker.py はCLIで実行できるようにしてください。

例:

python structure_aware_chunker.py input.txt --out chunks.json

オプション:

- --out chunks.json
- --debug
- --print-tree
- --encoding utf-8

標準出力には以下を出してください。

- total chunks
- heading_typeごとの件数
- levelごとの件数
- chunk_roleごとの件数
- 最初の数チャンクの path_text

--debug時は以下も出してください。

- heading candidates一覧
- marker_typeごとのgroup_stats
- 推定level一覧
- fallbackが使われたか
- 簡易ツリー表示

# 11. structure_aware_vector_search_policy.md の内容

以下の内容をMarkdownでまとめてください。

タイトル:
Structure Aware Chunking Vector Search Policy

含める内容:

1. このchunking方式の概要
2. metadata項目の説明
   - parent_id
   - children_ids
   - root_id
   - level
   - chunk_role
   - heading
   - heading_type
   - path
   - path_text
   - structure_confidence
   - inference_reason
3. ベクトル検索時の基本方針
   - mainチャンクを主検索対象にする
   - childチャンクでhitした場合はparent_idを使ってmain/parentを展開する
   - parentチャンクは広い文脈補完に使う
4. chunk_roleの使い方
   - parent
   - main
   - child
   - fallback
5. levelだけに依存しない理由
   - 文書ごとに推定levelが変わるため
   - chunk_roleとheading_typeを併用する
6. metadata filter例
   - chunk_role == "main"
   - heading_type == "japanese_article"
   - structure_confidence >= 0.6
7. rerank時の方針
   - 同一parent_idの重複をまとめる
   - 同一path配下のchildが複数hitしたら親を優先する
   - mainとchildの両方がhitしたらmainを代表にする
8. 回答生成時の方針
   - path_textをプロンプトに含める
   - 回答に章・条・項などの根拠を出せるようにする
9. structure_confidenceが低い場合の扱い
   - fallbackチャンクとして扱う
   - 必要に応じて広めの親文脈を添える
10. 将来的なhybrid search
   - BM25 + vector
   - heading/path_textへのキーワード重み付け
   - metadata filterとrerankの組み合わせ

# 12. 品質要件

- 標準ライブラリ中心
- 1つの.pyファイルで完結
- 型ヒントを付ける
- dataclassを使う
- JSON serializableなmetadataにする
- 例外で落ちにくくする
- 空文字でも落ちない
- 見出しがない文書でも最低1チャンク返す
- 章だけ、条だけ、番号だけの文書でも動く
- 特定文書専用のif文を増やさない
- ただし日本語規程文書でよくある「第n章」「第n条」は強いシグナルとして扱う
- デバッグしやすい構造にする
- KISS原則に従い、過剰な抽象化は避ける

# 13. 期待される今回文書での動作

今回の就業規則txtでは、文書全体の分布解析の結果として、概ね以下になることを期待します。

- document title: 株式会社エックス就業規則
- level 1: 第n章
- level 2: 第n条
- level 3: 1. / 2. / 3. など
- appendix: 附則 は level 1 相当、または appendix 専用type
- chunk_role:
  - 第n章: parent
  - 第n条: main
  - 条文内の番号・箇条書き: child

ただし、これは固定指定ではなく、文書全体の解析結果としてそう判断されるようにしてください。