from pathlib import Path

# -----------------------------------------------------------------------------
# 役割: Settings のデフォルト値（定数）のみを保持し、config から参照される。
# 主な呼び出し元: app.core.config の Settings フィールド既定値。
# 流れ: 定数参照 → get_settings 経由でアプリ全体に伝播（アプリロジックは持たない）。
# -----------------------------------------------------------------------------

_BYTES_PER_MIB = 1024 * 1024  ## MiB 換算（サイズ上限の計算用）

DEFAULT_APP_ENV = "development"
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

DEFAULT_LLM_PROVIDER = "ollama"
# llm_provider.<name> の import 先（論理 LLM_PROVIDER とは別。同 API の実装差し替え用）
DEFAULT_LLM_ADAPTER_SUBPACKAGE = "ollama"
DEFAULT_LLM_API_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_LLM_TEMPERATURE = 0.7
# LLM への HTTP 待ち上限（秒）。長文生成で切れないよう長めにする（環境変数で上書き可）。
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 600.0

DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# VECTOR_STORE_PROVIDER の既定。.env 未設定時および Settings の検証で許す実装 ID（Compose のボリューム名 `${値}_data` と一致させる）
DEFAULT_VECTOR_STORE_PROVIDER = "chroma"
# vectordb.<name> の import 先（接続先・VECTOR_STORE_PROVIDER とは別）
DEFAULT_VECTOR_DB_ADAPTER_SUBPACKAGE = "chroma"
DEFAULT_VECTOR_STORE_SERVER_HOST = ""
DEFAULT_VECTOR_STORE_SERVER_PORT = 8000

## 埋め込み時のベクトルストアのデータ保存先。docker運用時のHTTP接続では未使用。テストでも使う。
DEFAULT_VECTOR_STORE_PERSIST_DIR = Path("data/vector_store")

DEFAULT_RAG_COLLECTION_NAME = "rag_documents"
DEFAULT_RAG_TOP_K = 4
# ベクトル検索の既定件数（キーワード既定は RAG_TOP_K）。API で k を省略した vector モードでも利用。
DEFAULT_RAG_VECTOR_TOP_K = 4
DEFAULT_RAG_KEYWORD_WEIGHT = 1.0
# hybrid_search 窓口が委譲する実検索（vector_search | keyword_search）。将来の統合実装までの切替用。
DEFAULT_RAG_HYBRID_DELEGATE = "vector_search"
DEFAULT_RAG_CHUNK_SIZE = 1000
DEFAULT_RAG_CHUNK_OVERLAP = 200
# RAG チャット API のシステムプロンプト（app.rag.logic.prompt.prompt_<id>）
DEFAULT_RAG_PROMPT_LOGIC_ID = "logic_01"

DEFAULT_RAG_PDF_MAX_FILES_PER_REQUEST = 30
DEFAULT_RAG_PDF_MAX_BYTES_PER_FILE = 25 * _BYTES_PER_MIB

# txt/md キュー・同期アップロード（RAG_TEXT_MD_*）
DEFAULT_RAG_TEXT_MD_MAX_FILES_PER_REQUEST = 30
DEFAULT_RAG_TEXT_MD_MAX_BYTES_PER_FILE = 25 * _BYTES_PER_MIB

# PDF 抽出モード（native / ocr / auto）
DEFAULT_PDF_EXTRACTION_MODE = "auto"
DEFAULT_PDF_OCR_AUTO_MIN_CHARS_PER_PAGE = 50
DEFAULT_PDF_OCR_LANG = "jpn+eng"
DEFAULT_PDF_OCR_OEM = 3
DEFAULT_PDF_OCR_PSM = 3
DEFAULT_PDF_OCR_DPI = 300
DEFAULT_PDF_OCR_PREPROCESS = True
