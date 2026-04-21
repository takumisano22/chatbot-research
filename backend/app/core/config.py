from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlparse, urlunparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core import field_defaults as FD

# -----------------------------------------------------------------------------
# 役割: 環境変数と .env からアプリ設定（Pydantic Settings）を読み込み、リポジトリルートを推定する。
# 主な呼び出し元: 実験 runner・RAG 層・Langfuse（get_settings() 経由）。
# 流れ: get_settings（lru_cache）が Settings を生成 → 検証済みフィールドと resolve_* を各所が利用。
# Docker 内では LANGFUSE_HOST のループバックを host.docker.internal に寄せる（model_validator）。
# -----------------------------------------------------------------------------

__all__ = ["Settings", "get_settings", "get_repo_root"]

_CONFIG_FILE = Path(__file__).resolve() ## このファイル自身の絶対パスを取得
BACKEND_ROOT = _CONFIG_FILE.parents[2] ## このファイルの親ディレクトリの親ディレクトリのパスを取得
_FILESYSTEM_ROOT = Path("/") ## ルートディレクトリのパスを取得

## 端的にいえば、config.pyの三階層親がファイルシステムだとこまるから、REPOをBACKENDにするということ。通常は、REPOはプロジェクトルート
if _CONFIG_FILE.parents[3] == _FILESYSTEM_ROOT and (BACKEND_ROOT / "app").is_dir(): 
    REPO_ROOT = BACKEND_ROOT
else:
    REPO_ROOT = _CONFIG_FILE.parents[3]

## 設定したREPO_ROOT を返す関数
def get_repo_root() -> Path:
    return REPO_ROOT


class Settings(BaseSettings):
    ## 設定ファイルの読み込み設定、通常はREPOに.envを置くがbackendにもあれば重複は上書きする。
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ## 設定ファイルで見つからない場合のデフォルトフィールド設定。（FD*で省略記法しているfield_defaultsが構文。※app.coreからインポート）

    app_env: str = FD.DEFAULT_APP_ENV
    api_host: str = FD.DEFAULT_API_HOST
    api_port: int = FD.DEFAULT_API_PORT
    cors_origins: str = FD.DEFAULT_CORS_ORIGINS

    llm_provider: str = FD.DEFAULT_LLM_PROVIDER
    llm_api_base_url: str = FD.DEFAULT_LLM_API_BASE_URL
    llm_model: str = FD.DEFAULT_LLM_MODEL
    llm_temperature: float = FD.DEFAULT_LLM_TEMPERATURE
    llm_request_timeout_seconds: float = Field(
        default=FD.DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
        ge=1.0,
        description="LLM HTTP クライアントのタイムアウト（秒）。長い生成に合わせて調整する。",
    )
    llm_adapter_subpackage: str = Field(
        default=FD.DEFAULT_LLM_ADAPTER_SUBPACKAGE,
        description="llm_provider 配下の Python サブパッケージ名（環境変数 LLM_ADAPTER_SUBPACKAGE）。",
    )
    embedding_provider: str = FD.DEFAULT_EMBEDDING_PROVIDER
    embedding_base_url: str = FD.DEFAULT_EMBEDDING_BASE_URL
    embedding_model: str = FD.DEFAULT_EMBEDDING_MODEL

    vector_store_provider: str = FD.DEFAULT_VECTOR_STORE_PROVIDER
    vector_db_adapter_subpackage: str = Field(
        default=FD.DEFAULT_VECTOR_DB_ADAPTER_SUBPACKAGE,
        description="vectordb 配下の Python サブパッケージ名（環境変数 VECTOR_DB_ADAPTER_SUBPACKAGE）。",
    )
    vector_store_server_host: str = FD.DEFAULT_VECTOR_STORE_SERVER_HOST
    vector_store_server_port: int = FD.DEFAULT_VECTOR_STORE_SERVER_PORT
    vector_store_persist_dir: Path = Field(default=FD.DEFAULT_VECTOR_STORE_PERSIST_DIR)
    rag_collection_name: str = FD.DEFAULT_RAG_COLLECTION_NAME
    rag_top_k: int = FD.DEFAULT_RAG_TOP_K
    rag_vector_top_k: int = FD.DEFAULT_RAG_VECTOR_TOP_K
    rag_keyword_weight: float = FD.DEFAULT_RAG_KEYWORD_WEIGHT
    rag_hybrid_delegate: Literal["vector_search", "keyword_search"] = Field(
        default="vector_search",
        description="hybrid_search 窓口の委譲先（RAG_HYBRID_DELEGATE）。将来の統合実装までの切替用。",
    )
    rag_chunk_size: int = FD.DEFAULT_RAG_CHUNK_SIZE
    rag_chunk_overlap: int = FD.DEFAULT_RAG_CHUNK_OVERLAP
    rag_prompt_logic_id: str = Field(
        default=FD.DEFAULT_RAG_PROMPT_LOGIC_ID,
        min_length=1,
        description="HTTP RAG のシステムプロンプト（prompt_logic_<id>）。",
    )

    # Langfuse（別ホスト想定。未設定・無効時は観測を送らない）
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None
    langfuse_environment: str | None = None

    experiment_research_pairs_dir: Path = Field(
        default=Path("research_pairs"),
        description="research_pair ファイルのルート（相対なら REPO_ROOT 基準）。",
    )
    experiment_qa_datasets_dir: Path = Field(
        default=Path("qa_datasets"),
        description="QA JSON の配置ディレクトリ。",
    )
    experiment_ingest_document_dir: Path = Field(
        default=Path("ingest_document"),
        description="document_set_id ごとの PDF ルート。",
    )
    experiment_outputs_dir: Path = Field(
        default=Path("outputs"),
        description="実験 CSV 出力先。",
    )

    rag_pdf_max_files_per_request: int = FD.DEFAULT_RAG_PDF_MAX_FILES_PER_REQUEST
    rag_pdf_max_bytes_per_file: int = FD.DEFAULT_RAG_PDF_MAX_BYTES_PER_FILE
    rag_text_md_max_files_per_request: int = FD.DEFAULT_RAG_TEXT_MD_MAX_FILES_PER_REQUEST
    rag_text_md_max_bytes_per_file: int = FD.DEFAULT_RAG_TEXT_MD_MAX_BYTES_PER_FILE
    pdf_extraction_mode: str = FD.DEFAULT_PDF_EXTRACTION_MODE
    pdf_ocr_auto_min_chars_per_page: int = Field(
        default=FD.DEFAULT_PDF_OCR_AUTO_MIN_CHARS_PER_PAGE,
        ge=0,
    )
    pdf_ocr_lang: str = FD.DEFAULT_PDF_OCR_LANG
    pdf_ocr_oem: int = Field(default=FD.DEFAULT_PDF_OCR_OEM, ge=0)
    pdf_ocr_psm: int = Field(default=FD.DEFAULT_PDF_OCR_PSM, ge=0)
    pdf_ocr_dpi: int = Field(default=FD.DEFAULT_PDF_OCR_DPI, ge=72)
    pdf_ocr_preprocess: bool = FD.DEFAULT_PDF_OCR_PREPROCESS

    @field_validator("rag_hybrid_delegate", mode="before")
    @classmethod
    def _rag_hybrid_delegate(cls, v: object) -> str:
        s = str(v or "vector_search").strip().lower()
        if s in ("keyword", "keyword_search"):
            return "keyword_search"
        return "vector_search"

    @field_validator("llm_adapter_subpackage")
    @classmethod
    def _llm_adapter_subpackage(cls, v: str) -> str:
        allowed = frozenset({"ollama"})
        x = (v or FD.DEFAULT_LLM_ADAPTER_SUBPACKAGE).strip().lower()
        if x not in allowed:
            raise ValueError(
                f"llm_adapter_subpackage（LLM_ADAPTER_SUBPACKAGE）は {sorted(allowed)} のいずれかにしてください（現在: {v!r}）"
            )
        return x

    @field_validator("vector_db_adapter_subpackage")
    @classmethod
    def _vector_db_adapter_subpackage(cls, v: str) -> str:
        allowed = frozenset({"chroma"})
        x = (v or FD.DEFAULT_VECTOR_DB_ADAPTER_SUBPACKAGE).strip().lower()
        if x not in allowed:
            raise ValueError(
                "vector_db_adapter_subpackage（VECTOR_DB_ADAPTER_SUBPACKAGE）は "
                f"{sorted(allowed)} のいずれかにしてください（現在: {v!r}）"
            )
        return x

    @field_validator("pdf_extraction_mode")
    @classmethod
    def _pdf_extraction_mode(cls, v: str) -> str:
        allowed = frozenset({"native", "ocr", "auto"})
        x = (v or FD.DEFAULT_PDF_EXTRACTION_MODE).strip().lower()
        if x not in allowed:
            raise ValueError(
                "pdf_extraction_mode（PDF_EXTRACTION_MODE）は "
                f"{sorted(allowed)} のいずれかにしてください（現在: {v!r}）"
            )
        return x

    def resolve_vector_store_persist_dir(self) -> Path:
        p = self.vector_store_persist_dir
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    def resolve_experiment_research_pairs_dir(self) -> Path:
        p = self.experiment_research_pairs_dir
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    def resolve_experiment_qa_datasets_dir(self) -> Path:
        p = self.experiment_qa_datasets_dir
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    def resolve_experiment_ingest_document_dir(self) -> Path:
        p = self.experiment_ingest_document_dir
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    def resolve_experiment_outputs_dir(self) -> Path:
        p = self.experiment_outputs_dir
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()

    @model_validator(mode="after")
    def _vector_store_provider_supported(self) -> Self:
        name = (self.vector_store_provider or "").strip().lower()
        allowed = (FD.DEFAULT_VECTOR_STORE_PROVIDER or "").strip().lower()
        if name != allowed:
            raise ValueError(
                f"VECTOR_STORE_PROVIDER={self.vector_store_provider!r} は未対応です。"
                f" 現状は {FD.DEFAULT_VECTOR_STORE_PROVIDER!r} のみ指定してください。"
            )
        return self

    @model_validator(mode="after")
    def _embedding_provider_supported(self) -> Self:
        name = (self.embedding_provider or "").strip().lower()
        allowed = frozenset({"ollama", "ruri_http"})
        if name not in allowed:
            raise ValueError(
                f"EMBEDDING_PROVIDER={self.embedding_provider!r} は未対応です。"
                f" {sorted(allowed)} のいずれかを指定してください。"
            )
        return self

    @model_validator(mode="after")
    def _langfuse_host_loopback_in_docker(self) -> Self:
        # compose の extra_hosts と組み合わせ、コンテナ内からホストの Langfuse（OTEL 含む）へ届ける。
        new_host = _langfuse_host_replace_loopback_if_docker(self.langfuse_host)
        if new_host == self.langfuse_host:
            return self
        return self.model_copy(update={"langfuse_host": new_host})

## 毎処理にセッティングを読み込むと負荷が高いので、キャッシュを使用して再利用する。
## lru_cacheは利用頻度順でメモリを制御してくれる。

# -----------------------------------------------------------------------------
# 補助: Docker コンテナ内でのみ Langfuse のループバック URL を書き換える。
# -----------------------------------------------------------------------------

_DOCKERENV_PATH = Path("/.dockerenv")


def _langfuse_host_replace_loopback_if_docker(host: str | None) -> str | None:
    if not host or not _DOCKERENV_PATH.is_file():
        return host
    raw = host.strip()
    parsed = urlparse(raw)
    hn = (parsed.hostname or "").lower()
    if hn not in ("localhost", "127.0.0.1"):
        return host
    port = f":{parsed.port}" if parsed.port else ""
    auth = ""
    if parsed.username is not None or parsed.password is not None:
        user = parsed.username or ""
        passwd = f":{parsed.password}" if parsed.password else ""
        auth = f"{user}{passwd}@"
    new_netloc = f"{auth}host.docker.internal{port}"
    return urlunparse(
        (parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
