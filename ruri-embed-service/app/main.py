# Ruri 系モデル用の最小埋め込み HTTP API。
# 主要: SentenceTransformer でテキストをベクトル化し、検索向け prefix を付与する。
# エンドポイント: GET /health（稼働確認）、POST /embed（一括埋め込み）。
# 流れ: startup でモデルを 1 回ロード（/health はロード済みを前提）→ /embed で encode。

import os
from functools import lru_cache

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.schemas import EmbedRequest, EmbedResponse, EmbeddingInputType

MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "cl-nagoya/ruri-v3-310m")
DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

app = FastAPI(title="Ruri Embedding Service")


@app.on_event("startup")
def startup_load_model() -> None:
    # 初回 /embed だけでなく起動時にロードし、ST 3.x の AutoProcessor 問題は requirements の ST 2.x 固定で回避する。
    get_model()


@app.get("/health")
def health() -> dict[str, str]:
    # モデルは startup でロード済み。ここでは設定値のみ返す。
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    model = get_model()
    texts = [apply_input_prefix(t, req.input_type) for t in req.texts]
    raw = model.encode(
        texts,
        normalize_embeddings=req.normalize,
        convert_to_numpy=True,
    )
    dim = int(raw.shape[1])
    return EmbedResponse(
        model=MODEL_NAME,
        dimensions=dim,
        vectors=raw.tolist(),
    )


## ---------------------------------------------------------------------------
## 補助
## ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)


def apply_input_prefix(text: str, input_type: EmbeddingInputType) -> str:
    if input_type == "query":
        return f"検索クエリ: {text}"
    if input_type == "document":
        return f"検索文書: {text}"
    return text
