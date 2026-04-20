# [bolt-002 P1 Docker Compose] FastAPI バックエンドのコンテナイメージ（ルートを build context にし pip install + uvicorn）。
# ビルドはリポジトリルートを context にし、dockerfile を backend/Dockerfile に指定する（compose.yaml 参照）
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# RAG PDF は MarkItDown（Python 依存）。旧 OCR 用の poppler/tesseract は不要。

COPY backend/pyproject.toml .
COPY llm_definitions ../llm_definitions
COPY vector_stores ../vector_stores
COPY backend/alembic.ini .
COPY backend/alembic ./alembic
COPY backend/app ./app
COPY init /app/init
RUN chmod +x /app/init/run.sh
RUN pip install --no-cache-dir ".[migrate]"

EXPOSE 8000

# メモ：pythonのdockerイメージには既定起動がないので、CMDで指定する。（compose.yamlに書いてもいいらしい）
# メモ：uvicornを使ってFastAPIを起動。ASGIサーバーというやつ。指定のIPでTCPを待ち受ける。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
