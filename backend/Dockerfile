# 実験バッチ用: Chroma（別サービス） + 本イメージで runner のみ実行する想定。
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxcb1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml .
COPY llm_definitions ../llm_definitions
COPY vector_stores ../vector_stores

RUN pip install --no-cache-dir "."
COPY backend/app ./app


# 既定はヘルプ表示。compose では experiment_runner が runner を上書きする。
CMD ["python", "-m", "app.experiment.runner", "--help"]
