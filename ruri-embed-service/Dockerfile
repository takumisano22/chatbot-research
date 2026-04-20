FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    EMBEDDING_MODEL=cl-nagoya/ruri-v3-310m \
    EMBEDDING_DEVICE=cpu

WORKDIR /app

COPY requirements.txt .
# 既定 PyPI の torch は CUDA 依存が付くため、CPU ビルドを先に入れてイメージを最小化する。
RUN pip install --no-cache-dir "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
