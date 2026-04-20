# Chroma ClientAPI の生成と RAG コレクション解決。HTTP / 永続（pytest のみ）の切り替えはここだけ。
from __future__ import annotations

import os

import chromadb
from chromadb.api.models.Collection import Collection

from vectordb.chroma.config import VectorStoreConfig


def get_vector_store_client(config: VectorStoreConfig) -> chromadb.ClientAPI:
    host = (config.server_host or "").strip()
    if host:
        return chromadb.HttpClient(host=host, port=config.server_port)
    if _embedded_client_allowed_for_pytest():
        return chromadb.PersistentClient(path=str(config.persist_dir.resolve()))
    raise ValueError(
        "VECTOR_STORE_SERVER_HOST が未設定です。"
        " Docker Compose では .env にベクトルDBサービス名を設定してください（例: VECTOR_STORE_SERVER_HOST=vector_store）。"
    )


def get_rag_collection(config: VectorStoreConfig) -> Collection:
    client = get_vector_store_client(config)
    return client.get_or_create_collection(
        name=config.collection_name,
        metadata={"description": "RAG-collection"},
    )


# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------


def _embedded_client_allowed_for_pytest() -> bool:
    return bool(os.environ.get("PYTEST_VERSION"))
