# 互換: 旧名 RemoteVectorStoreConfig。実体は vectordb.chroma.VectorStoreConfig。
from __future__ import annotations

from vectordb.chroma.config import VectorStoreConfig as RemoteVectorStoreConfig

__all__ = ["RemoteVectorStoreConfig"]
