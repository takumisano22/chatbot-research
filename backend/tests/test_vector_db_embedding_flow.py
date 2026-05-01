from __future__ import annotations

from app.core.config import Settings
from app.rag.vectorstore import vector_db as vdb


def test_rag_write_session_add_chunks_passes_embeddings(monkeypatch) -> None:
    captured_embeddings: list[list[float]] | None = None
    captured_input_type: object = None

    class FakeInnerSession:
        def add_chunks(self, _records, embeddings) -> None:
            nonlocal captured_embeddings
            captured_embeddings = embeddings

        def delete_by_source(self, _source: str) -> None:
            return

    class FakeVectorStore:
        class ChunkRecord:
            def __init__(
                self,
                *,
                chunk_id: str,
                doc_id: str,
                source: str,
                chunk_text: str,
                document_lower: str,
                metadata: dict | None = None,
            ) -> None:
                self.chunk_id = chunk_id
                self.doc_id = doc_id
                self.source = source
                self.chunk_text = chunk_text
                self.document_lower = document_lower
                self.metadata = metadata or {}

        class VectorStoreConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        @staticmethod
        def RagWriteSession(_config) -> FakeInnerSession:
            return FakeInnerSession()

        @staticmethod
        def is_embedding_dimension_mismatch_error(_exc: Exception) -> bool:
            return False

        @staticmethod
        def reset_rag_collection(_config) -> None:
            return

    class FakeEmbeddingService:
        def embed_texts(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            nonlocal captured_input_type
            assert texts == ["hello world"]
            captured_input_type = kwargs.get("input_type")
            return [[0.11, 0.22]]

    monkeypatch.setattr(vdb, "_vector_store", lambda _settings: FakeVectorStore)
    monkeypatch.setattr(vdb, "build_embedding_service", lambda _settings: FakeEmbeddingService())

    settings = Settings.model_construct(
        vector_db_adapter_subpackage="chroma",
        vector_store_server_host="",
        vector_store_server_port=8000,
        rag_collection_name="rag_documents",
    )
    session = vdb.rag_write_session(settings)
    chunks = [
        vdb.ChunkForStore(
            chunk_id="c1",
            doc_id="d1",
            source="uploaded/a.md",
            chunk_text="Hello World",
            document_lower="hello world",
        )
    ]
    session.add_chunks(chunks)

    assert captured_embeddings == [[0.11, 0.22]]
    assert captured_input_type == "document"


def test_rag_write_session_retries_after_dimension_mismatch(monkeypatch) -> None:
    call_count = 0
    reset_called = False

    class FakeInnerSession:
        def add_chunks(self, _records, _embeddings) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Collection expecting embedding with dimension of 384, got 768")

        def delete_by_source(self, _source: str) -> None:
            return

    class FakeVectorStore:
        class ChunkRecord:
            def __init__(
                self,
                *,
                chunk_id: str,
                doc_id: str,
                source: str,
                chunk_text: str,
                document_lower: str,
                metadata: dict | None = None,
            ) -> None:
                self.chunk_id = chunk_id
                self.doc_id = doc_id
                self.source = source
                self.chunk_text = chunk_text
                self.document_lower = document_lower
                self.metadata = metadata or {}

        class VectorStoreConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        @staticmethod
        def RagWriteSession(_config) -> FakeInnerSession:
            return FakeInnerSession()

        @staticmethod
        def is_embedding_dimension_mismatch_error(exc: Exception) -> bool:
            return "dimension" in str(exc).lower()

        @staticmethod
        def reset_rag_collection(_config) -> None:
            nonlocal reset_called
            reset_called = True

    class FakeEmbeddingService:
        def embed_texts(self, _texts: list[str], **_: object) -> list[list[float]]:
            return [[0.11, 0.22]]

    monkeypatch.setattr(vdb, "_vector_store", lambda _settings: FakeVectorStore)
    monkeypatch.setattr(vdb, "build_embedding_service", lambda _settings: FakeEmbeddingService())

    settings = Settings.model_construct(
        vector_db_adapter_subpackage="chroma",
        vector_store_server_host="",
        vector_store_server_port=8000,
        rag_collection_name="rag_documents",
    )
    session = vdb.rag_write_session(settings)
    session.add_chunks(
        [
            vdb.ChunkForStore(
                chunk_id="c1",
                doc_id="d1",
                source="uploaded/a.md",
                chunk_text="Hello World",
                document_lower="hello world",
            )
        ]
    )

    assert reset_called is True
    assert call_count == 2
