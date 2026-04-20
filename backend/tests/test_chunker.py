# [bolt-005] chunker: doc_id 安定性・分割数
from pathlib import Path

from app.rag.vectorstore.chunker import _stable_doc_id, build_chunks_for_source, to_repo_relative


def test_stable_doc_id_same_path_same_id() -> None:
    assert _stable_doc_id("data/rag/a.md") == _stable_doc_id("data/rag/a.md")
    assert _stable_doc_id("data/rag/a.md") != _stable_doc_id("data/rag/b.md")


def test_build_chunks_respects_chunk_size() -> None:
    text = "0123456789" * 4  # 40 文字 → 10 文字×4 チャンク
    chunks = build_chunks_for_source(
        repo_relative_source="sample.md",
        full_text=text,
        chunk_size=10,
        chunk_overlap=0,
    )
    assert len(chunks) == 4
    doc_id = chunks[0].doc_id
    assert all(c.doc_id == doc_id for c in chunks)
    assert chunks[0].chunk_id == f"{doc_id}:0"
    assert chunks[-1].chunk_id == f"{doc_id}:3"


def test_to_repo_relative_under_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "docs" / "x.md"
    sub.parent.mkdir(parents=True)
    sub.touch()
    assert to_repo_relative(sub, root) == "docs/x.md"
