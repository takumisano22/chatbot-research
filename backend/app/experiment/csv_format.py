from __future__ import annotations

from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: 実験 CSV の固定列名生成とセル値整形（top_k 展開列）。
# -----------------------------------------------------------------------------

_MAX_SNIPPET = 800


def experiment_csv_fieldnames(top_k: int) -> list[str]:
    fixed = [
        "research_pair_id",
        "research_pair_spec",
        "question_index",
        "document_set_id",
        "dataset_name",
        "input",
        "output",
        "rag_latency_ms",
        "total_latency_ms",
        "top_k",
        "chunking_logic_id",
        "tokenizer_logic_id",
        "search_logic_id",
        "reranking_logic_id",
        "prompt_logic_id",
        "llm_model",
        "embedding_provider",
        "embedding_model",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
    ]
    dyn: list[str] = []
    for i in range(1, top_k + 1):
        dyn.extend(
            [
                f"retrieved_source_{i}",
                f"retrieved_chunk_id_{i}",
                f"retrieved_distance_{i}",
                f"retrieved_text_{i}",
            ]
        )
    return fixed + dyn


def distance_cell(chunk: RetrievedChunk) -> str:
    if chunk.vector_score_raw is not None:
        return f"{float(chunk.vector_score_raw):.8g}"
    return f"{float(chunk.keyword_score_raw):.8g}"


def snippet_cell(text: str) -> str:
    t = text.replace("\r", " ").replace("\n", " ").strip()
    if len(t) <= _MAX_SNIPPET:
        return t
    return t[: _MAX_SNIPPET - 3] + "..."
