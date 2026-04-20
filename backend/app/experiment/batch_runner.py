from __future__ import annotations

import csv
import io
import time
from typing import Any

from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.experiment.logic_fingerprints import validate_logic_fingerprints
from app.experiment.ragas_eval import run_ragas_faithfulness_row
from app.experiment.runtime_settings import build_runtime_settings
from app.experiment.schemas import ExperimentBatchManifest
from app.rag.ingest_pipeline.service import run_queued_upload_batch
from app.rag.prompts import RAG_NO_DOCUMENTS_REPLY, RAG_SYSTEM_MESSAGE, build_rag_user_message, chunks_to_context_lines
from app.rag.retrieval_service import search_documents
from app.rag.schemas import RagSearchMode, RetrievedChunk
from app.rag.vectorstore.vector_db import rag_reset_collection
from app.services.chat_service import run_chat
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 役割: Chroma リセット → 既存 ingest 経路でファイル投入 → RAG 検索 + LLM を順に実行し CSV バイト列を返す。
# 流れ: validate → runtime Settings → reset → run_queued_upload_batch → 質問ループで timing / RAGAS / CSV。
# 要点: Langfuse は RAGAS 経路のみ使用。グローバル get_settings は変更しない。
# ## ingest_worker と同時稼働すると Chroma への同時書き込みが競合し得るため、実験時は worker を止めるか別 compose にする。
# -----------------------------------------------------------------------------

_CSV_FIELDS: tuple[str, ...] = (
    "index",
    "input",
    "output",
    "rag_latency_ms",
    "total_latency_ms",
    "search_results",
    "vector_distances",
    "prompt",
    "rag_search_mode",
    "rag_hybrid_delegate",
    "ragas_faithfulness",
    "chunking_logic_id",
    "tokenizer_logic_id",
    "search_logic_id",
    "llm_model",
    "embedding_provider",
    "embedding_model",
)


def run_experiment_batch_sync(
    manifest: ExperimentBatchManifest,
    upload_items: list[tuple[str, bytes]],
) -> bytes:
    validate_logic_fingerprints(
        {
            "chunking_logic_id": manifest.chunking_logic_id,
            "tokenizer_logic_id": manifest.tokenizer_logic_id,
            "search_logic_id": manifest.search_logic_id,
        }
    )
    base = get_settings()
    runtime = build_runtime_settings(base, manifest)
    rag_reset_collection(runtime)
    ingest_rows = run_queued_upload_batch(runtime, upload_items)
    _raise_if_ingest_failed(ingest_rows)

    try:
        llm = get_chat_model(runtime)
    except ValueError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    logic_meta: dict[str, str] = {
        "chunking_logic_id": manifest.chunking_logic_id,
        "tokenizer_logic_id": manifest.tokenizer_logic_id,
        "search_logic_id": manifest.search_logic_id,
        "llm_model": runtime.llm_model,
        "embedding_provider": runtime.embedding_provider,
        "embedding_model": runtime.embedding_model,
        "rag_search_mode": str(manifest.rag_search_mode),
        "rag_hybrid_delegate": str(runtime.rag_hybrid_delegate),
    }
    for i, question in enumerate(manifest.questions):
        row = _run_one_question(
            runtime,
            llm,
            i,
            question,
            logic_meta,
            manifest.rag_search_mode,
            manifest.run_ragas_metrics,
        )
        writer.writerow(row)
    return buf.getvalue().encode("utf-8-sig")


# --- 補助 ---


def _raise_if_ingest_failed(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if not r.get("ok"):
            raise HTTPException(
                status_code=400,
                detail={"error": "ingest_failed", "source_name": r.get("source_name"), "message": r.get("error")},
            )


def _run_one_question(
    runtime: Settings,
    llm: BaseChatModel,
    index: int,
    question: str,
    logic_meta: dict[str, str],
    rag_mode: RagSearchMode,
    run_ragas: bool,
) -> dict[str, object]:
    t0 = time.monotonic()
    chunks = search_documents(runtime, question, top_k=None, rag_search_mode=rag_mode)
    t1 = time.monotonic()
    rag_ms = (t1 - t0) * 1000.0
    contexts = [c.chunk_text for c in chunks]
    ragas_score = ""
    if not chunks:
        total_ms = rag_ms
        if run_ragas:
            ragas_score = ""
        return {
            "index": index,
            "input": question,
            "output": RAG_NO_DOCUMENTS_REPLY,
            "rag_latency_ms": f"{rag_ms:.3f}",
            "total_latency_ms": f"{total_ms:.3f}",
            "search_results": "",
            "vector_distances": "",
            "prompt": "",
            "ragas_faithfulness": ragas_score,
            **logic_meta,
        }
    user_augmented = build_rag_user_message(question, chunks)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_MESSAGE},
        {"role": "user", "content": user_augmented},
    ]
    prompt_text = _format_prompt(messages)
    answer = run_chat(llm, messages)
    t3 = time.monotonic()
    total_ms = (t3 - t0) * 1000.0
    if run_ragas:
        ragas_score = run_ragas_faithfulness_row(
            runtime,
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            row_index=index,
        )
    return {
        "index": index,
        "input": question,
        "output": answer,
        "rag_latency_ms": f"{rag_ms:.3f}",
        "total_latency_ms": f"{total_ms:.3f}",
        "search_results": chunks_to_context_lines(chunks),
        "vector_distances": _format_vector_distances(chunks),
        "prompt": prompt_text,
        "ragas_faithfulness": ragas_score,
        **logic_meta,
    }


def _format_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"### {m['role']}\n{m.get('content', '')}")
    return "\n\n".join(parts)


def _format_vector_distances(chunks: list[RetrievedChunk]) -> str:
    pairs: list[str] = []
    for c in chunks:
        if c.vector_score_raw is not None:
            pairs.append(f"{c.chunk_id}:{float(c.vector_score_raw):.8g}")
    return "|".join(pairs)
