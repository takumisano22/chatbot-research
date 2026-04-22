from __future__ import annotations

# 研究ペア単位で Chroma リセット・取り込み・検索・再ランク・LLM・（任意）RAGAS を行い CSV を返す。
# runtime Settings は research_pair から 1 回だけ構築し、全問で同一コレクションを使う。
# QA ごとに stdout へ進捗（%）を出す（# で始まる行。ingest・モデル初期化は含まない。runner の最終行は CSV パス）。

import csv
import io
import sys
import time
from contextlib import ExitStack
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.experiment.csv_format import distance_cell, experiment_csv_fieldnames, snippet_cell
from app.experiment.logic_registry import (
    call_rerank,
    call_retrieve,
    load_rag_system_message,
    load_rerank_fn,
    load_retrieve_fn,
    load_split_for_rag,
    load_tokenize_query,
)
from app.experiment.ragas_eval import run_ragas_row_metrics
from app.experiment.research_pair_schema import ResearchPair
from app.rag.ingest_batch import run_upload_items_batch
from app.rag.logic.experiment_context import active_chunking_split, active_tokenizer
from app.rag.prompts import RAG_NO_DOCUMENTS_REPLY, build_rag_user_message
from app.langfuse.tracer import observe_llm_chat_turn
from app.rag.vectorstore.vector_db import rag_reset_collection
from app.services.chat_service import run_chat
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 例外
# -----------------------------------------------------------------------------


class ExperimentIngestError(Exception):
    def __init__(self, message: str, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.rows = rows or []

# -----------------------------------------------------------------------------
# 公開 API（エントリ）
# -----------------------------------------------------------------------------


def preflight_logic(
    *,
    chunking_logic_id: str,
    tokenizer_logic_id: str,
    search_logic_id: str,
    reranking_logic_id: str,
    prompt_logic_id: str,
) -> None:
    load_split_for_rag("chunking", chunking_logic_id)
    load_tokenize_query("tokenizer", tokenizer_logic_id)
    load_retrieve_fn("search", search_logic_id)
    load_rerank_fn("reranking", reranking_logic_id)
    _ = load_rag_system_message(prompt_logic_id)


def run_research_pair_batch_bytes(
    rp: ResearchPair,
    upload_items: list[tuple[str, bytes]],
    questions: list[str],
    *,
    dataset_name: str | None,
) -> bytes:
    preflight_logic(
        chunking_logic_id=rp.chunking_logic_id,
        tokenizer_logic_id=rp.tokenizer_logic_id,
        search_logic_id=rp.search_logic_id,
        reranking_logic_id=rp.reranking_logic_id,
        prompt_logic_id=rp.prompt_logic_id,
    )
    base = get_settings()
    runtime = _runtime_from_research_pair(base, rp)
    return _run_batch_with_logic(
        runtime=runtime,
        rp=rp,
        upload_items=upload_items,
        questions=questions,
        chunking_logic_id=rp.chunking_logic_id,
        tokenizer_logic_id=rp.tokenizer_logic_id,
        search_logic_id=rp.search_logic_id,
        reranking_logic_id=rp.reranking_logic_id,
        prompt_logic_id=rp.prompt_logic_id,
        run_ragas=rp.ragas_enabled,
        dataset_name=dataset_name,
        top_k=int(rp.top_k),
    )

# -----------------------------------------------------------------------------
# バッチ本処理（研究ペア単位・全問）
# -----------------------------------------------------------------------------


def _run_batch_with_logic(
    *,
    runtime: Settings,
    rp: ResearchPair,
    upload_items: list[tuple[str, bytes]],
    questions: list[str],
    chunking_logic_id: str,
    tokenizer_logic_id: str,
    search_logic_id: str,
    reranking_logic_id: str,
    prompt_logic_id: str,
    run_ragas: bool,
    dataset_name: str | None,
    top_k: int,
) -> bytes:
    split_fn = load_split_for_rag("chunking", chunking_logic_id)
    tok_fn = load_tokenize_query("tokenizer", tokenizer_logic_id)
    with ExitStack() as stack:
        stack.enter_context(active_chunking_split(split_fn))
        stack.enter_context(active_tokenizer(tok_fn))
        rag_reset_collection(runtime)
        ingest_rows = run_upload_items_batch(runtime, upload_items)
        _raise_if_ingest_failed(ingest_rows)

        try:
            llm = get_chat_model(runtime)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        rag_system_message = load_rag_system_message(prompt_logic_id)

        fieldnames = experiment_csv_fieldnames(top_k)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        spec_json = rp.spec_json_for_csv()
        logic_meta: dict[str, str] = {
            "research_pair_spec": spec_json,
            "chunking_logic_id": chunking_logic_id,
            "tokenizer_logic_id": tokenizer_logic_id,
            "search_logic_id": search_logic_id,
            "reranking_logic_id": reranking_logic_id,
            "prompt_logic_id": prompt_logic_id,
            "llm_model": runtime.llm_model,
            "embedding_provider": runtime.embedding_provider,
            "embedding_model": runtime.embedding_model,
        }
        n_questions = len(questions)
        if n_questions == 0:
            _emit_qa_progress(0, 0, rp.research_pair_id)
        for i, question in enumerate(questions):
            row = _run_one_question(
                runtime=runtime,
                llm=llm,
                index=i,
                question=question,
                logic_meta=logic_meta,
                search_logic_id=search_logic_id,
                reranking_logic_id=reranking_logic_id,
                rag_system_message=rag_system_message,
                run_ragas=run_ragas,
                research_pair_id=rp.research_pair_id,
                document_set_id=rp.document_set_id.strip(),
                dataset_name=dataset_name,
                top_k=top_k,
            )
            writer.writerow(row)
            _emit_qa_progress(i + 1, n_questions, rp.research_pair_id)
        return buf.getvalue().encode("utf-8-sig")

# -----------------------------------------------------------------------------
# 1 問の実行
# -----------------------------------------------------------------------------


def _run_one_question(
    *,
    runtime: Settings,
    llm: BaseChatModel,
    index: int,
    question: str,
    logic_meta: dict[str, str],
    search_logic_id: str,
    reranking_logic_id: str,
    rag_system_message: str,
    run_ragas: bool,
    research_pair_id: str,
    document_set_id: str,
    dataset_name: str | None,
    top_k: int,
) -> dict[str, object]:
    t0 = time.monotonic()
    chunks = call_retrieve(
        "search",
        search_logic_id,
        runtime,
        question,
        top_k=top_k,
    )
    chunks = call_rerank("reranking", reranking_logic_id, runtime, question, chunks)
    chunks = chunks[:top_k]
    t1 = time.monotonic()
    rag_ms = (t1 - t0) * 1000.0
    contexts = [c.chunk_text for c in chunks]
    ragas_faith = ""
    ragas_relev = ""
    base_row: dict[str, object] = {
        "research_pair_id": research_pair_id,
        "question_index": index,
        "document_set_id": document_set_id,
        "dataset_name": dataset_name or "",
        "input": question,
        "rag_latency_ms": f"{rag_ms:.3f}",
        "top_k": top_k,
        "ragas_faithfulness": ragas_faith,
        "ragas_answer_relevancy": ragas_relev,
        **logic_meta,
    }
    for k in range(1, top_k + 1):
        base_row[f"retrieved_source_{k}"] = ""
        base_row[f"retrieved_chunk_id_{k}"] = ""
        base_row[f"retrieved_distance_{k}"] = ""
        base_row[f"retrieved_text_{k}"] = ""
    for idx, c in enumerate(chunks):
        n = idx + 1
        base_row[f"retrieved_source_{n}"] = c.source
        base_row[f"retrieved_chunk_id_{n}"] = c.chunk_id
        base_row[f"retrieved_distance_{n}"] = distance_cell(c)
        base_row[f"retrieved_text_{n}"] = snippet_cell(c.chunk_text)

    if not chunks:
        total_ms = rag_ms
        base_row["output"] = RAG_NO_DOCUMENTS_REPLY
        base_row["total_latency_ms"] = f"{total_ms:.3f}"
        return base_row

    user_augmented = build_rag_user_message(question, chunks)
    messages = [
        {"role": "system", "content": rag_system_message},
        {"role": "user", "content": user_augmented},
    ]
    answer = observe_llm_chat_turn(runtime, messages, lambda: run_chat(llm, messages))
    t3 = time.monotonic()
    total_ms = (t3 - t0) * 1000.0
    if run_ragas:
        ragas_faith, ragas_relev = run_ragas_row_metrics(
            runtime,
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            row_index=index,
        )
        base_row["ragas_faithfulness"] = ragas_faith
        base_row["ragas_answer_relevancy"] = ragas_relev

    base_row["output"] = answer
    base_row["total_latency_ms"] = f"{total_ms:.3f}"
    return base_row

# -----------------------------------------------------------------------------
# 補助
# -----------------------------------------------------------------------------


def _runtime_from_research_pair(base: Settings, rp: ResearchPair) -> Settings:
    patch: dict[str, object] = {}
    if rp.llm_model is not None:
        patch["llm_model"] = rp.llm_model
    if rp.llm_api_base_url is not None:
        patch["llm_api_base_url"] = rp.llm_api_base_url
    if rp.llm_temperature is not None:
        patch["llm_temperature"] = rp.llm_temperature
    if rp.llm_request_timeout_seconds is not None:
        patch["llm_request_timeout_seconds"] = rp.llm_request_timeout_seconds

    if rp.embedding_provider is not None:
        patch["embedding_provider"] = rp.embedding_provider
    if rp.embedding_base_url is not None:
        patch["embedding_base_url"] = rp.embedding_base_url
    if rp.embedding_model is not None:
        patch["embedding_model"] = rp.embedding_model

    if rp.rag_hybrid_delegate is not None:
        patch["rag_hybrid_delegate"] = rp.rag_hybrid_delegate

    if rp.langfuse_enabled is not None:
        patch["langfuse_enabled"] = rp.langfuse_enabled

    patch["rag_vector_top_k"] = rp.top_k
    patch["rag_top_k"] = rp.top_k

    return base.model_copy(update=patch)


def _raise_if_ingest_failed(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if not r.get("ok"):
            raise ExperimentIngestError("ingest_failed", rows)


def _emit_qa_progress(done: int, total: int, research_pair_id: str) -> None:
    ## Docker や IDE によっては stderr がログに出にくいため、進捗は stdout（# 行）と stderr の両方へ出す。
    if total <= 0:
        msg = f"[{research_pair_id}] 進捗: 質問 0 件（データ行なし）"
        print(f"# experiment progress: {msg}", flush=True)
        print(msg, file=sys.stderr, flush=True)
        return
    pct = 100.0 * done / total
    msg = f"[{research_pair_id}] 進捗: {done}/{total} ({pct:.1f}%)"
    print(f"# experiment progress: {msg}", flush=True)
    print(msg, file=sys.stderr, flush=True)
