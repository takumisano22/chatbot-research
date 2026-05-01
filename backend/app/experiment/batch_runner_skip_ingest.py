from __future__ import annotations

# batch_runner.py から Chroma リセット・取り込みを省いたバリアント。
# 前ペアと chunking_logic_id / document_set_id / ingest_pipeline_id が全て同一の場合に
# Chroma コレクションを再利用して推論バッチのみを実行する。

import csv
import io
from contextlib import ExitStack

from app.core.config import get_settings
from app.experiment.batch_runner import (
    _emit_qa_progress,
    _run_one_question,
    _runtime_from_research_pair,
)
from app.experiment.csv_format import experiment_csv_fieldnames
from app.experiment.ingest_pipeline_registry import (
    IngestPipelineModule,
    is_superseded,
    load_ingest_pipeline,
)
from app.experiment.logic_registry import (
    load_rag_system_message,
    load_rerank_fn,
    load_retrieve_fn,
    load_split_for_rag_with_metadata,
    load_tokenize_query,
)
from app.experiment.research_pair_schema import ResearchPair
from app.rag.logic.experiment_context import (
    active_chunking_split_with_metadata,
    active_tokenizer,
)
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 公開 API（エントリ）
# -----------------------------------------------------------------------------


def run_research_pair_batch_skip_ingest(
    rp: ResearchPair,
    questions: list[str],
    *,
    dataset_name: str | None,
) -> tuple[bytes, list[dict[str, str]]]:
    # Chroma リセット・取り込みを行わず、前ペアのコレクションを再利用して推論バッチのみ実行する。
    pipeline = load_ingest_pipeline(rp.ingest_pipeline_id) if rp.ingest_pipeline_id else None
    if pipeline is None or not is_superseded(pipeline.superseded, "chunking"):
        load_split_for_rag_with_metadata("chunking", rp.chunking_logic_id)
    load_tokenize_query("tokenizer", rp.tokenizer_logic_id)
    load_retrieve_fn("search", rp.search_logic_id)
    load_rerank_fn("reranking", rp.reranking_logic_id)
    _ = load_rag_system_message(rp.prompt_logic_id)

    base = get_settings()
    runtime = _runtime_from_research_pair(base, rp)
    return _run_batch_skip_ingest(
        runtime=runtime,
        rp=rp,
        questions=questions,
        chunking_logic_id=rp.chunking_logic_id,
        tokenizer_logic_id=rp.tokenizer_logic_id,
        search_logic_id=rp.search_logic_id,
        reranking_logic_id=rp.reranking_logic_id,
        prompt_logic_id=rp.prompt_logic_id,
        run_ragas=rp.ragas_enabled,
        dataset_name=dataset_name,
        top_k=int(rp.top_k),
        pipeline=pipeline,
    )

# -----------------------------------------------------------------------------
# バッチ本処理（ingest なし・全問）
# -----------------------------------------------------------------------------


def _run_batch_skip_ingest(
    *,
    runtime,
    rp: ResearchPair,
    questions: list[str],
    chunking_logic_id: str,
    tokenizer_logic_id: str,
    search_logic_id: str,
    reranking_logic_id: str,
    prompt_logic_id: str,
    run_ragas: bool,
    dataset_name: str | None,
    top_k: int,
    pipeline: IngestPipelineModule | None = None,
) -> tuple[bytes, list[dict[str, str]]]:
    chunking_overridden = pipeline is not None and is_superseded(pipeline.superseded, "chunking")
    split_fn = (
        None if chunking_overridden
        else load_split_for_rag_with_metadata("chunking", chunking_logic_id)
    )
    tok_fn = load_tokenize_query("tokenizer", tokenizer_logic_id)
    with ExitStack() as stack:
        if split_fn is not None:
            stack.enter_context(active_chunking_split_with_metadata(split_fn))
        stack.enter_context(active_tokenizer(tok_fn))
        # rag_reset_collection と run_upload_items_batch は呼ばない（前ペアのコレクションを再利用）。

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
            "ingest_pipeline_id": pipeline.name if pipeline else "",
            "llm_model": runtime.llm_model,
            "embedding_provider": runtime.embedding_provider,
            "embedding_model": runtime.embedding_model,
        }
        n_questions = len(questions)
        qa_items: list[dict[str, str]] = []
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
            qa_items.append(
                {
                    "question": str(row.get("input", "")),
                    "answer": str(row.get("output", "")),
                }
            )
            _emit_qa_progress(i + 1, n_questions, rp.research_pair_id)
        return buf.getvalue().encode("utf-8-sig"), qa_items
