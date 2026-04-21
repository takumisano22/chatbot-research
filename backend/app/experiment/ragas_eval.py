from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from app.core.config import Settings
from app.langfuse.client import get_langfuse_client, safe_flush
from app.langfuse.config import is_langfuse_configured
from app.langfuse.tracer import safe_span_update
from app.services.llm_factory import get_chat_model

# -----------------------------------------------------------------------------
# 役割: 実験 1 ターン分の RAGAS（Faithfulness）を実行し、Langfuse に span として記録する。
# 流れ: Langfuse 有効時は experiment.ragas 観測の子として evaluate → スコアを output に載せる。
# 要点: ragas 未導入・コンテキスト空は no-op。失敗時はログのみで CSV 列は空にする。
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def run_ragas_row_metrics(
    settings: Settings,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    row_index: int,
) -> tuple[str, str]:
    """Faithfulness と Answer relevancy（利用可能なら）を 1 回の evaluate で取得。失敗時は空文字。"""
    if not retrieved_contexts or not response.strip():
        return "", ""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics.collections.faithfulness import Faithfulness
    except ImportError:
        logger.warning("ragas が未インストールのため RAGAS 評価をスキップします")
        return "", ""

    answer_relevancy_cls = None
    try:
        from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

        answer_relevancy_cls = AnswerRelevancy
    except ImportError:
        pass

    def _pick_score(df_columns: Any, row: Any, *substrings: str) -> str:
        for col in df_columns:
            c = str(col).lower()
            if all(s in c for s in substrings):
                val = row[col]
                if val is None:
                    return ""
                try:
                    return f"{float(val):.6f}"
                except (TypeError, ValueError):
                    return str(val)
        return ""

    def _run_eval() -> tuple[str, str]:
        chat = get_chat_model(settings)
        llm = LangchainLLMWrapper(_as_language_model(chat))
        metrics: list[Any] = [Faithfulness(llm=llm)]
        if answer_relevancy_cls is not None:
            try:
                metrics.append(answer_relevancy_cls(llm=llm))
            except Exception:
                logger.exception("AnswerRelevancy 初期化に失敗（Faithfulness のみ続行）")
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        ds = EvaluationDataset.from_list([sample])
        result = evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            show_progress=False,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        if df is None or df.empty:
            return "", ""
        row = df.iloc[0]
        faith = _pick_score(df.columns, row, "faith")
        relev = _pick_score(df.columns, row, "answer", "relev") or _pick_score(
            df.columns, row, "relevancy"
        )
        return faith, relev

    def _wrap_langfuse() -> tuple[str, str]:
        if not is_langfuse_configured(settings):
            try:
                return _run_eval()
            except Exception:
                logger.exception("RAGAS 評価に失敗しました（Langfuse なし）")
                return "", ""

        client = get_langfuse_client(settings)
        if client is None:
            try:
                return _run_eval()
            except Exception:
                logger.exception("RAGAS 評価に失敗しました")
                return "", ""

        try:
            cm = client.start_as_current_observation(
                name="experiment.ragas",
                as_type="span",
                input={
                    "row_index": row_index,
                    "user_input_preview": user_input[:500],
                    "contexts_count": len(retrieved_contexts),
                },
                metadata={"kind": "ragas_row_metrics"},
            )
        except Exception:
            logger.exception("Langfuse experiment.ragas 開始に失敗（評価はローカルのみ試行）")
            try:
                return _run_eval()
            except Exception:
                logger.exception("RAGAS 評価に失敗しました")
                return "", ""

        try:
            with cm as span:
                faith, relev = _run_eval()
                safe_span_update(
                    span,
                    output={"faithfulness": faith or None, "answer_relevancy": relev or None},
                )
                return faith, relev
        except Exception:
            logger.exception("RAGAS 評価に失敗しました")
            return "", ""
        finally:
            safe_flush(client)

    return _wrap_langfuse()


def run_ragas_faithfulness_row(
    settings: Settings,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    row_index: int,
) -> str:
    f, _ = run_ragas_row_metrics(
        settings,
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
        row_index=row_index,
    )
    return f


def _as_language_model(chat: Any) -> BaseLanguageModel:
    return chat  # type: ignore[return-value]
