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


def run_ragas_faithfulness_row(
    settings: Settings,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    row_index: int,
) -> str:
    if not retrieved_contexts or not response.strip():
        return ""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics.collections.faithfulness import Faithfulness
    except ImportError:
        logger.warning("ragas が未インストールのため RAGAS 評価をスキップします")
        return ""

    def _run_eval() -> str:
        chat = get_chat_model(settings)
        llm = LangchainLLMWrapper(_as_language_model(chat))
        metric = Faithfulness(llm=llm)
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        ds = EvaluationDataset.from_list([sample])
        result = evaluate(
            ds,
            metrics=[metric],
            llm=llm,
            show_progress=False,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        if df is None or df.empty:
            return ""
        for col in df.columns:
            if "faith" in str(col).lower():
                val = df.iloc[0][col]
                if val is None:
                    return ""
                return f"{float(val):.6f}"
        return ""

    if not is_langfuse_configured(settings):
        try:
            return _run_eval()
        except Exception:
            logger.exception("RAGAS 評価に失敗しました（Langfuse なし）")
            return ""

    client = get_langfuse_client(settings)
    if client is None:
        try:
            return _run_eval()
        except Exception:
            logger.exception("RAGAS 評価に失敗しました")
            return ""

    try:
        cm = client.start_as_current_observation(
            name="experiment.ragas",
            as_type="span",
            input={
                "row_index": row_index,
                "user_input_preview": user_input[:500],
                "contexts_count": len(retrieved_contexts),
            },
            metadata={"kind": "ragas_faithfulness"},
        )
    except Exception:
        logger.exception("Langfuse experiment.ragas 開始に失敗（評価はローカルのみ試行）")
        try:
            return _run_eval()
        except Exception:
            logger.exception("RAGAS 評価に失敗しました")
            return ""

    try:
        with cm as span:
            score = _run_eval()
            safe_span_update(span, output={"faithfulness": score or None})
            return score
    except Exception:
        logger.exception("RAGAS 評価に失敗しました")
        return ""
    finally:
        safe_flush(client)


def _as_language_model(chat: Any) -> BaseLanguageModel:
    return chat  # type: ignore[return-value]
