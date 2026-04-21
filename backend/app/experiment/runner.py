from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.experiment.batch_runner import run_research_pair_batch_bytes
from app.experiment.loaders import load_pdf_upload_items, load_qa_questions
from app.experiment.research_pair_schema import load_research_pair_by_id

# -----------------------------------------------------------------------------
# 役割: research_pair を指定して 1 回のバッチ実験を実行し CSV を outputs に保存する CLI。
# 流れ: 設定読込 → RP / QA / PDF 読込 → バッチ実行 → ファイル書き出し。
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research pair RAG batch experiment runner")
    parser.add_argument(
        "--research-pair",
        required=True,
        help="research_pair_id（例: RP-0001）またはファイル名",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    rp = load_research_pair_by_id(settings, args.research_pair)
    qa_path = rp.qa_dataset_path(settings)
    questions, ds_name = load_qa_questions(qa_path)
    pdfs = load_pdf_upload_items(settings, rp.document_set_id)

    csv_bytes = run_research_pair_batch_bytes(
        rp,
        pdfs,
        questions,
        dataset_name=ds_name or Path(rp.qa_dataset).stem,
    )

    out_dir = settings.resolve_experiment_outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{rp.research_pair_id}_{ts}.csv"
    out_path.write_bytes(csv_bytes)
    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
