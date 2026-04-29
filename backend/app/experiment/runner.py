from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.experiment.batch_runner import run_research_pair_batch
from app.experiment.loaders import load_pdf_upload_items, load_qa_questions
from app.experiment.research_pair_schema import load_research_pair_by_id

# -----------------------------------------------------------------------------
# 役割: research_pair を指定して 1 回のバッチ実験を実行し、CSV を outputs/csv/、
#       入力と推論 LLM 出力のみを抜き出した JSON を outputs/json/ に保存する CLI。
# 流れ: 設定読込 → RP / QA / PDF 読込 → バッチ実行（QA ごとに進捗%）→ CSV/JSON 保存 →
#       完了を stderr に通知。stdout 最終行は CSV パス（既存挙動を維持）。
# 命名規則: CSV と JSON は同一ステム ({research_pair_id}_{ts}) で対応関係を表現。
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

    dataset_name = ds_name or Path(rp.qa_dataset).stem
    csv_bytes, qa_items = run_research_pair_batch(
        rp,
        pdfs,
        questions,
        dataset_name=dataset_name,
    )

    # CSV/JSON は outputs 配下のサブディレクトリに分けて出力し、ファイル名ステムを揃える。
    base_dir = settings.resolve_experiment_outputs_dir()
    csv_dir = base_dir / "csv"
    json_dir = base_dir / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{rp.research_pair_id}_{ts}"
    csv_path = csv_dir / f"{stem}.csv"
    json_path = json_dir / f"{stem}.json"

    csv_path.write_bytes(csv_bytes)
    json_payload = {"dataset_name": dataset_name, "items": qa_items}
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"完了: CSV/JSON を出力しました（{rp.research_pair_id}）: "
        f"{csv_path.resolve()} / {json_path.resolve()}",
        file=sys.stderr,
        flush=True,
    )
    print(str(csv_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
