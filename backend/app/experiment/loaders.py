from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import Settings

# -----------------------------------------------------------------------------
# 役割: qa_datasets/*.json と ingest_document/<document_set_id>/*.pdf を読み込む。
# -----------------------------------------------------------------------------


class QaDatasetItem(BaseModel):
    question: str = Field(..., min_length=1)
    reference_answer: str | None = None


class QaDatasetFile(BaseModel):
    dataset_name: str | None = None
    items: list[QaDatasetItem] = Field(..., min_length=1)


def load_qa_questions(path: Path) -> tuple[list[str], str | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        ds = QaDatasetFile(dataset_name=None, items=[QaDatasetItem.model_validate(x) for x in raw])
    else:
        ds = QaDatasetFile.model_validate(raw)
    names = [x.question for x in ds.items]
    return names, ds.dataset_name


def load_pdf_upload_items(settings: Settings, document_set_id: str) -> list[tuple[str, bytes]]:
    rid = document_set_id.strip()
    if not rid:
        raise ValueError("document_set_id が空です")
    base = settings.resolve_experiment_ingest_document_dir().resolve()
    root = (base / rid).resolve()
    try:
        root.relative_to(base)
    except ValueError as e:
        raise ValueError("document_set_id が ingest ルート外を指しています") from e
    if not root.is_dir():
        raise FileNotFoundError(f"document set ディレクトリがありません: {root}")
    pdfs = sorted(root.rglob("*.pdf")) + sorted(root.rglob("*.PDF"))
    if not pdfs:
        raise FileNotFoundError(f"PDF がありません: {root}")
    out: list[tuple[str, bytes]] = []
    for p in pdfs:
        rel = p.relative_to(root).as_posix()
        out.append((rel, p.read_bytes()))
    return out
