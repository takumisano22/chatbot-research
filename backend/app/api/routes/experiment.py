from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.experiment.activity import mark_activity
from app.experiment.batch_runner import run_experiment_batch_sync
from app.experiment.logic_fingerprints import get_logic_fingerprints
from app.experiment.schemas import ExperimentBatchManifest

# -----------------------------------------------------------------------------
# 役割: 実験バッチ（質問セット + ドキュメント群）の指紋取得と CSV 応答。
# 流れ: POST は asyncio.Lock で直列化 → manifest 検証 → to_thread で同期バッチ → CSV 返却。
# 要点: 認証トークンは任意。アクティビティは main のミドルウェアで記録（本ルートでも mark_activity を呼ぶ）。
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/experiment", tags=["experiment"])

_batch_lock = asyncio.Lock()


def _require_experiment_enabled(settings: Settings) -> None:
    if not settings.experiment_batch_enabled:
        raise HTTPException(status_code=404, detail="experiment batch API is disabled")


def _verify_bearer(
    settings: Settings,
    authorization: str | None,
) -> None:
    expected = (settings.experiment_api_token or "").strip()
    if not expected:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer が必要です")
    got = authorization[7:].strip()
    if got != expected:
        raise HTTPException(status_code=403, detail="トークンが一致しません")


@router.get("/logic-fingerprints")
def get_fingerprints(
    settings: Settings = Depends(get_settings),
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    _require_experiment_enabled(settings)
    _verify_bearer(settings, authorization)
    mark_activity()
    return get_logic_fingerprints()


@router.post("/batch")
async def post_batch(
    manifest: Annotated[str, Form(description="ExperimentBatchManifest の JSON 文字列")],
    files: Annotated[list[UploadFile], File(description="取り込む .pdf / .txt / .md")],
    settings: Settings = Depends(get_settings),
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    _require_experiment_enabled(settings)
    _verify_bearer(settings, authorization)
    mark_activity()
    try:
        parsed = ExperimentBatchManifest.model_validate_json(manifest)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    items: list[tuple[str, bytes]] = []
    for uf in files:
        raw = await uf.read()
        items.append((uf.filename or "unnamed", raw))
    if not items:
        raise HTTPException(status_code=400, detail="アップロードファイルが 1 件以上必要です")

    async with _batch_lock:
        csv_bytes = await asyncio.to_thread(run_experiment_batch_sync, parsed, items)
    mark_activity()
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="experiment_results.csv"'},
    )
