from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, NamedTuple

from fastapi import HTTPException, UploadFile

from app.core.config import Settings

# -----------------------------------------------------------------------------
# 役割: FastAPI の UploadFile 列からファイル名・本文・種別を組み立て、件数・サイズ上限を検証する。
# 主な呼び出し元: ingest API ルート。
# 流れ: 空チェック → 拡張子で種別判定 → read とサイズ検証 → 種別ごとの件数上限。
# -----------------------------------------------------------------------------

UploadKind = Literal["pdf", "txt_md"]


class _KindRule(NamedTuple):
    extensions: frozenset[str]
    max_bytes: Callable[[Settings], int]
    max_files: Callable[[Settings], int]
    batch_count_mixed: Callable[[int, int], str]
    mixed_file_size_detail: Callable[[int, str], str]


def _pdf_batch_msg(n: int, m: int) -> str:
    return f"一度に送信できる PDF は最大 {m} 件です（現在 {n} 件）"


_RULES: dict[UploadKind, _KindRule] = {
    "pdf": _KindRule(
        extensions=frozenset({".pdf"}),
        max_bytes=lambda s: s.rag_pdf_max_bytes_per_file,
        max_files=lambda s: s.rag_pdf_max_files_per_request,
        batch_count_mixed=_pdf_batch_msg,
        mixed_file_size_detail=lambda lim, name: f"PDF のファイルサイズ上限は {lim} バイトです: {name}",
    ),
    "txt_md": _KindRule(
        extensions=frozenset({".txt", ".md"}),
        max_bytes=lambda s: s.rag_text_md_max_bytes_per_file,
        max_files=lambda s: s.rag_text_md_max_files_per_request,
        batch_count_mixed=lambda n, m: f"一度に送信できる .txt/.md は最大 {m} 件です（現在 {n} 件）",
        mixed_file_size_detail=lambda lim, name: f"テキストファイルのサイズ上限は {lim} バイトです: {name}",
    ),
}


def classify_upload_kind(filename: str) -> UploadKind | None:
    suf = Path(filename).suffix.lower()
    for kind, rule in _RULES.items():
        if suf in rule.extensions:
            return kind
    return None


async def collect_mixed_ordered_items(
    files: list[UploadFile],
    settings: Settings,
) -> list[tuple[str, bytes, UploadKind]]:
    _raise_if_empty(files, "1 件以上のファイルが必要です")
    ordered: list[tuple[str, bytes, UploadKind]] = []
    counts: dict[UploadKind, int] = {k: 0 for k in _RULES}
    hint = " ".join(sorted({e for r in _RULES.values() for e in r.extensions}))
    for f in files:
        raw_name = f.filename or ""
        name = Path(raw_name).name or "unnamed"
        kind = classify_upload_kind(name)
        if kind is None:
            raise HTTPException(
                status_code=422,
                detail=f"未対応の拡張子です: {name}（{hint} のみ）",
            )
        rule = _RULES[kind]
        data = await f.read()
        max_b = rule.max_bytes(settings)
        _raise_if_too_large(len(data), max_b, rule.mixed_file_size_detail(max_b, name))
        counts[kind] += 1
        ordered.append((name, data, kind))
    _enforce_batch_limits(settings, counts)
    return ordered


def _raise_if_empty(files: list[UploadFile], detail: str) -> None:
    if not files:
        raise HTTPException(status_code=422, detail=detail)


def _raise_if_count_exceeded(n: int, max_n: int, detail_fn: Callable[[int, int], str]) -> None:
    if n > max_n:
        raise HTTPException(status_code=413, detail=detail_fn(n, max_n))


def _raise_if_too_large(size: int, max_b: int, detail: str) -> None:
    if size > max_b:
        raise HTTPException(status_code=413, detail=detail)


def _enforce_batch_limits(settings: Settings, counts: Mapping[UploadKind, int]) -> None:
    for kind, n in counts.items():
        rule = _RULES[kind]
        _raise_if_count_exceeded(n, rule.max_files(settings), rule.batch_count_mixed)
