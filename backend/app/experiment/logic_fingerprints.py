from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import app.rag.logic.chunking as chunking_mod
import app.rag.logic.hybrid_search as hybrid_mod
import app.rag.logic.tokenizer as tokenizer_mod
from fastapi import HTTPException

# -----------------------------------------------------------------------------
# 役割: RAG ロジック実装ファイルの内容指紋（SHA-256）を返し、リクエスト値との一致検証を行う。
# 流れ: GET で指紋公開 → クライアントが同じ値を POST manifest に載せる → validate_logic_fingerprints。
# 要点: 実装が変われば指紋も変わり、実験条件の取り違えを防ぐ（KISS: 単一実装の検証のみ）。
# -----------------------------------------------------------------------------

_LOGIC_FILES: Final[dict[str, Path]] = {
    "chunking_logic_id": Path(chunking_mod.__file__),
    "tokenizer_logic_id": Path(tokenizer_mod.__file__),
    "search_logic_id": Path(hybrid_mod.__file__),
}


def get_logic_fingerprints() -> dict[str, str]:
    return {key: _sha256_file(path) for key, path in _LOGIC_FILES.items()}


def validate_logic_fingerprints(body: dict[str, str]) -> None:
    expected = get_logic_fingerprints()
    for key, path in _LOGIC_FILES.items():
        got = (body.get(key) or "").strip().lower()
        exp = expected[key].lower()
        if got != exp:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "logic_fingerprint_mismatch",
                    "key": key,
                    "path": str(path),
                    "expected": exp,
                    "got": got or None,
                },
            )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
