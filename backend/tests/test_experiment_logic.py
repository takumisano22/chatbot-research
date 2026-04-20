from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.experiment.logic_fingerprints import get_logic_fingerprints, validate_logic_fingerprints


def test_get_logic_fingerprints_has_three_keys() -> None:
    fp = get_logic_fingerprints()
    assert set(fp.keys()) == {
        "chunking_logic_id",
        "tokenizer_logic_id",
        "search_logic_id",
    }
    for v in fp.values():
        assert len(v) == 64


def test_validate_logic_rejects_mismatch() -> None:
    fp = get_logic_fingerprints()
    bad = {**fp, "chunking_logic_id": "0" * 64}
    with pytest.raises(HTTPException) as ei:
        validate_logic_fingerprints(bad)
    assert ei.value.status_code == 400


def test_validate_logic_accepts_current_files() -> None:
    validate_logic_fingerprints(get_logic_fingerprints())
