from __future__ import annotations

from app.experiment.logic_registry import get_logic_registry_info


def test_get_logic_registry_info_keys() -> None:
    info = get_logic_registry_info()
    assert set(info.keys()) == {
        "chunking_logic_ids",
        "tokenizer_logic_ids",
        "search_logic_ids",
        "reranking_logic_ids",
        "prompt_logic_ids",
    }
    for k, v in info.items():
        assert "logic_01" in v
        if k == "search_logic_ids":
            assert "logic_02" in v
