# 取り込み全文の正規化（logic.normalizer）
from __future__ import annotations

from app.rag.logic.normalizer import normalize_document_text


def test_normalize_unifies_crlf_and_limits_blank_lines() -> None:
    out = normalize_document_text("a\r\nb\n\n\nc")
    assert "a\nb" in out
    assert "\n\n\n" not in out


def test_normalize_strips_invisible_chars() -> None:
    s = "x" + chr(0x200B) + "y"
    out = normalize_document_text(s)
    assert chr(0x200B) not in out
    assert "xy" in out


def test_normalize_collapses_horizontal_space_runs() -> None:
    assert normalize_document_text("a  \t  b") == "a b"


def test_normalize_merges_intrusive_linebreaks() -> None:
    assert normalize_document_text("当\n社\nは") == "当社は"


def test_normalize_collapses_intraword_spaces() -> None:
    assert normalize_document_text("株 式 会 社") == "株式会社"


def test_normalize_keeps_space_after_sentence_punct() -> None:
    out = normalize_document_text("ある。 次")
    assert "。" in out and "次" in out


def test_normalize_keeps_newline_after_sentence_punctuation() -> None:
    out = normalize_document_text("ある。\n次")
    assert "。\n次" in out
