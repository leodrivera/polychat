"""Tests for the file loaders (TXT + CSV).

PDF loading is exercised via the integration suite — generating a valid PDF
purely in unit-test territory would require pulling in a heavy dep.
"""

from __future__ import annotations

import io

from polychat.rag.loaders.files import load_files


def _named_bytes(name: str, data: bytes) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name  # type: ignore[attr-defined]
    return buf


def test_txt_loader_sets_source_and_reads_content() -> None:
    fh = _named_bytes("notes.txt", b"hello world\nsecond line\n")
    docs = load_files(kind="txt", files=[fh])
    assert len(docs) == 1
    assert "hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == "notes.txt"


def test_csv_loader_returns_row_per_document() -> None:
    csv_bytes = b"name,age\nAlice,30\nBob,25\n"
    fh = _named_bytes("people.csv", csv_bytes)
    docs = load_files(kind="csv", files=[fh])
    assert len(docs) == 2
    assert any("Alice" in d.page_content for d in docs)
    assert all(d.metadata["source"] == "people.csv" for d in docs)


def test_loader_rejects_empty_input() -> None:
    import pytest

    with pytest.raises(ValueError, match="At least one file"):
        load_files(kind="txt", files=[])
