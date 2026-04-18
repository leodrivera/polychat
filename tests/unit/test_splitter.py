"""Tests for the chunking layer."""

from __future__ import annotations

from langchain_core.documents import Document

from polychat.rag.splitter import split_documents


def test_empty_input_returns_empty_list() -> None:
    assert split_documents([]) == []


def test_long_document_is_split() -> None:
    text = ("paragraph " * 300).strip()
    docs = [Document(page_content=text, metadata={"source": "doc.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.metadata["source"] == "doc.txt"


def test_short_document_is_not_fragmented() -> None:
    docs = [Document(page_content="short", metadata={"source": "x.txt"})]
    chunks = split_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "short"
