"""Tests for the Chroma-only vector store module."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from polychat.rag.vector_store import (
    EmbeddingsMismatchError,
    has_persisted_index,
    open_chroma_index,
    read_fingerprint,
    reset_chroma_index,
)


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


def test_ephemeral_does_not_create_persist_dir(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    open_chroma_index(
        embeddings=fake_embeddings,
        embeddings_fingerprint="test:v1",
        persist_dir=None,
    )
    assert not d.exists()


def test_open_creates_index_and_fingerprint(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    open_chroma_index(
        embeddings=fake_embeddings,
        embeddings_fingerprint="test:v1",
        persist_dir=d,
    )
    assert (d / "chroma.sqlite3").is_file()
    assert (d / ".fingerprint").is_file()
    assert read_fingerprint(d) == "test:v1"


def test_has_persisted_index_false_for_empty_dir(tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    d.mkdir()
    assert not has_persisted_index(d)


def test_has_persisted_index_true_after_build(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    assert has_persisted_index(d)


def test_round_trip_search(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    idx = open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    idx.add_documents([_doc("apple pie recipe")])

    idx2 = open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    results = idx2.similarity_search("apple", k=1)
    assert results
    assert "apple" in results[0].page_content


def test_append_grows_index(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    idx1 = open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    idx1.add_documents([_doc("doc A")])

    idx2 = open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    idx2.add_documents([_doc("doc B")])

    idx3 = open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    results = idx3.similarity_search("doc", k=5)
    contents = {r.page_content for r in results}
    assert "doc A" in contents
    assert "doc B" in contents


def test_mismatched_fingerprint_raises(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="openai:v1", persist_dir=d)
    with pytest.raises(EmbeddingsMismatchError):
        open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="hf:v1", persist_dir=d)


def test_reset_clears_directory(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    open_chroma_index(embeddings=fake_embeddings, embeddings_fingerprint="v1", persist_dir=d)
    assert has_persisted_index(d)
    reset_chroma_index(d)
    assert not has_persisted_index(d)
    assert read_fingerprint(d) is None


def test_read_fingerprint_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_fingerprint(tmp_path / "nonexistent") is None
