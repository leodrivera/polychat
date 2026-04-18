"""Vector-store factory tests.

Only FAISS is exercised here — Chroma and Qdrant would start their own
storage processes, which is out of scope for fast unit tests. The persistence
fingerprint logic is the meaningful bit and is covered backend-agnostically.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from polychat.rag.vector_store import (
    EmbeddingsMismatchError,
    VectorStoreFactory,
)


def _docs() -> list[Document]:
    return [
        Document(page_content="apple pie recipe", metadata={"source": "a.txt"}),
        Document(page_content="banana bread recipe", metadata={"source": "b.txt"}),
        Document(page_content="car maintenance manual", metadata={"source": "c.txt"}),
    ]


def test_ephemeral_faiss_does_not_create_persist_dir(
    fake_embeddings: Embeddings, tmp_path: Path
) -> None:
    persist_dir = tmp_path / "faiss"
    factory = VectorStoreFactory(backend="faiss", persist=False, persist_dir=persist_dir)
    store = factory.build(_docs(), fake_embeddings, embeddings_fingerprint="test:v1")
    assert store is not None
    assert not persist_dir.exists()


def test_persisted_faiss_round_trip(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    persist_dir = tmp_path / "faiss"
    factory = VectorStoreFactory(backend="faiss", persist=True, persist_dir=persist_dir)
    factory.build(_docs(), fake_embeddings, embeddings_fingerprint="test:v1")

    reloaded = factory.load(fake_embeddings, embeddings_fingerprint="test:v1")
    assert reloaded is not None
    results = reloaded.similarity_search("apple", k=1)
    assert results
    assert "apple" in results[0].page_content


def test_load_returns_none_when_nothing_persisted(
    fake_embeddings: Embeddings, tmp_path: Path
) -> None:
    factory = VectorStoreFactory(backend="faiss", persist=True, persist_dir=tmp_path / "nope")
    assert factory.load(fake_embeddings) is None


def test_mismatched_fingerprint_rejects_reuse(fake_embeddings: Embeddings, tmp_path: Path) -> None:
    persist_dir = tmp_path / "faiss"
    factory = VectorStoreFactory(backend="faiss", persist=True, persist_dir=persist_dir)
    factory.build(_docs(), fake_embeddings, embeddings_fingerprint="openai:v1")

    with pytest.raises(EmbeddingsMismatchError):
        factory.load(fake_embeddings, embeddings_fingerprint="hf:v1")


def test_zero_documents_is_rejected(fake_embeddings: Embeddings) -> None:
    factory = VectorStoreFactory(backend="faiss", persist=False)
    with pytest.raises(ValueError, match="zero documents"):
        factory.build([], fake_embeddings)
