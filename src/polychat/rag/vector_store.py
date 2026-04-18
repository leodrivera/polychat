"""Pluggable vector-store factory.

Three backends behind one interface:

- ``chroma`` (default) — cheapest switch between ephemeral and persistent modes;
  ``persist=True`` creates a ``chroma_db/`` directory.
- ``faiss`` — fastest in-memory; persistence uses ``save_local`` / ``load_local``.
- ``qdrant`` — in-memory via ``:memory:`` or file-backed via a directory.

A tiny JSON sidecar (``.fingerprint``) records which embedding model produced
the index so we can refuse to reuse it with a mismatched embedder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

VectorStoreBackend = Literal["chroma", "faiss", "qdrant"]

_FINGERPRINT_FILE = ".fingerprint"


class EmbeddingsMismatchError(RuntimeError):
    """A persisted index was built with a different embedding model."""


class VectorStoreFactory:
    """Builder for the supported backends.

    Parameters
    ----------
    backend:
        Which library to use. Defaults to ``"chroma"``.
    persist:
        When ``True``, write the index to ``persist_dir`` and reload it on
        subsequent runs. When ``False``, the index lives only in memory.
    persist_dir:
        Directory for on-disk storage. Ignored when ``persist=False``.
    """

    def __init__(
        self,
        backend: VectorStoreBackend = "chroma",
        *,
        persist: bool = False,
        persist_dir: str | Path = "./chroma_db",
    ) -> None:
        self.backend: VectorStoreBackend = backend
        self.persist: bool = persist
        self.persist_dir: Path = Path(persist_dir)

    # ------------------------------------------------------------------ build

    def build(
        self,
        docs: list[Document],
        embeddings: Embeddings,
        *,
        embeddings_fingerprint: str | None = None,
    ) -> VectorStore:
        """Index ``docs`` and return a ready-to-query :class:`VectorStore`."""
        if not docs:
            raise ValueError("Cannot build a vector store from zero documents.")

        if self.persist:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "chroma":
            store: VectorStore = self._build_chroma(docs, embeddings)
        elif self.backend == "faiss":
            store = self._build_faiss(docs, embeddings)
        elif self.backend == "qdrant":
            store = self._build_qdrant(docs, embeddings)
        else:
            raise ValueError(f"Unknown vector store backend: {self.backend!r}")

        if self.persist and embeddings_fingerprint is not None:
            self._write_fingerprint(embeddings_fingerprint)
        return store

    # ------------------------------------------------------------------- load

    def load(
        self,
        embeddings: Embeddings,
        *,
        embeddings_fingerprint: str | None = None,
    ) -> VectorStore | None:
        """Return a previously persisted index, or ``None`` if none exists.

        Raises :class:`EmbeddingsMismatchError` if the persisted fingerprint
        doesn't match ``embeddings_fingerprint``.
        """
        if not self.persist or not self.persist_dir.is_dir():
            return None

        stored = self._read_fingerprint()
        if (
            embeddings_fingerprint is not None
            and stored is not None
            and stored != embeddings_fingerprint
        ):
            raise EmbeddingsMismatchError(
                f"Persisted index was built with {stored!r} "
                f"but current embedder is {embeddings_fingerprint!r}."
            )

        if self.backend == "chroma":
            return self._load_chroma(embeddings)
        if self.backend == "faiss":
            return self._load_faiss(embeddings)
        if self.backend == "qdrant":
            return self._load_qdrant(embeddings)
        raise ValueError(f"Unknown vector store backend: {self.backend!r}")

    # ---------------------------------------------------------------- Chroma

    def _build_chroma(self, docs: list[Document], embeddings: Embeddings) -> VectorStore:
        from langchain_chroma import Chroma

        if self.persist:
            return Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=str(self.persist_dir),
            )
        return Chroma.from_documents(documents=docs, embedding=embeddings)

    def _load_chroma(self, embeddings: Embeddings) -> VectorStore | None:
        from langchain_chroma import Chroma

        if not any(self.persist_dir.iterdir()):
            return None
        return Chroma(persist_directory=str(self.persist_dir), embedding_function=embeddings)

    # ----------------------------------------------------------------- FAISS

    def _build_faiss(self, docs: list[Document], embeddings: Embeddings) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(documents=docs, embedding=embeddings)
        if self.persist:
            store.save_local(str(self.persist_dir))
        return store

    def _load_faiss(self, embeddings: Embeddings) -> VectorStore | None:
        from langchain_community.vectorstores import FAISS

        index_file = self.persist_dir / "index.faiss"
        if not index_file.is_file():
            return None
        return FAISS.load_local(
            str(self.persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # ---------------------------------------------------------------- Qdrant

    def _build_qdrant(self, docs: list[Document], embeddings: Embeddings) -> VectorStore:
        from langchain_qdrant import QdrantVectorStore

        collection = "polychat"
        if self.persist:
            return QdrantVectorStore.from_documents(
                documents=docs,
                embedding=embeddings,
                path=str(self.persist_dir),
                collection_name=collection,
            )
        return QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            location=":memory:",
            collection_name=collection,
        )

    def _load_qdrant(self, embeddings: Embeddings) -> VectorStore | None:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        collection = "polychat"
        client = QdrantClient(path=str(self.persist_dir))
        collections = {c.name for c in client.get_collections().collections}
        if collection not in collections:
            return None
        return QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)

    # ---------------------------------------------------------- fingerprint

    def _fingerprint_path(self) -> Path:
        return self.persist_dir / _FINGERPRINT_FILE

    def _write_fingerprint(self, value: str) -> None:
        self._fingerprint_path().write_text(json.dumps({"embeddings": value}), encoding="utf-8")

    def _read_fingerprint(self) -> str | None:
        path = self._fingerprint_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = data.get("embeddings")
        return None if not isinstance(value, str) else value


__all__ = [
    "EmbeddingsMismatchError",
    "VectorStoreBackend",
    "VectorStoreFactory",
]
