"""Chroma vector store — single backend, two modes (ephemeral / persistent).

Public API
----------
open_chroma_index   — open or create an index; idempotent.
has_persisted_index — True if a non-empty Chroma index exists on disk.
read_fingerprint    — read the embeddings fingerprint sidecar.
reset_chroma_index  — delete the persist_dir entirely.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

_FINGERPRINT_FILE = ".fingerprint"
_COLLECTION_NAME = "polychat"


class EmbeddingsMismatchError(RuntimeError):
    """Persisted index was built with a different embedder."""


def open_chroma_index(
    *,
    embeddings: Embeddings,
    embeddings_fingerprint: str,
    persist_dir: Path | None = None,
) -> Chroma:
    """Open (or create) a Chroma index. Idempotent for the same arguments.

    Parameters
    ----------
    embeddings:
        The embedding function to attach to the index.
    embeddings_fingerprint:
        A short string identifying the embedder (e.g. ``"openai:text-embedding-3-small"``).
        Written to a ``.fingerprint`` sidecar on creation; checked on reopen.
    persist_dir:
        ``None`` → in-memory (ephemeral, lost on process exit).
        ``Path`` → file-backed via Chroma's SQLite store.

    Raises
    ------
    EmbeddingsMismatchError
        If *persist_dir* contains an index built with a different fingerprint.
    """
    if persist_dir is None:
        return Chroma(collection_name=_COLLECTION_NAME, embedding_function=embeddings)

    persist_dir.mkdir(parents=True, exist_ok=True)
    stored = _read_fingerprint(persist_dir)
    if stored is not None and stored != embeddings_fingerprint:
        raise EmbeddingsMismatchError(
            f"Persisted index was built with {stored!r}, "
            f"current embedder is {embeddings_fingerprint!r}."
        )
    if stored is None:
        _write_fingerprint(persist_dir, embeddings_fingerprint)

    return Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def has_persisted_index(persist_dir: Path) -> bool:
    """True iff *persist_dir* contains a non-empty Chroma index."""
    return persist_dir.is_dir() and (persist_dir / "chroma.sqlite3").is_file()


def read_fingerprint(persist_dir: Path) -> str | None:
    """Return the stored embeddings fingerprint, or ``None`` if absent."""
    return _read_fingerprint(persist_dir)


def reset_chroma_index(persist_dir: Path) -> None:
    """Delete *persist_dir* and all its contents.

    The caller **must** clear any ``st.cache_resource`` handle that points at
    this directory before calling this function; otherwise cached objects will
    reference deleted files.
    """
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


# ---------------------------------------------------------------- private


def _fingerprint_path(persist_dir: Path) -> Path:
    return persist_dir / _FINGERPRINT_FILE


def _write_fingerprint(persist_dir: Path, value: str) -> None:
    _fingerprint_path(persist_dir).write_text(json.dumps({"embeddings": value}), encoding="utf-8")


def _read_fingerprint(persist_dir: Path) -> str | None:
    path = _fingerprint_path(persist_dir)
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
    "has_persisted_index",
    "open_chroma_index",
    "read_fingerprint",
    "reset_chroma_index",
]
