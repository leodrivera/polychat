"""File loaders for PDF / CSV / TXT uploads.

Streamlit's ``UploadedFile`` is a file-like bytes object; LangChain's loaders
want a path, so we persist each upload to a temporary file and delete it once
the loader has finished reading.
"""

from __future__ import annotations

import tempfile
from contextlib import suppress
from pathlib import Path
from typing import IO, Literal

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

FileKind = Literal["pdf", "csv", "txt"]

_SUFFIX: dict[FileKind, str] = {"pdf": ".pdf", "csv": ".csv", "txt": ".txt"}


def load_files(*, kind: FileKind, files: list[IO[bytes]]) -> list[Document]:
    """Load every uploaded file of ``kind`` and return concatenated Documents."""
    if not files:
        raise ValueError("At least one file is required.")

    docs: list[Document] = []
    for fh in files:
        filename = _guess_filename(fh, kind)
        tmp_path = _persist_to_tempfile(fh, suffix=_SUFFIX[kind])
        try:
            loader_docs = _load_one(kind, tmp_path)
        finally:
            with suppress(OSError):
                tmp_path.unlink()
        for doc in loader_docs:
            # Override whatever the loader stored (tmp-file path) with the
            # original filename so citations are meaningful.
            doc.metadata["source"] = filename
            docs.append(doc)
    return docs


def _load_one(kind: FileKind, path: Path) -> list[Document]:
    if kind == "pdf":
        return PyPDFLoader(str(path)).load()
    if kind == "csv":
        return CSVLoader(file_path=str(path), encoding="utf-8").load()
    if kind == "txt":
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file kind: {kind}")


def _persist_to_tempfile(fh: IO[bytes], *, suffix: str) -> Path:
    data = fh.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _guess_filename(fh: IO[bytes], kind: FileKind) -> str:
    name = getattr(fh, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"uploaded{_SUFFIX[kind]}"


__all__ = ["FileKind", "load_files"]
