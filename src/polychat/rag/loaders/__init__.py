"""Source-type dispatcher. The UI passes the ``kind`` and the raw inputs.

Each kind resolves to a dedicated loader module:

- ``pdf`` / ``csv`` / ``txt``  → :mod:`.files`
- ``site``                     → :mod:`.website`
- ``youtube``                  → :mod:`.youtube`

All loaders return ``list[Document]`` with ``metadata['source']`` set so the
retriever can cite it back to the user.
"""

from __future__ import annotations

from typing import IO, Literal, NoReturn, cast

from langchain_core.documents import Document

from . import files as _files
from . import website as _website
from . import youtube as _youtube
from .files import FileKind

SourceKind = Literal["pdf", "csv", "txt", "site", "youtube"]
_FILE_KINDS: set[SourceKind] = {"pdf", "csv", "txt"}


def load_source(
    kind: SourceKind,
    inputs: list[IO[bytes]] | str,
    *,
    locale: str = "en",
    proxy: dict[str, str] | None = None,
) -> list[Document]:
    """Dispatch to the right loader.

    ``inputs`` is either a list of file-like objects (for ``pdf``/``csv``/``txt``)
    or a URL string (for ``site``/``youtube``). The ``locale`` is used only by
    the YouTube loader to pick a preferred transcript language; ``proxy`` (also
    YouTube-only) is a dict with ``http``/``https`` keys.
    """
    if kind in _FILE_KINDS:
        if not isinstance(inputs, list):
            raise TypeError(
                f"{kind} loader expects a list of file-like objects, got {type(inputs).__name__}"
            )
        return _files.load_files(kind=cast(FileKind, kind), files=inputs)

    if kind == "site":
        if not isinstance(inputs, str):
            raise TypeError("site loader expects a URL string")
        return _website.load_website(inputs)

    if kind == "youtube":
        if not isinstance(inputs, str):
            raise TypeError("youtube loader expects a URL string")
        return _youtube.load_youtube(inputs, locale=locale, proxy=proxy)

    _unsupported_kind(kind)


def _unsupported_kind(value: object) -> NoReturn:
    raise ValueError(f"Unsupported source kind: {value!r}")


__all__ = ["SourceKind", "load_source"]
