"""Website loader using LangChain's WebBaseLoader + bs4."""

from __future__ import annotations

from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


class InvalidURLError(ValueError):
    """Raised when the provided URL is not a well-formed http(s) URL."""


def load_website(url: str) -> list[Document]:
    """Fetch ``url`` and return LangChain Documents.

    Raises :class:`InvalidURLError` if the URL is not http(s). Network errors
    bubble up from :class:`WebBaseLoader` unchanged so the UI can show a
    localized message.
    """
    _validate_url(url)
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", url)
    return docs


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidURLError(url)


__all__ = ["InvalidURLError", "load_website"]
