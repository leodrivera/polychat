"""Embeddings factory.

Two providers:

- ``openai``             → cloud ``text-embedding-3-small`` (requires ``OPENAI_API_KEY``)
- ``huggingface_local``  → ``paraphrase-multilingual-MiniLM-L12-v2`` on CPU, free/offline
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.embeddings import Embeddings

EmbeddingsProvider = Literal["openai", "huggingface_local"]

OPENAI_MODEL = "text-embedding-3-small"
HUGGINGFACE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider needs an API key that isn't set."""


def get_embeddings(provider: EmbeddingsProvider) -> Embeddings:
    """Build and return the LangChain ``Embeddings`` for ``provider``."""
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OpenAI")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=OPENAI_MODEL)

    if provider == "huggingface_local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def fingerprint(provider: EmbeddingsProvider) -> str:
    """A stable identifier for the embedding model, used to detect mismatches
    between a persisted index and the currently selected provider.
    """
    if provider == "openai":
        return f"openai:{OPENAI_MODEL}"
    if provider == "huggingface_local":
        return f"hf:{HUGGINGFACE_MODEL}"
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


__all__ = [
    "HUGGINGFACE_MODEL",
    "OPENAI_MODEL",
    "EmbeddingsProvider",
    "MissingAPIKeyError",
    "fingerprint",
    "get_embeddings",
]
