"""Tests for the LLM and embeddings factories."""

from __future__ import annotations

import pytest

from polychat.rag import embeddings, llm


def test_available_models_returns_copies() -> None:
    a = llm.available_models("openai")
    a.clear()
    b = llm.available_models("openai")
    assert b  # the factory's DEFAULT_MODELS was not mutated


def test_get_llm_raises_missing_key_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(embeddings.MissingAPIKeyError, match="OpenAI"):
        llm.get_llm("openai", model="gpt-4o")


def test_get_llm_raises_missing_key_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(embeddings.MissingAPIKeyError, match="Anthropic"):
        llm.get_llm("anthropic", model="claude-sonnet-4-6")


def test_get_llm_raises_missing_key_for_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(embeddings.MissingAPIKeyError, match="Groq"):
        llm.get_llm("groq", model="llama-3.1-8b-instant")


def test_embeddings_fingerprint_distinguishes_providers() -> None:
    assert embeddings.fingerprint("openai") != embeddings.fingerprint("huggingface_local")


def test_embeddings_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(embeddings.MissingAPIKeyError, match="OpenAI"):
        embeddings.get_embeddings("openai")
