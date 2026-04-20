"""Tests for the LLM and embeddings factories."""

from __future__ import annotations

from typing import Any

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


@pytest.fixture
def fake_chat_ollama(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChatOllama)
    return captured


def test_get_llm_ollama_forwards_advanced_params(fake_chat_ollama: dict[str, Any]) -> None:
    llm.get_llm(
        "ollama",
        model="llama3.1:8b",
        ollama_num_ctx=4096,
        ollama_num_predict=512,
        ollama_keep_alive="5m",
    )
    assert fake_chat_ollama["num_ctx"] == 4096
    assert fake_chat_ollama["num_predict"] == 512
    assert fake_chat_ollama["keep_alive"] == "5m"


def test_get_llm_ollama_omits_none_advanced_params(fake_chat_ollama: dict[str, Any]) -> None:
    llm.get_llm("ollama", model="llama3.1:8b")
    assert "num_ctx" not in fake_chat_ollama
    assert "num_predict" not in fake_chat_ollama
    assert "keep_alive" not in fake_chat_ollama


def test_embeddings_fingerprint_distinguishes_providers() -> None:
    assert embeddings.fingerprint("openai") != embeddings.fingerprint("huggingface_local")


def test_embeddings_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(embeddings.MissingAPIKeyError, match="OpenAI"):
        embeddings.get_embeddings("openai")
