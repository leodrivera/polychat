"""LLM factory: pluggable OpenAI / Anthropic / Groq / Ollama."""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel

from .embeddings import MissingAPIKeyError

LLMProvider = Literal["openai", "anthropic", "groq", "ollama"]

DEFAULT_MODELS: dict[LLMProvider, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    "ollama": ["llama3.1:8b", "mistral:7b", "phi3:mini", "qwen2.5:7b"],
}


def available_models(provider: LLMProvider) -> list[str]:
    """Return the default model list surfaced in the UI for ``provider``."""
    return list(DEFAULT_MODELS[provider])


def get_llm(
    provider: LLMProvider,
    *,
    model: str,
    temperature: float = 0.3,
) -> BaseChatModel:
    """Build the LangChain ``BaseChatModel`` for ``provider``.

    Raises :class:`MissingAPIKeyError` when a cloud provider's env key is absent.
    """
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OpenAI")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise MissingAPIKeyError("Anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=model, temperature=temperature, timeout=60, stop=None)

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise MissingAPIKeyError("Groq")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    raise ValueError(f"Unknown LLM provider: {provider!r}")


__all__ = ["DEFAULT_MODELS", "LLMProvider", "available_models", "get_llm"]
