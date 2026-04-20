"""Tests for Ollama ResponseError handling in the i18n layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LOCALE_DIR = Path("src/polychat/i18n")
_LOCALES = ["en.json", "pt_br.json"]

_OLLAMA_SIDEBAR_KEYS = [
    "sidebar.ollama.advanced",
    "sidebar.ollama.num_ctx",
    "sidebar.ollama.num_predict",
    "sidebar.ollama.keep_alive",
]


@pytest.mark.parametrize("locale_file", _LOCALES)
def test_ollama_model_not_found_key_exists(locale_file: str) -> None:
    data: dict[str, str] = json.loads((_LOCALE_DIR / locale_file).read_text(encoding="utf-8"))
    assert "errors.ollama_model_not_found" in data, (
        f"{locale_file} is missing key 'errors.ollama_model_not_found'"
    )


@pytest.mark.parametrize("locale_file", _LOCALES)
def test_ollama_model_not_found_message_contains_placeholder(locale_file: str) -> None:
    data: dict[str, str] = json.loads((_LOCALE_DIR / locale_file).read_text(encoding="utf-8"))
    msg = data.get("errors.ollama_model_not_found", "")
    assert "{model}" in msg, (
        f"{locale_file}: 'errors.ollama_model_not_found' must contain {{model}} placeholder"
    )


@pytest.mark.parametrize("locale_file", _LOCALES)
@pytest.mark.parametrize("key", _OLLAMA_SIDEBAR_KEYS)
def test_ollama_sidebar_keys_exist(locale_file: str, key: str) -> None:
    data: dict[str, str] = json.loads((_LOCALE_DIR / locale_file).read_text(encoding="utf-8"))
    assert key in data, f"{locale_file} is missing key '{key}'"
