"""Tests for the i18n helpers."""

from __future__ import annotations

import json
from pathlib import Path

from polychat import i18n


def test_available_locales_discovers_files(tiny_locale_dir: Path) -> None:
    locales = i18n.available_locales()
    assert locales == {"en": "English", "pt_br": "Português (Brasil)"}


def test_t_returns_active_locale_string(tiny_locale_dir: Path) -> None:
    i18n.set_locale("pt_br")
    assert i18n.t("app.title") == "Olá"


def test_t_falls_back_to_english_for_missing_key(tiny_locale_dir: Path) -> None:
    i18n.set_locale("pt_br")
    # "greeting" only exists in en.json
    assert i18n.t("greeting", name="Leo") == "Hi, Leo!"


def test_t_returns_key_when_missing_everywhere(tiny_locale_dir: Path) -> None:
    i18n.set_locale("en")
    assert i18n.t("does.not.exist") == "does.not.exist"


def test_set_locale_rejects_unknown(tiny_locale_dir: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown locale"):
        i18n.set_locale("xx_yy")


def test_adding_new_locale_requires_no_code_change(tiny_locale_dir: Path) -> None:
    (tiny_locale_dir / "es.json").write_text(
        json.dumps({"_meta": {"name": "Español", "code": "es"}, "app.title": "Hola"}),
        encoding="utf-8",
    )
    i18n.clear_cache()
    locales = i18n.available_locales()
    assert locales["es"] == "Español"
    i18n.set_locale("es")
    assert i18n.t("app.title") == "Hola"
