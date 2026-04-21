"""Lightweight i18n for the Streamlit UI.

Locales are plain JSON files in this package. Adding a new language means
dropping an ``xx_yy.json`` file next to the existing ones — no code changes.

The active locale code is stored in ``streamlit.session_state['locale']`` when
Streamlit is available; otherwise it lives in a module-level variable so the
helpers are usable from unit tests without a Streamlit runtime.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_LOCALE: Final[str] = "en"
_LOCALES_DIR: Final[Path] = Path(__file__).parent
_fallback_module_locale: str = DEFAULT_LOCALE


def _locales_dir() -> Path:
    """Indirection so tests can monkeypatch the locale directory."""
    return _LOCALES_DIR


@lru_cache(maxsize=32)
def _load_locale(code: str) -> dict[str, Any]:
    path = _locales_dir() / f"{code}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Locale file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def available_locales() -> dict[str, str]:
    """Return ``{code: display_name}`` discovered from JSON files on disk.

    The display name comes from each file's ``_meta.name`` field; the code
    comes from its filename.
    """
    out: dict[str, str] = {}
    for file in sorted(_locales_dir().glob("*.json")):
        code = file.stem
        try:
            data = _load_locale(code)
            meta = data.get("_meta", {})
            name = str(meta.get("name", code))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable locale %s: %s", file, exc)
            continue
        out[code] = name
    return out


def locale_flag(code: str) -> str:
    """Return the flag emoji for a locale code, or an empty string if not defined."""
    try:
        data = _load_locale(code)
        return str(data.get("_meta", {}).get("flag", ""))
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def get_active_locale() -> str:
    """Return the active locale code, using Streamlit session_state if possible."""
    try:
        import streamlit as st

        return str(st.session_state.get("locale", _fallback_module_locale))
    except (ImportError, RuntimeError):
        return _fallback_module_locale


def set_locale(code: str) -> None:
    """Set the active locale, both in Streamlit session_state and in-process."""
    global _fallback_module_locale
    if code not in available_locales():
        raise ValueError(f"Unknown locale: {code!r}")
    _fallback_module_locale = code
    try:
        import streamlit as st

        st.session_state["locale"] = code
    except (ImportError, RuntimeError):
        pass


def t(key: str, **kwargs: Any) -> str:
    """Translate ``key`` using the active locale.

    Falls back to ``DEFAULT_LOCALE`` when a key is missing, then to the literal
    key string. Values may contain ``{placeholders}`` interpolated by
    ``str.format(**kwargs)``.
    """
    active = get_active_locale()
    value = _lookup(active, key)
    if value is None and active != DEFAULT_LOCALE:
        value = _lookup(DEFAULT_LOCALE, key)
        if value is not None:
            logger.warning(
                "i18n key %r missing in locale %r; falling back to %r",
                key,
                active,
                DEFAULT_LOCALE,
            )
    if value is None:
        logger.warning("i18n key %r missing in all locales; returning the key itself", key)
        return key
    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError) as exc:
            logger.warning("i18n key %r: format error with %r: %s", key, kwargs, exc)
            return value
    return value


def _lookup(code: str, key: str) -> str | None:
    try:
        data = _load_locale(code)
    except FileNotFoundError:
        return None
    raw = data.get(key)
    return None if raw is None else str(raw)


def clear_cache() -> None:
    """Invalidate the locale cache. Useful in tests and after adding locale files."""
    _load_locale.cache_clear()


__all__ = [
    "DEFAULT_LOCALE",
    "available_locales",
    "clear_cache",
    "get_active_locale",
    "locale_flag",
    "set_locale",
    "t",
]
