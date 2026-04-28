from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

_PREFS_PATH = Path("./polychat_prefs.json")
_PREF_KEYS = ("llm_provider", "llm_model", "embeddings_provider", "temperature")


def load_prefs() -> dict[str, object]:
    """Read prefs from disk. Returns {} if file is missing or corrupt."""
    try:
        return {k: v for k, v in json.loads(_PREFS_PATH.read_text()).items() if k in _PREF_KEYS}
    except Exception:
        return {}


def save_prefs(state: Mapping[str, object]) -> None:
    """Write only _PREF_KEYS subset of state to disk."""
    _PREFS_PATH.write_text(json.dumps({k: state[k] for k in _PREF_KEYS if k in state}, indent=2))
