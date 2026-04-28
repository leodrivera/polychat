import json
from pathlib import Path

import pytest

from polychat.rag.prefs import load_prefs, save_prefs


def test_load_prefs_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_prefs() == {}


def test_load_prefs_corrupt_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "polychat_prefs.json").write_text("not json")
    assert load_prefs() == {}


def test_save_and_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-5-haiku-20241022",
        "embeddings_provider": "openai",
        "temperature": 0.7,
        "some_other_key": "should_be_ignored",
    }
    save_prefs(state)
    loaded = load_prefs()
    assert loaded == {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-5-haiku-20241022",
        "embeddings_provider": "openai",
        "temperature": 0.7,
    }


def test_save_prefs_only_writes_pref_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    save_prefs({"llm_provider": "groq", "rag_chain": object()})
    raw = json.loads((tmp_path / "polychat_prefs.json").read_text())
    assert set(raw.keys()) == {"llm_provider"}
