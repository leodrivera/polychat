"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings


class DeterministicEmbeddings(Embeddings):
    """Tiny embedding that hashes text into a fixed-length float vector.

    Not semantically meaningful — only used to exercise the vector-store
    round-trip in unit tests without downloading any model.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        buckets = [0.0] * self.dim
        for i, ch in enumerate(text):
            buckets[(ord(ch) + i) % self.dim] += 1.0
        # Normalize so cosine-similarity works predictably.
        total = sum(x * x for x in buckets) ** 0.5 or 1.0
        return [x / total for x in buckets]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


@pytest.fixture
def fake_embeddings() -> Embeddings:
    return DeterministicEmbeddings()


@pytest.fixture
def tiny_locale_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Spin up a locale directory with just ``en.json`` and ``pt_br.json``."""
    d = tmp_path / "locales"
    d.mkdir()
    (d / "en.json").write_text(
        json.dumps(
            {
                "_meta": {"name": "English", "code": "en"},
                "app.title": "Hello",
                "greeting": "Hi, {name}!",
            }
        ),
        encoding="utf-8",
    )
    (d / "pt_br.json").write_text(
        json.dumps(
            {
                "_meta": {"name": "Português (Brasil)", "code": "pt_br"},
                "app.title": "Olá",
                # deliberately missing "greeting" to test fallback
            }
        ),
        encoding="utf-8",
    )
    from polychat import i18n

    monkeypatch.setattr(i18n, "_LOCALES_DIR", d)
    monkeypatch.setattr(i18n, "_locales_dir", lambda: d)
    i18n.clear_cache()
    i18n.set_locale("en")
    yield d
    i18n.clear_cache()


class _FakeSnippet:
    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


class _FakeFetched:
    def __init__(
        self, language_code: str, is_generated: bool, snippets: list[_FakeSnippet]
    ) -> None:
        self.language_code = language_code
        self.is_generated = is_generated
        self._snippets = snippets

    def __iter__(self) -> Any:
        return iter(self._snippets)


class _FakeTranscript:
    def __init__(self, fetched: _FakeFetched) -> None:
        self._fetched = fetched

    def fetch(self) -> _FakeFetched:
        return self._fetched


class FakeTranscriptList:
    """Test double for ``TranscriptList`` with pluggable behaviour."""

    def __init__(
        self,
        *,
        manual: _FakeTranscript | None = None,
        generated: _FakeTranscript | None = None,
        manual_error: BaseException | None = None,
        generated_error: BaseException | None = None,
    ) -> None:
        self._manual = manual
        self._generated = generated
        self._manual_error = manual_error
        self._generated_error = generated_error

    def find_manually_created_transcript(self, _langs: list[str]) -> _FakeTranscript:
        if self._manual_error is not None:
            raise self._manual_error
        if self._manual is None:
            raise RuntimeError("no manual transcript")
        return self._manual

    def find_generated_transcript(self, _langs: list[str]) -> _FakeTranscript:
        if self._generated_error is not None:
            raise self._generated_error
        if self._generated is None:
            raise RuntimeError("no generated transcript")
        return self._generated


class FakeYouTubeAPI:
    """Minimal fake implementing our ``_YouTubeTranscriptAPILike`` protocol."""

    def __init__(
        self,
        *,
        transcript_list: FakeTranscriptList | None = None,
        list_error: BaseException | None = None,
    ) -> None:
        self._transcript_list = transcript_list
        self._list_error = list_error

    def list(self, video_id: str) -> FakeTranscriptList:
        if self._list_error is not None:
            raise self._list_error
        assert self._transcript_list is not None
        return self._transcript_list


def make_transcript(
    *,
    language_code: str = "en",
    is_generated: bool = False,
    snippets: list[tuple[str, float, float]] | None = None,
) -> _FakeTranscript:
    default = [("hello world", 0.0, 2.0), ("second line", 2.0, 2.5)]
    items = default if snippets is None else snippets
    return _FakeTranscript(
        _FakeFetched(
            language_code=language_code,
            is_generated=is_generated,
            snippets=[_FakeSnippet(text=t, start=s, duration=d) for t, s, d in items],
        )
    )
