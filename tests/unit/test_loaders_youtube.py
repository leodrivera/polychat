"""Tests for the YouTube loader."""

from __future__ import annotations

import pytest

from polychat.rag.loaders import youtube
from polychat.rag.loaders.youtube import (
    InvalidYouTubeURLError,
    TranscriptUnavailableError,
    VideoUnavailableError,
    extract_video_id,
    load_youtube,
    preferred_langs,
)
from tests.conftest import FakeTranscriptList, FakeYouTubeAPI, make_transcript


class TestExtractVideoID:
    @pytest.mark.parametrize(
        "text",
        [
            "dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ&t=30s",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/v/dQw4w9WgXcQ",
            "https://youtube.com/live/dQw4w9WgXcQ",
        ],
    )
    def test_accepts_all_common_formats(self, text: str) -> None:
        assert extract_video_id(text) == "dQw4w9WgXcQ"

    @pytest.mark.parametrize(
        "text",
        ["", "not a url", "https://example.com/watch?v=abc", "https://youtube.com/"],
    )
    def test_rejects_invalid(self, text: str) -> None:
        with pytest.raises(InvalidYouTubeURLError):
            extract_video_id(text)


class TestPreferredLangs:
    def test_pt_br_maps_to_portuguese_variants(self) -> None:
        assert preferred_langs("pt_br") == ["pt-BR", "pt", "en"]

    def test_english_single_entry(self) -> None:
        assert preferred_langs("en") == ["en"]

    def test_unknown_locale_derives_code_plus_english(self) -> None:
        assert preferred_langs("fr_fr") == ["fr-fr", "en"]


class TestLoadYouTube:
    def test_returns_one_document_per_snippet_with_timestamped_source(self) -> None:
        api = FakeYouTubeAPI(
            transcript_list=FakeTranscriptList(manual=make_transcript(language_code="en")),
        )
        docs = load_youtube(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            api=api,
        )
        assert len(docs) == 2
        first = docs[0]
        assert first.page_content == "hello world"
        assert first.metadata["video_id"] == "dQw4w9WgXcQ"
        assert first.metadata["source"].endswith("?v=dQw4w9WgXcQ&t=0s")
        assert docs[1].metadata["source"].endswith("&t=2s")

    def test_falls_back_to_generated_when_manual_missing(self) -> None:
        api = FakeYouTubeAPI(
            transcript_list=FakeTranscriptList(
                manual_error=RuntimeError("none"),
                generated=make_transcript(is_generated=True),
            ),
        )
        docs = load_youtube("dQw4w9WgXcQ", api=api)
        assert docs[0].metadata["is_generated"] is True

    def test_translates_no_transcript_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from youtube_transcript_api import TranscriptsDisabled

        exc = TranscriptsDisabled("dQw4w9WgXcQ")
        api = FakeYouTubeAPI(list_error=exc)
        with pytest.raises(TranscriptUnavailableError):
            load_youtube("dQw4w9WgXcQ", api=api)

    def test_translates_video_unavailable_error(self) -> None:
        from youtube_transcript_api import VideoUnavailable

        exc = VideoUnavailable("dQw4w9WgXcQ")
        api = FakeYouTubeAPI(list_error=exc)
        with pytest.raises(VideoUnavailableError):
            load_youtube("dQw4w9WgXcQ", api=api)

    def test_rejects_invalid_url(self) -> None:
        with pytest.raises(InvalidYouTubeURLError):
            load_youtube("not a url")

    def test_empty_transcript_raises(self) -> None:
        api = FakeYouTubeAPI(
            transcript_list=FakeTranscriptList(manual=make_transcript(snippets=[])),
        )
        with pytest.raises(TranscriptUnavailableError):
            load_youtube("dQw4w9WgXcQ", api=api)

    def test_module_exposes_public_error_types(self) -> None:
        # Smoke-check that the public surface is stable.
        assert issubclass(youtube.InvalidYouTubeURLError, Exception)
        assert issubclass(youtube.RequestBlockedError, Exception)
