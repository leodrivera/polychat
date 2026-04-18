"""YouTube transcript loader.

Uses ``youtube-transcript-api`` v1.2.4+ directly instead of LangChain's
``YoutubeLoader`` for tighter control over language fallback, auto-vs-manual
transcript preference, proxy configuration, and error mapping.

Each transcript snippet becomes its own LangChain ``Document`` whose
``metadata['source']`` is a deep link to the exact timestamp, so citations can
jump straight into the video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MIN_PATH_PARTS_FOR_ID = 2
_PATH_PREFIXES_WITH_ID = {"shorts", "embed", "v", "live"}


class YouTubeLoaderError(Exception):
    """Base class for translated YouTube loader failures."""


class InvalidYouTubeURLError(YouTubeLoaderError):
    """The given URL is not a recognisable YouTube video URL."""


class TranscriptUnavailableError(YouTubeLoaderError):
    """The video exists but has no accessible transcript."""


class VideoUnavailableError(YouTubeLoaderError):
    """The video itself is private, deleted, or otherwise inaccessible."""


class RequestBlockedError(YouTubeLoaderError):
    """YouTube blocked the request (IP ban, rate limit, etc.)."""


class AgeRestrictedError(YouTubeLoaderError):
    """The video is age-restricted and requires authentication."""


@runtime_checkable
class _YouTubeTranscriptAPILike(Protocol):
    """The tiny slice of the library's API we rely on. Helps testing."""

    def list(self, video_id: str) -> Any: ...


@dataclass(frozen=True)
class YouTubeFetchOptions:
    """Options for a single transcript fetch."""

    preferred_languages: list[str]
    proxy: dict[str, str] | None = None


def preferred_langs(locale: str) -> list[str]:
    """Map a UI locale to an ordered list of BCP-47 transcript languages.

    English is always appended as a safe fallback, even for non-English UIs.
    """
    base_by_locale: dict[str, list[str]] = {
        "pt_br": ["pt-BR", "pt"],
        "en": ["en"],
        "es": ["es", "es-419"],
    }
    base = base_by_locale.get(locale)
    if base is None:
        base = [locale.replace("_", "-")]
    seen: set[str] = set()
    out: list[str] = []
    for code in [*base, "en"]:
        if code not in seen:
            out.append(code)
            seen.add(code)
    return out


def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL or a bare 11-char video ID and return the ID.

    Supports ``youtube.com/watch?v=…``, ``youtu.be/…``, ``youtube.com/shorts/…``,
    and ``youtube.com/embed/…``.
    """
    text = url_or_id.strip()
    if _VIDEO_ID_RE.match(text):
        return text

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidYouTubeURLError(url_or_id)

    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate

    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            values = parse_qs(parsed.query).get("v", [])
            if values and _VIDEO_ID_RE.match(values[0]):
                return values[0]
        parts = [p for p in parsed.path.split("/") if p]
        if (
            len(parts) >= _MIN_PATH_PARTS_FOR_ID
            and parts[0] in _PATH_PREFIXES_WITH_ID
            and _VIDEO_ID_RE.match(parts[1])
        ):
            return parts[1]

    raise InvalidYouTubeURLError(url_or_id)


def load_youtube(
    url_or_id: str,
    *,
    locale: str = "en",
    proxy: dict[str, str] | None = None,
    api: _YouTubeTranscriptAPILike | None = None,
) -> list[Document]:
    """Fetch a video's transcript and return one Document per snippet.

    ``api`` is injectable so tests can supply a fake. In production it's
    built on demand from :func:`_build_api`.
    """
    video_id = extract_video_id(url_or_id)
    client = api if api is not None else _build_api(proxy)
    transcript = _select_transcript(client, video_id, preferred_langs(locale))
    fetched = _invoke(transcript, "fetch")

    language_code = getattr(fetched, "language_code", None) or ""
    is_generated = bool(getattr(fetched, "is_generated", False))

    docs: list[Document] = []
    for snippet in fetched:
        text = getattr(snippet, "text", "")
        if not text:
            continue
        start = float(getattr(snippet, "start", 0.0))
        duration = float(getattr(snippet, "duration", 0.0))
        source = f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "video_id": video_id,
                    "language": language_code,
                    "is_generated": is_generated,
                    "start": start,
                    "duration": duration,
                },
            )
        )
    if not docs:
        raise TranscriptUnavailableError(f"Empty transcript for video {video_id}.")
    return docs


def _build_api(proxy: dict[str, str] | None) -> _YouTubeTranscriptAPILike:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.proxies import GenericProxyConfig

    proxy_config = None
    if proxy and (proxy.get("http") or proxy.get("https")):
        proxy_config = GenericProxyConfig(
            http_url=proxy.get("http") or None,
            https_url=proxy.get("https") or None,
        )
    return YouTubeTranscriptApi(proxy_config=proxy_config)


def _select_transcript(
    api: _YouTubeTranscriptAPILike,
    video_id: str,
    languages: list[str],
) -> Any:
    try:
        transcript_list = api.list(video_id)
    except Exception as exc:
        _translate_error(exc)

    try:
        return transcript_list.find_manually_created_transcript(languages)
    except Exception:
        try:
            return transcript_list.find_generated_transcript(languages)
        except Exception as exc:
            _translate_error(exc)


def _invoke(obj: Any, method: str) -> Any:
    try:
        return getattr(obj, method)()
    except Exception as exc:
        _translate_error(exc)


def _translate_error(exc: BaseException) -> NoReturn:
    """Map library exceptions to our own taxonomy. Always re-raises."""
    # Import lazily so tests can run without the library installed.
    try:
        from youtube_transcript_api import (
            AgeRestricted,
            IpBlocked,
            NoTranscriptFound,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError:
        raise YouTubeLoaderError(str(exc)) from exc

    if isinstance(exc, VideoUnavailable):
        raise VideoUnavailableError(str(exc)) from exc
    if isinstance(exc, AgeRestricted):
        raise AgeRestrictedError(str(exc)) from exc
    if isinstance(exc, RequestBlocked | IpBlocked):
        raise RequestBlockedError(str(exc)) from exc
    if isinstance(exc, TranscriptsDisabled | NoTranscriptFound):
        raise TranscriptUnavailableError(str(exc)) from exc

    raise YouTubeLoaderError(str(exc)) from exc


__all__ = [
    "AgeRestrictedError",
    "InvalidYouTubeURLError",
    "RequestBlockedError",
    "TranscriptUnavailableError",
    "VideoUnavailableError",
    "YouTubeFetchOptions",
    "YouTubeLoaderError",
    "extract_video_id",
    "load_youtube",
    "preferred_langs",
]
