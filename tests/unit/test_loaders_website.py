"""Website loader URL validation tests.

Network-dependent behaviour lives in the integration suite.
"""

from __future__ import annotations

import pytest

from polychat.rag.loaders.website import InvalidURLError, load_website


@pytest.mark.parametrize("url", ["", "not-a-url", "ftp://example.com", "example.com"])
def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(InvalidURLError):
        load_website(url)
