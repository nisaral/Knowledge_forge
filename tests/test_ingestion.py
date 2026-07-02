import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch

from services.ingestion import _extract_youtube_id, get_youtube_transcript, ingest_text_source, is_youtube_url


class TestYouTubeParsing:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("not-a-url", None),
    ])
    def test_extract_id(self, url, expected):
        assert _extract_youtube_id(url) == expected

    def test_transcript_fetch_live(self):
        """Requires network + youtube-transcript-api >= 1.2."""
        text, err = get_youtube_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
        assert err == ""
        assert text and len(text) > 20

    def test_playlist_url_extracts_video_id(self):
        url = "https://www.youtube.com/watch?v=dAQVTNnRrEg&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=33"
        assert _extract_youtube_id(url) == "dAQVTNnRrEg"
        assert is_youtube_url(url)

    def test_web_type_auto_detects_youtube(self):
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        with patch("services.ingestion.get_youtube_transcript", return_value=("hello transcript text " * 10, "")):
            text, detail, stype = ingest_text_source(url, "web")
            assert stype == "youtube"
            assert detail == ""
            assert text and "hello" in text

    def test_web_crawler_rejects_youtube(self):
        from services.ingestion import crawl_url
        assert crawl_url("https://www.youtube.com/watch?v=jNQXAC9IVRw") is None