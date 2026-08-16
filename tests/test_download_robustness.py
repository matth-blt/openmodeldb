"""
Tests for downloader robustness: atomic writes, network error wrapping,
the Google Drive "can't scan for viruses" interstitial fix, and the
MediaFire direct-link scraper.

All network I/O is faked by monkeypatching urllib.request.urlopen with
lightweight stand-ins. No real network access happens in this file.
"""
import base64
import json
import os
import urllib.error
import urllib.request

import pytest

from openmodeldb import (
    DownloadError,
    FormatNotFoundError,
    ModelNotFoundError,
    OpenModelDBError,
)
from openmodeldb.downloader import (
    _download_with_progress,
    download_direct,
    download_mediafire,
    download_mega,
    smart_download,
)

# ─── Fakes ───────────────────────────────────────────────────────────────

class _FakeResp:
    """A minimal stand-in for the object returned by urllib.request.urlopen.

    `.read()` returns the whole body on the first call and b"" thereafter
    (matching the "read until falsy" loop in _download_with_progress), and
    the object supports the `with ... as resp:` protocol used throughout
    downloader.py.
    """

    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self._served = False
        self.headers = headers or {}

    def read(self, size=-1):
        if self._served:
            return b""
        self._served = True
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FailingFirstChunkResp:
    """Serves one good chunk, then raises on the next .read() call."""

    def __init__(self, first_chunk: bytes = b"partial-bytes"):
        self._first_chunk = first_chunk
        self._served_first = False
        self.headers = {}

    def read(self, size=-1):
        if not self._served_first:
            self._served_first = True
            return self._first_chunk
        raise ConnectionError("simulated connection drop")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


GDRIVE_INTERSTITIAL_HTML = """<!DOCTYPE html>
<html><head><title>Google Drive - Virus scan warning</title></head>
<body>
<div>Google Drive can't scan this file for viruses.</div>
<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
<input type="hidden" name="id" value="FILEID123">
<input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t">
<input type="hidden" name="uuid" value="abc-def-uuid">
<button type="submit" id="uc-download-link">Download anyway</button>
</form>
</body></html>"""

GDRIVE_UNPARSEABLE_HTML = """<!DOCTYPE html>
<html><body><p>Sorry, this file is too large for Google Drive to scan.</p>
<p>No download form here, just a dead end.</p></body></html>"""


def _mega_key_str() -> str:
    """A syntactically valid 16-byte Mega key, base64url-encoded, unpadded."""
    return base64.urlsafe_b64encode(b"0123456789abcdef").decode().rstrip("=")


# ─── Atomic writes (_download_with_progress) ───────────────────────────────

def test_download_with_progress_success_leaves_dest_no_part(tmp_path):
    dest = str(tmp_path / "model.pth")
    resp = _FakeResp(b"hello world")

    _download_with_progress(resp, dest, total=None, quiet=True)

    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == b"hello world"
    assert not os.path.exists(dest + ".part")


def test_download_with_progress_failure_leaves_no_dest_no_part(tmp_path):
    dest = str(tmp_path / "model.pth")
    resp = _FailingFirstChunkResp()

    with pytest.raises(ConnectionError):
        _download_with_progress(resp, dest, total=None, quiet=True)

    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_download_with_progress_failure_does_not_clobber_existing_file(tmp_path):
    dest = str(tmp_path / "model.pth")
    with open(dest, "wb") as f:
        f.write(b"already-complete-file")

    resp = _FailingFirstChunkResp()
    with pytest.raises(ConnectionError):
        _download_with_progress(resp, dest, total=None, quiet=True)

    with open(dest, "rb") as f:
        assert f.read() == b"already-complete-file"
    assert not os.path.exists(dest + ".part")


# ─── Error wrapping ─────────────────────────────────────────────────────

def test_download_direct_wraps_urlerror_as_download_error(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError) as exc_info:
        download_direct("https://example.com/model.pth", str(tmp_path / "model.pth"))

    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)


def test_download_direct_wraps_httperror_as_download_error(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.com/model.pth", 404, "Not Found", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError) as exc_info:
        download_direct("https://example.com/model.pth", str(tmp_path / "model.pth"))

    assert isinstance(exc_info.value.__cause__, urllib.error.HTTPError)
    assert "404" in str(exc_info.value)


def test_download_direct_wraps_timeout_as_download_error(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError) as exc_info:
        download_direct("https://example.com/model.pth", str(tmp_path / "model.pth"))

    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_download_direct_normal_path_still_works(monkeypatch, tmp_path):
    """Non-Drive, non-HTML responses are downloaded directly, unaffected
    by the interstitial-sniffing logic."""
    payload = b"totally-a-model-file"

    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            payload, headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload))
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "model.pth")
    download_direct("https://example.com/model.pth", dest, quiet=True)

    with open(dest, "rb") as f:
        assert f.read() == payload
    assert not os.path.exists(dest + ".part")


# ─── download_mega error wrapping ──────────────────────────────────────

def test_download_mega_unparseable_url_raises_download_error(tmp_path):
    with pytest.raises(DownloadError):
        download_mega("https://mega.nz/not-a-valid-mega-link", str(tmp_path / "model.pth"))


def test_download_mega_api_error_raises_download_error(monkeypatch, tmp_path):
    url = f"https://mega.nz/file/FILEID#{_mega_key_str()}"

    def fake_urlopen(req, timeout=None):
        return _FakeResp(json.dumps(-9).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError, match="Mega API error"):
        download_mega(url, str(tmp_path / "model.pth"))


def test_download_mega_wraps_urlerror(monkeypatch, tmp_path):
    url = f"https://mega.nz/file/FILEID#{_mega_key_str()}"

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError) as exc_info:
        download_mega(url, str(tmp_path / "model.pth"))
    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)


# ─── download_mediafire: direct-link scraper ───────────────────────────
#
# download_mediafire fetches the MediaFire download page itself (bounded
# timeout), parses the #downloadButton anchor's href, and downloads that
# direct URL. urlopen is faked: first call serves the HTML page, the
# second serves the file.

MEDIAFIRE_PAGE_HTML = """<!DOCTYPE html>
<html><body>
<a id="downloadButton" class="download" href="https://download123.mediafire.com/abc123/model.pth">Download</a>
</body></html>"""

# Same page, but attributes in the other order and an escaped ampersand in
# the URL (as real MediaFire pages serve it).
MEDIAFIRE_PAGE_HTML_REVERSED = """<html><body>
<a href="https://download99.mediafire.com/x?a=1&amp;b=2" id="downloadButton">Download</a>
</body></html>"""

MEDIAFIRE_PAGE_NO_BUTTON = "<html><body><p>file not found</p></body></html>"

MEDIAFIRE_PAGE_JS_HREF = (
    '<html><body><a id="downloadButton" href="javascript:void(0)">Download</a></body></html>'
)


def _fake_mediafire_urlopen(monkeypatch, page_html: str, payload: bytes):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            return _FakeResp(
                page_html.encode(),
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        return _FakeResp(
            payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload))
            },
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_download_mediafire_success_path(monkeypatch, tmp_path):
    payload = b"mediafire-model-bytes"
    calls = _fake_mediafire_urlopen(monkeypatch, MEDIAFIRE_PAGE_HTML, payload)

    dest = str(tmp_path / "model.pth")
    download_mediafire("https://www.mediafire.com/file/xyz/model.pth", dest, quiet=True)

    assert calls[0] == "https://www.mediafire.com/file/xyz/model.pth"
    assert calls[1] == "https://download123.mediafire.com/abc123/model.pth"
    with open(dest, "rb") as f:
        assert f.read() == payload


def test_download_mediafire_attribute_order_and_entity_escaping(monkeypatch, tmp_path):
    payload = b"model-bytes"
    calls = _fake_mediafire_urlopen(
        monkeypatch, MEDIAFIRE_PAGE_HTML_REVERSED, payload
    )

    dest = str(tmp_path / "model.pth")
    download_mediafire("https://www.mediafire.com/file/xyz/model.pth", dest, quiet=True)

    assert calls[1] == "https://download99.mediafire.com/x?a=1&b=2"
    with open(dest, "rb") as f:
        assert f.read() == payload


def test_download_mediafire_page_without_button_raises(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            MEDIAFIRE_PAGE_NO_BUTTON.encode(),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError, match="[Mm]edia[Ff]ire"):
        download_mediafire(
            "https://www.mediafire.com/file/xyz/model.pth",
            str(tmp_path / "model.pth"),
        )


def test_download_mediafire_non_http_href_raises(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            MEDIAFIRE_PAGE_JS_HREF.encode(),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError):
        download_mediafire(
            "https://www.mediafire.com/file/xyz/model.pth",
            str(tmp_path / "model.pth"),
        )


# ─── URL scheme allowlist ──────────────────────────────────────────────

def test_smart_download_rejects_file_scheme_without_network(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise AssertionError("urlopen must not be called for a rejected scheme")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError, match="scheme"):
        smart_download("file:///etc/passwd", str(tmp_path / "leak"))


def test_smart_download_rejects_ftp_scheme(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise AssertionError("urlopen must not be called for a rejected scheme")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(DownloadError, match="scheme"):
        smart_download("ftp://evil.com/model.pth", str(tmp_path / "m.pth"))


# ─── scraped-URL destination validation ────────────────────────────────

GDRIVE_INTERSTITIAL_EVIL_ACTION = """<!DOCTYPE html>
<html><body>
<form id="download-form" action="https://evil.com/exfil" method="get">
<input type="hidden" name="id" value="FILEID123">
</form>
</body></html>"""

GDRIVE_INTERSTITIAL_FILE_ACTION = """<!DOCTYPE html>
<html><body>
<form id="download-form" action="file:///etc/passwd" method="get">
<input type="hidden" name="id" value="FILEID123">
</form>
</body></html>"""


def _serve_first_response(monkeypatch, body: bytes, content_type: str):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(body, headers={"Content-Type": content_type})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_gdrive_interstitial_action_to_foreign_host_rejected(monkeypatch, tmp_path):
    _serve_first_response(
        monkeypatch,
        GDRIVE_INTERSTITIAL_EVIL_ACTION.encode(),
        "text/html; charset=utf-8",
    )

    with pytest.raises(DownloadError, match="[Rr]efus"):
        download_direct(
            "https://drive.google.com/file/d/FILEID123/view",
            str(tmp_path / "m.pth"),
        )


def test_gdrive_interstitial_action_to_file_scheme_rejected(monkeypatch, tmp_path):
    _serve_first_response(
        monkeypatch,
        GDRIVE_INTERSTITIAL_FILE_ACTION.encode(),
        "text/html; charset=utf-8",
    )

    with pytest.raises(DownloadError):
        download_direct(
            "https://drive.google.com/file/d/FILEID123/view",
            str(tmp_path / "m.pth"),
        )


def test_mediafire_href_to_foreign_host_rejected(monkeypatch, tmp_path):
    html = (
        '<html><body><a id="downloadButton" '
        'href="https://evil.com/steal">Download</a></body></html>'
    )
    _serve_first_response(monkeypatch, html.encode(), "text/html; charset=utf-8")

    with pytest.raises(DownloadError):
        download_mediafire(
            "https://www.mediafire.com/file/xyz/model.pth",
            str(tmp_path / "m.pth"),
        )


def test_mediafire_http_href_downgrade_rejected(monkeypatch, tmp_path):
    html = (
        '<html><body><a id="downloadButton" '
        'href="http://download123.mediafire.com/abc">Download</a></body></html>'
    )
    _serve_first_response(monkeypatch, html.encode(), "text/html; charset=utf-8")

    with pytest.raises(DownloadError):
        download_mediafire(
            "https://www.mediafire.com/file/xyz/model.pth",
            str(tmp_path / "m.pth"),
        )


# ─── Google Drive interstitial fix ─────────────────────────────────────

def test_download_direct_gdrive_interstitial_is_followed(monkeypatch, tmp_path):
    payload = b"the-actual-model-bytes"
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            return _FakeResp(
                GDRIVE_INTERSTITIAL_HTML.encode(),
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        return _FakeResp(
            payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload))
            },
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "model.pth")
    download_direct("https://drive.google.com/file/d/FILEID123/view", dest, quiet=True)

    assert len(calls) == 2
    assert calls[1].startswith("https://drive.usercontent.google.com/download?")
    assert "confirm=t" in calls[1]
    assert "uuid=abc-def-uuid" in calls[1]

    with open(dest, "rb") as f:
        assert f.read() == payload
    assert not os.path.exists(dest + ".part")


def test_download_direct_gdrive_unparseable_html_raises_download_error(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            GDRIVE_UNPARSEABLE_HTML.encode(),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "model.pth")
    with pytest.raises(DownloadError):
        download_direct("https://drive.google.com/file/d/FILEID123/view", dest, quiet=True)

    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_non_gdrive_html_response_is_not_sniffed(monkeypatch, tmp_path):
    """Only Drive URLs get the HTML-interstitial treatment; a normal host
    that happens to serve text/html is downloaded as-is (not penalized)."""
    html_body = b"<html><body>just some html file, not Drive</body></html>"

    def fake_urlopen(req, timeout=None):
        return _FakeResp(html_body, headers={"Content-Type": "text/html"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "page.html")
    download_direct("https://example.com/page.html", dest, quiet=True)

    with open(dest, "rb") as f:
        assert f.read() == html_body


# ─── Exceptions module ───────────────────────────────────────────────────

def test_exceptions_importable_from_all_three_paths():
    import openmodeldb
    import openmodeldb.client as client_mod
    import openmodeldb.exceptions as exceptions_mod

    for name in ("OpenModelDBError", "ModelNotFoundError", "FormatNotFoundError", "DownloadError"):
        top_level = getattr(openmodeldb, name)
        via_client = getattr(client_mod, name)
        via_exceptions = getattr(exceptions_mod, name)
        assert top_level is via_client is via_exceptions


def test_exception_hierarchy_unchanged():
    assert issubclass(ModelNotFoundError, OpenModelDBError)
    assert issubclass(FormatNotFoundError, OpenModelDBError)
    assert issubclass(DownloadError, OpenModelDBError)
