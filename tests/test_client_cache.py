"""
Tests for OpenModelDB._fetch/_fetch_remote/refresh cache resilience:
offline/stale-cache fallback, refresh() failing loudly, conditional
requests (ETag/Last-Modified + 304 handling), and corrupt-cache recovery.

All network I/O is faked by monkeypatching urllib.request.urlopen. No real
network access happens in this file.
"""
import json
import os
import time
import urllib.error
import urllib.request

import pytest

from openmodeldb import OpenModelDB, OpenModelDBError

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MODELS_FIXTURE = os.path.join(FIXTURES_DIR, "models.json")

with open(MODELS_FIXTURE, "r", encoding="utf-8") as _f:
    FIXTURE_DATA = json.load(_f)


# ─── Fakes ───────────────────────────────────────────────────────────────

class _FakeResp:
    """Minimal stand-in for urllib.request.urlopen()'s return value."""

    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    def read(self, size=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_client(tmp_path, seed_cache: bool = False, expired: bool = False,
                  meta: dict | None = None, corrupt_cache: bool = False):
    """Build an OpenModelDB pointed at an isolated tmp cache dir.

    seed_cache: write the fixture data (or corrupt bytes) as the cache file.
    expired: back-date the cache file's mtime past CACHE_MAX_AGE.
    meta: if given, also write a models.json.meta sidecar.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    db = OpenModelDB(cache_dir=str(cache_dir), download_dir=str(tmp_path / "downloads"))

    if seed_cache:
        cache_file = cache_dir / "models.json"
        if corrupt_cache:
            cache_file.write_text("{not valid json", encoding="utf-8")
        else:
            cache_file.write_text(json.dumps(FIXTURE_DATA), encoding="utf-8")

        if expired:
            old = time.time() - 999999
            os.utime(cache_file, (old, old))
        else:
            now = time.time()
            os.utime(cache_file, (now, now))

        if meta is not None:
            meta_file = cache_dir / "models.json.meta"
            meta_file.write_text(json.dumps(meta), encoding="utf-8")

    return db


# ─── Offline / stale-cache fallback ────────────────────────────────────

def test_expired_cache_falls_back_with_warning_on_urlerror(tmp_path, capsys, monkeypatch):
    db = _make_client(tmp_path, seed_cache=True, expired=True)

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = db.models  # triggers _fetch() -> _fetch_remote() fails -> fallback

    assert db._raw_data == FIXTURE_DATA
    assert len(models) == len(FIXTURE_DATA) - 1  # one fixture entry has an excluded arch

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "warning: failed to fetch openmodeldb API (" in captured.err
    assert "using stale cache from" in captured.err


def test_connection_reset_during_read_falls_back_with_warning(tmp_path, capsys, monkeypatch):
    """A ConnectionResetError raised mid-stream by resp.read() (not by
    urlopen() itself) must still be wrapped as OpenModelDBError and trigger
    the same stale-cache fallback as any other network failure."""
    db = _make_client(tmp_path, seed_cache=True, expired=True)

    class _BrokenReadResp:
        headers = {}

        def read(self, size=-1):
            raise ConnectionResetError("Connection reset by peer")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_urlopen(req, timeout=None):
        return _BrokenReadResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = db.models  # triggers _fetch() -> _fetch_remote() fails mid-read -> fallback

    assert db._raw_data == FIXTURE_DATA
    assert len(models) == len(FIXTURE_DATA) - 1  # one fixture entry has an excluded arch

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "warning: failed to fetch openmodeldb API (" in captured.err
    assert "using stale cache from" in captured.err


def test_no_cache_raises_openmodeldb_error_on_urlerror(tmp_path, monkeypatch):
    db = _make_client(tmp_path, seed_cache=False)

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OpenModelDBError) as exc_info:
        db.models

    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)


def test_no_cache_raises_openmodeldb_error_on_connection_reset_during_read(tmp_path, monkeypatch):
    db = _make_client(tmp_path, seed_cache=False)

    class _BrokenReadResp:
        headers = {}

        def read(self, size=-1):
            raise ConnectionResetError("Connection reset by peer")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_urlopen(req, timeout=None):
        return _BrokenReadResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OpenModelDBError) as exc_info:
        db.models

    assert isinstance(exc_info.value.__cause__, ConnectionResetError)


# ─── Conditional requests / 304 handling ───────────────────────────────

def test_304_reuses_cache_and_refreshes_mtime(tmp_path, monkeypatch):
    db = _make_client(
        tmp_path, seed_cache=True, expired=True, meta={"etag": '"abc123"'},
    )
    cache_file = tmp_path / "cache" / "models.json"
    mtime_before = os.path.getmtime(cache_file)

    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        raise urllib.error.HTTPError(
            "https://openmodeldb.info/api/v1/models.json", 304, "Not Modified", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    db.models

    assert db._raw_data == FIXTURE_DATA
    assert len(captured_requests) == 1
    # urllib.request.Request stores header keys via str.capitalize(), i.e.
    # only the leading character is upper-cased ("If-none-match").
    assert captured_requests[0].get_header("If-none-match") == '"abc123"'

    mtime_after = os.path.getmtime(cache_file)
    assert mtime_after > mtime_before
    assert (time.time() - mtime_after) < 5  # freshly refreshed


def test_200_stores_meta_sidecar_when_headers_present(tmp_path, monkeypatch):
    db = _make_client(tmp_path, seed_cache=True, expired=True)

    body = json.dumps(FIXTURE_DATA).encode()

    def fake_urlopen(req, timeout=None):
        return _FakeResp(body, headers={"ETag": '"new-etag"', "Last-Modified": "Fri, 10 Jul 2026 00:00:00 GMT"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    db.models
    assert db._raw_data == FIXTURE_DATA

    meta_file = tmp_path / "cache" / "models.json.meta"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta == {"etag": '"new-etag"', "last_modified": "Fri, 10 Jul 2026 00:00:00 GMT"}


# ─── Corrupt cache recovery ─────────────────────────────────────────────

def test_corrupt_cache_file_is_discarded_and_refetched(tmp_path, monkeypatch):
    db = _make_client(tmp_path, seed_cache=True, expired=False, corrupt_cache=True)

    body = json.dumps(FIXTURE_DATA).encode()

    def fake_urlopen(req, timeout=None):
        return _FakeResp(body, headers={})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    db.models  # cache file has invalid JSON but a "valid" (fresh) mtime

    assert db._raw_data == FIXTURE_DATA

    # Cache file should now contain valid, parseable data.
    cache_file = tmp_path / "cache" / "models.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        assert json.load(f) == FIXTURE_DATA


# ─── refresh() fails loudly (no fallback) ──────────────────────────────

def test_refresh_propagates_network_failure_without_fallback(tmp_path, monkeypatch):
    db = _make_client(tmp_path, seed_cache=True, expired=False)  # valid, fresh cache

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OpenModelDBError) as exc_info:
        db.refresh()

    assert isinstance(exc_info.value.__cause__, urllib.error.URLError)
