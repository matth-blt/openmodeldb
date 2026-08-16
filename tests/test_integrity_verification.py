"""
Tests for post-download SHA-256 verification.

The OpenModelDB API provides a `sha256` per resource, computed over the model
file itself (for zip resources: over the model file *inside* the archive —
verified empirically against the live API). Every download path must verify
it when present and treat a mismatch as tampering/corruption: raise
DownloadError and delete the bad file.

No network access: smart_download is monkeypatched to write crafted bytes.
"""
import hashlib
import os
import zipfile

import pytest

from openmodeldb import DownloadError

DUMMY = b"dummy-model-bytes"
DUMMY_SHA = hashlib.sha256(DUMMY).hexdigest()
PTH_CONTENT = b"fake-pth-weights"


def _patch_download(monkeypatch, body: bytes):
    """Replace smart_download with a fake writing *body* to dest."""
    calls = []

    def _fake(url, dest, quiet=False):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(body)

    monkeypatch.setattr("openmodeldb.downloader.smart_download", _fake)
    return calls


def _patch_zip_download(monkeypatch, tmp_path, members: dict):
    """Replace smart_download with a fake delivering a crafted zip archive."""
    zip_path = tmp_path / "crafted.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)

    import shutil
    calls = []

    def _fake(url, dest, quiet=False):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(zip_path, dest)

    monkeypatch.setattr("openmodeldb.downloader.smart_download", _fake)
    return calls


# ─── sha256_of_file helper ─────────────────────────────────────────────

def test_sha256_of_file_matches_hashlib(tmp_path):
    from openmodeldb.downloader import sha256_of_file

    p = tmp_path / "blob.bin"
    p.write_bytes(b"some binary content")
    assert sha256_of_file(str(p)) == hashlib.sha256(b"some binary content").hexdigest()


def test_sha256_of_file_streams_large_files(tmp_path):
    """Files bigger than one chunk still hash correctly (chunked read)."""
    from openmodeldb.downloader import sha256_of_file

    payload = os.urandom(5 * 1024 * 1024)  # 5 MB, several 1 MB chunks
    p = tmp_path / "big.bin"
    p.write_bytes(payload)
    assert sha256_of_file(str(p)) == hashlib.sha256(payload).hexdigest()


# ─── direct (non-zip) downloads ────────────────────────────────────────

def test_download_with_correct_sha_succeeds(db, monkeypatch):
    calls = _patch_download(monkeypatch, DUMMY)

    path = db.download("4x-ExamplePth", quiet=True)

    assert os.path.exists(path)
    assert len(calls) == 1


def test_download_tampered_file_raises_and_deletes(db, monkeypatch):
    _patch_download(monkeypatch, b"tampered-payload")

    with pytest.raises(DownloadError, match="SHA-256"):
        db.download("4x-ExamplePth", quiet=True)

    assert not os.path.exists(os.path.join(db.download_dir, "examplenet.pth"))


def test_download_without_sha_in_api_skips_verification(db, monkeypatch):
    _patch_download(monkeypatch, b"whatever-bytes")

    path = db.download("2x-NoScale", quiet=True)

    assert os.path.exists(path)


# ─── zip downloads (sha256 is over the inner model file) ───────────────

def test_download_zip_with_correct_inner_sha_succeeds(db, monkeypatch, tmp_path):
    _patch_zip_download(monkeypatch, tmp_path, {"ziparchive.pth": PTH_CONTENT})

    path = db.download("4x-ZipArchive", quiet=True)

    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == PTH_CONTENT


def test_download_zip_tampered_inner_file_raises_and_cleans_up(
    db, monkeypatch, tmp_path
):
    _patch_zip_download(
        monkeypatch, tmp_path, {"ziparchive.pth": b"evil-tampered-weights"}
    )

    with pytest.raises(DownloadError, match="SHA-256"):
        db.download("4x-ZipArchive", quiet=True)

    extracted = os.path.join(db.download_dir, "ziparchive.pth")
    cached_zip = os.path.join(db.cache_dir, "ziparchive.zip")
    assert not os.path.exists(extracted)
    assert not os.path.exists(cached_zip)


# ─── cache revalidation ────────────────────────────────────────────────

def test_cached_file_with_wrong_sha_is_redownloaded(db, monkeypatch, tmp_path):
    """A pre-existing cache entry whose hash no longer matches the API's
    sha256 must be discarded and re-fetched, not trusted."""
    calls = _patch_download(monkeypatch, DUMMY)

    cache_file = os.path.join(db.cache_dir, "dual.pth")
    os.makedirs(db.cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        f.write(b"stale-or-poisoned-bytes")

    def _fake_convert(
        model_path: str,
        output_path: str | None = None,
        target: str = "safetensors",
        quiet: bool = False,
    ):
        assert output_path is not None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"converted")
        return output_path

    monkeypatch.setattr(
        "openmodeldb.converter.convert_format", _fake_convert
    )

    path = db.download("4x-DualFormat", format="safetensors", quiet=True)

    assert os.path.exists(path)
    assert len(calls) == 1


def test_cached_file_with_correct_sha_is_reused(db, monkeypatch):
    calls = _patch_download(monkeypatch, DUMMY)

    cache_file = os.path.join(db.cache_dir, "dual.pth")
    os.makedirs(db.cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        f.write(DUMMY)

    def _fake_convert(
        model_path: str,
        output_path: str | None = None,
        target: str = "safetensors",
        quiet: bool = False,
    ):
        assert output_path is not None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"converted")
        return output_path

    monkeypatch.setattr(
        "openmodeldb.converter.convert_format", _fake_convert
    )

    db.download("4x-DualFormat", format="safetensors", quiet=True)

    assert calls == []
