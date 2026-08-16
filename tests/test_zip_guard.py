"""
Tests for the zip extraction expansion limit (zip-bomb guard).

The API's `sha256` protects file *contents*, but nothing bounds the
*volume* a malicious archive can write to disk. Both zip extraction paths
therefore enforce an uncompressed-size ceiling derived from the archive
size (with an absolute floor), configurable via module constants for
testing.
"""
import os
import zipfile

import pytest

from openmodeldb import DownloadError
from openmodeldb import client as client_mod


def _make_zip(path, members: dict) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return str(path)


# ─── _extract_from_zip: single-entry limit ─────────────────────────────

def test_extract_rejects_entry_beyond_expansion_limit(db, tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "ZIP_MAX_EXPANSION_RATIO", 2)
    monkeypatch.setattr(client_mod, "ZIP_MIN_EXPANSION_LIMIT", 1024)

    zip_path = _make_zip(tmp_path / "bomb.zip", {"bomb.pth": b"\x00" * (1 << 20)})
    res = {"type": "pth", "size": 1 << 20, "sha256": None}

    with pytest.raises(DownloadError, match="expansion limit"):
        db._extract_from_zip(zip_path, res, str(tmp_path / "out"))

    assert not os.path.exists(tmp_path / "out" / "bomb.pth")


def test_extract_allows_entry_within_limit(db, tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "ZIP_MAX_EXPANSION_RATIO", 2)
    monkeypatch.setattr(client_mod, "ZIP_MIN_EXPANSION_LIMIT", 4 << 20)

    zip_path = _make_zip(tmp_path / "ok.zip", {"ok.pth": b"\x00" * (1 << 20)})
    res = {"type": "pth", "size": 1 << 20, "sha256": None}

    out = db._extract_from_zip(zip_path, res, str(tmp_path / "out"))
    assert os.path.exists(out)


# ─── _extract_all_from_zip: cumulative limit ───────────────────────────

def test_extract_all_enforces_cumulative_limit(db, tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "ZIP_MAX_EXPANSION_RATIO", 2)
    monkeypatch.setattr(client_mod, "ZIP_MIN_EXPANSION_LIMIT", 1024)

    zip_path = _make_zip(tmp_path / "bomb.zip", {
        "a.pth": b"\x00" * (700 * 1024),
        "b.pth": b"\x00" * (700 * 1024),
    })
    res = {"type": "pth", "size": 700 * 1024, "sha256": None}

    with pytest.raises(DownloadError, match="expansion limit"):
        db._extract_all_from_zip(zip_path, str(tmp_path / "out"), None, res)

    assert not os.path.exists(tmp_path / "out" / "a.pth")
    assert not os.path.exists(tmp_path / "out" / "b.pth")
    assert not os.path.exists(zip_path)


def test_extract_all_normal_archives_are_unaffected(db, tmp_path):
    content = b"normal-model-weights"
    zip_path = _make_zip(tmp_path / "ok.zip", {"ok.pth": content})
    res = {"type": "pth", "size": len(content), "sha256": None}

    paths = db._extract_all_from_zip(zip_path, str(tmp_path / "out"), None, res)

    assert len(paths) == 1
    with open(paths[0], "rb") as f:
        assert f.read() == content
