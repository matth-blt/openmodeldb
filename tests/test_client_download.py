"""
Tests for OpenModelDB.download() and model resolution, backed by the offline
`db` fixture (tests/conftest.py) built from tests/fixtures/models.json.

No network access: openmodeldb.downloader.smart_download is monkeypatched to
write dummy bytes (or, for the zip case, copy a locally-crafted zip archive)
instead of hitting the network.
"""
import os
import zipfile

import pytest

from openmodeldb import ModelNotFoundError, OpenModelDB

# ─── _resolve_model ──────────────────────────────────────────────────────

def test_resolve_exact_id(db):
    m = db._resolve_model("4x-ExamplePth")
    assert m.id == "4x-ExamplePth"


def test_resolve_exact_id_case_insensitive(db):
    m = db._resolve_model("4x-examplepth")
    assert m.id == "4x-ExamplePth"


def test_resolve_exact_name(db):
    m = db._resolve_model("ExampleNet Pth")
    assert m.id == "4x-ExamplePth"


def test_resolve_unique_partial_match(db):
    m = db._resolve_model("SafeOnly")
    assert m.id == "2x-SafeOnly"


def test_resolve_ambiguous_partial_match_raises(db):
    with pytest.raises(ModelNotFoundError) as exc_info:
        db._resolve_model("4x-")

    message = str(exc_info.value)
    assert message.startswith("Ambiguous model name '4x-': matches")
    for candidate in (
        "4x-ExamplePth",
        "4x-DualFormat",
        "4x-ZipArchive",
        "4x-ListAuthor",
    ):
        assert candidate in message


def test_resolve_unknown_raises(db):
    with pytest.raises(ModelNotFoundError) as exc_info:
        db._resolve_model("totally-nonexistent-model")
    assert "totally-nonexistent-model" in str(exc_info.value)


def test_resolve_ambiguous_partial_match_truncates_with_total_count(db):
    from openmodeldb import Model

    fake_models = [
        Model(
            id=f"4x-Fake{i}", name=f"Fake{i}", author="tester",
            architecture="esrgan", scale=4,
        )
        for i in range(7)
    ]
    db._models = fake_models

    with pytest.raises(ModelNotFoundError) as exc_info:
        db._resolve_model("4x-Fake")

    message = str(exc_info.value)
    assert message.startswith("Ambiguous model name '4x-Fake': matches")
    assert "(7 matches)" in message
    assert message.endswith(", ... (7 matches)")
    for i in range(5):
        assert f"4x-Fake{i}" in message
    for i in range(5, 7):
        assert f"4x-Fake{i}" not in message


# ─── include_all opt-out of EXCLUDED_ARCHS ───────────────────────────────

def test_include_all_false_excludes_cain(db):
    ids = {m.id for m in db.models}
    assert "cain-excluded-model" not in ids


def test_include_all_true_includes_cain(tmp_path):
    import shutil
    import time

    models_fixture = os.path.join(
        os.path.dirname(__file__), "fixtures", "models.json"
    )

    cache_dir = tmp_path
    cache_file = cache_dir / "models.json"
    shutil.copyfile(models_fixture, cache_file)
    now = time.time()
    os.utime(cache_file, (now, now))

    db = OpenModelDB(
        cache_dir=str(cache_dir),
        download_dir=str(tmp_path / "downloads"),
        include_all=True,
    )
    ids = {m.id for m in db.models}
    assert "cain-excluded-model" in ids


# ─── download(): direct (non-zip) path ───────────────────────────────────

@pytest.fixture
def fake_smart_download(monkeypatch):
    """Monkeypatch openmodeldb.downloader.smart_download.

    Records every call and, instead of hitting the network, writes dummy
    bytes to the destination path (creating parent dirs as needed).
    """
    calls = []

    def _fake(url, dest, quiet=False):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"dummy-model-bytes")

    monkeypatch.setattr("openmodeldb.downloader.smart_download", _fake)
    return calls


def test_download_direct_writes_to_download_dir(db, fake_smart_download):
    path = db.download("4x-ExamplePth", quiet=True)

    assert os.path.exists(path)
    assert path == os.path.join(db.download_dir, "examplenet.pth")
    assert len(fake_smart_download) == 1


def test_download_direct_second_call_uses_existing_file(db, fake_smart_download):
    first = db.download("4x-ExamplePth", quiet=True)
    second = db.download("4x-ExamplePth", quiet=True)

    assert first == second
    assert os.path.exists(second)
    assert len(fake_smart_download) == 1


# ─── download(): zip path ────────────────────────────────────────────────

@pytest.fixture
def zip_with_pth(tmp_path):
    """A small real zip archive containing a single .pth member."""
    zip_path = tmp_path / "crafted.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ziparchive.pth", b"fake-pth-weights")
    return str(zip_path)


@pytest.fixture
def fake_smart_download_zip(monkeypatch, zip_with_pth):
    """Like fake_smart_download, but copies a crafted zip into place for
    .zip destinations (simulating the archive download) and writes dummy
    bytes otherwise."""
    import shutil

    calls = []

    def _fake(url, dest, quiet=False):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if dest.endswith(".zip"):
            shutil.copyfile(zip_with_pth, dest)
        else:
            with open(dest, "wb") as f:
                f.write(b"dummy-model-bytes")

    monkeypatch.setattr("openmodeldb.downloader.smart_download", _fake)
    return calls


def test_download_zip_extracts_member_and_removes_cached_zip(
    db, fake_smart_download_zip
):
    path = db.download("4x-ZipArchive", quiet=True)

    assert os.path.exists(path)
    assert path.endswith(".pth")
    assert os.path.dirname(path) == db.download_dir

    with open(path, "rb") as f:
        assert f.read() == b"fake-pth-weights"

    cache_zip_path = os.path.join(db.cache_dir, "ziparchive.zip")
    assert not os.path.exists(cache_zip_path)

    assert len(fake_smart_download_zip) == 1


# ─── download(): untrusted model.id must not reach the filesystem ─────

def test_download_format_convert_sanitizes_traversal_model_id(db, monkeypatch):
    """A malicious model id (e.g. from a poisoned DB entry) must never
    escape the download dir when it becomes an output filename."""
    import hashlib

    from openmodeldb import Model

    dummy = b"dummy-model-bytes"
    evil = Model(
        id="evil/../../4x-DualFormat",
        name="Evil Net",
        author="mallory",
        architecture="span",
        scale=4,
        resources=[{
            "platform": "pytorch",
            "type": "pth",
            "size": len(dummy),
            "sha256": hashlib.sha256(dummy).hexdigest(),
            "urls": ["https://huggingface.co/mallory/evil/resolve/main/dual.pth"],
        }],
    )
    db._models = [evil]

    def _fake(url, dest, quiet=False):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(dummy)

    monkeypatch.setattr("openmodeldb.downloader.smart_download", _fake)

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

    monkeypatch.setattr("openmodeldb.converter.convert_format", _fake_convert)

    path = db.download(evil, format="safetensors", quiet=True)

    assert ".." not in path
    assert os.path.dirname(path) == db.download_dir
    assert os.path.exists(path)


# ─── download_dir / cache_dir / cache_is_valid() ─────────────────────────

def test_download_dir_property(db, tmp_path):
    assert db.download_dir == str(tmp_path / "downloads")


def test_cache_dir_property(db, tmp_path):
    assert db.cache_dir == str(tmp_path)


def test_cache_is_valid_true_for_fresh_fixture(db):
    assert db.cache_is_valid() is True


def test_cache_is_valid_false_when_no_cache(tmp_path):
    empty_db = OpenModelDB(
        cache_dir=str(tmp_path / "no-such-cache"),
        download_dir=str(tmp_path / "downloads"),
    )
    assert empty_db.cache_is_valid() is False
