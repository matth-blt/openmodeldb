"""
Tests for openmodeldb.client.OpenModelDB query/lookup behavior, backed by
the offline `db` fixture (tests/conftest.py) built from
tests/fixtures/models.json. No network access.
"""
import json
import os
import time

from openmodeldb import OpenModelDB

# ─── find() ──────────────────────────────────────────────────────────────

def test_find_by_scale(db):
    results = db.find(scale=4)
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-DualFormat", "4x-ZipArchive", "4x-ListAuthor"}


def test_find_by_architecture(db):
    results = db.find(architecture="compact")
    ids = {m.id for m in results}
    assert ids == {"2x-SafeOnly", "4x-ListAuthor"}


def test_find_by_architecture_case_insensitive(db):
    results = db.find(architecture="ESRGAN")
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-ZipArchive", "2x-NoScale"}


def test_find_by_tag(db):
    results = db.find(tag="photo")
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-DualFormat", "2x-NoScale"}


def test_find_by_tag_case_insensitive(db):
    results = db.find(tag="PHOTO")
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-DualFormat", "2x-NoScale"}


def test_find_combined_scale_and_architecture(db):
    results = db.find(scale=4, architecture="esrgan")
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-ZipArchive"}


def test_find_combined_scale_and_tag(db):
    results = db.find(scale=4, tag="photo")
    ids = {m.id for m in results}
    assert ids == {"4x-ExamplePth", "4x-DualFormat"}


def test_find_no_filters_returns_all(db):
    assert len(db.find()) == len(db.models)


def test_find_no_match(db):
    assert db.find(scale=99) == []


def test_find_excludes_cain(db):
    ids = {m.id for m in db.find()}
    assert "cain-excluded-model" not in ids


def test_find_missing_scale_defaults_to_zero(db):
    results = db.find(scale=0)
    ids = {m.id for m in results}
    assert "2x-NoScale" in ids


# ─── search() ────────────────────────────────────────────────────────────

def test_search_matches_name(db):
    results = db.search("ExampleNet")
    ids = {m.id for m in results}
    assert "4x-ExamplePth" in ids


def test_search_matches_author(db):
    results = db.search("carol")
    ids = {m.id for m in results}
    assert "4x-DualFormat" in ids


def test_search_matches_joined_list_author(db):
    results = db.search("erin")
    ids = {m.id for m in results}
    assert "4x-ListAuthor" in ids


def test_search_matches_tag(db):
    results = db.search("sharp")
    ids = {m.id for m in results}
    assert "4x-DualFormat" in ids


def test_search_matches_description(db):
    results = db.search("zip archive")
    ids = {m.id for m in results}
    assert "4x-ZipArchive" in ids


def test_search_case_insensitive(db):
    results = db.search("ANIME")
    ids = {m.id for m in results}
    assert "2x-SafeOnly" in ids


def test_search_no_match(db):
    assert db.search("nonexistent-query-xyz") == []


def test_search_excludes_cain(db):
    results = db.search("frame-interpolation")
    ids = {m.id for m in results}
    assert "cain-excluded-model" not in ids


# ─── architectures() / tags() ─────────────────────────────────────────────

def test_architectures_sorted_and_deduped(db):
    assert db.architectures() == ["compact", "esrgan", "span"]


def test_architectures_excludes_cain(db):
    assert "cain" not in db.architectures()


def test_tags_sorted_and_deduped(db):
    expected = sorted(
        {"photo", "denoise", "anime", "sharp", "archive"}
    )
    assert db.tags() == expected


# ─── dunder methods ───────────────────────────────────────────────────────

def test_contains_by_name(db):
    assert "ExampleNet" in db


def test_contains_by_id(db):
    assert "safeonly" in db


def test_contains_false(db):
    assert "totally-not-a-model" not in db


def test_getitem_exact_id(db):
    model = db["2x-SafeOnly"]
    assert model.id == "2x-SafeOnly"
    assert model.name == "SafeOnly Net"


def test_getitem_exact_name_case_insensitive(db):
    model = db["examplenet pth"]
    assert model.id == "4x-ExamplePth"


def test_len(db):
    assert len(db) == 6


def test_cain_absent_from_models(db):
    ids = {m.id for m in db.models}
    assert "cain-excluded-model" not in ids


# ─── field parsing ─────────────────────────────────────────────────────────

def test_list_author_joined_to_comma_string(db):
    model = db["4x-ListAuthor"]
    assert model.author == "erin, frank"


def test_single_string_author_unchanged(db):
    model = db["4x-ExamplePth"]
    assert model.author == "alice"


# ─── terminal-injection sanitization ────────────────────────────────────

def test_model_fields_strip_terminal_control_sequences(tmp_path):
    """Names/descriptions/tags come from the remote DB: escape sequences
    (ANSI, OSC clipboard/title tricks) must never reach the terminal."""
    payload = {
        "4x-Evil": {
            "name": "Evil\x1b]52;c;pwned\x07Net",
            "author": "mallory\x1b[A",
            "architecture": "esrgan",
            "scale": 4,
            "license": "MIT\x1b[2K",
            "tags": ["photo", "tag\x1b]0;title\x07"],
            "description": "desc with \x9b31m control\x00 chars",
            "resources": [],
        }
    }
    cache_dir = tmp_path
    cache_file = cache_dir / "models.json"
    with open(cache_file, "w") as f:
        json.dump(payload, f)
    now = time.time()
    os.utime(cache_file, (now, now))

    db = OpenModelDB(cache_dir=str(cache_dir), download_dir=str(tmp_path / "d"))
    m = db["4x-Evil"]

    for field in (m.id, m.name, m.author, m.license, m.description):
        assert "\x1b" not in field and "\x9b" not in field and "\x00" not in field
    assert all("\x1b" not in t and "\x9b" not in t for t in m.tags)
