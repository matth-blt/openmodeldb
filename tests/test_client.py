"""
Tests for openmodeldb.client.OpenModelDB query/lookup behavior, backed by
the offline `db` fixture (tests/conftest.py) built from
tests/fixtures/models.json. No network access.
"""


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
    # cain is excluded at load time; even an unfiltered find() must never
    # surface it.
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
    # 7 fixture entries minus the 1 excluded "cain" architecture model.
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
