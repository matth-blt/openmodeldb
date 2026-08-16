"""
Tests for the pure/offline-safe helpers in openmodeldb.downloader.

None of these exercise smart_download/download_mega/download_mediafire/
download_direct: those perform network I/O and are covered in
test_download_robustness.py via monkeypatched urllib.
"""
import re

from openmodeldb.downloader import (
    _convert_gdrive_url,
    build_filename,
    fmt_size,
    is_gdrive_url,
    is_mediafire_url,
    is_mega_url,
    pick_best_url,
    safe_filename_component,
)

# ─── pick_best_url ──────────────────────────────────────────────────────

def test_pick_best_url_github_over_mega():
    urls = [
        "https://mega.nz/file/abc#def",
        "https://github.com/user/repo/releases/download/v1/model.pth",
    ]
    assert pick_best_url(urls) == "https://github.com/user/repo/releases/download/v1/model.pth"


def test_pick_best_url_huggingface_over_mediafire():
    urls = [
        "https://www.mediafire.com/file/xyz/model.pth",
        "https://huggingface.co/user/model/resolve/main/model.pth",
    ]
    assert pick_best_url(urls) == "https://huggingface.co/user/model/resolve/main/model.pth"


def test_pick_best_url_objectstorage_is_highest_priority():
    urls = [
        "https://github.com/user/repo/releases/download/v1/model.pth",
        "https://objectstorage.example.com/model.pth",
    ]
    assert pick_best_url(urls) == "https://objectstorage.example.com/model.pth"


def test_pick_best_url_gdrive_over_mediafire_and_mega():
    urls = [
        "https://mega.nz/file/abc#def",
        "https://www.mediafire.com/file/xyz/model.pth",
        "https://drive.google.com/file/d/abc123/view",
    ]
    assert pick_best_url(urls) == "https://drive.google.com/file/d/abc123/view"


def test_pick_best_url_fallback_to_first_when_no_priority_host_matches():
    urls = [
        "https://example.com/a/model.pth",
        "https://example.org/b/model.pth",
    ]
    assert pick_best_url(urls) == urls[0]


def test_pick_best_url_single_url():
    urls = ["https://example.com/model.pth"]
    assert pick_best_url(urls) == urls[0]


# ─── build_filename ─────────────────────────────────────────────────────

def test_build_filename_extracts_from_url():
    url = "https://github.com/alice/example/releases/download/v1/examplenet.pth"
    assert build_filename(url, "some-model", "pth") == "examplenet.pth"


def test_build_filename_strips_query_string():
    url = "https://example.com/dir/model.pth?raw=true&x=1"
    assert build_filename(url, "some-model", "pth") == "model.pth"


def test_build_filename_falls_back_for_gdrive_uc_url():
    url = "https://drive.google.com/uc?export=download&id=ABC123"
    assert build_filename(url, "my-model", "pth") == "my-model.pth"


def test_build_filename_falls_back_for_extensionless_url():
    url = "https://example.com/download"
    assert build_filename(url, "my-model", "onnx") == "my-model.onnx"


def test_build_filename_falls_back_for_short_filename():
    url = "https://example.com/a.b"
    assert build_filename(url, "my-model", "pth") == "my-model.pth"


def test_build_filename_suspicious_extension_fallback_is_sanitized():
    url = "https://example.com/download"
    fn = build_filename(url, "my-model", "../evil")
    assert fn == "my-model.evil"


# ─── safe_filename_component (untrusted model ids/extensions) ──────────

def test_safe_filename_component_returns_simple_names_unchanged():
    assert safe_filename_component("4xNomos8k_atd_jpg") == "4xNomos8k_atd_jpg"
    assert safe_filename_component("4x-AnimeJaNai.v2") == "4x-AnimeJaNai.v2"


def test_safe_filename_component_strips_path_traversal():
    assert safe_filename_component("../../.cache/openmodeldb/models.json") == "models.json"
    assert safe_filename_component("sub/../../4xNomos8k") == "4xNomos8k"
    assert safe_filename_component("a\\..\\..\\b") == "b"


def test_safe_filename_component_never_returns_traversal_or_hidden():
    for evil in ("..", ".", "...", ".bashrc", "", "/"):
        out = safe_filename_component(evil)
        assert out not in ("", ".", ".."), evil
        assert not out.startswith("."), evil
        assert "/" not in out and "\\" not in out, evil


def test_safe_filename_component_replaces_unsafe_characters():
    out = safe_filename_component("4x Nomos (hd); rm -rf")
    assert out == "4x_Nomos__hd___rm_-rf"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", out)


# ─── fmt_size ────────────────────────────────────────────────────────────

def test_fmt_size_zero():
    assert fmt_size(0) == "?"


def test_fmt_size_none():
    assert fmt_size(None) == "?"


def test_fmt_size_bytes():
    assert fmt_size(500) == "500 B"


def test_fmt_size_kb():
    assert fmt_size(2048) == "2.0 KB"


def test_fmt_size_mb():
    assert fmt_size(5 * 1048576) == "5.0 MB"


def test_fmt_size_gb():
    assert fmt_size(2 * 1073741824) == "2.00 GB"


# ─── host detectors ──────────────────────────────────────────────────────

def test_is_mega_url():
    assert is_mega_url("https://mega.nz/file/abc#def") is True
    assert is_mega_url("https://mega.co.nz/file/abc#def") is True
    assert is_mega_url("https://example.com/model.pth") is False


def test_is_mediafire_url():
    assert is_mediafire_url("https://www.mediafire.com/file/xyz/model.pth") is True
    assert is_mediafire_url("https://example.com/model.pth") is False


def test_is_gdrive_url():
    assert is_gdrive_url("https://drive.google.com/file/d/abc123/view") is True
    assert is_gdrive_url("https://example.com/model.pth") is False


def test_host_detectors_match_hostname_not_substring():
    assert is_gdrive_url("https://evil.com/drive.google.com/x") is False
    assert is_gdrive_url("https://evil.com/?next=drive.google.com") is False
    assert is_mediafire_url("https://evil.com/mediafire.com/x") is False
    assert is_mediafire_url("https://evil.com/?u=mediafire.com") is False
    assert is_mega_url("https://evil.com/mega.nz/file") is False
    assert is_mega_url("https://evil.com/?file=mega.nz") is False


def test_host_detectors_accept_subdomains():
    assert is_mediafire_url("https://download123.mediafire.com/abc") is True
    assert is_gdrive_url("https://drive.usercontent.google.com/download?id=x") is True
    assert is_mega_url("https://g123.mega.co.nz/file") is True


# ─── _convert_gdrive_url ──────────────────────────────────────────────────

def test_convert_gdrive_url_file_d_form():
    url = "https://drive.google.com/file/d/1A2B3C4D5E/view?usp=sharing"
    assert _convert_gdrive_url(url) == (
        "https://drive.google.com/uc?export=download&id=1A2B3C4D5E"
    )


def test_convert_gdrive_url_open_id_form():
    url = "https://drive.google.com/open?id=1A2B3C4D5E"
    assert _convert_gdrive_url(url) == (
        "https://drive.google.com/uc?export=download&id=1A2B3C4D5E"
    )


def test_convert_gdrive_url_non_gdrive_passthrough():
    url = "https://github.com/user/repo/releases/download/v1/model.pth"
    assert _convert_gdrive_url(url) == url
