"""
Tests for the scriptable CLI (openmodeldb.cli.main).

Fully offline: OpenModelDB is monkeypatched inside openmodeldb.cli with a
recording stub, so no network access, cache I/O, or InquirerPy prompts are
ever touched.
"""
import pytest

from openmodeldb import cli
from openmodeldb.exceptions import ModelNotFoundError


class StubModel:
    def __init__(self, id, name, author, architecture, scale):
        self.id = id
        self.name = name
        self.author = author
        self.architecture = architecture
        self.scale = scale


DEFAULT_INTEGRITY_RESULT = {
    "similarity": 100.0,
    "identical": True,
    "matched": 1,
    "total_a": 1,
    "max_diff": 0.0,
    "mean_diff": 0.0,
    "mean_rel_diff": 0.0,
}


class StubDB:
    """Records every call made to it in place of a real OpenModelDB."""

    instances = []
    integrity_result = DEFAULT_INTEGRITY_RESULT

    def __init__(self, *args, **kwargs):
        self.calls = []
        StubDB.instances.append(self)

    def search(self, query):
        self.calls.append(("search", query))
        return [StubModel("4x-Foo", "FooNet", "someone", "esrgan", 4)]

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return []

    def download(self, model, dest=None, format=None, quiet=False, half=False):
        self.calls.append((
            "download",
            dict(model=model, dest=dest, format=format, quiet=quiet, half=half),
        ))
        return "/tmp/downloaded"

    def download_all(self, model, dest=None, format=None, quiet=False):
        self.calls.append((
            "download_all",
            dict(model=model, dest=dest, format=format, quiet=quiet),
        ))
        return ["/tmp/downloaded"]

    def test_integrity(self, file_path, quiet=False):
        self.calls.append(("test_integrity", file_path))
        return StubDB.integrity_result


@pytest.fixture
def stub_db(monkeypatch):
    """Replace openmodeldb.cli.OpenModelDB with the recording stub."""
    StubDB.instances.clear()
    StubDB.integrity_result = DEFAULT_INTEGRITY_RESULT
    monkeypatch.setattr(cli, "OpenModelDB", StubDB)
    return StubDB


# ─── search ──────────────────────────────────────────────────────────────

def test_search_calls_db_search_and_prints(stub_db, capsys):
    cli.main(["search", "foo"])

    db = StubDB.instances[-1]
    assert db.calls == [("search", "foo")]

    out = capsys.readouterr().out
    assert "4x-Foo" in out
    assert "FooNet — someone (esrgan, 4x)" in out


# ─── list ────────────────────────────────────────────────────────────────

def test_list_forwards_filters(stub_db):
    cli.main(["list", "--scale", "4", "--arch", "compact"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("list", {"scale": 4, "architecture": "compact", "tag": None})
    ]


def test_list_no_filters(stub_db):
    cli.main(["list"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("list", {"scale": None, "architecture": None, "tag": None})
    ]


# ─── download ────────────────────────────────────────────────────────────

def test_download_forwards_format_and_half(stub_db):
    cli.main(["download", "x", "--format", "onnx", "--half"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("download", dict(model="x", dest=None, format="onnx", quiet=False, half=True))
    ]


def test_download_forwards_dest(stub_db):
    cli.main(["download", "x", "--dest", "/somewhere"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("download", dict(model="x", dest="/somewhere", format=None, quiet=False, half=False))
    ]


def test_download_all_flag_routes_to_download_all(stub_db):
    cli.main(["download", "x", "--all", "--format", "pth"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("download_all", dict(model="x", dest=None, format="pth", quiet=False))
    ]


def test_download_all_with_half_prints_notice_and_still_calls_download_all(stub_db, capsys):
    cli.main(["download", "x", "--all", "--half"])

    db = StubDB.instances[-1]
    assert db.calls == [
        ("download_all", dict(model="x", dest=None, format=None, quiet=False))
    ]

    err = capsys.readouterr().err
    assert "note: --half is ignored with --all" in err


def test_download_all_without_half_prints_no_notice(stub_db, capsys):
    cli.main(["download", "x", "--all"])

    err = capsys.readouterr().err
    assert "--half is ignored" not in err


# ─── no args → interactive ────────────────────────────────────────────────

def test_no_args_routes_to_interactive(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "interactive", lambda *a, **kw: called.append((a, kw)))

    cli.main([])

    assert called == [((), {})]


# ─── check ───────────────────────────────────────────────────────────────

def test_check_identical_exits_0(stub_db):
    StubDB.integrity_result = dict(DEFAULT_INTEGRITY_RESULT, similarity=100.0, identical=True)

    # Should complete without raising SystemExit.
    cli.main(["check", "somefile.pth"])

    db = StubDB.instances[-1]
    assert db.calls == [("test_integrity", "somefile.pth")]


def test_check_similarity_42_exits_1(stub_db):
    StubDB.integrity_result = dict(
        DEFAULT_INTEGRITY_RESULT, similarity=42.0, identical=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["check", "somefile.pth"])

    assert exc_info.value.code == 1


def test_check_similarity_just_above_threshold_exits_0(stub_db):
    StubDB.integrity_result = dict(
        DEFAULT_INTEGRITY_RESULT, similarity=99.95, identical=False,
    )

    cli.main(["check", "somefile.pth"])  # no SystemExit


# ─── error handling ─────────────────────────────────────────────────────

def test_model_not_found_error_exits_1_with_message_no_traceback(monkeypatch, capsys):
    class RaisingDB:
        def __init__(self, *a, **kw):
            pass

        def search(self, query):
            raise ModelNotFoundError("Model not found: 'x'")

    monkeypatch.setattr(cli, "OpenModelDB", RaisingDB)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["search", "x"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Model not found" in err
    assert "Traceback" not in err


def test_file_not_found_error_exits_1_with_message(monkeypatch, capsys):
    class RaisingDB:
        def __init__(self, *a, **kw):
            pass

        def test_integrity(self, file_path, quiet=False):
            raise FileNotFoundError(file_path)

    monkeypatch.setattr(cli, "OpenModelDB", RaisingDB)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["check", "missing.pth"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "missing.pth" in err


def test_import_error_exits_1_with_message_no_traceback(monkeypatch, capsys):
    class RaisingDB:
        def __init__(self, *a, **kw):
            pass

        def test_integrity(self, file_path, quiet=False):
            raise ImportError("torch is required for integrity checks; install with `pip install torch`")

    monkeypatch.setattr(cli, "OpenModelDB", RaisingDB)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["check", "somefile.pth"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "torch is required" in err
    assert "Traceback" not in err


def test_keyboard_interrupt_exits_130_quietly(monkeypatch, capsys):
    class RaisingDB:
        def __init__(self, *a, **kw):
            pass

        def search(self, query):
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "OpenModelDB", RaisingDB)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["search", "x"])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert err == ""


# ─── --version / --help ───────────────────────────────────────────────────

def test_version_prints_version(capsys):
    cli.main(["--version"])

    out = capsys.readouterr().out.strip()
    assert out  # non-empty version string printed


def test_help_exits_0(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for sub in ("list", "search", "download", "check"):
        assert sub in out
