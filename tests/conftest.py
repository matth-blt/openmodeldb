"""
Shared pytest fixtures for the openmodeldb test suite.

Tests must never touch the network. The `db` fixture below builds an
OpenModelDB instance backed entirely by a local, hand-written fixture file
(tests/fixtures/models.json) copied into a fresh cache directory with a
current mtime, so OpenModelDB._cache_is_valid() reports the cache as valid
and no HTTP request is ever attempted.
"""
import os
import shutil
import time

import pytest

from openmodeldb import OpenModelDB

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MODELS_FIXTURE = os.path.join(FIXTURES_DIR, "models.json")


@pytest.fixture
def db(tmp_path):
    """An OpenModelDB instance pre-seeded from the local fixture, offline."""
    cache_dir = tmp_path
    cache_file = cache_dir / "models.json"
    shutil.copyfile(MODELS_FIXTURE, cache_file)
    # Ensure a fresh mtime so OpenModelDB._cache_is_valid() is True and the
    # client never attempts to hit the network.
    now = time.time()
    os.utime(cache_file, (now, now))

    return OpenModelDB(cache_dir=str(cache_dir), download_dir=str(tmp_path / "downloads"))
