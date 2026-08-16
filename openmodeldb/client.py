"""
OpenModelDB Client — core API for fetching and querying models.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import overload

from openmodeldb.downloader import USER_AGENT
from openmodeldb.exceptions import (
    DownloadError,
    FormatNotFoundError,
    ModelNotFoundError,
    OpenModelDBError,
)
from openmodeldb.exceptions import UnsafeModelError as UnsafeModelError

API_URL = "https://openmodeldb.info/api/v1/models.json"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "openmodeldb")
DEFAULT_DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
CACHE_MAX_AGE = 3600
EXCLUDED_ARCHS = {"cain", "cain-yuv"}

# Zip-bomb guard: refuse to extract more than ZIP_MAX_EXPANSION_RATIO times
# the archive's size (with an absolute floor for tiny archives). Model
# weights barely compress, so legitimate archives stay far below this.
ZIP_MAX_EXPANSION_RATIO = 100
ZIP_MIN_EXPANSION_LIMIT = 1 << 30  # 1 GiB


def _zip_expansion_limit(archive_size: int) -> int:
    return max(archive_size * ZIP_MAX_EXPANSION_RATIO, ZIP_MIN_EXPANSION_LIMIT)


# C0/C1 control characters (ANSI/OSC escape sequences included) — remote
# data must never reach the terminal raw: OSC 52 can touch the clipboard,
# OSC 0/2 rewrite the window title, CSI sequences move/clear the screen.
_TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_text(value):
    """Strip terminal control sequences from untrusted remote strings."""
    if isinstance(value, str):
        return _TERMINAL_CONTROL_RE.sub("", value)
    return value


@dataclass
class Model:
    """A single upscaling model from OpenModelDB."""
    id: str
    name: str
    author: str
    architecture: str
    scale: int
    license: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""
    resources: list[dict] = field(default_factory=list)
    data: dict = field(default_factory=dict, repr=False)

    def __str__(self):
        return f"{self.name} by {self.author} ({self.architecture}, {self.scale}x)"


class OpenModelDB:
    """
    Client for the OpenModelDB model database.

    Usage:
        db = OpenModelDB()
        models = db.list(scale=4)
        db.download(models[0])
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        download_dir: str | None = None,
        include_all: bool = False,
    ):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_file = os.path.join(self._cache_dir, "models.json")
        self._meta_file = self._cache_file + ".meta"
        self._download_dir = download_dir or DEFAULT_DOWNLOAD_DIR
        self._include_all = include_all
        self._raw_data: dict | None = None
        self._models: list[Model] | None = None

    # ─── Public accessors ──────────────────────────────────────────────────

    @property
    def download_dir(self) -> str:
        """Directory where downloaded model files are saved."""
        return self._download_dir

    @property
    def cache_dir(self) -> str:
        """Directory used to cache the API response and intermediate downloads."""
        return self._cache_dir

    def cache_is_valid(self) -> bool:
        """Whether the local models.json cache exists and is still fresh."""
        return self._cache_is_valid()

    # ─── Data loading ────────────────────────────────────────────────────

    def _cache_is_valid(self) -> bool:
        if not os.path.exists(self._cache_file):
            return False
        return (time.time() - os.path.getmtime(self._cache_file)) < CACHE_MAX_AGE

    def _load_cache(self) -> dict:
        with open(self._cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_meta(self) -> dict | None:
        """Load the cache sidecar metadata (etag/last-modified), if any and valid."""
        if not os.path.exists(self._meta_file):
            return None
        try:
            with open(self._meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_meta(self, headers) -> None:
        """Save (or clear) the cache sidecar metadata from response headers."""
        meta = {}
        if headers is not None:
            etag = headers.get("ETag")
            last_modified = headers.get("Last-Modified")
            if etag:
                meta["etag"] = etag
            if last_modified:
                meta["last_modified"] = last_modified

        if meta:
            with open(self._meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        elif os.path.exists(self._meta_file):
            try:
                os.remove(self._meta_file)
            except OSError:
                pass

    def _save_cache(self, data: dict, headers=None):
        os.makedirs(self._cache_dir, exist_ok=True)
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self._save_meta(headers)

    def _discard_cache(self) -> None:
        """Remove a corrupt/stale cache file and its meta sidecar."""
        for path in (self._cache_file, self._meta_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _fetch_remote(self) -> dict:
        """Fetch fresh data from the API and update the cache.

        If a cache meta sidecar exists, sends conditional request headers
        (If-None-Match / If-Modified-Since). A 304 response means the
        existing cache is still current: it is reused and its mtime is
        refreshed instead of re-downloading the body.

        Any network failure (URLError, non-304 HTTPError, timeout, or a
        connection error mid-read such as ConnectionResetError or
        IncompleteRead) or response JSON decode error is wrapped as
        OpenModelDBError.
        """
        headers = {"User-Agent": USER_AGENT}
        meta = self._load_meta() if os.path.exists(self._cache_file) else None
        if meta:
            etag = meta.get("etag")
            last_modified = meta.get("last_modified")
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        req = urllib.request.Request(API_URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode()
                resp_headers = resp.headers
        except urllib.error.HTTPError as e:
            if e.code == 304:
                now = time.time()
                os.utime(self._cache_file, (now, now))
                try:
                    return self._load_cache()
                except json.JSONDecodeError as je:
                    raise OpenModelDBError(
                        f"Cache file corrupt after 304 response from {API_URL}"
                    ) from je
            raise OpenModelDBError(f"Failed to fetch {API_URL}: {e}") from e
        except urllib.error.URLError as e:
            raise OpenModelDBError(f"Failed to fetch {API_URL}: {e.reason}") from e
        except TimeoutError as e:
            raise OpenModelDBError(f"Timed out while fetching {API_URL}: {e}") from e
        except (OSError, http.client.HTTPException) as e:
            raise OpenModelDBError(f"Connection error while fetching {API_URL}: {e}") from e

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise OpenModelDBError(f"Invalid JSON received from {API_URL}: {e}") from e

        self._save_cache(data, resp_headers)
        return data

    def _fetch(self) -> dict:
        """Fetch all models from the API or cache.

        Uses a valid local cache when present. Otherwise fetches from the
        API; if that fetch fails and a (possibly expired) cache file
        exists, falls back to it with a warning printed to stderr rather
        than raising. Only raises OpenModelDBError when there is no cache
        to fall back to.
        """
        if self._raw_data is not None:
            return self._raw_data

        if self._cache_is_valid():
            try:
                self._raw_data = self._load_cache()
                return self._raw_data
            except json.JSONDecodeError:
                self._discard_cache()

        try:
            self._raw_data = self._fetch_remote()
        except OpenModelDBError as e:
            if os.path.exists(self._cache_file):
                try:
                    data = self._load_cache()
                except json.JSONDecodeError:
                    self._discard_cache()
                    raise
                mtime = os.path.getmtime(self._cache_file)
                date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                reason = str(e.__cause__) if e.__cause__ is not None else str(e)
                print(
                    f"warning: failed to fetch openmodeldb API ({reason}), "
                    f"using stale cache from {date_str}",
                    file=sys.stderr,
                )
                self._raw_data = data
            else:
                raise

        return self._raw_data

    def refresh(self):
        """Force re-fetch from the API, ignoring cache freshness.

        Unlike the implicit fetch used by ``models``/etc., this does not
        fall back to a stale cache on failure — an explicit refresh must
        fail loudly, propagating as OpenModelDBError.
        """
        self._raw_data = None
        self._models = None
        self._raw_data = self._fetch_remote()

    def clear_cache(self):
        """Delete the local cache file (and its metadata sidecar)."""
        self._discard_cache()
        self._raw_data = None
        self._models = None

    def _build_models(self) -> list[Model]:
        """Parse raw data into Model objects."""
        if self._models is not None:
            return self._models

        raw = self._fetch()
        models = []
        for model_id, data in raw.items():
            arch = _sanitize_text(data.get("architecture") or "other").lower()
            if not self._include_all and arch in EXCLUDED_ARCHS:
                continue
            author = data.get("author", "unknown")
            if isinstance(author, list):
                author = ", ".join(author)
            tags = [_sanitize_text(t) for t in data.get("tags", [])]
            models.append(Model(
                id=_sanitize_text(model_id),
                name=_sanitize_text(data.get("name", model_id)),
                author=_sanitize_text(str(author)),
                architecture=arch,
                scale=data.get("scale", 0),
                license=_sanitize_text(data.get("license", "")),
                tags=tags,
                description=_sanitize_text(data.get("description", "")),
                resources=data.get("resources", []),
                data=data,
            ))

        models.sort(key=lambda m: (m.architecture, m.name.lower()))
        self._models = models
        return models

    # ─── Dunder methods ───────────────────────────────────────────────────

    @property
    def models(self) -> list[Model]:
        """All models in the database."""
        return self._build_models()

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        return f"<OpenModelDB: {len(self)} models>"

    def __iter__(self):
        return iter(self.models)

    def __contains__(self, item: str) -> bool:
        """Check if a model name or id exists: '4xNomos8k' in db."""
        q = item.lower()
        return any(q in m.name.lower() or q in m.id.lower() for m in self.models)

    def __getitem__(self, key: str) -> Model:
        """Get a model by name or id: db['4xNomos8k_atd_jpg']."""
        return self._resolve_model(key)

    # ─── Public API ──────────────────────────────────────────────────────

    def find(
        self,
        scale: int | None = None,
        architecture: str | None = None,
        tag: str | None = None,
    ) -> list[Model]:
        """
        Find models matching the given filters.

        Args:
            scale: Filter by scale factor (1, 2, 4, etc.)
            architecture: Filter by architecture name (esrgan, compact, span, etc.)
            tag: Filter by tag (denoise, anime, photo, etc.)

        Returns:
            List of matching Model objects.
        """
        result = self.models
        if scale is not None:
            result = [m for m in result if m.scale == scale]
        if architecture is not None:
            arch = architecture.lower()
            result = [m for m in result if m.architecture == arch]
        if tag is not None:
            t = tag.lower()
            result = [m for m in result if t in [x.lower() for x in m.tags]]
        return result

    def list(
        self,
        scale: int | None = None,
        architecture: str | None = None,
        tag: str | None = None,
    ) -> list[Model]:
        """
        Display models in a formatted table and return them.

        Args:
            scale: Filter by scale factor (1, 2, 4, etc.)
            architecture: Filter by architecture name (esrgan, compact, span, etc.)
            tag: Filter by tag (denoise, anime, photo, etc.)

        Returns:
            List of matching Model objects.
        """
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        results = self.find(scale=scale, architecture=architecture, tag=tag)

        table = Table(
            title=None,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            row_styles=["", "dim"],
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Name", style="bold white", min_width=20)
        table.add_column("Author", style="white")
        table.add_column("Arch", style="magenta")
        table.add_column("Scale", style="cyan", justify="center")
        table.add_column("Tags", style="dim")

        for i, m in enumerate(results, 1):
            tags = ", ".join(m.tags[:3])
            if len(m.tags) > 3:
                tags += f" +{len(m.tags) - 3}"
            table.add_row(
                str(i),
                escape(m.name),
                escape(m.author),
                escape(m.architecture),
                f"{m.scale}x",
                escape(tags),
            )

        console = Console()
        console.print()
        console.print(table)

        filters = []
        if scale is not None:
            filters.append(f"x{scale}")
        if architecture is not None:
            filters.append(architecture)
        if tag is not None:
            filters.append(f"#{tag}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        console.print(
            f"  [bold]{len(results)}[/bold] models{filter_str}"
            f"  [dim]· {len(self.models)} total[/dim]\n"
        )
        return results

    def search(self, query: str) -> list[Model]:
        """
        Search models by name, author, tags, or description.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching Model objects.
        """
        q = query.lower()
        results = []
        for m in self.models:
            if (q in m.name.lower()
                or q in str(m.author).lower()
                or q in str(m.description).lower()
                or any(q in t.lower() for t in m.tags)):
                results.append(m)
        return results

    def architectures(self) -> list[str]:
        """List all unique architectures in the database."""
        return sorted(set(m.architecture for m in self.models))

    def tags(self) -> list[str]:
        """List all unique tags in the database."""
        all_tags: set[str] = set()
        for m in self.models:
            all_tags.update(m.tags)
        return sorted(all_tags)

    def _resolve_model(self, name: str) -> Model:
        """Resolve a string name/id to a Model object.

        Tries an exact (case-insensitive) id/name match first. If none is
        found, falls back to a partial (substring) match: exactly one
        partial match is returned, but several partial matches are treated
        as an ambiguous query and raise rather than silently picking the
        first one.
        """
        q = name.lower()
        for m in self.models:
            if m.id.lower() == q or m.name.lower() == q:
                return m
        partial = [m for m in self.models if q in m.name.lower() or q in m.id.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            ids = [m.id for m in partial[:5]]
            suffix = f", ... ({len(partial)} matches)" if len(partial) > 5 else ""
            raise ModelNotFoundError(
                f"Ambiguous model name '{name}': matches {', '.join(ids)}{suffix}"
            )
        raise ModelNotFoundError(f"Model not found: '{name}'")

    def get_url(self, model: Model | str, format: str | None = None) -> str:
        """
        Get the download URL for a model.

        Args:
            model: The Model or a model name/id string.
            format: File format ('pth', 'safetensors', 'onnx'). Default: first available.

        Returns:
            Download URL string.
        """
        from openmodeldb.downloader import pick_best_url

        if isinstance(model, str):
            model = self._resolve_model(model)

        res = self._find_resource(model, format)
        urls = res.get("urls", [])
        if not urls:
            raise ValueError(f"No download URLs for {model.name}")
        return pick_best_url(urls)

    def _find_resource(self, model: Model, fmt: str | None) -> dict:
        """Find the best resource matching the requested format."""
        if not model.resources:
            raise ValueError(f"No resources available for {model.name}")

        if fmt is None:
            return model.resources[0]

        fmt = fmt.lower().lstrip(".")
        for res in model.resources:
            if res.get("type", "").lower() == fmt:
                return res

        available = [r.get("type", "?") for r in model.resources]
        raise FormatNotFoundError(f"Format '{fmt}' not found for {model.name}. Available: {', '.join(available)}")

    def _find_convertible_resource(self, model: Model) -> dict:
        """Find a pth or safetensors resource suitable for ONNX conversion."""
        for preferred in ("pth", "safetensors"):
            for res in model.resources:
                if res.get("type", "").lower() == preferred:
                    return res
        for res in model.resources:
            if res.get("type", "").lower() != "onnx":
                return res
        raise FormatNotFoundError(
            f"No PyTorch format available for {model.name} to convert to ONNX."
        )

    def _is_zip_url(self, url: str) -> bool:
        """Check if a URL points to a zip archive."""
        return url.split("?")[0].lower().endswith(".zip")

    @overload
    def _extract_from_zip(
        self, zip_path: str, res: dict, dest_dir: str, target_ext: None = None,
    ) -> str: ...

    @overload
    def _extract_from_zip(
        self, zip_path: str, res: dict, dest_dir: str, target_ext: str,
    ) -> str | None: ...

    def _extract_from_zip(
        self, zip_path: str, res: dict, dest_dir: str, target_ext: str | None = None,
    ) -> str | None:
        """Extract a model file from a zip archive.

        Args:
            zip_path: 
                Path to the zip file.
            res: 
                Resource dict (with 'size' and 'type').
            dest_dir: 
                Directory to write the extracted file.
            target_ext: 
                If set (e.g. ".onnx"), look for a sibling file with this
                extension instead of the resource's own type.
                Returns None if no such file is found. Sibling files
                are not sha256-checked: the API's hash covers the
                resource's own file only.

        Returns:
            Path to the extracted file, or None when target_ext is set and
            no matching file was found. When *res* carries a ``sha256`` and
            ``target_ext`` is None, the extracted file is verified against it
            (mismatch raises DownloadError and removes the file and archive).

        Raises:
            DownloadError: When target_ext is not set and no file can be found.
        """
        import shutil
        import zipfile

        from openmodeldb.downloader import verify_sha256

        expected_size = res.get("size")
        expected_ext = f".{res.get('type', 'pth')}"
        expected_sha = res.get("sha256")
        expansion_limit = _zip_expansion_limit(os.path.getsize(zip_path))

        def _check_expansion(entry, extra: int = 0):
            if entry.file_size + extra > expansion_limit:
                raise DownloadError(
                    f"Refusing to extract {os.path.basename(entry.filename)} "
                    f"({entry.file_size} bytes) from {os.path.basename(zip_path)} "
                    f"({os.path.getsize(zip_path)} bytes): exceeds expansion "
                    f"limit {expansion_limit} bytes (possible zip bomb)."
                )

        def _write(zf, info):
            _check_expansion(info)
            out_name = os.path.basename(info.filename)
            out_path = os.path.join(dest_dir, out_name)
            os.makedirs(dest_dir, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return out_path

        def _write_and_verify(zf, info):
            """Extract *info* and verify it against the resource's sha256.

            The API's hash covers the model file itself, so it applies to the
            resource's own entry (not to sibling files like a pre-built ONNX).
            On mismatch the extracted file and the archive are removed: the
            download was corrupted or tampered with.
            """
            out_path = _write(zf, info)
            if expected_sha:
                try:
                    verify_sha256(out_path, expected_sha)
                except DownloadError:
                    self._cleanup(out_path, zip_path)
                    raise
            return out_path

        with zipfile.ZipFile(zip_path) as zf:
            entries = [i for i in zf.infolist() if not i.is_dir()]

            if target_ext is not None:
                target_ext = target_ext.lower()
                stem = None
                if expected_size:
                    for info in entries:
                        if info.file_size == expected_size:
                            stem = os.path.splitext(os.path.basename(info.filename))[0]
                            break
                if stem:
                    for info in entries:
                        name = os.path.basename(info.filename)
                        fstem, fext = os.path.splitext(name)
                        if fstem == stem and fext.lower() == target_ext:
                            return _write(zf, info)
                return None
            
            if expected_size:
                for info in entries:
                    if info.file_size == expected_size:
                        return _write_and_verify(zf, info)

            for info in entries:
                if os.path.basename(info.filename).lower().endswith(expected_ext):
                    return _write_and_verify(zf, info)

            if entries:
                return _write_and_verify(zf, entries[0])

        raise DownloadError(f"No model file found inside {os.path.basename(zip_path)}")

    def _extract_all_from_zip(
        self, zip_path: str, dest_dir: str, ext_filter: str | None = None,
        res: dict | None = None,
    ) -> list[str]:
        """Extract all model files from a zip archive.

        Args:
            zip_path: Path to the zip file.
            dest_dir: Directory to write extracted files.
            ext_filter: If set (e.g. ".onnx"), only extract files with this extension.
            res: Resource dict. When it carries a ``sha256``/``size``, the
                extracted entry matching that size is verified against the
                hash; a mismatch removes every extracted file and raises.

        Returns:
            List of paths to extracted files.
        """
        import shutil
        import zipfile

        from openmodeldb.downloader import sha256_of_file

        MODEL_EXTS = {".pth", ".safetensors", ".onnx", ".pt", ".bin", ".ckpt"}
        expected_sha = (res or {}).get("sha256")
        expected_size = (res or {}).get("size")
        paths = []
        os.makedirs(dest_dir, exist_ok=True)
        archive_size = os.path.getsize(zip_path)
        expansion_limit = _zip_expansion_limit(archive_size)
        written = 0

        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                _, fext = os.path.splitext(name)
                fext = fext.lower()

                if ext_filter is not None:
                    if fext != ext_filter.lower():
                        continue
                elif fext not in MODEL_EXTS:
                    continue

                if written + info.file_size > expansion_limit:
                    self._cleanup(*paths, zip_path)
                    raise DownloadError(
                        f"Refusing to extract {name} ({info.file_size} bytes, "
                        f"{written} already extracted) from "
                        f"{os.path.basename(zip_path)} "
                        f"({archive_size} bytes): exceeds "
                        f"expansion limit {expansion_limit} bytes "
                        f"(possible zip bomb)."
                    )

                out_path = os.path.join(dest_dir, name)
                with zf.open(info) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                paths.append(out_path)
                written += info.file_size

                if (
                    expected_sha
                    and expected_size is not None
                    and info.file_size == expected_size
                ):
                    actual = sha256_of_file(out_path)
                    if actual != expected_sha.lower():
                        self._cleanup(*paths, zip_path)
                        raise DownloadError(
                            f"SHA-256 mismatch for extracted {name}: expected "
                            f"{expected_sha.lower()}, got {actual}. The archive "
                            f"may be corrupted or tampered with; extracted "
                            f"files have been removed."
                        )

        return paths

    def _print_downloading(
        self, model: Model, file_ext: str, extra: str = "", quiet: bool = False,
    ) -> None:
        """Print the bold 'Downloading ...' status line for a model."""
        if not quiet:
            print(
                f"  Downloading \033[1m{model.name}\033[0m by {model.author} "
                f"({model.architecture}, {model.scale}x) [{file_ext}{extra}]"
            )

    def _download_to_cache(
        self, url: str, file_name: str, res: dict | None = None, quiet: bool = False,
    ) -> str:
        """Download *url* into ``self._cache_dir`` unless a valid copy is cached.

        Returns the path to the cached file. Prints the 'Using cached ...'
        line when a previously-downloaded copy is reused.

        When the resource carries a ``sha256`` (over the model file itself,
        not a zip container) cached copies are revalidated against it: a
        stale or poisoned entry is discarded and re-fetched, and fresh
        downloads are verified before being returned.
        """
        from openmodeldb.downloader import sha256_of_file, smart_download, verify_sha256

        cache_path = os.path.join(self._cache_dir, file_name)
        sha = (res or {}).get("sha256")
        verifiable = bool(sha) and not self._is_zip_url(url)

        if os.path.exists(cache_path):
            if verifiable and sha is not None and sha256_of_file(cache_path) != sha.lower():
                self._cleanup(cache_path)
            else:
                if not quiet:
                    print(f"  Using cached \033[2m{cache_path}\033[0m")
                return cache_path

        smart_download(url, cache_path, quiet=quiet)
        if verifiable and sha is not None:
            verify_sha256(cache_path, sha)
        return cache_path

    def _cleanup(self, *paths: str) -> None:
        """Best-effort removal of one or more paths (duplicates skipped)."""
        for p in set(paths):
            try:
                os.remove(p)
            except OSError:
                pass

    def download(
        self,
        model: Model | str,
        dest: str | None = None,
        format: str | None = None,
        quiet: bool = False,
        half: bool = False,
    ) -> str:
        """
        Download a model file.

        Args:
            model: 
                The Model to download, or a model name/id string.
            dest: 
                Destination directory (default: ./downloads/).
            format: 
                File format to download ('pth', 'safetensors', 'onnx').
                If 'onnx' is requested but not available, downloads
                a PyTorch format and converts automatically.
            quiet: 
                If True, download silently (no prints or progress bar).
            half: 
                If True and converting to ONNX, export in FP16 instead of FP32.

        Returns:
            Path to the downloaded file.
        """
        from openmodeldb.downloader import (
            build_filename,
            pick_best_url,
            safe_filename_component,
            smart_download,
            verify_sha256,
        )

        if isinstance(model, str):
            model = self._resolve_model(model)

        dest_dir = dest or self._download_dir
        need_onnx_convert = False
        need_format_convert = False

        try:
            res = self._find_resource(model, format)
        except FormatNotFoundError:
            fmt_lower = (format or "").lower().lstrip(".")
            if fmt_lower == "onnx":
                res = self._find_convertible_resource(model)
                need_onnx_convert = True
            elif fmt_lower in ("pth", "safetensors"):
                other = "safetensors" if fmt_lower == "pth" else "pth"
                try:
                    res = self._find_resource(model, other)
                    need_format_convert = True
                except FormatNotFoundError:
                    raise
            else:
                raise

        urls = res.get("urls", [])
        if not urls:
            raise ValueError(f"No download URLs for {model.name}")

        dl_url = pick_best_url(urls)
        file_ext = res.get("type", "pth")
        file_name = build_filename(dl_url, model.id, file_ext)
        is_zip = self._is_zip_url(dl_url)

        if need_onnx_convert:
            self._print_downloading(model, file_ext, quiet=quiet)
            cache_path = self._download_to_cache(dl_url, file_name, res=res, quiet=quiet)
            if is_zip:
                onnx_from_zip = self._extract_from_zip(cache_path, res, dest_dir, target_ext=".onnx")
                if onnx_from_zip:
                    self._cleanup(cache_path)
                    if not quiet:
                        size_mb = os.path.getsize(onnx_from_zip) / 1048576
                        print(f"  \033[92m✓\033[0m Extracted \033[2m{onnx_from_zip}\033[0m ({size_mb:.1f} MB)\n")
                    return onnx_from_zip

                if not quiet:
                    print(f"  No ONNX in archive, converting from {file_ext}...")
                model_path = self._extract_from_zip(cache_path, res, self._cache_dir)
            else:
                model_path = cache_path

            onnx_name = safe_filename_component(model.id)
            onnx_path = os.path.join(dest_dir, f"{onnx_name}.onnx")

            if os.path.exists(onnx_path):
                self._cleanup(cache_path, model_path)
                if not quiet:
                    print(
                        f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m "
                        f"ONNX already exists \033[2m({onnx_path})\033[0m"
                    )
                return onnx_path

            from openmodeldb.converter import convert_to_onnx

            onnx_path = convert_to_onnx(
                model_path=model_path,
                output_path=onnx_path,
                half=half,
                quiet=quiet,
            )

            self._cleanup(cache_path, model_path)

            if not quiet:
                print()
            return onnx_path

        if need_format_convert:
            assert format is not None  # need_format_convert implies a format was requested
            fmt_lower = format.lower().lstrip(".")
            self._print_downloading(model, file_ext, extra=f" → {fmt_lower}", quiet=quiet)
            cache_path = self._download_to_cache(dl_url, file_name, res=res, quiet=quiet)

            if is_zip:
                model_path = self._extract_from_zip(cache_path, res, self._cache_dir)
            else:
                model_path = cache_path

            # Build output path
            out_name = f"{safe_filename_component(model.id)}.{fmt_lower}"
            out_path = os.path.join(dest_dir, out_name)

            if os.path.exists(out_path):
                self._cleanup(cache_path, model_path)
                if not quiet:
                    print(f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m already exists \033[2m({out_path})\033[0m")
                return out_path

            # Convert
            from openmodeldb.converter import convert_format

            out_path = convert_format(
                model_path=model_path,
                output_path=out_path,
                target=fmt_lower,
                quiet=quiet,
            )

            self._cleanup(cache_path, model_path)

            if not quiet:
                print()
            return out_path

        if is_zip:
            self._print_downloading(model, file_ext, quiet=quiet)
            cache_path = self._download_to_cache(dl_url, file_name, res=res, quiet=quiet)

            if not quiet:
                print("  Extracting from archive...")
            file_path = self._extract_from_zip(cache_path, res, dest_dir)

            self._cleanup(cache_path)

            if os.path.exists(file_path):
                if not quiet:
                    print(f"  \033[92m✓\033[0m Saved to \033[2m{file_path}\033[0m\n")
            return file_path

        file_path = os.path.join(dest_dir, file_name)

        if os.path.exists(file_path):
            if not quiet:
                print(f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m already exists \033[2m({file_path})\033[0m")
            return file_path

        self._print_downloading(model, file_ext, quiet=quiet)
        smart_download(dl_url, file_path, quiet=quiet)
        if res.get("sha256"):
            verify_sha256(file_path, res["sha256"])
        if not quiet:
            print(f"  \033[92m✓\033[0m Saved to \033[2m{file_path}\033[0m\n")
        return file_path

    def download_all(
        self,
        model: Model | str,
        dest: str | None = None,
        format: str | None = None,
        quiet: bool = False,
    ) -> list[str]:
        """
        Download all files for a model.

        When the resource is a zip archive, extracts all model files from it.
        Use ``format`` to filter by extension (e.g. ``"onnx"``).

        Args:
            model: 
                The Model or a model name/id string.
            dest: 
                Destination directory (default: ./downloads/).
            format: 
                Only extract files with this extension (e.g. "onnx", "pth").
            quiet: 
                If True, suppress output.

        Returns:
            List of paths to downloaded/extracted files.
        """
        from openmodeldb.downloader import build_filename, pick_best_url, smart_download

        if isinstance(model, str):
            model = self._resolve_model(model)

        dest_dir = dest or self._download_dir
        ext_filter = f".{format.lower().lstrip('.')}" if format else None
        all_paths: list[str] = []
        seen_urls: set[str] = set()

        for res in model.resources:
            urls = res.get("urls", [])
            if not urls:
                continue

            dl_url = pick_best_url(urls)
            if dl_url in seen_urls:
                continue
            seen_urls.add(dl_url)

            file_ext = res.get("type", "pth")
            file_name = build_filename(dl_url, model.id, file_ext)

            if self._is_zip_url(dl_url):
                cache_path = os.path.join(self._cache_dir, file_name)

                if not quiet:
                    label = ext_filter or "all"
                    print(f"  Downloading \033[1m{model.name}\033[0m [{label}]")

                if not os.path.exists(cache_path):
                    smart_download(dl_url, cache_path, quiet=quiet)

                if not quiet:
                    print("  Extracting from archive...")

                try:
                    extracted = self._extract_all_from_zip(cache_path, dest_dir, ext_filter, res)
                finally:
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                all_paths.extend(extracted)

                if not quiet:
                    for p in extracted:
                        size_mb = os.path.getsize(p) / 1048576
                        print(f"  \033[92m✓\033[0m \033[2m{os.path.basename(p)}\033[0m ({size_mb:.1f} MB)")
                    print()
            else:
                if ext_filter and not file_name.lower().endswith(ext_filter):
                    continue
                all_paths.append(self.download(model, dest=dest, format=file_ext, quiet=quiet))

        return all_paths

    def test_integrity(
        self,
        file_path: str,
        quiet: bool = False,
    ) -> dict:
        """
        Compare the weights of a local model file against the reference from
        the database.

        Downloads the reference model to the cache directory, compares all
        weight tensors, then cleans up the cached file.

        Args:
            file_path: Path to the local model file (.pth, .safetensors, or .onnx).
            quiet: If True, suppress progress output.

        Returns:
            A dict with keys: ``matched``, ``total_a``, ``total_b``,
            ``max_diff``, ``mean_diff``, ``identical``, ``similarity``.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ModelNotFoundError: If the model cannot be resolved from the filename.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        from openmodeldb.converter import compare_weights
        from openmodeldb.downloader import build_filename, pick_best_url, smart_download, verify_sha256

        basename = os.path.basename(file_path)
        stem = os.path.splitext(basename)[0]

        model = self._resolve_model(stem)

        local_ext = os.path.splitext(basename)[1].lower().lstrip(".")
        ref_res = None
        for res in model.resources:
            if res.get("type", "pth").lower() == local_ext:
                ref_res = res
                break
        if ref_res is None:
            for res in model.resources:
                urls = res.get("urls", [])
                if urls and not self._is_zip_url(pick_best_url(urls)):
                    ref_res = res
                    break
        if ref_res is None:
            for res in model.resources:
                if res.get("urls"):
                    ref_res = res
                    break

        if ref_res is None:
            raise ValueError(f"No downloadable resource for {model.name}")

        urls = ref_res.get("urls", [])
        dl_url = pick_best_url(urls)
        file_ext = ref_res.get("type", "pth")
        ref_name = build_filename(dl_url, model.id, file_ext)
        ref_path = os.path.join(self._cache_dir, ref_name)

        is_zip = self._is_zip_url(dl_url)

        if not quiet:
            print(f"  Checking \033[1m{model.name}\033[0m integrity...")

        if not os.path.exists(ref_path):
            smart_download(dl_url, ref_path, quiet=quiet)
            if not is_zip and ref_res.get("sha256"):
                verify_sha256(ref_path, ref_res["sha256"])
        elif not quiet:
            print(f"  Using cached \033[2m{ref_path}\033[0m")

        if is_zip:
            extracted = self._extract_from_zip(ref_path, ref_res, self._cache_dir)
            try:
                os.remove(ref_path)
            except OSError:
                pass
            ref_path = extracted

        result = compare_weights(file_path, ref_path, quiet=quiet)

        try:
            os.remove(ref_path)
        except OSError:
            pass

        if not quiet:
            sim = result["similarity"]
            status = "\033[92m✓ PASS\033[0m" if result["identical"] else (
                "\033[93m~ CLOSE\033[0m" if sim > 99.9 else "\033[91m✗ FAIL\033[0m"
            )
            print(f"  {status}  similarity={sim:.6f}  "
                  f"matched={result['matched']}/{result['total_a']}  "
                  f"max_diff={result['max_diff']:.2e}  "
                  f"mean_diff={result['mean_diff']:.2e}")
            print()

        return result

    def interactive(self):
        """Launch the interactive CLI for browsing and downloading models."""
        from openmodeldb.cli import interactive as _interactive
        _interactive(self)
