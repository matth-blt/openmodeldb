"""
OpenModelDB Client — core API for fetching and querying models.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field


API_URL = "https://openmodeldb.info/api/v1/models.json"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "openmodeldb")
DEFAULT_DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
CACHE_MAX_AGE = 3600
EXCLUDED_ARCHS = {"cain", "cain-yuv"}


# ─── Exceptions ──────────────────────────────────────────────────────────

class OpenModelDBError(Exception):
    """Base exception for OpenModelDB."""

class ModelNotFoundError(OpenModelDBError):
    """Raised when a model name/id cannot be resolved."""

class FormatNotFoundError(OpenModelDBError):
    """Raised when a requested format is not available for a model."""

class DownloadError(OpenModelDBError):
    """Raised when a download fails."""


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

    def __init__(self, cache_dir: str | None = None, download_dir: str | None = None):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_file = os.path.join(self._cache_dir, "models.json")
        self._download_dir = download_dir or DEFAULT_DOWNLOAD_DIR
        self._raw_data: dict | None = None
        self._models: list[Model] | None = None

    # ─── Data loading ────────────────────────────────────────────────────

    def _cache_is_valid(self) -> bool:
        if not os.path.exists(self._cache_file):
            return False
        return (time.time() - os.path.getmtime(self._cache_file)) < CACHE_MAX_AGE

    def _load_cache(self) -> dict:
        with open(self._cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, data: dict):
        os.makedirs(self._cache_dir, exist_ok=True)
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _fetch(self) -> dict:
        """Fetch all models from the API or cache."""
        if self._raw_data is not None:
            return self._raw_data

        if self._cache_is_valid():
            self._raw_data = self._load_cache()
        else:
            req = urllib.request.Request(API_URL, headers={"User-Agent": "OpenModelDB-Py/1.0"})
            resp = urllib.request.urlopen(req, timeout=60)
            self._raw_data = json.loads(resp.read().decode())
            self._save_cache(self._raw_data)

        return self._raw_data

    def refresh(self):
        """Force re-fetch from the API, ignoring cache."""
        self._raw_data = None
        self._models = None
        req = urllib.request.Request(API_URL, headers={"User-Agent": "OpenModelDB-Py/1.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        self._raw_data = json.loads(resp.read().decode())
        self._save_cache(self._raw_data)

    def clear_cache(self):
        """Delete the local cache file."""
        if os.path.exists(self._cache_file):
            os.remove(self._cache_file)
        self._raw_data = None
        self._models = None

    def _build_models(self) -> list[Model]:
        """Parse raw data into Model objects."""
        if self._models is not None:
            return self._models

        raw = self._fetch()
        models = []
        for model_id, data in raw.items():
            arch = (data.get("architecture") or "other").lower()
            if arch in EXCLUDED_ARCHS:
                continue
            author = data.get("author", "unknown")
            if isinstance(author, list):
                author = ", ".join(author)
            models.append(Model(
                id=model_id,
                name=data.get("name", model_id),
                author=str(author),
                architecture=arch,
                scale=data.get("scale", 0),
                license=data.get("license", ""),
                tags=data.get("tags", []),
                description=data.get("description", ""),
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
                m.name,
                m.author,
                m.architecture,
                f"{m.scale}x",
                tags,
            )

        console = Console()
        console.print()
        console.print(table)

        # Summary line
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
        """Resolve a string name/id to a Model object."""
        q = name.lower()
        # Exact match first
        for m in self.models:
            if m.id.lower() == q or m.name.lower() == q:
                return m
        # Partial match
        for m in self.models:
            if q in m.name.lower() or q in m.id.lower():
                return m
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
        # Prefer pth, then safetensors
        for preferred in ("pth", "safetensors"):
            for res in model.resources:
                if res.get("type", "").lower() == preferred:
                    return res
        # Fallback to first non-onnx resource
        for res in model.resources:
            if res.get("type", "").lower() != "onnx":
                return res
        raise FormatNotFoundError(
            f"No PyTorch format available for {model.name} to convert to ONNX."
        )

    def _is_zip_url(self, url: str) -> bool:
        """Check if a URL points to a zip archive."""
        return url.split("?")[0].lower().endswith(".zip")

    def _extract_from_zip(
        self, zip_path: str, res: dict, dest_dir: str, target_ext: str | None = None,
    ) -> str | None:
        """Extract a model file from a zip archive.

        Args:
            zip_path: Path to the zip file.
            res: Resource dict (with 'size' and 'type').
            dest_dir: Directory to write the extracted file.
            target_ext: If set (e.g. ".onnx"), look for a sibling file with this
                        extension instead of the resource's own type.
                        Returns None if no such file is found.

        Returns:
            Path to the extracted file, or None when target_ext is set and
            no matching file was found.

        Raises:
            DownloadError: When target_ext is not set and no file can be found.
        """
        import zipfile

        expected_size = res.get("size")
        expected_ext = f".{res.get('type', 'pth')}"

        def _write(zf, info):
            out_name = os.path.basename(info.filename)
            out_path = os.path.join(dest_dir, out_name)
            os.makedirs(dest_dir, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            return out_path

        with zipfile.ZipFile(zip_path) as zf:
            entries = [i for i in zf.infolist() if not i.is_dir()]

            if target_ext is not None:
                # Looking for a sibling file (e.g. .onnx next to .pth).
                # Find the resource file by size to get its stem, then
                # look for stem + target_ext.
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

            # Normal extraction: match by size → extension → first file
            if expected_size:
                for info in entries:
                    if info.file_size == expected_size:
                        return _write(zf, info)

            for info in entries:
                if os.path.basename(info.filename).lower().endswith(expected_ext):
                    return _write(zf, info)

            if entries:
                return _write(zf, entries[0])

        raise DownloadError(f"No model file found inside {os.path.basename(zip_path)}")

    def _extract_all_from_zip(
        self, zip_path: str, dest_dir: str, ext_filter: str | None = None,
    ) -> list[str]:
        """Extract all model files from a zip archive.

        Args:
            zip_path: Path to the zip file.
            dest_dir: Directory to write extracted files.
            ext_filter: If set (e.g. ".onnx"), only extract files with this extension.

        Returns:
            List of paths to extracted files.
        """
        import zipfile

        MODEL_EXTS = {".pth", ".safetensors", ".onnx", ".pt", ".bin", ".ckpt"}
        paths = []
        os.makedirs(dest_dir, exist_ok=True)

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

                out_path = os.path.join(dest_dir, name)
                with zf.open(info) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                paths.append(out_path)

        return paths

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
            model: The Model to download, or a model name/id string.
            dest: Destination directory (default: ./downloads/).
            format: File format to download ('pth', 'safetensors', 'onnx').
                    If 'onnx' is requested but not available, downloads
                    a PyTorch format and converts automatically.
            quiet: If True, download silently (no prints or progress bar).
            half: If True and converting to ONNX, export in FP16 instead of FP32.

        Returns:
            Path to the downloaded file.
        """
        from openmodeldb.downloader import smart_download, pick_best_url, build_filename

        # Resolve string to Model
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
                # Try to find the other PyTorch format and convert
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
            if not quiet:
                print(f"  Downloading \033[1m{model.name}\033[0m by {model.author} ({model.architecture}, {model.scale}x) [{file_ext}]")

            # Download to cache
            cache_path = os.path.join(self._cache_dir, file_name)
            if not os.path.exists(cache_path):
                smart_download(dl_url, cache_path, quiet=quiet)
            elif not quiet:
                print(f"  Using cached \033[2m{cache_path}\033[0m")

            # If zip, try to find a pre-built ONNX inside first
            if is_zip:
                onnx_from_zip = self._extract_from_zip(cache_path, res, dest_dir, target_ext=".onnx")
                if onnx_from_zip:
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                    if not quiet:
                        size_mb = os.path.getsize(onnx_from_zip) / 1048576
                        print(f"  \033[92m✓\033[0m Extracted \033[2m{onnx_from_zip}\033[0m ({size_mb:.1f} MB)\n")
                    return onnx_from_zip

                # No ONNX in archive, extract source model for conversion
                if not quiet:
                    print(f"  No ONNX in archive, converting from {file_ext}...")
                model_path = self._extract_from_zip(cache_path, res, self._cache_dir)
            else:
                model_path = cache_path

            # Build ONNX output path
            onnx_name = model.id
            onnx_path = os.path.join(dest_dir, f"{onnx_name}.onnx")

            if os.path.exists(onnx_path):
                for p in {cache_path, model_path}:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                if not quiet:
                    print(f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m ONNX already exists \033[2m({onnx_path})\033[0m")
                return onnx_path

            # Convert to ONNX
            from openmodeldb.converter import convert_to_onnx

            onnx_path = convert_to_onnx(
                model_path=model_path,
                output_path=onnx_path,
                half=half,
                quiet=quiet,
            )

            # Clean up cached files
            for p in {cache_path, model_path}:
                try:
                    os.remove(p)
                except OSError:
                    pass

            if not quiet:
                print()
            return onnx_path

        if need_format_convert:
            fmt_lower = format.lower().lstrip(".")
            if not quiet:
                print(f"  Downloading \033[1m{model.name}\033[0m by {model.author} ({model.architecture}, {model.scale}x) [{file_ext} → {fmt_lower}]")

            # Download source to cache
            cache_path = os.path.join(self._cache_dir, file_name)
            if not os.path.exists(cache_path):
                smart_download(dl_url, cache_path, quiet=quiet)
            elif not quiet:
                print(f"  Using cached \033[2m{cache_path}\033[0m")

            if is_zip:
                model_path = self._extract_from_zip(cache_path, res, self._cache_dir)
            else:
                model_path = cache_path

            # Build output path
            out_name = f"{model.id}.{fmt_lower}"
            out_path = os.path.join(dest_dir, out_name)

            if os.path.exists(out_path):
                for p in {cache_path, model_path}:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
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

            # Clean up cached files
            for p in {cache_path, model_path}:
                try:
                    os.remove(p)
                except OSError:
                    pass

            if not quiet:
                print()
            return out_path

        # Normal (non-convert) download
        if is_zip:
            # Download zip to cache, extract the model to dest
            cache_path = os.path.join(self._cache_dir, file_name)

            if not quiet:
                print(f"  Downloading \033[1m{model.name}\033[0m by {model.author} ({model.architecture}, {model.scale}x) [{file_ext}]")

            if not os.path.exists(cache_path):
                smart_download(dl_url, cache_path, quiet=quiet)

            if not quiet:
                print(f"  Extracting from archive...")
            file_path = self._extract_from_zip(cache_path, res, dest_dir)

            # Clean up the zip
            try:
                os.remove(cache_path)
            except OSError:
                pass

            if os.path.exists(file_path):
                if not quiet:
                    print(f"  \033[92m✓\033[0m Saved to \033[2m{file_path}\033[0m\n")
            return file_path

        file_path = os.path.join(dest_dir, file_name)

        if os.path.exists(file_path):
            if not quiet:
                print(f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m already exists \033[2m({file_path})\033[0m")
            return file_path

        if not quiet:
            print(f"  Downloading \033[1m{model.name}\033[0m by {model.author} ({model.architecture}, {model.scale}x) [{file_ext}]")
        smart_download(dl_url, file_path, quiet=quiet)
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
            model: The Model or a model name/id string.
            dest: Destination directory (default: ./downloads/).
            format: Only extract files with this extension (e.g. "onnx", "pth").
            quiet: If True, suppress output.

        Returns:
            List of paths to downloaded/extracted files.
        """
        from openmodeldb.downloader import smart_download, pick_best_url, build_filename

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
                # Download zip to cache, extract all (or filtered) model files
                cache_path = os.path.join(self._cache_dir, file_name)

                if not quiet:
                    label = ext_filter or "all"
                    print(f"  Downloading \033[1m{model.name}\033[0m [{label}]")

                if not os.path.exists(cache_path):
                    smart_download(dl_url, cache_path, quiet=quiet)

                if not quiet:
                    print(f"  Extracting from archive...")

                extracted = self._extract_all_from_zip(cache_path, dest_dir, ext_filter)
                all_paths.extend(extracted)

                try:
                    os.remove(cache_path)
                except OSError:
                    pass

                if not quiet:
                    for p in extracted:
                        size_mb = os.path.getsize(p) / 1048576
                        print(f"  \033[92m✓\033[0m \033[2m{os.path.basename(p)}\033[0m ({size_mb:.1f} MB)")
                    print()
            else:
                # Non-zip: respect ext_filter
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
        from openmodeldb.downloader import smart_download, pick_best_url, build_filename

        basename = os.path.basename(file_path)
        stem = os.path.splitext(basename)[0]

        # Resolve model from filename
        model = self._resolve_model(stem)

        # Pick the best resource to use as reference
        local_ext = os.path.splitext(basename)[1].lower().lstrip(".")
        ref_res = None
        # Prefer same format
        for res in model.resources:
            if res.get("type", "pth").lower() == local_ext:
                ref_res = res
                break
        # Fallback to any non-zip resource
        if ref_res is None:
            for res in model.resources:
                urls = res.get("urls", [])
                if urls and not self._is_zip_url(pick_best_url(urls)):
                    ref_res = res
                    break
        # Fallback to first resource with URLs
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

        # Download reference to cache
        if not os.path.exists(ref_path):
            smart_download(dl_url, ref_path, quiet=quiet)
        elif not quiet:
            print(f"  Using cached \033[2m{ref_path}\033[0m")

        # If zip, extract
        if is_zip:
            extracted = self._extract_from_zip(ref_path, ref_res, self._cache_dir)
            try:
                os.remove(ref_path)
            except OSError:
                pass
            ref_path = extracted

        # Compare
        result = compare_weights(file_path, ref_path, quiet=quiet)

        # Clean up
        try:
            os.remove(ref_path)
        except OSError:
            pass

        # Print results
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
        from openmodeldb.cli import main
        main(self)
