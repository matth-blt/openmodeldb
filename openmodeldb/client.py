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

        try:
            res = self._find_resource(model, format)
        except FormatNotFoundError:
            if format and format.lower().lstrip(".") == "onnx":
                # ONNX not available — download a PyTorch format and convert
                res = self._find_convertible_resource(model)
                need_onnx_convert = True
            else:
                raise

        urls = res.get("urls", [])
        if not urls:
            raise ValueError(f"No download URLs for {model.name}")

        dl_url = pick_best_url(urls)
        file_ext = res.get("type", "pth")
        file_name = build_filename(dl_url, model.id, file_ext)

        if need_onnx_convert:
            # Download the intermediate PyTorch file into the cache directory
            cache_path = os.path.join(self._cache_dir, file_name)
            # Final ONNX goes into the destination directory
            onnx_name = os.path.splitext(file_name)[0]
            onnx_path = os.path.join(dest_dir, f"{onnx_name}.onnx")

            if os.path.exists(onnx_path):
                if not quiet:
                    print(f"  \033[92m✓\033[0m \033[1m{model.name}\033[0m ONNX already exists \033[2m({onnx_path})\033[0m")
                return onnx_path

            if not quiet:
                print(f"  Downloading \033[1m{model.name}\033[0m by {model.author} ({model.architecture}, {model.scale}x) [{file_ext}] → ONNX conversion")

            # Download to cache (may already be cached)
            if not os.path.exists(cache_path):
                smart_download(dl_url, cache_path, quiet=quiet)
            elif not quiet:
                print(f"  Using cached \033[2m{cache_path}\033[0m")

            # Convert to ONNX
            from openmodeldb.converter import convert_to_onnx

            onnx_path = convert_to_onnx(
                model_path=cache_path,
                output_path=onnx_path,
                half=half,
                quiet=quiet,
            )

            # Clean up the cached PyTorch file
            try:
                os.remove(cache_path)
            except OSError:
                pass

            if not quiet:
                print()
            return onnx_path

        # Normal (non-convert) download
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
        quiet: bool = False,
    ) -> list[str]:
        """
        Download all available formats for a model.

        Returns:
            List of paths to downloaded files.
        """
        if isinstance(model, str):
            model = self._resolve_model(model)

        paths = []
        for res in model.resources:
            fmt = res.get("type", "pth")
            paths.append(self.download(model, dest=dest, format=fmt, quiet=quiet))
        return paths

    def interactive(self):
        """Launch the interactive CLI for browsing and downloading models."""
        from openmodeldb.cli import main
        main(self)
