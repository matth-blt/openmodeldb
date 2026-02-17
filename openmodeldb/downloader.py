"""
Download handlers for various file hosting services.
"""

import os
import re
import urllib.request

from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def _download_with_progress(resp, dest: str, total: int | None = None, transform=None, quiet: bool = False):
    """Download response to file with optional rich progress bar.

    Args:
        resp: HTTP response object.
        dest: Destination file path.
        total: Total size in bytes (None if unknown).
        transform: Optional callable to transform each chunk (e.g. cipher.decrypt).
        quiet: If True, download silently without progress bar.
    """
    chunk_size = 64 * 1024
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if quiet:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                if transform:
                    chunk = transform(chunk)
                f.write(chunk)
        return

    if total:
        columns = (
            TextColumn("  "),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(elapsed_when_finished=True),
        )
    else:
        columns = (
            TextColumn("  "),
            SpinnerColumn("line", speed=1.5),
            FileSizeColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
        )

    progress = Progress(*columns, refresh_per_second=5)
    task_id = progress.add_task("", total=total or float("inf"))

    with progress, open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            if transform:
                chunk = transform(chunk)
            f.write(chunk)
            progress.update(task_id, advance=len(chunk))


def is_mega_url(url: str) -> bool:
    return "mega.nz" in url or "mega.co.nz" in url


def is_mediafire_url(url: str) -> bool:
    return "mediafire.com" in url


def is_gdrive_url(url: str) -> bool:
    return "drive.google.com" in url


def _convert_gdrive_url(url: str) -> str:
    """Convert Google Drive view/open links to direct download links."""
    match = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url


def download_mega(url: str, dest: str, quiet: bool = False):
    """Download from Mega.nz using native crypto (no external Mega lib)."""
    import base64
    import json
    import struct
    from Crypto.Cipher import AES

    def _mega_base64_decode(s):
        s += "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode(s)

    def _mega_key(key_str):
        key = _mega_base64_decode(key_str)
        if len(key) == 32:
            return bytes(a ^ b for a, b in zip(key[:16], key[16:]))
        return key[:16]

    def _mega_parse_url(url):
        """Extract file ID and key from a Mega.nz URL."""
        import re
        # Handle mega.nz/file/ID#KEY and mega.nz/#!ID!KEY formats
        m = re.search(r"mega\.nz/file/([^#]+)#(.+)", url)
        if m:
            return m.group(1), m.group(2)
        m = re.search(r"mega\.nz/#!([^!]+)!(.+)", url)
        if m:
            return m.group(1), m.group(2)
        raise ValueError(f"Cannot parse Mega URL: {url}")

    file_id, key_str = _mega_parse_url(url)
    key = _mega_key(key_str)

    # Get file info from Mega API
    api_url = "https://g.api.mega.co.nz/cs"
    payload = json.dumps([{"a": "g", "g": 1, "p": file_id}]).encode()
    req = urllib.request.Request(
        f"{api_url}?id=0", data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "OpenModelDB-Py/1.0"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode())

    if isinstance(result, int) or (isinstance(result, list) and isinstance(result[0], int)):
        raise RuntimeError(f"Mega API error: {result}")

    dl_url = result[0]["g"]

    # Download encrypted file
    req = urllib.request.Request(dl_url, headers={"User-Agent": "OpenModelDB-Py/1.0"})
    resp = urllib.request.urlopen(req, timeout=300)

    total = resp.headers.get("Content-Length")
    total = int(total) if total else None

    raw_key = _mega_base64_decode(key_str)
    if len(raw_key) == 32:
        k = [struct.unpack(">I", raw_key[i:i+4])[0] for i in range(0, 32, 4)]
        iv_ints = [k[4] ^ k[6], k[5] ^ k[7]]
    else:
        k = [struct.unpack(">I", raw_key[i:i+4])[0] for i in range(0, 16, 4)]
        iv_ints = [0, 0]

    iv = struct.pack(">II", iv_ints[0], iv_ints[1]) + b"\x00" * 8
    cipher = AES.new(key, AES.MODE_CTR, initial_value=iv, nonce=b"")

    _download_with_progress(resp, dest, total, transform=cipher.decrypt, quiet=quiet)


def download_mediafire(url: str, dest: str, quiet: bool = False):
    """Download from MediaFire using mediafiredl."""
    try:
        from mediafiredl import MediaFire
    except ImportError:
        raise ImportError("mediafiredl not installed. Run: pip install mediafiredl")

    mf = MediaFire()
    direct_url = mf.get_link(url)
    if not direct_url:
        raise RuntimeError("Could not extract MediaFire direct link")
    download_direct(direct_url, dest, quiet=quiet)


def download_direct(url: str, dest: str, quiet: bool = False):
    """Download a file with optional rich progress bar."""
    url = _convert_gdrive_url(url)

    req = urllib.request.Request(url, headers={"User-Agent": "OpenModelDB-Py/1.0"})
    resp = urllib.request.urlopen(req, timeout=120)

    total = resp.headers.get("Content-Length")
    total = int(total) if total else None

    _download_with_progress(resp, dest, total, quiet=quiet)


def smart_download(url: str, dest: str, quiet: bool = False):
    """Route to the appropriate download handler based on URL."""
    if is_mega_url(url):
        download_mega(url, dest, quiet=quiet)
    elif is_mediafire_url(url):
        download_mediafire(url, dest, quiet=quiet)
    else:
        download_direct(url, dest, quiet=quiet)


def pick_best_url(urls: list[str]) -> str:
    """Pick the best URL, preferring direct download hosts."""
    priority = [
        "objectstorage",
        "github.com",
        "huggingface.co",
        "drive.google.com",
        "mediafire.com",
        "mega.nz",
        "mega.co.nz",
    ]
    for host in priority:
        for url in urls:
            if host in url:
                return url
    return urls[0]


def build_filename(url: str, model_id: str, file_ext: str) -> str:
    """Build a sane filename from URL, model ID, and extension."""
    url_filename = url.split("/")[-1].split("?")[0]
    url_filename = os.path.basename(url_filename)  # prevent path traversal
    if url_filename and "." in url_filename and len(url_filename) > 3:
        if not url_filename.startswith("uc") and not url_filename.startswith("#"):
            return url_filename
    return f"{model_id}.{file_ext}"


def fmt_size(b: int) -> str:
    """Format bytes to human-readable size."""
    if not b:
        return "?"
    if b < 1024:
        return f"{b} B"
    if b < 1048576:
        return f"{b / 1024:.1f} KB"
    if b < 1073741824:
        return f"{b / 1048576:.1f} MB"
    return f"{b / 1073741824:.2f} GB"
