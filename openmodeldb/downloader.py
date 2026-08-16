"""
Download handlers for various file hosting services.
"""

import hashlib
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlparse

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

from openmodeldb.exceptions import DownloadError

USER_AGENT = "OpenModelDB-Py/1.2.0"


def _url_hostname(url: str) -> str:
    """Lowercased hostname of *url* ("" if unparseable)."""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _host_is(host: str, domain: str) -> bool:
    """True when *host* is exactly *domain* or a subdomain of it."""
    return host == domain or host.endswith("." + domain)


def _open_url(url: str, req: "urllib.request.Request | None" = None, timeout: float | None = None):
    """Open a URL via urllib, wrapping network errors as DownloadError.

    Only http/https URLs are accepted — urllib would happily fetch
    ``file://`` or ``ftp://`` URLs coming from untrusted API data otherwise.

    Returns the response object (a context manager) on success. Any
    HTTPError, URLError, or timeout is caught and re-raised as a
    DownloadError carrying the URL and underlying reason.
    """
    request = req if req is not None else urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    scheme = urlparse(request.full_url).scheme.lower()
    if scheme not in ("http", "https"):
        raise DownloadError(
            f"Unsupported URL scheme {scheme!r} (only http/https allowed): {url}"
        )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise DownloadError(f"HTTP error {e.code} while downloading {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise DownloadError(f"Failed to download {url}: {e.reason}") from e
    except TimeoutError as e:
        raise DownloadError(f"Timed out while downloading {url}: {e}") from e


def _download_with_progress(resp, dest: str, total: int | None = None, transform=None, quiet: bool = False):
    """Download response to file with optional rich progress bar.

    Writes to a ``dest + ".part"`` temp file and atomically renames it to
    ``dest`` only once the download completes successfully. On any error
    the partial file is removed (best-effort) and the exception re-raised,
    so a failed/interrupted download never leaves a truncated file at
    ``dest`` that a later call could mistake for a completed download.

    Args:
        resp: HTTP response object.
        dest: Destination file path.
        total: Total size in bytes (None if unknown).
        transform: Optional callable to transform each chunk (e.g. cipher.decrypt).
        quiet: If True, download silently without progress bar.
    """
    chunk_size = 64 * 1024
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part_path = dest + ".part"

    try:
        if quiet:
            with open(part_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    if transform:
                        chunk = transform(chunk)
                    f.write(chunk)
        else:
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

            with progress, open(part_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    if transform:
                        chunk = transform(chunk)
                    f.write(chunk)
                    progress.update(task_id, advance=len(chunk))
    except BaseException:
        try:
            os.remove(part_path)
        except OSError:
            pass
        raise

    os.replace(part_path, dest)


def sha256_of_file(path: str) -> str:
    """Compute the SHA-256 hex digest of a file (chunked read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str, expected: str, remove_on_mismatch: bool = True) -> None:
    """Verify a downloaded file's SHA-256 against the expected digest.

    Raises DownloadError on mismatch (deleting the file first, best-effort,
    unless *remove_on_mismatch* is False). Comparison is case-insensitive.
    """
    actual = sha256_of_file(path)
    if actual != expected.lower():
        if remove_on_mismatch:
            try:
                os.remove(path)
            except OSError:
                pass
        raise DownloadError(
            f"SHA-256 mismatch for {os.path.basename(path)}: expected "
            f"{expected.lower()}, got {actual}. The download may be corrupted "
            f"or tampered with; the file has been removed."
        )


def is_mega_url(url: str) -> bool:
    h = _url_hostname(url)
    return _host_is(h, "mega.nz") or _host_is(h, "mega.co.nz")


def is_mediafire_url(url: str) -> bool:
    return _host_is(_url_hostname(url), "mediafire.com")


def is_gdrive_url(url: str) -> bool:
    h = _url_hostname(url)
    return (
        _host_is(h, "drive.google.com")
        or _host_is(h, "docs.google.com")
        or _host_is(h, "drive.usercontent.google.com")
    )


def _convert_gdrive_url(url: str) -> str:
    """Convert Google Drive view/open links to direct download links."""
    match = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url


_GDRIVE_FORM_RE = re.compile(r'<form\b[^>]*\bid=["\']download-form["\'][^>]*>', re.I)
_HTML_TAG_ATTR_RE = re.compile(
    r"""([a-zA-Z][\w-]*)\s*=\s*"([^"]*)"|([a-zA-Z][\w-]*)\s*=\s*'([^']*)'""",
    re.S,
)
_INPUT_TAG_RE = re.compile(r"<input\b[^>]*>", re.I)
_ANCHOR_TAG_RE = re.compile(r"<a\b[^>]*>", re.I)


def _parse_html_attrs(tag: str) -> dict:
    """Parse attribute="value" pairs (either quote style) out of an HTML tag."""
    attrs = {}
    for m in _HTML_TAG_ATTR_RE.finditer(tag):
        if m.group(1) is not None:
            attrs[m.group(1).lower()] = m.group(2)
        else:
            attrs[m.group(3).lower()] = m.group(4)
    return attrs


def _parse_gdrive_confirm_form(html: str):
    """Parse Google Drive's "can't scan for viruses" interstitial page.

    Looks for ``<form id="download-form" action="...">`` plus its hidden
    ``<input>`` fields (typically ``id``, ``export``, ``confirm``, ``uuid``).

    Returns:
        (action_url, params_dict), or None if the form couldn't be found/parsed.
    """
    form_match = _GDRIVE_FORM_RE.search(html)
    if not form_match:
        return None

    form_attrs = _parse_html_attrs(form_match.group(0))
    action = form_attrs.get("action")
    if not action:
        return None
    action = action.replace("&amp;", "&")

    body_start = form_match.end()
    end_match = re.search(r"</form>", html[body_start:], re.I)
    form_body = html[body_start:body_start + end_match.start()] if end_match else html[body_start:]

    params = {}
    for input_tag in _INPUT_TAG_RE.findall(form_body):
        input_attrs = _parse_html_attrs(input_tag)
        if input_attrs.get("type", "").lower() != "hidden":
            continue
        name = input_attrs.get("name")
        if not name:
            continue
        params[name] = input_attrs.get("value", "")

    return action, params


# DOCS
# https://github.com/meganz/sdk — Mega file/key handling and AES-CTR layout
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
        m = re.search(r"mega\.nz/file/([^#]+)#(.+)", url)
        if m:
            return m.group(1), m.group(2)
        m = re.search(r"mega\.nz/#!([^!]+)!(.+)", url)
        if m:
            return m.group(1), m.group(2)
        raise DownloadError(f"Cannot parse Mega URL: {url}")

    file_id, key_str = _mega_parse_url(url)
    key = _mega_key(key_str)

    api_url = "https://g.api.mega.co.nz/cs"
    payload = json.dumps([{"a": "g", "g": 1, "p": file_id}]).encode()
    req = urllib.request.Request(
        f"{api_url}?id=0", data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with _open_url(f"{api_url}?id=0", req=req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    if isinstance(result, int) or (isinstance(result, list) and isinstance(result[0], int)):
        raise DownloadError(f"Mega API error: {result}")

    dl_url = result[0]["g"]

    req = urllib.request.Request(dl_url, headers={"User-Agent": USER_AGENT})
    with _open_url(dl_url, req=req, timeout=300) as resp:
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


def _mediafire_direct_url(page_url: str) -> str:
    """Fetch a MediaFire download page and extract the direct download link
    (the href of the ``#downloadButton`` anchor).

    The page HTML is untrusted: the extracted link is only followed when it
    is an https URL on mediafire.com (or a subdomain). Uses a bounded
    timeout and the same attribute parser as the Google Drive interstitial
    handler.
    """
    req = urllib.request.Request(page_url, headers={"User-Agent": USER_AGENT})
    with _open_url(page_url, req=req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    for tag in _ANCHOR_TAG_RE.findall(html):
        attrs = _parse_html_attrs(tag)
        if attrs.get("id") == "downloadButton":
            href = (attrs.get("href") or "").replace("&amp;", "&")
            if (
                href.startswith("https://")
                and _host_is(_url_hostname(href), "mediafire.com")
            ):
                return href
            break

    raise DownloadError(f"Could not extract MediaFire direct link for {page_url}")


def download_mediafire(url: str, dest: str, quiet: bool = False):
    """Download from MediaFire by scraping the direct link from the download
    page, then fetching it like any direct URL."""
    download_direct(_mediafire_direct_url(url), dest, quiet=quiet)


def download_direct(url: str, dest: str, quiet: bool = False):
    """Download a file with optional rich progress bar.

    Handles the Google Drive "can't scan this file for viruses" interstitial
    page that Drive serves instead of the file itself for large (~100 MB+)
    files: when the response for a Drive URL comes back as HTML, the
    confirmation form embedded in that page is parsed and re-submitted to
    get the actual file. The form's action URL is untrusted scraped HTML:
    it is only followed when it is an https URL on google.com (or a
    subdomain).
    """
    url = _convert_gdrive_url(url)
    gdrive = is_gdrive_url(url)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _open_url(url, req=req, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type", "")

        if gdrive and content_type.startswith("text/html"):
            html = resp.read().decode("utf-8", errors="replace")
            parsed = _parse_gdrive_confirm_form(html)
            if parsed is None:
                raise DownloadError(
                    "Google Drive returned an HTML page instead of the file; "
                    "the file is likely too large for Drive to scan for viruses "
                    "and the confirmation form could not be parsed."
                )
            action, params = parsed
            action_host = _url_hostname(action)
            if (
                not action.lower().startswith("https://")
                or not _host_is(action_host, "google.com")
            ):
                raise DownloadError(
                    "Refusing to follow Google Drive confirmation form to "
                    f"unexpected destination: {action}"
                )
            confirm_url = f"{action}?{urlencode(params)}"
            confirm_req = urllib.request.Request(confirm_url, headers={"User-Agent": USER_AGENT})
            with _open_url(confirm_url, req=confirm_req, timeout=120) as confirm_resp:
                total = confirm_resp.headers.get("Content-Length")
                total = int(total) if total else None
                _download_with_progress(confirm_resp, dest, total, quiet=quiet)
            return

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


_SAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename_component(name: str) -> str:
    """Reduce an untrusted string (e.g. a model id from the API) to a single
    safe filename component.

    Strips any path components (both separators, so Windows-style paths are
    covered too), collapses anything outside ``[A-Za-z0-9._-]`` to
    underscores, and never returns a traversal segment (``.``/``..``) or a
    hidden dotfile name.
    """
    seg = os.path.basename(str(name).replace("\\", "/"))
    seg = _SAFE_FILENAME_CHARS_RE.sub("_", seg)
    seg = seg.lstrip(".")
    return seg or "_"


def build_filename(url: str, model_id: str, file_ext: str) -> str:
    """Build a sane filename from URL, model ID, and extension."""
    url_filename = url.split("/")[-1].split("?")[0]
    url_filename = os.path.basename(url_filename)
    if url_filename and "." in url_filename and len(url_filename) > 3:
        if not url_filename.startswith("uc") and not url_filename.startswith("#"):
            return url_filename
    return f"{safe_filename_component(model_id)}.{safe_filename_component(file_ext)}"

def fmt_size(b: int | None) -> str:
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
