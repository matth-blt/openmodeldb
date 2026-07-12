"""
CLI for browsing and downloading OpenModelDB models — interactive picker
plus scriptable argparse subcommands.
"""

import argparse
import os
import sys
import threading
import time

from InquirerPy import inquirer
from InquirerPy.separator import Separator

from openmodeldb.client import OpenModelDB
from openmodeldb.downloader import (
    build_filename,
    fmt_size,
    is_gdrive_url,
    is_mediafire_url,
    is_mega_url,
    pick_best_url,
    smart_download,
)
from openmodeldb.exceptions import OpenModelDBError


# ─── COLORS ──────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"


class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, text="Loading…"):
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r  {C.CYAN}{frame}{C.RESET} {self.text}", end="", flush=True)
            i += 1
            time.sleep(0.08)

    def start(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def update(self, text):
        self.text = text

    def succeed(self, text):
        self._stop.set()
        self._thread.join()
        print(f"\r  {C.GREEN}✓{C.RESET} {text}                    ")

    def fail(self, text):
        self._stop.set()
        self._thread.join()
        print(f"\r  {C.RED}✗{C.RESET} {text}                    ")


def header():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════╗
║        OpenModelDB  ·  Model Downloader          ║
╚══════════════════════════════════════════════════╝{C.RESET}
""")


def interactive(db=None):
    """Run the interactive CLI."""
    if db is None:
        db = OpenModelDB()

    print("\033[2J\033[H", end="", flush=True)
    header()

    # 1) Choose scale
    scale = inquirer.select(
        message="Select upscale factor:",
        choices=[
            {"name": "x1", "value": 1},
            {"name": "x2", "value": 2},
            {"name": "x4", "value": 4},
        ],
    ).execute()

    # 2) Fetch models
    cached = db.cache_is_valid()
    spinner = Spinner(
        "Loading from cache…" if cached else "Fetching model database…"
    ).start()
    try:
        models = db.find(scale=scale)
    except Exception as e:
        spinner.fail(f"Failed: {e}")
        sys.exit(1)

    source = "cache" if cached else "API"
    spinner.succeed(
        f"Found {C.BOLD}{len(models)}{C.RESET} models for {C.CYAN}x{scale}{C.RESET}"
        f"  {C.DIM}({len(db.models)} total · from {source}){C.RESET}"
    )

    if not models:
        print(f"\n  {C.RED}No models found for x{scale}.{C.RESET}\n")
        sys.exit(1)

    # 3) Build choices grouped by architecture
    choices = []
    current_arch = None
    for m in models:
        if m.architecture != current_arch:
            current_arch = m.architecture
            choices.append(Separator(f"── {current_arch.upper()} ──"))
        label = f"{m.name}  — {m.author}"
        choices.append({"name": label, "value": m})

    # 4) Select model
    model = inquirer.select(
        message=f"Select a model ({len(models)} available):",
        choices=choices,
        max_height="70%",
    ).execute()

    # 5) Show model info
    print("\033[2J\033[H", end="", flush=True)
    header()
    print(f"  {C.BOLD}Model Details:{C.RESET}\n")
    print(f"    Name:          {C.CYAN}{C.BOLD}{model.name}{C.RESET}")
    print(f"    Author:        {C.WHITE}{model.author}{C.RESET}")
    print(f"    Architecture:  {C.WHITE}{model.architecture}{C.RESET}")
    print(f"    Scale:         {C.WHITE}{model.scale}x{C.RESET}")
    print(f"    License:       {C.WHITE}{model.license or '?'}{C.RESET}")

    if model.tags:
        print(f"    Tags:          {C.DIM}{', '.join(model.tags)}{C.RESET}")

    if model.description:
        short = "\n".join(model.description.strip().split("\n")[:2])
        print(f"    Description:   {C.DIM}{short}{C.RESET}")
    print()

    # 6) Choose resource if multiple
    resources = model.resources
    if not resources:
        print(f"  {C.RED}No download resources found.{C.RESET}\n")
        sys.exit(1)

    if len(resources) == 1:
        res = resources[0]
    else:
        res = inquirer.select(
            message="Select download format:",
            choices=[
                {
                    "name": f"{r.get('platform', '?')} .{r.get('type', '?')}  ({fmt_size(r.get('size', 0))})",
                    "value": r,
                }
                for r in resources
            ],
        ).execute()

    urls = res.get("urls", [])
    if not urls:
        print(f"  {C.RED}No URL found.{C.RESET}\n")
        sys.exit(1)

    dl_url = pick_best_url(urls)
    file_ext = res.get("type", "pth")
    file_name = build_filename(dl_url, model.id, file_ext)
    dest = os.path.join(db.download_dir, file_name)

    # Host tag
    if is_mega_url(dl_url):
        host_tag = f"{C.YELLOW}Mega.nz{C.RESET}"
    elif is_mediafire_url(dl_url):
        host_tag = f"{C.YELLOW}MediaFire{C.RESET}"
    elif is_gdrive_url(dl_url):
        host_tag = f"{C.YELLOW}Google Drive{C.RESET}"
    else:
        host_tag = f"{C.GREEN}Direct{C.RESET}"

    print(f"  {C.BOLD}Download:{C.RESET}\n")
    print(f"    File:  {C.WHITE}{file_name}{C.RESET}")
    print(f"    Size:  {C.WHITE}{fmt_size(res.get('size', 0))}{C.RESET}")
    print(f"    Host:  {host_tag}")
    print(f"    URL:   {C.DIM}{dl_url}{C.RESET}")
    print(f"    Dest:  {C.DIM}{dest}{C.RESET}")
    print()

    confirm = inquirer.confirm(message="Download now?", default=True).execute()

    if not confirm:
        print(f"\n  {C.DIM}Cancelled.{C.RESET}\n")
        return

    print()
    try:
        smart_download(dl_url, dest)
        print(f"\n  {C.GREEN}{C.BOLD}✓ Downloaded successfully!{C.RESET}")
        print(f"    {C.DIM}{dest}{C.RESET}\n")
    except Exception as e:
        print(f"\n  {C.RED}Download failed: {e}{C.RESET}\n")
        sys.exit(1)


# ─── SCRIPTABLE CLI ──────────────────────────────────────────────────────────

def _get_version() -> str:
    """Resolve the installed package version, falling back to __version__."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("openmodeldb")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    from openmodeldb import __version__
    return __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openmodeldb",
        description="Browse and download AI upscaling models from OpenModelDB.",
    )
    parser.add_argument(
        "--version", action="store_true", help="show the installed version and exit",
    )

    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List models (formatted table)")
    p_list.add_argument("--scale", type=int, help="filter by scale factor")
    p_list.add_argument("--arch", help="filter by architecture")
    p_list.add_argument("--tag", help="filter by tag")

    p_search = sub.add_parser("search", help="Search models by name, author, tags or description")
    p_search.add_argument("query", help="search query")

    p_download = sub.add_parser("download", help="Download a model")
    p_download.add_argument("name", help="model name or id")
    p_download.add_argument("--format", help="file format (pth, safetensors, onnx)")
    p_download.add_argument("--dest", help="destination directory")
    p_download.add_argument(
        "--half", action="store_true",
        help="export ONNX in FP16 instead of FP32 (only meaningful with --format onnx)",
    )
    p_download.add_argument(
        "--all", action="store_true", help="download/extract all files for the model",
    )

    p_check = sub.add_parser("check", help="Check a local model file's integrity against the reference")
    p_check.add_argument("file", help="path to the local model file")

    return parser


def main(argv=None):
    """Scriptable entry point: dispatches to a subcommand, or the interactive
    picker when called with no arguments."""
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(_get_version())
        return

    try:
        if args.command is None:
            interactive()
            return

        db = OpenModelDB()

        if args.command == "list":
            db.list(scale=args.scale, architecture=args.arch, tag=args.tag)
        elif args.command == "search":
            results = db.search(args.query)
            for m in results:
                print(
                    f"{C.DIM}{m.id}{C.RESET}  ·  "
                    f"{m.name} — {m.author} ({m.architecture}, {m.scale}x)"
                )
        elif args.command == "download":
            if args.all:
                if args.half:
                    print("note: --half is ignored with --all", file=sys.stderr)
                db.download_all(args.name, dest=args.dest, format=args.format)
            else:
                db.download(args.name, dest=args.dest, format=args.format, half=args.half)
        elif args.command == "check":
            result = db.test_integrity(args.file)
            if not (result["identical"] or result["similarity"] > 99.9):
                sys.exit(1)
    except (OpenModelDBError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
