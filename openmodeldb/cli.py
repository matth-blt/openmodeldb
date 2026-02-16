"""
Interactive CLI for browsing and downloading OpenModelDB models.
"""

import os
import sys
import threading
import time

from InquirerPy import inquirer
from InquirerPy.separator import Separator

from openmodeldb.downloader import (
    fmt_size, is_mega_url, is_mediafire_url, is_gdrive_url,
    smart_download, pick_best_url, build_filename,
)


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


def main(db=None):
    """Run the interactive CLI."""
    if db is None:
        from openmodeldb.client import OpenModelDB
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
    cached = db._cache_is_valid()
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
    dest = os.path.join(db._download_dir, file_name)

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
