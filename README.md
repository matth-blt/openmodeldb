# OpenModelDB

[![CI](https://github.com/matth-blt/openmodeldb/actions/workflows/ci.yml/badge.svg)](https://github.com/matth-blt/openmodeldb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openmodeldb)](https://pypi.org/project/openmodeldb/)
[![Python](https://img.shields.io/pypi/pyversions/openmodeldb)](https://pypi.org/project/openmodeldb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Browse and download AI upscaling models from [OpenModelDB](https://openmodeldb.info).

- **Browse** — search and filter 650+ super-resolution models by scale, architecture, or tag
- **Download** — from any host (direct, GitHub, Hugging Face, Google Drive, MediaFire, Mega.nz), with atomic writes and resume-safe caching
- **Convert** — automatic pth ↔ safetensors ↔ ONNX conversion when the requested format isn't published
- **Verify** — compare a local file's weights against the database reference

## Install

```bash
pip install openmodeldb
```

## CLI

```bash
openmodeldb                    # interactive: select scale → pick a model → download

openmodeldb list [--scale N] [--arch ARCH] [--tag TAG]
openmodeldb search QUERY
openmodeldb download NAME [--format FMT] [--dest DIR] [--half] [--all]
openmodeldb check FILE         # verify a local file against the reference
openmodeldb --version
```

## Python API

```python
from openmodeldb import OpenModelDB

db = OpenModelDB()
# <OpenModelDB: 658 models>

# List models (formatted table)
db.list(scale=4)
db.list(scale=1, architecture="compact")

# Find models (returns list[Model])
models = db.find(scale=4)
compacts = db.find(scale=1, architecture="compact")

# Search by name, author, tags or description
results = db.search("denoise")

# Download by name or Model object
db.download("4xNomos8k_atd_jpg")
db.download(models[0])
db.download(models[0], dest="./my_models/")

# Download a specific format (pth, safetensors, onnx)
db.download("4xNomos8k_atd_jpg", format="safetensors")

# Auto-conversion between pth and safetensors
# If the requested format is unavailable, downloads the other and converts
db.download("2x-HFA2kAVCCompact", format="safetensors")  # only pth available → auto-convert
db.download("1x-SuperScale", format="pth")                # only safetensors → auto-convert

# Download as ONNX with auto-conversion
# If no ONNX file is available, downloads .pth/.safetensors and converts automatically
db.download("4xNomos8k_atd_jpg", format="onnx")
db.download("2x-DigitalFlim-SuperUltraCompact", format="onnx", half=True)  # FP16 export

# Download all available formats
db.download_all("4xNomos8k_atd_jpg")
db.download_all("4xNomos8k_atd_jpg", format="pth")  # only .pth files

# Verify model integrity (compare weights against database reference)
db.test_integrity("downloads/4xNomos8k_atd_jpg.pth")
# ✓ PASS  similarity=100.000000  matched=53/53  max_diff=0.00e+00  mean_diff=0.00e+00

# Silent mode (no output, for use as a library)
path = db.download("4xNomos8k_atd_jpg", quiet=True)

# Get download URL (for custom download logic)
url = db.get_url("4xNomos8k_atd_jpg")
url = db.get_url("4xNomos8k_atd_jpg", format="safetensors")

# Dict-style access
model = db["4xNomos8k_atd_jpg"]
print(model.name, model.author, model.scale, model.architecture)

# Check if a model exists
"4xNomos8k" in db  # True

# Browse architectures and tags
db.architectures()  # ['atd', 'compact', 'cugan', 'dat', ...]
db.tags()           # ['anime', 'denoise', 'photo', ...]

# Iterate
for model in db:
    print(model)

# Launch interactive CLI
db.interactive()
```

### Options

```python
db = OpenModelDB(
    cache_dir="/path/to/cache",   # model index + temp files (default: ~/.cache/openmodeldb)
    download_dir="./my_models",   # default destination for downloads (default: ./downloads)
    include_all=True,             # include archs excluded by default (cain, cain-yuv)
)

db.download_dir      # resolved download directory
db.cache_dir         # resolved cache directory
db.cache_is_valid()  # True if the cached index is fresh (< 1 hour)

db.refresh()      # force re-fetch of the model index from the API
db.clear_cache()  # delete the cached index
```

The model index is cached for 1 hour (with ETag revalidation). If the API is
unreachable, the client falls back to the stale cache with a warning on stderr.

### Error handling

All errors inherit from `OpenModelDBError`:

```python
from openmodeldb import (
    OpenModelDBError,      # base — also raised when the API is unreachable with no cache
    ModelNotFoundError,    # unknown or ambiguous model name
    FormatNotFoundError,   # requested format unavailable (and not convertible)
    DownloadError,         # network/host failure during download
)

try:
    db.download("4xNomos8k_atd_jpg", format="onnx")
except OpenModelDBError as e:
    print(f"failed: {e}")
```

Name resolution is strict: an exact id/name match wins, a unique partial match
is accepted, and an ambiguous partial match raises `ModelNotFoundError` listing
the candidates.

## Dependencies

- [InquirerPy](https://github.com/kazhala/InquirerPy) — interactive prompts
- [rich](https://github.com/Textualize/rich) — progress bars and tables
- [pycryptodome](https://github.com/Legrandin/pycryptodome) — Mega.nz decryption
- [mediafiredl](https://github.com/Gann4/mediafiredl) — MediaFire direct-link extraction

### Conversion (optional)

```bash
pip install openmodeldb[convert]
```

Enables automatic conversion between formats: pth ↔ safetensors ↔ ONNX.

- [PyTorch](https://pytorch.org/) — model loading and ONNX export
- [safetensors](https://github.com/huggingface/safetensors) — safe tensor serialization
- [onnx](https://github.com/onnx/onnx) — ONNX model format
- [onnxruntime](https://github.com/microsoft/onnxruntime) — graph optimization
- [spandrel](https://github.com/chaiNNer-org/spandrel) — universal model loader

## Development

```bash
git clone https://github.com/matth-blt/openmodeldb
cd openmodeldb
pip install -e .[dev]

python -m pytest tests/ -q   # run the test suite (offline, no network needed)
ruff check .                 # lint
```

Tests covering format conversion require the `convert` extras
(`pip install -e .[dev,convert]`) and are skipped automatically without them.

## Credits

- [OpenModelDB](https://openmodeldb.info) — the open model database
- All model authors and contributors
