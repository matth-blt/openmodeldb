# OpenModelDB

Browse and download AI upscaling models from [OpenModelDB](https://openmodeldb.info).

## Install

```bash
pip install openmodeldb
```

## CLI

```bash
openmodeldb
```

Select scale → pick a model → download.

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

# Download as ONNX with auto-conversion
# If no ONNX file is available, downloads .pth/.safetensors and converts automatically
db.download("4xNomos8k_atd_jpg", format="onnx")
db.download("2x-DigitalFlim-SuperUltraCompact", format="onnx", half=True)  # FP16 export

# Download all available formats
db.download_all("4xNomos8k_atd_jpg")

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

## Dependencies

- [InquirerPy](https://github.com/kazhala/InquirerPy) — interactive prompts
- [rich](https://github.com/Textualize/rich) — progress bars and tables
- [pycryptodome](https://github.com/Legrandin/pycryptodome) — Mega.nz decryption

### ONNX conversion (optional)

```bash
pip install openmodeldb[onnx]
```

> **Tip:** The conversion runs on CPU only. For a lighter install (~200 MB instead of ~2 GB), install the CPU-only build of PyTorch **before** installing the extras:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install openmodeldb[onnx]
> ```

- [PyTorch](https://pytorch.org/) — model loading and ONNX export (CPU only, no GPU needed)
- [onnx](https://github.com/onnx/onnx) — ONNX model format
- [onnxruntime](https://github.com/microsoft/onnxruntime) — graph optimization
- [spandrel](https://github.com/chaiNNer-org/spandrel) — universal model loader

## Credits

- [OpenModelDB](https://openmodeldb.info) — the open model database
- All model authors and contributors
