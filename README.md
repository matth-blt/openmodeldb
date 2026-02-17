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

## Dependencies

- [InquirerPy](https://github.com/kazhala/InquirerPy) — interactive prompts
- [rich](https://github.com/Textualize/rich) — progress bars and tables
- [pycryptodome](https://github.com/Legrandin/pycryptodome) — Mega.nz decryption

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

## Credits

- [OpenModelDB](https://openmodeldb.info) — the open model database
- All model authors and contributors
