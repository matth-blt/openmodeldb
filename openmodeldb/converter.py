"""
Conversion utilities for OpenModelDB.

- pth ↔ safetensors: state_dict re-serialization
- pth / safetensors → ONNX: spandrel + torch.onnx.export
"""

import os
import time


def _check_dependencies():
    """Check that ONNX conversion dependencies are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    try:
        import spandrel
    except ImportError:
        missing.append("spandrel")
    if missing:
        raise ImportError(
            f"ONNX conversion requires additional packages: {', '.join(missing)}. "
            f"Install them with: pip install openmodeldb[convert]"
        )


def _make_progress(quiet: bool):
    """Create a rich Status context manager (or a no-op if quiet)."""
    if quiet:
        from contextlib import nullcontext
        return nullcontext()
    from rich.console import Console
    return Console().status("", spinner="dots")


def convert_format(
    model_path: str,
    output_path: str | None = None,
    target: str = "safetensors",
    quiet: bool = False,
) -> str:
    """
    Convert between pth and safetensors formats.

    Args:
        model_path: Path to the source model file (.pth or .safetensors).
        output_path: Path for the output file.
                     If None, replaces the extension automatically.
        target: Target format ("safetensors" or "pth").
        quiet: If True, suppress all output.

    Returns:
        Path to the converted file.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for format conversion. pip install torch")

    target = target.lower().lstrip(".")
    base_name = os.path.basename(model_path)

    if output_path is None:
        base = os.path.splitext(model_path)[0]
        output_path = f"{base}.{target}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    status = _make_progress(quiet)
    with status:
        if not quiet:
            status.update(f"  Loading {base_name}...")

        # Load state dict
        src_ext = os.path.splitext(model_path)[1].lower()
        if src_ext == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError("safetensors is required. pip install safetensors")
            state_dict = load_file(model_path, device="cpu")
        else:
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            # Handle nested state dicts (e.g. {"state_dict": {...}, "params_ema": {...}})
            if isinstance(state_dict, dict):
                for key in ("params_ema", "params", "state_dict", "model", "model_state_dict"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break

        if not quiet:
            status.update(f"  Converting to {target}...")

        start = time.time()

        # Save in target format
        if target == "safetensors":
            try:
                from safetensors.torch import save_file
            except ImportError:
                raise ImportError("safetensors is required. pip install safetensors")
            save_file(state_dict, output_path)
        else:
            torch.save(state_dict, output_path)

        elapsed = time.time() - start

    if not quiet:
        size_mb = os.path.getsize(output_path) / 1048576
        print(
            f"  \033[92m✓\033[0m Converted to {target} \033[2m{output_path}\033[0m"
            f" ({size_mb:.1f} MB, {elapsed:.1f}s)"
        )

    return output_path


def compare_weights(
    path_a: str,
    path_b: str,
    quiet: bool = False,
) -> dict:
    """
    Compare weights between two model files (pth, safetensors, or onnx).

    Returns a dict with:
        - matched: number of matched tensor names
        - total_a / total_b: tensor counts
        - max_diff: worst-case absolute difference
        - mean_diff: average absolute difference
        - identical: True when max_diff == 0
        - similarity: percentage (100.0 = identical)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for weight comparison. pip install torch")

    def _load(path: str) -> dict[str, "torch.Tensor"]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file
            return load_file(path, device="cpu")
        if ext == ".onnx":
            import onnx
            from onnx import numpy_helper
            model = onnx.load(path)
            return {
                init.name: torch.from_numpy(numpy_helper.to_array(init))
                for init in model.graph.initializer
            }
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for key in ("params_ema", "params", "state_dict", "model", "model_state_dict"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        return sd

    weights_a = _load(path_a)
    weights_b = _load(path_b)

    common = set(weights_a.keys()) & set(weights_b.keys())
    if not common:
        # No shared keys — try matching by shape (ONNX renames layers)
        by_shape_a = {}
        for k, v in weights_a.items():
            key = (tuple(v.shape), v.dtype)
            by_shape_a.setdefault(key, []).append((k, v))
        by_shape_b = {}
        for k, v in weights_b.items():
            key = (tuple(v.shape), v.dtype)
            by_shape_b.setdefault(key, []).append((k, v))

        max_diff = 0.0
        total_diff = 0.0
        count = 0
        for key in by_shape_a:
            if key in by_shape_b:
                for (_, va), (_, vb) in zip(by_shape_a[key], by_shape_b[key]):
                    diff = (va.float() - vb.float()).abs()
                    md = diff.max().item()
                    max_diff = max(max_diff, md)
                    total_diff += diff.mean().item()
                    count += 1

        if count == 0:
            return {
                "matched": 0,
                "total_a": len(weights_a),
                "total_b": len(weights_b),
                "max_diff": float("inf"),
                "mean_diff": float("inf"),
                "identical": False,
                "similarity": 0.0,
            }

        mean_diff = total_diff / count
        similarity = max(0.0, 100.0 * (1.0 - mean_diff))
        return {
            "matched": count,
            "total_a": len(weights_a),
            "total_b": len(weights_b),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "identical": max_diff == 0.0,
            "similarity": round(similarity, 4),
        }

    max_diff = 0.0
    total_diff = 0.0
    for name in common:
        a = weights_a[name].float()
        b = weights_b[name].float()
        if a.shape != b.shape:
            continue
        diff = (a - b).abs()
        md = diff.max().item()
        max_diff = max(max_diff, md)
        total_diff += diff.mean().item()

    n = len(common) or 1
    mean_diff = total_diff / n
    similarity = max(0.0, 100.0 * (1.0 - mean_diff))

    return {
        "matched": len(common),
        "total_a": len(weights_a),
        "total_b": len(weights_b),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "identical": max_diff == 0.0,
        "similarity": round(similarity, 4),
    }


def convert_to_onnx(
    model_path: str,
    output_path: str | None = None,
    opset: int = 20,
    half: bool = False,
    optimize: bool = True,
    quiet: bool = False,
) -> str:
    """
    Convert a PyTorch model (.pth / .safetensors) to ONNX.

    Args:
        model_path: Path to the source PyTorch model file.
        output_path: Path for the output ONNX file.
                     If None, replaces the extension with .onnx in the same directory.
        opset: ONNX opset version (default: 20).
        half: If True, convert to FP16. Otherwise FP32.
        optimize: If True, apply ORT graph optimizations.
        quiet: If True, suppress all output.

    Returns:
        Path to the saved ONNX file.
    """
    _check_dependencies()

    import io
    import sys
    import warnings

    import torch
    import onnxruntime as ort
    from spandrel import ModelLoader, ImageModelDescriptor

    precision = "FP16" if half else "FP32"
    base_name = os.path.basename(model_path)

    status = _make_progress(quiet)
    with status:
        # ── Step 1: Load model ────────────────────────────────────────
        if not quiet:
            status.update(f"  Loading {base_name}...")

        # Install extra architecture support if available
        try:
            import spandrel_extra_arches
            spandrel_extra_arches.install()
        except ImportError:
            pass

        # Suppress spandrel's verbose weight dump during loading
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            loader = ModelLoader()
            model_desc = loader.load_from_file(model_path)
        finally:
            sys.stdout, sys.stderr = _old_stdout, _old_stderr

        assert isinstance(model_desc, ImageModelDescriptor), (
            f"Unsupported model type: expected ImageModelDescriptor, got {type(model_desc).__name__}"
        )

        # Extract the raw nn.Module and prepare for export
        model = model_desc.model.to("cpu").eval()
        model.requires_grad_(False)  # Detach all gradients for ONNX export

        # Check FP16 support
        if half and not model_desc.supports_half:
            if not quiet:
                print(
                    f"  \033[93m⚠\033[0m {model_desc.architecture.name} does not support FP16 "
                    f"— exporting as FP32 instead."
                )
            half = False
            precision = "FP32"

        # Determine output path
        if output_path is None:
            base = os.path.splitext(model_path)[0]
            output_path = f"{base}.onnx"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Dummy input — use FP16 if half (PyTorch-level conversion, more reliable)
        if half:
            model = model.half()
            torch_input = torch.randn(1, model_desc.input_channels, 32, 32, device="cpu", dtype=torch.float16)
        else:
            torch_input = torch.randn(1, model_desc.input_channels, 32, 32, device="cpu")

        # Wrap the raw nn.Module (fixes issues with various architectures, from chaiNNer)
        class _SpandrelWrapper(torch.nn.Module):
            def __init__(self, m: torch.nn.Module):
                super().__init__()
                self.model = m

            def forward(self, x: torch.Tensor):
                return self.model(x)

        wrapped = _SpandrelWrapper(model)

        # ── Step 2: ONNX export ──────────────────────────────────────
        if not quiet:
            status.update(f"  Exporting to ONNX ({precision})...")

        start = time.time()

        # Suppress TracerWarnings (noisy but harmless)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message=".*TrainingMode.EVAL.*")

        # Patch torch.sort ONNX symbolic: the built-in handler wrongly
        # treats the `stable` kwarg as the `out` parameter and rejects it.
        # We monkey-patch _sort_helper to ignore `out` and just do TopK.
        from torch.onnx import symbolic_helper as _sh

        _orig_sort_helper = _sh._sort_helper

        def _patched_sort_helper(g, input, dim, decending=True, out=None):
            shape_ = g.op("Shape", input)
            dim_size_ = g.op(
                "Gather",
                shape_,
                g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
            )
            return g.op(
                "TopK", input, dim_size_,
                axis_i=dim, largest_i=decending, outputs=2,
            )

        _sh._sort_helper = _patched_sort_helper

        try:
            torch.onnx.export(
                wrapped,
                (torch_input,),
                output_path,
                dynamo=False,
                verbose=False,
                opset_version=opset,
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size", 2: "height", 3: "width"},
                },
                input_names=["input"],
                output_names=["output"],
            )
        finally:
            # Restore original sort helper
            _sh._sort_helper = _orig_sort_helper

        export_time = time.time() - start

        # ── Step 3: ORT graph optimization ───────────────────────────
        if optimize:
            if not quiet:
                status.update("  Optimizing ONNX graph...")

            optimized_path = output_path.replace(".onnx", "_opt.onnx")
            session_opt = ort.SessionOptions()
            session_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            session_opt.optimized_model_filepath = optimized_path
            ort.InferenceSession(output_path, session_opt, providers=["CPUExecutionProvider"])
            os.replace(optimized_path, output_path)

    # Final summary (outside the spinner)
    if not quiet:
        size_mb = os.path.getsize(output_path) / 1048576
        print(
            f"  \033[92m✓\033[0m ONNX {precision} saved to \033[2m{output_path}\033[0m"
            f" ({size_mb:.1f} MB, export {export_time:.1f}s)"
        )

    return output_path
