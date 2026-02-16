"""
ONNX conversion utilities for OpenModelDB.

Converts PyTorch (.pth / .safetensors) models to ONNX format
using spandrel for model loading and torch.onnx for export.
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
            f"Install them with: pip install openmodeldb[onnx]"
        )


def _make_progress(quiet: bool):
    """Create a rich Status context manager (or a no-op if quiet)."""
    if quiet:
        from contextlib import nullcontext
        return nullcontext()
    from rich.console import Console
    return Console().status("", spinner="dots")


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
