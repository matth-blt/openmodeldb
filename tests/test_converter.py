"""
Tests for openmodeldb.converter — offline, no network access.

Skips cleanly (module-level) when the `convert` extras (torch, safetensors)
are not installed, since compare_weights/convert_format/convert_to_onnx all
require them.
"""
import os

import pytest

from openmodeldb.converter import compare_weights, convert_format
from openmodeldb.exceptions import OpenModelDBError, UnsafeModelError

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from safetensors.torch import save_file


def _save_state_dict(path, state_dict):
    save_file(state_dict, path)


# ─── compare_weights: similarity metric ────────────────────────────────

def test_compare_weights_identical_files(tmp_path):
    state_dict = {
        "a": torch.ones(4, 4),
        "b": torch.full((3,), 2.0),
        "c": torch.arange(10, dtype=torch.float32),
    }
    path_a = str(tmp_path / "a.safetensors")
    path_b = str(tmp_path / "b.safetensors")
    _save_state_dict(path_a, state_dict)
    _save_state_dict(path_b, state_dict)

    result = compare_weights(path_a, path_b, quiet=True)

    assert result["similarity"] == 100.0
    assert result["identical"] is True
    assert result["mean_rel_diff"] == 0
    assert result["matched"] == 3
    assert result["total_a"] == 3
    assert result["total_b"] == 3


def test_compare_weights_perturbed_copy(tmp_path):
    state_dict = {
        "a": torch.ones(4, 4),
        "b": torch.full((3,), 2.0),
        "c": torch.arange(10, dtype=torch.float32) + 1.0,
    }
    perturbed = {k: v + 0.1 for k, v in state_dict.items()}

    path_a = str(tmp_path / "a.safetensors")
    path_b = str(tmp_path / "b.safetensors")
    _save_state_dict(path_a, state_dict)
    _save_state_dict(path_b, perturbed)

    result = compare_weights(path_a, path_b, quiet=True)

    assert result["similarity"] < 100.0
    assert result["identical"] is False


def test_compare_weights_scale_invariance(tmp_path):
    """
    Regression test: two completely different tensors with tiny magnitude
    (~1e-4) must score LOW similarity. The old metric (100*(1-mean_abs_diff))
    scored this ~99.9 because the absolute diff was tiny even though the
    tensors are entirely different (scale-dependent bug).
    """
    state_dict_a = {
        "w": torch.full((8, 8), 1e-4),
    }
    state_dict_b = {
        "w": torch.full((8, 8), -1e-4),
    }

    path_a = str(tmp_path / "a.safetensors")
    path_b = str(tmp_path / "b.safetensors")
    _save_state_dict(path_a, state_dict_a)
    _save_state_dict(path_b, state_dict_b)

    result = compare_weights(path_a, path_b, quiet=True)

    assert result["similarity"] < 50.0
    assert result["identical"] is False


# ─── unsafe pickle refusal (no weights_only=False fallback) ────────────

def _make_load_fail_once(monkeypatch):
    """Patch torch.load so the weights_only=True attempt fails, and record
    every call's weights_only value (to prove no unsafe retry happens)."""
    real_load = torch.load
    calls = []

    def _fake_load(path, *args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        if kwargs.get("weights_only"):
            raise RuntimeError("simulated weights_only failure")
        kwargs["weights_only"] = False
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _fake_load)
    return calls


def test_compare_weights_refuses_unsafe_pickle(tmp_path, monkeypatch):
    path_a = str(tmp_path / "a.pth")
    path_b = str(tmp_path / "b.pth")
    torch.save({"model": {"w": torch.ones(2, 2)}}, path_a)
    torch.save({"model": {"w": torch.ones(2, 2)}}, path_b)

    calls = _make_load_fail_once(monkeypatch)

    with pytest.raises(UnsafeModelError, match="weights_only"):
        compare_weights(path_a, path_b, quiet=True)

    assert calls == [True]


def test_convert_format_refuses_unsafe_pickle(tmp_path, monkeypatch):
    src = str(tmp_path / "src.pth")
    out = str(tmp_path / "out.safetensors")
    torch.save({"model": {"w": torch.ones(2, 2)}}, src)

    calls = _make_load_fail_once(monkeypatch)

    with pytest.raises(UnsafeModelError, match="weights_only"):
        convert_format(src, output_path=out, quiet=True)

    assert calls == [True]

    assert not os.path.exists(out)


def test_unsafe_model_error_is_openmodeldb_error():
    assert issubclass(UnsafeModelError, OpenModelDBError)


# ─── torch version safety check (CVE-2025-32434) ──────────────────────

def _tiny_pth(tmp_path):
    path = str(tmp_path / "tiny.pth")
    torch.save({"w": torch.ones(2, 2)}, path)
    return path


def test_warns_on_torch_below_2_6(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(torch, "__version__", "2.5.1", raising=False)

    convert_format(_tiny_pth(tmp_path), output_path=str(tmp_path / "o.st"), quiet=True)

    assert any("CVE-2025-32434" in str(w.message) for w in recwarn.list)


def test_no_cve_warning_on_torch_2_6_plus(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(torch, "__version__", "2.6.0", raising=False)

    convert_format(_tiny_pth(tmp_path), output_path=str(tmp_path / "o.st"), quiet=True)

    assert not any("CVE-2025-32434" in str(w.message) for w in recwarn.list)


def test_compare_weights_also_warns_on_old_torch(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(torch, "__version__", "2.0.1+cpu", raising=False)

    path = _tiny_pth(tmp_path)
    compare_weights(path, path, quiet=True)

    assert any("CVE-2025-32434" in str(w.message) for w in recwarn.list)
