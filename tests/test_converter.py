"""
Tests for openmodeldb.converter — offline, no network access.

Skips cleanly (module-level) when the `convert` extras (torch, safetensors)
are not installed, since compare_weights/convert_format/convert_to_onnx all
require them.
"""
import argparse

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from openmodeldb.converter import compare_weights  # noqa: E402


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
        "c": torch.arange(10, dtype=torch.float32) + 1.0,  # avoid zeros
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

    # rel = mean(|1e-4 - (-1e-4)|) / (mean(|-1e-4|) + 1e-12) = 2e-4/1e-4 = 2.0
    # similarity = max(0, 100*(1-2.0)) = 0.0
    assert result["similarity"] < 50.0
    assert result["identical"] is False


# ─── unsafe pickle load warning ────────────────────────────────────────

def test_compare_weights_warns_on_unsafe_pickle_fallback(tmp_path, monkeypatch):
    # Payload contains a non-tensor object (argparse.Namespace) nested under
    # a known state-dict key so it gets unwrapped before comparison, but the
    # actual weights_only=True/False split is simulated deterministically
    # via monkeypatch (real torch's weights_only allow-list varies by
    # version, so this keeps the test stable).
    path_a = str(tmp_path / "a.pth")
    path_b = str(tmp_path / "b.pth")
    payload = {
        "cfg": argparse.Namespace(foo="bar"),
        "model": {"w": torch.ones(2, 2)},
    }
    torch.save(payload, path_a)
    torch.save(payload, path_b)

    real_load = torch.load

    def _fake_load(path, *args, **kwargs):
        if kwargs.get("weights_only", False):
            raise RuntimeError("simulated weights_only failure")
        kwargs["weights_only"] = False
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _fake_load)

    with pytest.warns(UserWarning, match="unsafe pickle load"):
        result = compare_weights(path_a, path_b, quiet=True)

    assert result["identical"] is True
