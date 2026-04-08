# test_migration.py — Grader for the hard task (TensorFlow → PyTorch)
#
# 3 layers of tests for partial reward signal:
#   LAYER 1 — Import checks (~30%): old lib removed, new lib present
#   LAYER 2 — Structure checks (~30%): correct PyTorch patterns used
#   LAYER 3 — Runtime checks (~40%): code actually runs and produces correct output
#
# Total: 10 tests. All pass → reward ≈ 1.0
# Only Layer 1 passes → reward ≈ 0.3
# Layers 1+2 pass → reward ≈ 0.6

import ast
import os
import sys
import importlib.util
import pytest
import numpy as np

REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "repo")
MODEL_FILE = os.path.join(REPO_DIR, "model.py")
TRAIN_FILE = os.path.join(REPO_DIR, "train.py")
DATA_FILE = os.path.join(REPO_DIR, "data.py")
PREDICT_FILE = os.path.join(REPO_DIR, "predict.py")

ALL_FILES = [MODEL_FILE, TRAIN_FILE, DATA_FILE, PREDICT_FILE]

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def _get_source(filepath):
    with open(filepath, "r") as f:
        return f.read()


def _get_imports(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _load_module(name, filepath):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Add repo dir to the module's search path so intra-repo imports work
    mod.__path__ = [REPO_DIR] if hasattr(mod, "__path__") else []
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# LAYER 1 — Import Checks (3 tests)
# Agent gets partial credit just for replacing imports correctly.
# ===========================================================================
class TestImportCleanliness:
    """Verify TF is gone and PyTorch is present in all files."""

    def test_no_tensorflow_in_model(self):
        imports = _get_imports(MODEL_FILE)
        assert "tensorflow" not in imports, "model.py still imports tensorflow"

    def test_no_tensorflow_in_train(self):
        imports = _get_imports(TRAIN_FILE)
        assert "tensorflow" not in imports, "train.py still imports tensorflow"

    def test_no_tensorflow_in_data(self):
        imports = _get_imports(DATA_FILE)
        assert "tensorflow" not in imports, "data.py still imports tensorflow"


# ===========================================================================
# LAYER 2 — Structure Checks (3 tests)
# Agent gets more credit for using correct PyTorch patterns.
# ===========================================================================
class TestStructure:
    """Verify correct PyTorch patterns are present in source."""

    def test_model_uses_nn_module(self):
        """tf.keras.Model → nn.Module"""
        src = _get_source(MODEL_FILE)
        assert "nn.Module" in src, "model.py should inherit from nn.Module"

    def test_train_uses_dataloader(self):
        """tf.data.Dataset → DataLoader"""
        src = _get_source(DATA_FILE)
        assert "DataLoader" in src, "data.py should use DataLoader"

    def test_train_uses_backward(self):
        """tf.GradientTape → loss.backward() + optimizer.step()"""
        src = _get_source(TRAIN_FILE)
        assert "backward()" in src or ".backward()" in src, (
            "train.py should use loss.backward()"
        )
        assert "optimizer.step()" in src or ".step()" in src, (
            "train.py should use optimizer.step()"
        )


# ===========================================================================
# LAYER 3 — Runtime Checks (4 tests)
# Agent gets full credit when the migrated code actually runs.
# ===========================================================================
class TestRuntime:
    """Actually import and run the migrated code."""

    def test_model_instantiates(self):
        """Model class can be created with expected arguments."""
        try:
            mod = _load_module("model", MODEL_FILE)
            model = mod.MultiTaskClassifier(
                input_dim=10, hidden_units=64, num_classes=3
            )
            assert model is not None
        except ImportError as e:
            if "tensorflow" in str(e):
                pytest.fail("model.py still requires tensorflow at runtime")
            raise
        except Exception as e:
            pytest.fail(f"Failed to instantiate model: {e}")

    def test_forward_pass_shape(self):
        """Model forward pass produces correct output shape."""
        try:
            import torch
            mod = _load_module("model", MODEL_FILE)
            model = mod.MultiTaskClassifier(
                input_dim=10, hidden_units=64, num_classes=3
            )
            # Set to eval mode (standard PyTorch pattern)
            if hasattr(model, "eval"):
                model.eval()
            x = torch.randn(4, 10)  # batch of 4, input_dim=10
            with torch.no_grad():
                out = model(x)
            assert out.shape == (4, 3), (
                f"Expected output shape (4, 3), got {out.shape}"
            )
        except ImportError as e:
            if "tensorflow" in str(e):
                pytest.fail("model.py still requires tensorflow at runtime")
            raise
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_dataloader_yields_batches(self):
        """Data pipeline produces batched tensors of correct shape."""
        try:
            import torch
            mod = _load_module("data", DATA_FILE)
            x = np.random.randn(100, 10).astype(np.float32)
            y = np.random.randint(0, 3, size=100).astype(np.int64)
            loader = mod.create_dataset(x, y, batch_size=16, shuffle=False)
            batch = next(iter(loader))
            assert len(batch) == 2, "Batch should be a tuple of (x, y)"
            x_batch, y_batch = batch
            assert x_batch.shape == (16, 10), f"x_batch shape {x_batch.shape}"
            assert y_batch.shape == (16,), f"y_batch shape {y_batch.shape}"
            # Verify types are torch tensors
            assert isinstance(x_batch, torch.Tensor), "x_batch should be a torch.Tensor"
            assert isinstance(y_batch, torch.Tensor), "y_batch should be a torch.Tensor"
        except ImportError as e:
            if "tensorflow" in str(e):
                pytest.fail("data.py still requires tensorflow at runtime")
            raise
        except Exception as e:
            pytest.fail(f"DataLoader test failed: {e}")

    def test_predict_single_returns_numpy(self):
        """Prediction function returns numpy array of correct shape."""
        try:
            import torch
            model_mod = _load_module("model", MODEL_FILE)
            predict_mod = _load_module("predict", PREDICT_FILE)

            model = model_mod.MultiTaskClassifier(
                input_dim=10, hidden_units=64, num_classes=3
            )
            if hasattr(model, "eval"):
                model.eval()

            x = np.random.randn(10).astype(np.float32)
            result = predict_mod.predict_single(model, x)
            assert isinstance(result, np.ndarray), (
                f"Expected numpy array, got {type(result)}"
            )
            assert result.shape == (3,), (
                f"Expected shape (3,), got {result.shape}"
            )
        except ImportError as e:
            if "tensorflow" in str(e):
                pytest.fail("predict.py still requires tensorflow at runtime")
            raise
        except Exception as e:
            pytest.fail(f"predict_single failed: {e}")
