# predict.py — REFERENCE SOLUTION (PyTorch)
import torch
import torch.nn.functional as F
import numpy as np


def load_trained_model(path, model_class, **kwargs):
    """Load saved model weights."""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_single(model, x):
    """Predict class probabilities for a single input vector."""
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = x_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = model(x_tensor)
    return predictions.numpy()[0]


def predict_batch(model, x_batch):
    """Predict on a batch of inputs."""
    model.eval()
    x_tensor = torch.tensor(x_batch, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_tensor)
    probs = predictions.numpy()
    classes = torch.argmax(predictions, dim=1).numpy()
    return {"probabilities": probs, "classes": classes}


def get_top_k(model, x, k=2):
    """Get top-k predictions for an input."""
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = model(x_tensor)
    top_k = torch.topk(predictions, k=k)
    return {"values": top_k.values.numpy(), "indices": top_k.indices.numpy()}
