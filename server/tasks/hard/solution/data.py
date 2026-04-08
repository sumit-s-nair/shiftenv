# data.py — REFERENCE SOLUTION (PyTorch)
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_dataset(x, y, batch_size=32, shuffle=True):
    """Create a DataLoader from numpy arrays."""
    x_t = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.float()
    y_t = torch.tensor(y, dtype=torch.long) if isinstance(y, np.ndarray) else y.long()
    dataset = TensorDataset(x_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def normalize_features(x):
    """Normalize features to zero mean and unit variance."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return (x - mean) / (std + 1e-7)


def split_dataset(x, y, val_fraction=0.2):
    """Split data into train/val sets."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)
    n = len(x)
    val_size = int(n * val_fraction)
    indices = torch.randperm(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
