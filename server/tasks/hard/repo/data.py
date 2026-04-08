# data.py — Data pipeline utilities
# Currently uses TensorFlow tf.data and tf ops. Must be migrated to PyTorch.
#
# Migration challenges:
#   - tf.data.Dataset.from_tensor_slices → TensorDataset + DataLoader
#   - .batch().shuffle().prefetch() → DataLoader(batch_size, shuffle)
#   - tf.cast(x, tf.float32) → torch.tensor(x, dtype=torch.float32)
#   - tf.reduce_mean → torch.mean
#   - tf.math.reduce_std → torch.std
#   - tf.random.shuffle → torch.randperm
#   - tf.gather → tensor indexing
#   - tf.data.AUTOTUNE → no equivalent needed

import tensorflow as tf
import numpy as np


def create_dataset(x, y, batch_size=32, shuffle=True):
    """Create a batched dataset from numpy arrays.
    
    Uses tf.data.Dataset pipeline with optional shuffling and prefetch.
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(x, tf.float32),
        tf.cast(y, tf.int32),
    ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def normalize_features(x):
    """Normalize features to zero mean and unit variance."""
    if isinstance(x, np.ndarray):
        x = tf.constant(x, dtype=tf.float32)
    mean = tf.reduce_mean(x, axis=0)
    std = tf.math.reduce_std(x, axis=0)
    return (x - mean) / (std + 1e-7)


def split_dataset(x, y, val_fraction=0.2):
    """Split data into train/val sets using random shuffled indices."""
    if isinstance(x, np.ndarray):
        x = tf.constant(x, dtype=tf.float32)
    if isinstance(y, np.ndarray):
        y = tf.constant(y, dtype=tf.int32)
    n = tf.shape(x)[0]
    val_size = tf.cast(tf.cast(n, tf.float32) * val_fraction, tf.int32)
    indices = tf.random.shuffle(tf.range(n))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    x_train = tf.gather(x, train_idx)
    y_train = tf.gather(y, train_idx)
    x_val = tf.gather(x, val_idx)
    y_val = tf.gather(y, val_idx)
    return (x_train, y_train), (x_val, y_val)
