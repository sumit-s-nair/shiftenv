# predict.py — Inference / prediction utilities
# Currently uses TensorFlow ops. Must be migrated to PyTorch.
#
# Migration challenges:
#   - tf.keras.models.load_model → torch.load + model.load_state_dict
#   - tf.convert_to_tensor → torch.tensor or torch.from_numpy
#   - tf.expand_dims → torch.unsqueeze
#   - tf.argmax → torch.argmax
#   - tf.math.top_k → torch.topk
#   - .numpy() on tf.Tensor → .detach().numpy() on torch.Tensor
#   - model(x, training=False) → model.eval(); with torch.no_grad(): model(x)

import tensorflow as tf
import numpy as np


def load_trained_model(path):
    """Load a saved TF model from disk."""
    return tf.keras.models.load_model(path)


def predict_single(model, x):
    """Predict class probabilities for a single input vector.
    
    Args:
        model: trained model
        x: numpy array of shape (input_dim,)
    Returns:
        numpy array of shape (num_classes,) with probabilities
    """
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    x_tensor = tf.expand_dims(x_tensor, axis=0)
    predictions = model(x_tensor, training=False)
    return predictions.numpy()[0]


def predict_batch(model, x_batch):
    """Predict on a batch of inputs.
    
    Args:
        model: trained model
        x_batch: numpy array of shape (batch_size, input_dim)
    Returns:
        dict with 'probabilities' and 'classes' arrays
    """
    x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    predictions = model(x_tensor, training=False)
    probs = predictions.numpy()
    classes = tf.argmax(predictions, axis=1).numpy()
    return {"probabilities": probs, "classes": classes}


def get_top_k(model, x, k=2):
    """Get top-k predictions for an input.
    
    Args:
        model: trained model
        x: numpy array of shape (input_dim,) or (batch, input_dim)
        k: number of top predictions
    Returns:
        dict with 'values' and 'indices' arrays
    """
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    if len(x_tensor.shape) == 1:
        x_tensor = tf.expand_dims(x_tensor, axis=0)
    predictions = model(x_tensor, training=False)
    top_k = tf.math.top_k(predictions, k=k)
    return {"values": top_k.values.numpy(), "indices": top_k.indices.numpy()}
