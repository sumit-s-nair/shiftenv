# train.py — Custom training loop
# Currently uses TensorFlow GradientTape. Must be migrated to PyTorch.
#
# Migration challenges:
#   - @tf.function decorator → remove entirely
#   - tf.GradientTape → loss.backward() + optimizer.step()
#   - tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2) →
#     torch.optim.Adam(params, lr, betas=(b1,b2))
#   - tf.keras.losses.SparseCategoricalCrossentropy() →
#     torch.nn.CrossEntropyLoss() — TAKES LOGITS not softmax output!
#   - tf.keras.metrics.Mean / SparseCategoricalAccuracy → manual computation
#   - model(x, training=True) → model.train(); model(x)
#   - model.trainable_variables → model.parameters()
#   - model.save(path, save_format='tf') → torch.save(model.state_dict(), path)

import tensorflow as tf
import numpy as np
from model import MultiTaskClassifier
from data import create_dataset, normalize_features, split_dataset


@tf.function
def train_step(model, x_batch, y_batch, loss_fn, optimizer):
    """Single training step with gradient tape."""
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


def evaluate(model, dataset, loss_fn):
    """Evaluate model on a dataset, return loss and accuracy."""
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_batch, y_batch in dataset:
        predictions = model(x_batch, training=False)
        loss = loss_fn(y_batch, predictions)
        loss_metric.update_state(loss)
        acc_metric.update_state(y_batch, predictions)
    return float(loss_metric.result()), float(acc_metric.result())


def train(model, train_dataset, val_dataset, epochs=10, lr=1e-3):
    """Full training loop with metrics tracking."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()

        for x_batch, y_batch in train_dataset:
            loss, preds = train_step(model, x_batch, y_batch, loss_fn, optimizer)
            train_loss_metric.update_state(loss)
            train_acc_metric.update_state(y_batch, preds)

        val_loss, val_acc = evaluate(model, val_dataset, loss_fn)

        history["train_loss"].append(float(train_loss_metric.result()))
        history["train_acc"].append(float(train_acc_metric.result()))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def save_model(model, path):
    """Save the full model in TF SavedModel format."""
    model.save(path, save_format="tf")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    x = np.random.randn(200, 10).astype(np.float32)
    y = np.random.randint(0, 3, size=200).astype(np.int32)

    (x_train, y_train), (x_val, y_val) = split_dataset(x, y)
    x_train = normalize_features(x_train)
    x_val = normalize_features(x_val)

    train_ds = create_dataset(x_train, y_train, batch_size=16)
    val_ds = create_dataset(x_val, y_val, batch_size=16, shuffle=False)

    model = MultiTaskClassifier(input_dim=10, hidden_units=64, num_classes=3)
    history = train(model, train_ds, val_ds, epochs=5)
    save_model(model, "saved_model")
    print("Training complete. History:", history)
