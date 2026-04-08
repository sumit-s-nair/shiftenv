# train.py — REFERENCE SOLUTION (PyTorch)
import torch
import torch.nn as nn
import numpy as np
from model import MultiTaskClassifier
from data import create_dataset, normalize_features, split_dataset


def train_step(model, x_batch, y_batch, loss_fn, optimizer):
    """Single training step."""
    model.train()
    predictions = model(x_batch)
    loss = loss_fn(predictions, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, predictions


def evaluate(model, loader, loss_fn):
    """Evaluate model on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)
    avg_loss = total_loss / total if total else 0
    accuracy = correct / total if total else 0
    return avg_loss, accuracy


def train(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """Full training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            loss, preds = train_step(model, x_batch, y_batch, loss_fn, optimizer)
            train_losses.append(loss.item())
            predicted = torch.argmax(preds, dim=1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += len(y_batch)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["train_acc"].append(train_correct / train_total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history


def save_model(model, path):
    """Save model weights."""
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    x = np.random.randn(200, 10).astype(np.float32)
    y = np.random.randint(0, 3, size=200).astype(np.int64)

    (x_train, y_train), (x_val, y_val) = split_dataset(x, y)
    x_train = normalize_features(x_train)
    x_val = normalize_features(x_val)

    train_loader = create_dataset(x_train, y_train, batch_size=16)
    val_loader = create_dataset(x_val, y_val, batch_size=16, shuffle=False)

    model = MultiTaskClassifier(input_dim=10, hidden_units=64, num_classes=3)
    history = train(model, train_loader, val_loader, epochs=5)
    save_model(model, "model.pth")
    print("Training complete. History:", history)
