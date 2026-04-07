# model.py — REFERENCE SOLUTION (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskClassifier(nn.Module):
    """A multi-layer classifier with dropout and batch normalization."""

    def __init__(self, input_dim, hidden_units, num_classes, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.dense1 = nn.Linear(input_dim, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.dense2 = nn.Linear(hidden_units, hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(hidden_units // 2)
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(hidden_units // 2, num_classes)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.dense2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)

        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_units": self.dense1.in_features,
            "num_classes": self.num_classes,
        }
