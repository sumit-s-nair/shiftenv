# model.py — Neural network model definition
# Currently uses TensorFlow/Keras. Must be migrated to PyTorch.
#
# Migration challenges:
#   - tf.keras.Model → torch.nn.Module
#   - call() → forward()  
#   - Dense(units=N) → nn.Linear(in_features, out_features) — must specify input dim
#   - BatchNormalization() → nn.BatchNorm1d(num_features)
#   - Dropout(rate=r) → nn.Dropout(p=r) — parameter renamed
#   - tf.nn.relu → torch.nn.functional.relu or F.relu
#   - tf.nn.softmax → F.softmax(x, dim=-1) — needs dim argument
#   - training=False flag → model.eval() / model.train()

import tensorflow as tf


class MultiTaskClassifier(tf.keras.Model):
    """A multi-layer classifier with dropout and batch normalization.
    
    Architecture: Dense → ReLU → BN → Dropout → Dense → ReLU → BN → Dropout → Dense → Softmax
    """

    def __init__(self, input_dim, hidden_units, num_classes, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # First hidden block
        self.dense1 = tf.keras.layers.Dense(hidden_units, input_shape=(input_dim,))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # Second hidden block
        self.dense2 = tf.keras.layers.Dense(hidden_units // 2)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        # Output
        self.output_layer = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            training: Whether to run in training mode (enables dropout/BN updates)
        """
        # Block 1
        x = self.dense1(inputs)
        x = tf.nn.relu(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        # Block 2
        x = self.dense2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        # Output with softmax
        logits = self.output_layer(x)
        return tf.nn.softmax(logits)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_units": self.dense1.units,
            "num_classes": self.num_classes,
        }
