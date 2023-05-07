"""Functions for the inception resnet model."""

from tensorflow.keras import layers


def apply_top_layers(x):
    """Apply the top layers of the inception resnet model."""
    x = layers.Dense(512, activation="selu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="selu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="selu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dropout(0.5)(x)
    return x
