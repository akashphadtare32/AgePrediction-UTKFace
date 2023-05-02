"""Functions for the VGG model."""

from tensorflow.keras.layers import Dense, Dropout, Flatten


def apply_top_layers(x):
    """Apply the top layers of the VGG model."""
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    return x
