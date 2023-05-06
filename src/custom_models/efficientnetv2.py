"""Functions for the efficient net v2 model."""
from tensorflow.keras.layers import Dense, Dropout


def apply_top_layers(x):
    """Apply the top layers of the efficient net v2 model."""
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    return x
