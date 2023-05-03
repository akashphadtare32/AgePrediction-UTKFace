"""Functions for the efficient net v2 model."""
from tensorflow.keras.layers import Dropout


def apply_top_layers(x):
    """Apply the top layers of the efficient net v2 model."""
    x = Dropout(0.2)(x)
    return x
