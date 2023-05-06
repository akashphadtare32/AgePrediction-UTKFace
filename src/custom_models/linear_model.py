"""A linear model as a benchmark/baseline model."""

from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential


def get_base_model(input_shape=None):
    """Get the model."""
    model = Sequential()
    model.add(Flatten())
    return model
