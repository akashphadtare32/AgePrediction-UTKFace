"""Baseline CNN model."""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Rescaling,
)


def get_base_model(input_shape):
    """Get the baseline CNN model."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dropout(0.25),
        ]
    )
    return model


def preprocess_input(x):
    """Preprocessing for the baseline CNN."""
    # scale to 0-1 range
    x = Rescaling(scale=1.0 / 255)(x)
    return x
