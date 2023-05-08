"""A more sophisticated custom CNN model.

From the paper `Age and Gender prediction using Deep CNNs
and Transfer learning` by Sheoran et al.
"""


from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    SeparableConv2D,
    SpatialDropout2D,
)


def get_base_model(input_shape):
    """Get the custom CNN model which uses separable convolutional layers."""
    return Sequential(
        [
            Input(shape=input_shape),
            # ----
            SeparableConv2D(64, kernel_size=(3, 3)),
            BatchNormalization(),
            MaxPooling2D(),
            Activation("relu"),
            SpatialDropout2D(0.3),
            # ----
            SeparableConv2D(128, kernel_size=(2, 2)),
            # no batchnorm here because of dropout in the previous block
            Activation("relu"),
            MaxPooling2D(),
            # ----
            SeparableConv2D(128, kernel_size=(3, 3)),
            MaxPooling2D(),
            # ----
            SeparableConv2D(256, kernel_size=(3, 3)),
            BatchNormalization(),
            MaxPooling2D(),
            Activation("relu"),
            # ----
            SeparableConv2D(256, kernel_size=(3, 3)),
            BatchNormalization(),
            MaxPooling2D(),
            Activation("relu"),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )
