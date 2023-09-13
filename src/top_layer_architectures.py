"""Top layer architectures for the base models."""

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)


def pass_through(x):
    """Pass through the input."""
    return x


def fully_connected_with_dropout(x):
    """Apply Fully connected top layers with dropout."""
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    return x


def fully_connected(x):
    """Apply Fully connected top layers without dropout."""
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    return x


def conv_with_fc_top(x):
    """Apply Convolutional top layers with one fully connected layer."""
    x = Conv2D(128, (2, 2), activation="relu", padding="same")(x)
    x = Conv2D(256, (2, 2), activation="relu", padding="same")(x)
    x = Conv2D(512, (2, 2), activation="relu", padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)
    return x


def vgg_top(x):
    """Apply the top layers of the VGG model."""
    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    return x


# I don't know why I named this `resnet_top`
def resnet_top(x):
    """Apply the top layers of the (inception) resnet model."""
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="selu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="selu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="selu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="selu")(x)
    x = Dropout(0.5)(x)
    return x
