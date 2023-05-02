"""Functions for the VGG model."""

import tensorflow as tf


def apply_top_layers(x):
    """Apply the top layers of the VGG model."""
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    return x
