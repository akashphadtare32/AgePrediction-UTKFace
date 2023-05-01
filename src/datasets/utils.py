"""Utilities for datasets."""


import tensorflow as tf


def decode_image(raw_img, target_size, channels=3):
    """Decode a raw image and resize it."""
    img = tf.image.decode_jpeg(raw_img, channels=channels)
    img = tf.image.resize(img, [*target_size])
    img.set_shape([*target_size, channels])
    return img


def get_process_path_function(get_label, target_size):
    """Return a function for processing a path.

    Parameters
    ----------
    get_label : function
        A function that takes a path and returns a label.
    target_size : tuple
        The target size of the image.

    Returns
    -------
    function
        A function that takes a path and returns an image and a label.
    """

    def process_path(file_path):
        """Process a path."""
        label = get_label(file_path)
        raw_img = tf.io.read_file(file_path)
        img = decode_image(raw_img, target_size=target_size)
        return img, label

    return process_path
