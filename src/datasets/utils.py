"""Utilities for datasets."""

import glob
import os

import numpy as np
import tensorflow as tf


def get_dataset_labels_from_filepaths(filepaths, get_label):
    """Return the labels of the given filepaths.

    Parameters
    ----------
    filepaths : np.ndarray
        The filepaths of the images.
    get_label : Callable
        A function that returns the label of the image given the filepath.

    Returns
    -------
    np.ndarray
        The labels of the images.
    """
    return np.array([int(get_label(filepath)) for filepath in filepaths])


def get_dataset_filepaths(path) -> np.ndarray:
    """Return the filepaths of the images in the given directory.

    Parameters
    ----------
    path : str
        The path to the directory containing the images.

    Returns
    -------
    np.ndarray
        The filepaths of the images.
    """
    return np.array(glob.glob(os.path.join(path, "*.jpg")))


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
