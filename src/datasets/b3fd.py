"""The B3FD dataset."""

import os
from collections.abc import Sequence

import pandas as pd
import tensorflow as tf

from src.datasets.utils import get_process_path_function


def get_label_function(metadata):
    """Return a function for getting the label based on the metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata.
    """

    def get_label(file_path):
        """Get the label for the image."""
        parts = tf.strings.split(file_path, os.path.sep)[-2:]
        identifier = tf.strings.join(parts, os.sep)
        label = metadata[metadata.filepath == identifier.numpy().decode("utf-8")][
            "age"
        ].iloc[0]
        return label

    return get_label


def get_b3fd_dataset(
    data_path: str, target_size: Sequence[int] = (200, 200), metadata_path: str = None
):
    """Get the B3FD dataset."""
    ds = tf.data.Dataset.list_files(data_path + "*/*.jpg")
    metadata = pd.read_csv(
        metadata_path,
        sep=" ",
        header=None,
        names=["filepath", "age"],
    )
    get_label = get_label_function(metadata)
    process_path = get_process_path_function(get_label, target_size=target_size)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
