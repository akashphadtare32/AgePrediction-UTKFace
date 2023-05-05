"""The UTKFace dataset."""
import logging
import os
from collections.abc import Sequence

import tensorflow as tf

from src.datasets.utils import get_process_path_function

logger = logging.getLogger(__name__)


def get_label(file_path):
    """Get the label for the image."""
    filename = tf.strings.split(file_path, os.path.sep)[-1]
    label = tf.strings.split(filename, "_")[0]
    label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return label


def get_utkface_dataset(filepaths: list[str], target_size: Sequence[int] = (200, 200)):
    """Get the UTKFace dataset."""
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    process_path = get_process_path_function(get_label, target_size=target_size)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
