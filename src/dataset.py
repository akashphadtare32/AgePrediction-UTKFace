"""Datasets module."""

import logging
import os
from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras.layers import (
    RandomBrightness,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
)

logger = logging.getLogger(__name__)


def prepare_for_training(
    ds: tf.data.Dataset,
    batch_size: int = 32,
    shuffle=False,
    augment=False,
    data_augmentation_pipeline: tf.keras.Sequential = None,
) -> tf.data.Dataset:
    """Prepare a dataset for training.

    Parameters
    ----------
    ds : tf.data.Dataset
        The dataset to prepare.
    batch_size : int, default=32
        The batch size.
    shuffle : bool, default=False
        Whether to shuffle the dataset.
    augment : bool, default=False
        Whether to apply data augmentation to the images.
    data_augmentation_pipeline: tf.keras.Sequential
        The data augmentation pipeline to apply to the images.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        if data_augmentation_pipeline is None:
            raise ValueError(
                "data_augmentation_pipeline must be provided if augment=True"
            )
        ds = ds.map(
            lambda x, y: (data_augmentation_pipeline(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def get_dataset(
    name: str, data_path: str, target_size: Sequence[int] = (200, 200)
) -> tf.data.Dataset:
    """Get the datasets."""
    if name.lower() == "utkface":
        return get_utkface_dataset(data_path=data_path, target_size=target_size)
    else:
        raise ValueError(f"Invalid dataset name given: {name}")


def train_test_split(ds: tf.data.Dataset, split: float = 0.8):
    """Split the dataset into train and test sets."""
    train_size = int(len(ds) * split)
    test_size = len(ds) - train_size

    train_ds = ds.shuffle(1000).take(train_size)
    test_ds = ds.skip(train_size).take(test_size)
    logger.debug(f"Train size: {train_size}")
    logger.debug(f"Test size: {test_size}")
    return train_ds, test_ds


def get_data_augmentation_pipeline(
    random_rotation: float = 0.1,
    random_translation: float = 0.1,
    random_flip: str = "horizontal",
    random_brightness: float = 0.2,
):
    """Get the data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            RandomRotation(factor=random_rotation),
            RandomTranslation(
                width_factor=random_translation,
                height_factor=random_translation,
            ),
            RandomFlip(mode=random_flip),
            RandomBrightness(factor=random_brightness),
        ],
        name="data_augmentation",
    )


def get_utkface_dataset(data_path: str, target_size: Sequence[int] = (200, 200)):
    """Get the UTKFace dataset."""
    dataset = tf.data.Dataset.list_files(data_path + "*")

    def process_path(file_path):
        # read the age from the filename
        filename = tf.strings.split(file_path, os.sep)[-1]
        label = tf.strings.split(filename, "_")[0]
        label = tf.strings.to_number(label, out_type=tf.dtypes.int32)

        # read and decode the image
        raw = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(raw, channels=3)
        logger.debug("Initial shape: ", image.shape)
        image = tf.image.resize(image, [*target_size])
        image.set_shape([*target_size, 3])
        logger.debug("Final shape: ", image.shape)
        return image, label

    labeled_dataset = dataset.map(process_path)
    return labeled_dataset
