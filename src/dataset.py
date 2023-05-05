"""Datasets module."""

import logging
from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras.layers import (
    RandomBrightness,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
)

from src.datasets import get_b3fd_dataset, get_utkface_dataset

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
    data_augmentation = tf.keras.Sequential(name="data_augmentation")

    if random_rotation:
        data_augmentation.add(RandomRotation(factor=random_rotation))
    if random_translation:
        data_augmentation.add(
            RandomTranslation(
                width_factor=random_translation,
                height_factor=random_translation,
            )
        )
    if random_flip:
        data_augmentation.add(RandomFlip(mode=random_flip))
    if random_brightness:
        data_augmentation.add(RandomBrightness(factor=random_brightness))
    return data_augmentation


def get_dataset(
    name: str,
    data_path: str,
    target_size: Sequence[int] = (200, 200),
) -> tf.data.Dataset:
    """Get the datasets."""
    name = name.lower()
    if name == "utkface":
        return get_utkface_dataset(data_path=data_path, target_size=target_size)
    elif name == "b3fd":
        return get_b3fd_dataset(data_path=data_path, target_size=target_size)
    else:
        raise ValueError(f"Invalid dataset name given: {name}")
