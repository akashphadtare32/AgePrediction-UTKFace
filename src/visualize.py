"""Visualizations module."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_dataset(ds: tf.data.Dataset, figsize=(8, 8)) -> mpl.figure.Figure:
    """Visualize the dataset."""
    fig = plt.figure(figsize=figsize)
    for i, (image, label) in enumerate(ds.take(9)):
        _ = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("int32"))
        plt.title(int(label))
        plt.axis("off")
    plt.tight_layout()
    return fig


def visualize_augmented_image(
    train_ds: tf.data.Dataset, data_augmentation: tf.keras.Sequential, figsize=(8, 8)
) -> mpl.figure.Figure:
    """Visualize the augmented images."""
    fig = plt.figure(figsize=figsize)
    for image, _ in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(image, 0), training=True)
            plt.imshow(augmented_image[0].numpy().astype("int32"))
            plt.axis("off")
    plt.tight_layout()
    return fig


def visualize_predictions(y_pred, ds):
    """Visualize the predictions."""
    fig = plt.figure(figsize=(8, 8))
    for i, (image, label) in enumerate(ds.take(9)):
        _ = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("int32"))
        plt.title(f"True: {int(label)}\nPred: {int(y_pred[i])}")
        plt.axis("off")
    plt.tight_layout()
    return fig
