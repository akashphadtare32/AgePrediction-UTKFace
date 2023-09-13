"""Callbacks module.

This module offers a function to get the callbacks
for model training and implements a custom WandB Callback for Age Classification,
that logs example predictions during training to show the progress of the model.
"""

import tensorflow as tf
from wandb.keras import WandbEvalCallback, WandbMetricsLogger, WandbModelCheckpoint

import wandb


def get_callbacks(
    validation_data: tf.data.Dataset,
    early_stopping_patience: int = 5,
    monitor: str = "val_mae",
    initial_epoch: int = 0,
    use_wandb=True,
    model_ckpt=True,
    ckpt_filepath: str = "ckpt/model-{epoch:02d}-{val_mae:.2f}",
    visualize_predictions=True,
    with_wandb_ckpt=True,
    sync_tensorboard=False,
    tensorboard_log_dir="logs",
) -> list[tf.keras.callbacks.Callback]:
    """Return the callbacks for model training.

    Parameters
    ----------
    validation_data : tf.data.Dataset or None
        The validation dataset to use for visualization of predictions.
        If `visualize_predictions=False`, then this can be None.
    early_stopping_patience : int, default=5
        The number of epochs to wait before early stopping.
    monitor : str, default="val_mae"
        The metric to monitor for early stopping.
    initial_epoch : int, default=0
        The initial epoch number.
    use_wandb : bool, default=True
        Whether to use wandb callbacks.
    model_ckpt : bool, default=True
        Whether to save model checkpoints (locally or via wandb).
        If `with_wandb_ckpt=True`, then checkpoints will be saved to wandb.
        Otherwise they will be saved locally only.
    ckpt_filepath : str, default="ckpt/model-{epoch:02d}-{val_mae:.2f}"
        The filepath to save the model checkpoints.
    visualize_predictions : bool, default=True
        Whether to visualize predictions.
    with_wandb_ckpt : bool, default=True
        Whether to save model checkpoints to wandb.
        If `use_wandb=False`, then this is ignored.
    sync_tensorboard : bool, default=False
        Whether to sync the tensorboard logs to wandb.
    tensorboard_log_dir : str, default="logs"
        The directory to save the tensorboard logs.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=early_stopping_patience,
            monitor=monitor,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if use_wandb:
        if sync_tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    histogram_freq=1,
                    write_graph=False,
                    write_images=False,
                    write_steps_per_second=False,
                    log_dir=tensorboard_log_dir,
                )
            )
        else:
            callbacks.append(WandbMetricsLogger(initial_global_step=initial_epoch))
        if model_ckpt and with_wandb_ckpt:
            callbacks.append(
                WandbModelCheckpoint(
                    ckpt_filepath, monitor=monitor, save_best_only=True, verbose=1
                )
            )
        if visualize_predictions:
            callbacks.append(
                VisualizePredictionsWandbCallback(
                    validation_data=validation_data,
                    data_table_columns=["idx", "image", "label"],
                    pred_table_columns=["epoch", "idx", "image", "label", "pred"],
                    n_samples=8,
                ),
            )
    if model_ckpt and not (use_wandb and with_wandb_ckpt):
        # save checkpoints locally only
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_filepath, monitor=monitor, save_best_only=True
            )
        )

    return callbacks


class VisualizePredictionsWandbCallback(WandbEvalCallback):
    """Classification Evaluation Callback that logs predictions to Weights and biases.

    This Callback runs after each epoch and logs a single batch of predictions.
    """

    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, n_samples=8
    ):
        """Initialize the callback."""
        super().__init__(data_table_columns, pred_table_columns)

        self.data = validation_data

        self.n_samples = n_samples

    def add_ground_truth(self, logs=None):
        """Add ground truth data to the data table."""
        for images, labels in self.data.take(1).as_numpy_iterator():
            for idx, (img, label) in enumerate(zip(images, labels)):
                self.data_table.add_data(idx, wandb.Image(img), label)
                if idx == self.n_samples - 1:
                    return

    def add_model_predictions(self, epoch, logs=None):
        """Add model predictions to the predictions table."""
        preds = self.model.predict(self.data.take(1), verbose=0)

        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx][0]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )
