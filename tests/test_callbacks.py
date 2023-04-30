"""Tests for callbacks creation."""
import tensorflow as tf

import wandb
from src.callbacks import get_callbacks


def test_only_early_stopping():
    """Test retrieval of early stopping callback only."""
    callbacks = get_callbacks(
        validation_data=None,
        early_stopping_patience=5,
        monitor="val_mae",
        use_wandb=False,
        model_ckpt=False,
        with_wandb_ckpt=False,
    )
    assert len(callbacks) == 1
    early_stopping = callbacks[0]
    assert isinstance(early_stopping, tf.keras.callbacks.EarlyStopping)
    assert early_stopping.patience == 5
    assert early_stopping.monitor == "val_mae"


def test_es_with_local_ckpt():
    """Test early stopping with local checkpoint."""
    callbacks = get_callbacks(
        validation_data=None,
        use_wandb=False,
        model_ckpt=True,
        with_wandb_ckpt=False,
    )
    assert len(callbacks) == 2
    early_stopping = callbacks[0]
    model_ckpt = callbacks[1]
    assert isinstance(early_stopping, tf.keras.callbacks.EarlyStopping)
    assert isinstance(model_ckpt, tf.keras.callbacks.ModelCheckpoint)


def test_local_ckpt_with_wandb_callbacks():
    """Test local checkpoint with wandb callbacks."""
    run = wandb.init(mode="disabled")  # TODO: move to a pytest fixture
    callbacks = get_callbacks(
        validation_data=None,
        use_wandb=True,
        with_wandb_ckpt=False,
        visualize_predictions=False,
        model_ckpt=True,
    )
    assert len(callbacks) == 3
    [early_stopping, wandb_metrics, model_ckpt] = callbacks
    assert isinstance(early_stopping, tf.keras.callbacks.EarlyStopping)
    assert isinstance(wandb_metrics, wandb.keras.WandbMetricsLogger)
    assert isinstance(
        model_ckpt, tf.keras.callbacks.ModelCheckpoint
    )  # not WandbModelCheckpoint
    run.finish()


def test_wandb_ckpt_with_wandb_callbacks():
    """Test wandb checkpoint with wandb callbacks."""
    run = wandb.init(mode="disabled")  # TODO: move to a pytest fixture
    callbacks = get_callbacks(
        validation_data=None,
        use_wandb=True,
        with_wandb_ckpt=True,
        visualize_predictions=False,
        model_ckpt=True,
    )
    assert len(callbacks) == 3
    [early_stopping, wandb_metrics, wandb_ckpt] = callbacks
    assert isinstance(early_stopping, tf.keras.callbacks.EarlyStopping)
    assert isinstance(wandb_metrics, wandb.keras.WandbMetricsLogger)
    assert isinstance(wandb_ckpt, wandb.keras.WandbModelCheckpoint)
    run.finish()
