"""Model module."""
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense


def instantiate_base_model(model_instantiate_cfg: DictConfig) -> tf.keras.Model:
    """Get the model."""
    model = instantiate(model_instantiate_cfg)
    return model


def instantiate_preprocessing(preprocessing: DictConfig):
    """Instantiate the preprocessing function.

    Parameters
    ----------
    preprocessing : DictConfig or None
        The preprocessing function to instantiate with hydra. If None,
        then no preprocessing is applied.

    Returns
    -------
    a partial object
        The preprocessing function to apply to the inputs.
    """
    if preprocessing is not None:
        return instantiate(preprocessing)


def instantiate_top_layers(top_layer_architecture: DictConfig):
    """Instantiate the top layers."""
    if top_layer_architecture is not None:
        return instantiate(top_layer_architecture)


def get_complete_model(
    model_cfg: DictConfig, target_size=(200, 200), channels=3
) -> tf.keras.Model:
    """Get the complete model."""
    base_model = instantiate_base_model(model_cfg.instantiate)

    # if the model has a base model, then we freeze it
    # (e.g. VGG, ResNet, EfficientNetV2 ...)
    # for the baseline cnn, this is just the model itself
    base_model.trainable = not model_cfg.freeze_base

    preprocessing = instantiate_preprocessing(model_cfg.preprocessing)
    inputs = Input(shape=(*target_size, channels))
    x = preprocessing(inputs) if preprocessing is not None else inputs

    x = base_model(x, training=False)

    apply_top_layers = instantiate_top_layers(model_cfg.top_layer_architecture)
    x = apply_top_layers(x) if apply_top_layers is not None else x

    outputs = Dense(1, activation="relu")(x)
    model = Model(inputs, outputs)
    return model


def build_model_from_cfg(
    cfg: DictConfig, model=None, first_stage=True
) -> tf.keras.Model:
    """Build the model from the config dict."""
    if model is None:
        model = get_complete_model(
            cfg.model, target_size=cfg.train.target_size, channels=cfg.dataset.channels
        )

    lr_schedule_cfg = (
        cfg.lr_schedule.stage_1 if first_stage else cfg.lr_schedule.stage_2
    )

    optimizer = build_optimizer_from_cfg(cfg.optimizer, lr_schedule_cfg)
    loss_fn = cfg.train.loss
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=cfg.train.metrics,
    )
    return model


def build_optimizer_from_cfg(
    optim_cfg: DictConfig, lr_schedule_cfg: DictConfig
) -> tf.keras.optimizers.Optimizer:
    """Build the optimizer from the config dict ."""
    optimizer = instantiate(optim_cfg)
    lr_schedule = instantiate(lr_schedule_cfg)
    optimizer = optimizer(lr_schedule)
    return optimizer
