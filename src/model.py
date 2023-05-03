"""Model module."""
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from src.custom_models import convnext, efficientnetv2, resnet, vgg


def instantiate_base_model(model_instantiate_cfg: DictConfig) -> tf.keras.Model:
    """Get the model."""
    model = instantiate(model_instantiate_cfg)
    return model


def instantiate_preprocessing(preprocessing: DictConfig):
    """Instantiate the preprocessing function."""
    if preprocessing is not None:
        return instantiate(preprocessing)


def get_complete_model(model_cfg: DictConfig, channels=3) -> tf.keras.Model:
    """Get the complete model."""
    base_model = instantiate_base_model(model_cfg.instantiate)

    # if the model has a base model, then we freeze it
    # (e.g. VGG, ResNet, EfficientNetV2 ...)
    base_model.trainable = model_cfg.freeze_base

    preprocessing = instantiate_preprocessing(model_cfg.preprocessing)
    inputs = Input(shape=(*model_cfg.target_size, channels))
    x = preprocessing(inputs) if preprocessing is not None else inputs

    x = base_model(x, training=False)

    # determine the top layers based on the architecture
    model_architecture = model_cfg.architecture.lower()
    if model_architecture == "vgg":
        x = vgg.apply_top_layers(x)
    elif model_architecture == "resnet":
        x = resnet.apply_top_layers(x)
    elif model_architecture == "convnext":
        x = convnext.apply_top_layers(x)
    elif model_architecture == "efficientnetv2":
        x = efficientnetv2.apply_top_layers(x)

    outputs = Dense(1, activation="relu")(x)
    model = Model(inputs, outputs)
    return model


def build_model_from_cfg(
    cfg: DictConfig, model=None, first_stage=True
) -> tf.keras.Model:
    """Build the model from the config dict."""
    if model is None:
        model = get_complete_model(cfg.model, channels=cfg.dataset.channels)

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
