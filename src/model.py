"""Model module."""
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig


def instantiate_model(model_instantiate_cfg: DictConfig) -> tf.keras.Model:
    """Get the model."""
    model = instantiate(model_instantiate_cfg)
    return model


def build_model_from_cfg(
    cfg: DictConfig, model=None, first_stage=True
) -> tf.keras.Model:
    """Build the model from the config dict."""
    if model is None:
        model = instantiate_model(cfg.model.instantiate)

    # get the correct lr_schedule
    lr_schedule_cfg = (
        cfg.lr_schedule.stage_1 if first_stage else cfg.lr_schedule.stage_2
    )

    optimizer = build_optimizer_from_cfg(cfg.optimizer, lr_schedule_cfg)
    loss_fn = cfg.loss
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mae"],
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


def compile_model(model, optimizer, lr_schedule, loss_fn):
    """Build ."""
    optimizer = optimizer(lr_schedule)
    return model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mae"],
    )
