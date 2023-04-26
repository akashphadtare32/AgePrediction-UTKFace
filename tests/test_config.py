"""Test the config setup."""
import os

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig


@pytest.fixture(scope="session")
def defaults_config() -> DictConfig:
    """Get the config."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        cfg = compose(
            config_name="config",
        )
        return cfg


def test_with_initialize():
    """Test the config setup with initialize (compose API)."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        # config is relative to a module
        os.environ["WANDB_API_KEY"] = "test"
        cfg = compose(
            config_name="config",
            overrides=[
                "wandb.project=test_project",
            ],
        )
        assert cfg.wandb.project == "test_project"
        assert cfg.wandb.key == "test"


def test_defaults(defaults_config):
    """Test some default values in the config."""
    # model
    assert defaults_config.model.name == "EfficientNetV2B0"

    # dataset
    assert defaults_config.dataset.name == "UTKFace"


def test_instantiate_optimizer(defaults_config):
    """Test the optimizer instantiation."""
    # instantiate the optimizer
    optim_partial = instantiate(defaults_config.optimizer)
    assert optim_partial.__class__.__name__ == "partial"

    # finish the instantiation
    optimizer = optim_partial(1e-3)
    assert optimizer.learning_rate == 1e-3


def test_instantiate_lr_schedule(defaults_config):
    """Test the learning rate schedule instantiation."""
    # lr_schedule

    # instantiate the lr_schedule
    lr_schedule = instantiate(defaults_config.lr_schedule)
    assert (
        lr_schedule.initial_learning_rate
        == defaults_config.lr_schedule.initial_learning_rate
    )
