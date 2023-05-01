"""Test the model module."""

import pytest
from hydra import compose, initialize

from src.model import build_model_from_cfg, instantiate_base_model


@pytest.fixture(scope="session")
def efficientnet_config():
    """Return the config for the EfficientNet model."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        cfg = compose(config_name="config", overrides=["model=efficientnetv2"])
        return cfg


def test_instantiate_efficientnet(efficientnet_config):
    """Test that the model can be instantiated."""
    model = instantiate_base_model(efficientnet_config.model.instantiate)
    assert model.__class__.__name__ == "Functional"


def test_build_from_cfg(efficientnet_config):
    """Test that the model can be built from the config dict."""
    model = build_model_from_cfg(efficientnet_config, first_stage=True)
    assert model.__class__.__name__ == "Functional"
    assert model.optimizer.__class__.__name__ == "Adam"

    actual_lr = efficientnet_config.lr_schedule.stage_1.initial_learning_rate
    assert (
        model.optimizer.get_config()["learning_rate"]["config"]["initial_learning_rate"]
        == actual_lr
    )


def test_build_from_cfg_stage2(efficientnet_config):
    """Test that the stage-2 model can be built from the config dict."""
    model = build_model_from_cfg(efficientnet_config, first_stage=True)
    model = build_model_from_cfg(efficientnet_config, model=model, first_stage=False)
    assert model.__class__.__name__ == "Functional"
    assert model.optimizer.__class__.__name__ == "Adam"

    actual_lr = efficientnet_config.lr_schedule.stage_2.initial_learning_rate

    optim_config = model.optimizer.get_config()
    assert optim_config["learning_rate"]["config"]["initial_learning_rate"] == actual_lr
