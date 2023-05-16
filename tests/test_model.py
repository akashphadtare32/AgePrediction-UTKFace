"""Test the model module."""

import pytest
from hydra import compose, initialize
from tensorflow.keras import Input

from src.model import (
    build_model_from_cfg,
    instantiate_base_model,
    instantiate_preprocessing,
)


@pytest.fixture(scope="session")
def efficientnet_config():
    """Return the config for the EfficientNet model."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=efficientnetv2",
                "model.num_finetune_layers=all",
                "optimizer=adam",
            ],
        )
        return cfg


def test_instantiate_efficientnet(efficientnet_config):
    """Test that the model can be instantiated."""
    model = instantiate_base_model(efficientnet_config.model.instantiate)
    assert model.__class__.__name__ == "Functional"


def test_build_from_cfg(efficientnet_config):
    """Test that the model can be built from the config dict."""
    model, base_model = build_model_from_cfg(efficientnet_config, first_stage=True)
    assert model.__class__.__name__ == "Functional"
    assert model.optimizer.__class__.__name__ == "Adam"

    actual_lr = efficientnet_config.lr_schedule.stage_1.initial_learning_rate
    assert (
        model.optimizer.get_config()["learning_rate"]["config"]["initial_learning_rate"]
        == actual_lr
    )

    for layer in base_model.layers:
        # should be frozen because `model.freeze_base=True`
        assert layer.trainable is False


def test_build_from_cfg_stage2(efficientnet_config):
    """Test that the stage-2 model can be built from the config dict."""
    model, base_model = build_model_from_cfg(efficientnet_config, first_stage=True)
    model, base_model = build_model_from_cfg(
        efficientnet_config, model=model, base_model=base_model, first_stage=False
    )
    assert model.__class__.__name__ == "Functional"
    assert model.optimizer.__class__.__name__ == "Adam"

    actual_lr = efficientnet_config.lr_schedule.stage_2.initial_learning_rate

    optim_config = model.optimizer.get_config()
    assert optim_config["learning_rate"]["config"]["initial_learning_rate"] == actual_lr

    for layer in base_model.layers:
        # in second, stage, all layers should be trainable
        # because `model.num_finetune_layers=all`
        assert layer.trainable is True


def test_no_preprocessing(efficientnet_config):
    """Test that the preprocessing function can be applied."""
    preprocessing = instantiate_preprocessing(efficientnet_config.model.preprocessing)
    assert preprocessing is None


def test_with_preprocessing():
    """Test that the preprocessing function can be applied."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        cfg = compose(config_name="config", overrides=["model=resnet"])

        x = Input(shape=(224, 224, 3))
        preprocessing = instantiate_preprocessing(cfg.model.preprocessing)
        assert preprocessing is not None
        x = preprocessing(x)


def test_num_finetuning_layers():
    """Test that the number of finetuning layers can be set."""
    num_layers = 2

    with initialize(version_base="1.3", config_path="../src/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=vgg",
                f"model.num_finetune_layers={num_layers}",
            ],
        )
        _, base_model = build_model_from_cfg(cfg, first_stage=False)
        for layer in base_model.layers[-num_layers:]:
            assert layer.trainable is True
        for layer in base_model.layers[:-num_layers]:
            assert layer.trainable is False
