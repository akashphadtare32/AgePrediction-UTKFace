"""Test the config setup."""
import os

from hydra import compose, initialize


def test_with_initialize() -> None:
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
