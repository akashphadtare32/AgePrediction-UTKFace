"""Test the config setup."""
from hydra import compose, initialize


# 1. initialize will add config_path the config search path within the context
# 2. The module with your configs should be importable.
#    it needs to have a __init__.py (can be empty).
# 3. THe config path is relative to the file calling initialize (this file)
def test_with_initialize() -> None:
    """Test the config setup with initialize (compose API)."""
    with initialize(version_base="1.3", config_path="../src/conf"):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=[
                "wandb.project=test_project",
                "wandb.mode=offline",
                "wandb.group=test_group",
            ],
        )
        assert cfg.wandb == {
            "project": "test_project",
            "mode": "offline",
            "group": "test_group",
            "name": None,
        }
