"""Train the model."""
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    print(cfg)
