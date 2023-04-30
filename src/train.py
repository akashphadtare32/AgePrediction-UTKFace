"""Train the model."""
import logging
import multiprocessing

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.callbacks import get_callbacks
from src.dataset import get_dataset, train_test_split
from src.model import build_model_from_cfg

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    wandb.init(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    ds = get_dataset(
        name=cfg.dataset.name,
        data_path=cfg.dataset.path,
        target_size=cfg.model.target_size,
    )
    train_ds, test_ds = train_test_split(ds, split=0.9)
    train_ds, val_ds = train_test_split(train_ds, split=0.9)
    logger.info("Dataset prepared.")

    callbacks = get_callbacks(val_ds, **cfg.callbacks)

    model = build_model_from_cfg(cfg, first_stage=True)
    print(model.summary())
    model.fit(
        train_ds,
        epochs=cfg.model.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count(),
    )
    # second round of fine-tuning
    model = build_model_from_cfg(cfg, model=model, first_stage=False)
    model.trainable = True
    print(model.summary())
    model.fit(
        train_ds,
        epochs=cfg.model.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count(),
    )
    print(model.evaluate(test_ds))


if __name__ == "__main__":
    main()
