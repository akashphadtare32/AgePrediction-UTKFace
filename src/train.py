"""Train the model."""
import logging
import multiprocessing

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from src.callbacks import get_callbacks
from src.dataset import (
    get_data_augmentation_pipeline,
    get_dataset,
    prepare_for_training,
    train_test_split,
)
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

    if cfg.augment.active:
        data_augmentation_pipeline = get_data_augmentation_pipeline(
            **cfg.augment.factors
        )
    else:
        data_augmentation_pipeline = None

    train_ds = prepare_for_training(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        augment=cfg.augment.active,
        data_augmentation_pipeline=data_augmentation_pipeline,
    )
    val_ds = prepare_for_training(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        augment=False,
    )
    test_ds = prepare_for_training(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        augment=False,
    )

    callbacks = get_callbacks(val_ds, **cfg.callbacks)

    model = build_model_from_cfg(cfg, first_stage=True)
    print(model.summary())
    model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count(),
    )
    if cfg.model.fine_whole_model:
        # second round of fine-tuning
        model = build_model_from_cfg(cfg, model=model, first_stage=False)
        model.trainable = True
        print(model.summary())
        callbacks = get_callbacks(val_ds, initial_epoch=wandb.run.step, **cfg.callbacks)
        model.fit(
            train_ds,
            epochs=cfg.train.epochs,
            initial_epoch=wandb.run.step,
            validation_data=val_ds,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count(),
        )
    wandb.log({"test_loss": model.evaluate(test_ds)[0]})


if __name__ == "__main__":
    main()
