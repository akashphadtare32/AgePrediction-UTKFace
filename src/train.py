"""Train the model."""
import multiprocessing
from logging import Logger

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.callbacks import get_callbacks
from src.dataset import get_dataset, train_test_split
from src.model import get_model

logger = Logger(__name__)


def fit_model(cfg, model, train_ds, val_ds, callbacks):
    """Fit the model."""
    model.fit(
        train_ds,
        epochs=cfg.model.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count(),
    )

    if cfg.finetune_whole_model:
        # second round of fine-tuning

        model.trainable = True

        # TODO: clean this up
        lr_schedule = instantiate(cfg.model.lr_schedule_finetuning)
        optim_partial = instantiate(cfg.model.optimizer)
        optimizer = optim_partial(lr_schedule)

        loss = "mean_absolute_error"
        metrics = ["mae"]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary(print_fn=logger.info)
        model.fit(
            train_ds,
            epochs=cfg.model.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count(),
        )


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
    logger.info(f"Preparing dataset {cfg.dataset.name}...")
    ds = get_dataset(
        name=cfg.dataset.name,
        data_path=cfg.dataset.path,
        target_size=cfg.model.target_size,
    )
    train_ds, test_ds = train_test_split(ds, split=0.9)
    train_ds, val_ds = train_test_split(train_ds, split=0.9)
    logger.info("Dataset prepared.")

    callbacks = get_callbacks(val_ds, **cfg.callbacks)

    model = get_model()
    # first round of fine-tuning
    fit_model(cfg, model, train_ds, val_ds, callbacks)
    logger.info(model.evaluate(test_ds))


if __name__ == "__main__":
    main()
