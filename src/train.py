"""Train the model."""
import logging
import multiprocessing

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn import model_selection

import wandb
from src.callbacks import get_callbacks
from src.dataset import (
    get_data_augmentation_pipeline,
    get_dataset,
    prepare_for_training,
)
from src.datasets.utils import (
    get_dataset_filepaths,
    get_dataset_labels_from_filepaths,
    get_label_function_for,
)
from src.model import build_model_from_cfg
from src.utils import save_and_upload_model

logger = logging.getLogger(__name__)


def train(train_ds, val_ds, cfg):
    """Train the model on the given train and validation datasets.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset. Note that this dataset should be shuffled,
        but not prepared for training (no batching etc).
    val_ds : tf.data.Dataset
        Validation dataset.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    tf.keras.Model
        The trained model.
    """
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
        cache=cfg.train.cache_dataset,
    )
    val_ds = prepare_for_training(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        augment=False,
        cache=cfg.train.cache_dataset,
    )

    callbacks = get_callbacks(val_ds, **cfg.callbacks)

    model = build_model_from_cfg(cfg, first_stage=True)
    print(model.summary(expand_nested=True))
    model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count(),
    )
    if cfg.model.finetune_whole_model:
        # second round of fine-tuning
        model = build_model_from_cfg(cfg, model=model, first_stage=False)
        model.trainable = True
        print(model.summary(expand_nested=False))
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
    return model


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    if cfg.train.seed is not None:
        tf.random.set_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

    wandb.init(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    filepaths = get_dataset_filepaths(cfg.dataset.path)
    get_label_func = get_label_function_for(cfg.dataset.name)
    labels = get_dataset_labels_from_filepaths(filepaths, get_label_func)

    (
        train_filepaths,
        train_labels,
        test_filepaths,
        test_labels,
    ) = model_selection.train_test_split(
        filepaths,
        labels,
        test_size=0.2,
        random_state=cfg.train.seed,
        shuffle=True,
        stratify=labels,
    )
    test_ds = get_dataset(
        name=cfg.dataset.name,
        filepaths=test_filepaths,
        target_size=cfg.train.target_size,
    )
    test_ds = prepare_for_training(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        augment=False,
        cache=cfg.train.cache_dataset,
    )

    # perform Stratified K-Fold cross-validation
    cv = model_selection.StratifiedKFold(
        n_splits=cfg.train.n_splits, shuffle=True, random_state=cfg.train.seed
    )
    results = []
    models = []
    for i, (train_idx, val_idx) in enumerate(cv.split(train_filepaths)):
        print(f"Fitting model on Fold {i + 1}")
        train_ds = get_dataset(
            name=cfg.dataset.name,
            filepaths=train_filepaths[train_idx],
            target_size=cfg.train.target_size,
        )
        val_ds = get_dataset(
            name=cfg.dataset.name,
            filepaths=train_filepaths[val_idx],
            target_size=cfg.train.target_size,
        )

        model = train(train_ds, val_ds, cfg)
        results.append(model.evaluate(val_ds))
        models.append(model)

    print(f"Average Validation MAE: {np.mean(results)}")
    print(f"Validation MAE Std: {np.std(results)}")
    test_results = [model.evaluate(test_ds) for model in models]

    wandb.run.summary(
        {
            "avg_val_mae": np.mean(results),
            "val_mae_std": np.std(results),
            "avg_test_mae": np.mean(test_results),
            "test_mae_std": np.std(test_results),
        }
    )

    # save the model
    if not cfg.callbacks.model_ckpt:
        if cfg.wandb.mode == "online":
            model_name = f"run_{wandb.run.id}_model"
            model_dir = cfg.model_dir + "/" + wandb.run.id
            upload = True
        else:
            model_name = "model-best"
            model_dir = cfg.model_dir + "/" + model_name
            upload = False
        save_and_upload_model(model, model_dir, model_name, upload=upload)
    wandb.finish()


if __name__ == "__main__":
    main()
