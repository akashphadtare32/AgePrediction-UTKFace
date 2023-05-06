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
from src.datasets.utils import get_dataset_filepaths
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
    return model, train_ds, val_ds


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the model with K-Fold Cross Validation.

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
    # get_label_func = get_label_function_for(cfg.dataset.name)
    # labels = get_dataset_labels_from_filepaths(filepaths, get_label_func)

    (
        train_filepaths,
        test_filepaths,
    ) = model_selection.train_test_split(
        filepaths,
        test_size=0.1,
        random_state=cfg.train.seed,
        shuffle=True,
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
    cv = model_selection.KFold(
        n_splits=cfg.train.cv_folds, shuffle=True, random_state=cfg.train.seed
    )
    results = []
    models = []
    print(f"Starting {cfg.train.cv_folds}-Fold Cross-Validation...")
    print(80 * "=")
    for i, (train_idx, val_idx) in enumerate(cv.split(train_filepaths)):
        print(f"Fitting model on Fold {i + 1}")
        print(80 * "-")
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

        model, train_ds, val_ds = train(train_ds, val_ds, cfg)
        _, val_mae = model.evaluate(val_ds)
        results.append(val_mae)
        models.append(model)
        print(f"Validation MAE: {results[-1]}")
        # save the model
        if not cfg.callbacks.model_ckpt:
            if cfg.wandb.mode == "online":
                model_name = f"run_{wandb.run.id}_model"
                model_dir = f"{cfg.model_dir}/{wandb.run.id}/fold{i + 1}"
                upload = True
            else:
                model_name = "model-best"
                model_dir = cfg.model_dir + "/" + model_name + f"_fold{i + 1}"
                upload = False
            save_and_upload_model(model, model_dir, model_name, upload=upload)
        print(80 * "-")

    print(80 * "=")
    print(f"Average Validation MAE: {np.mean(results)}")
    print(f"Validation MAE Std: {np.std(results)}")
    test_results = [model.evaluate(test_ds)[1] for model in models]

    wandb.run.summary["avg_val_mae"] = np.mean(results)
    wandb.run.summary["val_mae_std"] = np.std(results)
    wandb.run.summary["avg_test_mae"] = np.mean(test_results)
    wandb.run.summary["test_mae_std"] = np.std(test_results)
    wandb.finish()


if __name__ == "__main__":
    main()
