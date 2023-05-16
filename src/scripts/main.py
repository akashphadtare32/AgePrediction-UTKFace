"""Main script to train and evaluate the model.

Uses K-Fold Cross Validation or just a simple train-test split and evaluates the model
or ensemble of models on a test dataset.
"""
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn import model_selection

import wandb
from src.dataset import get_dataset
from src.datasets.utils import get_dataset_filepaths
from src.ensemble import make_ensemble_prediction
from src.scripts.cross_validate import cross_validate
from src.scripts.evaluate import evaluate
from src.scripts.train import train
from src.utils import save_and_upload_model


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the model with K-Fold Cross Validation or just a simple train-test split.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    if cfg.train.seed is not None:
        tf.random.set_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

    if cfg.wandb.sync_tensorboard:
        wandb.tensorboard.patch(root_logdir=cfg.log_dir)

    wandb.init(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    filepaths = get_dataset_filepaths(cfg.dataset.path)

    train_filepaths, test_filepaths = model_selection.train_test_split(
        filepaths,
        test_size=cfg.train.test_size,
        random_state=cfg.train.seed,
        shuffle=True,
    )
    test_ds = get_dataset(cfg.dataset.name, test_filepaths, cfg.train.target_size)
    if cfg.train.cv_folds is None:
        train_ds = get_dataset(cfg.dataset.name, train_filepaths, cfg.train.target_size)
        model, train_ds, test_ds = train(train_ds, test_ds, cfg)
        if not cfg.callbacks.model_ckpt:
            if cfg.wandb.mode == "online":
                model_name = f"run_{wandb.run.id}_model"
                model_dir = f"{cfg.model_dir}/{wandb.run.id}"
                upload = True
            else:
                model_name = "model-best"
                model_dir = cfg.model_dir + "/" + model_name
                upload = False
            save_and_upload_model(model, model_dir, model_name, upload=upload)
        predictions = model.predict(test_ds)
        predictions = np.squeeze(predictions)
    elif cfg.train.cv_folds > 1:
        models, test_ds = cross_validate(train_filepaths, test_ds, cfg)
        predictions = make_ensemble_prediction(test_ds, models)
    else:
        raise ValueError(
            f"Invalid value for cv_folds: {cfg.train.cv_folds}. "
            "Must be None or an integer greater than 1."
        )
    evaluate(predictions, test_ds)
    wandb.finish()


if __name__ == "__main__":
    main()
