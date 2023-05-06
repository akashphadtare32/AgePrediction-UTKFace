"""Cross-Validation Script."""
import numpy as np
from sklearn import model_selection

import wandb
from src.dataset import get_dataset, prepare_for_training
from src.scripts.train import train
from src.utils import save_and_upload_model


def cross_validate(train_filepaths, test_ds, cfg):
    """Perform K-Fold Cross-Validation and return the fitted models.

    Parameters
    ----------
    train_filepaths : List[str]
        List of filepaths to the training images.
    test_ds : tf.data.Dataset
        Test dataset. It is expected that the images have already been split into
        train and test images. The training images will be used for K-Fold cross
        validation and the test images will be used for the final evaluation of
        the ensemble model.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    models : List[tf.keras.Model]
        List of trained models.
    """
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

    wandb.run.summary["avg_val_mae"] = np.mean(results)
    wandb.run.summary["val_mae_std"] = np.std(results)

    return models
