"""Utility functions."""
import tensorflow as tf
import wandb


def restore_model(run_id: str, version: int):
    """Restores the model.

    Restores the model from the run with the given
    id and version (does not equal epoch in general).

    Downloads the artifact from W&B and returns the model.
    Use this, if the kernel crashed.
    Otherwise you can use `wandb.restore(name=<model-name>)`
    """
    model_name = f"run_{run_id}_model:v{version}"
    artifact_name = f"moritzm00/UTKFace-Age-Regression/{model_name}"
    if wandb.run is not None:
        artifact = wandb.run.use_artifact(artifact_name, type="model")
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    model = tf.keras.models.load_model(artifact_dir)
    return model
