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


def save_and_upload_model(model, model_dir, artifact_name=None, upload=True):
    """Save the model and optionally upload it to W&B.

    Parameters
    ----------
    model : tf.keras.Model
        The model to save.
    artifact_name : str
        The name of the artifact. Ignored, if upload is False.
    model_dir : str
        The directory to save the model to.
    upload : bool, optional
        Whether to upload the model to W&B, by default True
    """
    print(f"Saving model to {model_dir}")
    model.save(model_dir)
    if upload:
        print("Uploading model artifact to W&B:", artifact_name)
        trained_model_artifact = wandb.Artifact(
            artifact_name,
            type="model",
        )
        trained_model_artifact.add_dir(model_dir)
        wandb.run.log_artifact(trained_model_artifact)
