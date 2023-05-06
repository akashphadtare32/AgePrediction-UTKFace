"""Ensemble methods for combining models."""

import numpy as np


def make_ensemble_prediction(X, models):
    """Make an ensemble prediction for a regression problem."""
    preds = [model.predict(X) for model in models]
    y_pred = np.mean(preds, axis=0).reshape(-1)
    return y_pred
