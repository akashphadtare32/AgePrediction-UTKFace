"""Evaluation functions for the age prediction model."""

import matplotlib.pyplot as plt
import seaborn as sns


def compare_age_distribution(y_true, y_pred):
    """Plot the distribution of ages in the predicted vs. the true ages."""
    fig = plt.figure(figsize=(10, 5))
    sns.kdeplot(y_true, label="Ground Truth")
    sns.kdeplot(y_pred, label="Predictions")
    plt.xlabel("Age")
    return fig


def compare_empirical_cdf(y_true, y_pred):
    """Plot the empirical CDF of the predicted vs. the true ages."""
    fig = plt.subplots(figsize=(10, 5))
    sns.ecdfplot(y_true, label="Ground Truth")
    sns.ecdfplot(y_pred, label="Predictions")
    plt.legend()
    plt.title("Empirical CDF of the predicted vs. the true ages.")
    return fig


def evaluate_age_distribution(y_true, y_pred):
    """Evaluate the distribution of ages in the predicted vs. the true ages."""
    pass


def plot_residuals(y_true, y_pred):
    """Plot the residuals of the predicted vs. the true ages."""
    pass
