"""Evaluation functions for the age prediction model."""

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import kstest


def compare_age_distribution(y_true, y_pred):
    """Plot the distribution of ages in the predicted vs. the true ages."""
    fig = plt.figure(figsize=(10, 5))
    sns.kdeplot(y_true, label="Ground Truth")
    sns.kdeplot(y_pred, label="Predictions")
    plt.legend()
    plt.xlabel("Age")
    return fig


def compare_age_distribution_plotly(y_true, y_pred):
    """Plot the distribution of ages in the predicted vs. the true ages with plotly."""
    # Group data together
    hist_data = [y_true, y_pred]

    group_labels = ["Ground Truth", "Predictions"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=1, show_rug=False)
    fig.layout.update({"title": "Predicted vs Real Age Distribution"})
    return fig


def compare_empirical_cdf(y_true, y_pred):
    """Plot the empirical CDF of the predicted vs. the true ages."""
    fig = plt.figure(figsize=(10, 5))
    sns.ecdfplot(y_true, label="Ground Truth")
    sns.ecdfplot(y_pred, label="Predictions")
    plt.legend()
    plt.title("Empirical CDF of the predicted vs. the true ages.")
    return fig


def compare_empirical_cdf_plotly(y_true, y_pred):
    """Plot the empirical CDF of the predicted vs. the true ages with plotly."""
    fig = go.Figure()
    fig.add_scatter(
        x=px.ecdf(y_true).data[0].x,
        y=px.ecdf(y_true, marginal="histogram").data[0].y,
        mode="lines",
        name="Ground Truth",
    )
    fig.add_scatter(
        x=px.ecdf(y_pred).data[0].x,
        y=px.ecdf(y_pred, marginal="histogram").data[0].y,
        mode="lines",
        name="Predictions",
    )
    fig.update_layout(title="Empirical CDF of the predicted vs. the true ages.")
    return fig


def evaluate_age_distribution(y_true, y_pred):
    """Evaluate the distribution of ages in the predicted vs. the true ages."""
    results = kstest(y_true, y_pred)
    return results


def plot_residuals(y_true, y_pred):
    """Plot the residuals of the predicted vs. the true ages."""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_true, y=residuals)
    plt.xlabel("Age")
    plt.ylabel("Residuals")
    return fig


def plot_residuals_plotly(y_true, y_pred):
    """Plot the residuals of the predicted vs. the true ages with plotly."""
    residuals = y_true - y_pred
    fig = px.scatter(x=y_true, y=residuals, labels={"x": "True Age", "y": "Residuals"})
    fig.update_layout(title="Residuals of the predicted vs. the true ages.")
    return fig
