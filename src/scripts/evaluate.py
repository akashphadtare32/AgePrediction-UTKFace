"""Script to evaluate the predictions on the test set."""


from sklearn.metrics import mean_absolute_error

import wandb
from src.evaluation import (
    compare_age_distribution_plotly,
    compare_empirical_cdf_plotly,
    evaluate_age_distribution,
    plot_residuals_plotly,
)
from src.visualize import visualize_predictions


def evaluate(predictions, test_ds):
    """Evaluate the predictions on the test dataset."""
    y_true = [y for _, y in test_ds.unbatch().as_numpy_iterator()]

    # Evaluate the model on the test set.
    test_mae = mean_absolute_error(y_true, predictions)
    print(f"Test MAE (ensemble prediction): {test_mae}")
    wandb.run.summary["test_mae"] = test_mae

    age_dist = compare_age_distribution_plotly(y_true, predictions)

    kstest_results = evaluate_age_distribution(y_true, predictions)
    print(f"Kolmogorov-Smirnov test results: {kstest_results}")

    empirical_cdfs = compare_empirical_cdf_plotly(y_true, predictions)

    residual_plot = plot_residuals_plotly(y_true, predictions)

    # some predictions
    fig = visualize_predictions(predictions, test_ds.unbatch())

    wandb.log(
        {
            "age_distribution": age_dist,
            "empirical_cdfs": empirical_cdfs,
            "residuals": residual_plot,
            "predictions": fig,
        }
    )
