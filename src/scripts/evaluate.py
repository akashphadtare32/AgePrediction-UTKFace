"""Script to evaluate the predictions on the test set."""


import numpy as np
from sklearn.metrics import mean_absolute_error

import wandb
from src.evaluation import (
    compare_age_distribution,
    compare_empirical_cdf,
    evaluate_age_distribution,
    plot_residuals,
)


def evaluate(predictions, test_ds):
    """Evaluate the predictions on the test dataset."""
    y_true = [y for _, y in test_ds.unbatch().as_numpy_iterator()]

    # Evaluate the model on the test set.
    test_mae = mean_absolute_error(y_true, predictions)
    print(f"Test MAE (ensemble prediction): {test_mae}")
    wandb.run.summary["test_mae"] = np.mean(test_mae)

    age_dist = compare_age_distribution(y_true, predictions)

    kstest_results = evaluate_age_distribution(y_true, predictions)
    print(f"Kolmogorov-Smirnov test results: {kstest_results}")

    empirical_cdfs = compare_empirical_cdf(y_true, predictions)

    residual_plot = plot_residuals(y_true, predictions)

    wandb.log(
        {
            "age_distribution": wandb.Image(age_dist),
            "empirical_cdfs": wandb.Image(empirical_cdfs),
            "residuals": wandb.Image(residual_plot),
        }
    )
