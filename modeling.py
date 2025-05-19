"""Model training and evaluation helpers for the forecasting pipeline."""

from prophet_analysis import (
    tune_prophet_hyperparameters as _tune_prophet_hyperparameters,
    train_prophet_model as _train_prophet_model,
    evaluate_prophet_model as _evaluate_prophet_model,
)


def tune_prophet_hyperparameters(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.tune_prophet_hyperparameters`."""
    return _tune_prophet_hyperparameters(*args, **kwargs)


def train_prophet_model(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.train_prophet_model`."""
    return _train_prophet_model(*args, **kwargs)


def evaluate_prophet_model(*args, **kwargs):
    """Proxy to :func:`prophet_analysis.evaluate_prophet_model`."""
    return _evaluate_prophet_model(*args, **kwargs)
