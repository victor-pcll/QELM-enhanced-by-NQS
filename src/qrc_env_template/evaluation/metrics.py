"""Metric names. No scores are computed."""

from __future__ import annotations


def metric_status(names: list[str]) -> str:
    """Report which metrics a run would compute.

    Args:
        names: Metric identifiers from ``cfg.evaluation.metrics``. Each
            entry is a scalar string. No score tensor is computed.

    Returns:
        A status line listing the names and ``status=scaffold``.
    """
    return "metrics=" + ",".join(names) + " status=scaffold"

def NMSE(y_true, y_pred):
    """Normalized Mean Squared Error (NMSE) metric.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        NMSE score.
    """
    mse = MSE(y_true, y_pred)
    norm = (y_true ** 2).mean()
    return mse / norm

def MSE(y_true, y_pred):
    """Mean Squared Error (MSE) metric.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MSE score.
    """
    return ((y_true - y_pred) ** 2).mean()
