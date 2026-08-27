"""
Standard forecast accuracy metrics used across every model in the
benchmarking framework, so results are always apples-to-apples.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error. Rows with y_true == 0 are excluded
    (undefined percentage error), since Rossmann demand data includes
    legitimate zero-sales closed-store days."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum(|error|) / sum(|actual|) -- robust to zeros and
    the standard metric for intermittent/promotional retail demand."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signed mean error as % of average actual demand. Positive = over-forecasting."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return float("nan")
    return float(np.mean(y_pred - y_true) / mean_actual * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
    }
