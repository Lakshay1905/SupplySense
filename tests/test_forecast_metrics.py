"""Tests for forecasting.evaluation.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.evaluation.metrics import mae, rmse, mape, wmape, forecast_bias, compute_all_metrics


def test_mae_perfect_forecast_is_zero():
    y = np.array([100, 200, 300])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 320])
    assert mae(y_true, y_pred) == pytest.approx((10 + 10 + 20) / 3)


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([100, 100, 100, 100])
    y_pred_uniform = np.array([110, 110, 110, 110])   # error 10 everywhere
    y_pred_spiky = np.array([100, 100, 100, 140])     # one big error of 40
    # Same total absolute error (40) but RMSE should be higher for the spiky case
    assert mae(y_true, y_pred_uniform) == mae(y_true, y_pred_spiky)
    assert rmse(y_true, y_pred_spiky) > rmse(y_true, y_pred_uniform)


def test_mape_excludes_zero_actuals():
    y_true = np.array([0, 100, 200])
    y_pred = np.array([50, 110, 180])
    # Only rows where y_true != 0 contribute
    expected = np.mean([abs(110 - 100) / 100, abs(180 - 200) / 200]) * 100
    assert mape(y_true, y_pred) == pytest.approx(expected)


def test_mape_all_zero_actuals_returns_nan():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 2, 3])
    assert np.isnan(mape(y_true, y_pred))


def test_wmape_handles_zero_actuals_gracefully():
    """Unlike MAPE, WMAPE is well-defined even with some zero actuals,
    since it is a ratio of sums rather than a mean of ratios."""
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 180])
    expected = (10 + 10 + 20) / (0 + 100 + 200) * 100
    assert wmape(y_true, y_pred) == pytest.approx(expected)


def test_wmape_all_zero_actuals_returns_nan():
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    assert np.isnan(wmape(y_true, y_pred))


def test_forecast_bias_detects_overforecasting():
    y_true = np.array([100, 100, 100])
    y_pred = np.array([110, 110, 110])  # consistently 10% over
    assert forecast_bias(y_true, y_pred) == pytest.approx(10.0)


def test_forecast_bias_detects_underforecasting():
    y_true = np.array([100, 100, 100])
    y_pred = np.array([90, 90, 90])
    assert forecast_bias(y_true, y_pred) == pytest.approx(-10.0)


def test_forecast_bias_zero_for_perfect_forecast():
    y = np.array([100, 200, 300])
    assert forecast_bias(y, y) == pytest.approx(0.0)


def test_compute_all_metrics_returns_expected_keys():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = compute_all_metrics(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "mape", "wmape", "bias"}
    assert all(isinstance(v, float) for v in result.values())
