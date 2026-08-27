"""Tests for forecasting.probabilistic."""
from __future__ import annotations

import numpy as np

from forecasting.probabilistic import (
    compute_residual_quantiles, apply_probabilistic_bands, build_probabilistic_forecast,
)


def test_compute_residual_quantiles_returns_zero_for_tiny_sample():
    residuals = np.array([1.0, 2.0, 3.0])  # fewer than 5 -> insufficient
    lower, upper = compute_residual_quantiles(residuals)
    assert lower == 0.0
    assert upper == 0.0


def test_compute_residual_quantiles_symmetric_distribution():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 100, size=1000)
    lower, upper = compute_residual_quantiles(residuals, 0.10, 0.90)
    assert lower < 0 < upper
    # roughly symmetric around zero for a symmetric distribution
    assert abs(abs(lower) - abs(upper)) < 30


def test_compute_residual_quantiles_ignores_nans():
    residuals = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    lower, upper = compute_residual_quantiles(residuals, 0.10, 0.90)
    assert np.isfinite(lower)
    assert np.isfinite(upper)


def test_apply_probabilistic_bands_ordering():
    point_forecast = np.array([100.0, 200.0, 300.0])
    bands = apply_probabilistic_bands(point_forecast, lower_offset=-20, upper_offset=30)
    assert (bands["p10"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p90"]).all()


def test_apply_probabilistic_bands_clips_at_zero():
    point_forecast = np.array([5.0, 10.0])
    bands = apply_probabilistic_bands(point_forecast, lower_offset=-100, upper_offset=50)
    assert (bands["p10"] >= 0).all()
    assert (bands["p50"] >= 0).all()


def test_apply_probabilistic_bands_p50_equals_point_forecast_when_nonnegative():
    point_forecast = np.array([100.0, 200.0])
    bands = apply_probabilistic_bands(point_forecast, lower_offset=-10, upper_offset=10)
    np.testing.assert_array_almost_equal(bands["p50"].values, point_forecast)


def test_build_probabilistic_forecast_end_to_end():
    point_forecast = np.array([100.0, 150.0, 200.0])
    rng = np.random.default_rng(1)
    residuals = rng.normal(0, 20, size=200)
    bands = build_probabilistic_forecast(point_forecast, residuals)
    assert list(bands.columns) == ["p10", "p50", "p90"]
    assert len(bands) == 3
    assert (bands["p10"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p90"]).all()
