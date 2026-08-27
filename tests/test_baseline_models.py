"""Tests for forecasting.baselines.baseline_models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.baselines.baseline_models import (
    naive_forecast, seasonal_naive_forecast, moving_average_forecast,
)


@pytest.fixture
def weekly_series() -> pd.Series:
    # Deterministic weekly pattern: Mon..Sun = 10,20,30,40,50,60,70 repeated 4x
    pattern = [10, 20, 30, 40, 50, 60, 70]
    values = pattern * 4
    idx = pd.date_range("2015-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_naive_forecast_repeats_last_value(weekly_series):
    horizon = 5
    forecast = naive_forecast(weekly_series, horizon)
    assert len(forecast) == horizon
    assert np.all(forecast == weekly_series.iloc[-1])


def test_seasonal_naive_forecast_repeats_last_season(weekly_series):
    horizon = 7
    forecast = seasonal_naive_forecast(weekly_series, horizon, season_length=7)
    np.testing.assert_array_almost_equal(forecast, weekly_series.iloc[-7:].values)


def test_seasonal_naive_forecast_cycles_for_longer_horizon(weekly_series):
    horizon = 10  # more than one season length
    forecast = seasonal_naive_forecast(weekly_series, horizon, season_length=7)
    assert len(forecast) == horizon
    # first 7 values match the last season, next 3 wrap around to the start of that season again
    np.testing.assert_array_almost_equal(forecast[:7], weekly_series.iloc[-7:].values)
    np.testing.assert_array_almost_equal(forecast[7:10], weekly_series.iloc[-7:-4].values)


def test_seasonal_naive_falls_back_to_naive_for_short_history():
    short_series = pd.Series([100, 200], index=pd.date_range("2015-01-01", periods=2))
    forecast = seasonal_naive_forecast(short_series, horizon=3, season_length=7)
    assert np.all(forecast == 200)


def test_moving_average_forecast_is_flat_mean(weekly_series):
    horizon = 4
    forecast = moving_average_forecast(weekly_series, horizon, window=7)
    expected_mean = weekly_series.iloc[-7:].mean()
    assert len(forecast) == horizon
    assert np.all(forecast == pytest.approx(expected_mean))


def test_moving_average_window_larger_than_history_uses_full_history():
    short_series = pd.Series([10, 20, 30], index=pd.date_range("2015-01-01", periods=3))
    forecast = moving_average_forecast(short_series, horizon=2, window=100)
    assert np.all(forecast == pytest.approx(20.0))
