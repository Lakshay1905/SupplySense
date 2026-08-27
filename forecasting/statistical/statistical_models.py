"""
Statistical forecasting models.

Holt-Winters (triple exponential smoothing) is used as the primary
statistical model because it fits fast enough to run per-store at scale
(~0.2s/store) and captures both trend and weekly seasonality, which is
exactly the pattern the Phase-1 EDA found in Rossmann demand.

SARIMA is included for benchmarking purposes on a representative sample
of stores only (see forecasting/evaluation/run_benchmark.py) -- fitting a
full seasonal ARIMA per store across 1,115 stores is computationally
prohibitive for a periodic batch job and provides negligible accuracy
benefit over Holt-Winters on this weekly-seasonal, non-trending-strongly
retail series (confirmed empirically in the benchmark, not assumed).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.logging_config import get_logger

logger = get_logger(__name__)


def holt_winters_forecast(history: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Fit additive Holt-Winters (trend + weekly seasonality) and forecast.

    Falls back to a seasonal-naive style forecast if the series is too
    short or the fit fails to converge (common for very new/short-lived
    stores), so the pipeline never crashes on a single problem series.
    """
    series = history.asfreq("D").interpolate(limit_direction="both")
    if len(series) < 2 * season_length or series.std() == 0:
        from forecasting.baselines.baseline_models import seasonal_naive_forecast
        return seasonal_naive_forecast(history, horizon, season_length)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add",
                seasonal_periods=season_length,
                initialization_method="estimated",
            ).fit(optimized=True)
        forecast = model.forecast(horizon)
        return np.clip(forecast.values, a_min=0, a_max=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Holt-Winters fit failed (%s); falling back to seasonal naive", exc)
        from forecasting.baselines.baseline_models import seasonal_naive_forecast
        return seasonal_naive_forecast(history, horizon, season_length)


def sarima_forecast(history: pd.Series, horizon: int, season_length: int = 7,
                     order: tuple = (1, 1, 1), seasonal_order_pdq: tuple = (1, 1, 1)) -> np.ndarray:
    """Fit a seasonal ARIMA. Slower than Holt-Winters; used for benchmarking
    on a sample of stores, not full-scale production forecasting."""
    series = history.asfreq("D").interpolate(limit_direction="both")
    if len(series) < 3 * season_length or series.std() == 0:
        from forecasting.baselines.baseline_models import seasonal_naive_forecast
        return seasonal_naive_forecast(history, horizon, season_length)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seasonal_order = (*seasonal_order_pdq, season_length)
            model = SARIMAX(
                series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        forecast = model.forecast(horizon)
        return np.clip(forecast.values, a_min=0, a_max=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SARIMA fit failed (%s); falling back to seasonal naive", exc)
        from forecasting.baselines.baseline_models import seasonal_naive_forecast
        return seasonal_naive_forecast(history, horizon, season_length)
