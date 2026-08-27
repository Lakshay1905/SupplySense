"""
Probabilistic forecasting.

Point forecasts alone don't tell an inventory planner what could
plausibly happen. We convert a point forecast into a P10/P50/P90 range
using the empirical distribution of that model's backtest residuals
(per store, or per demand segment when a store's own history is too
short) -- a standard, model-agnostic technique that works identically for
baselines, statistical models, and ML models.

    P50 = point forecast
    P10 = point forecast + (10th percentile of signed residuals)
    P90 = point forecast + (90th percentile of signed residuals)

Residuals are `actual - predicted` from backtesting, so this naturally
widens the interval for volatile/hard-to-forecast series and narrows it
for stable ones -- it is *learned* from the model's own track record, not
a fixed +/-X% band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


def compute_residual_quantiles(residuals: np.ndarray,
                                lower_q: float = 0.10,
                                upper_q: float = 0.90) -> tuple[float, float]:
    """Return (lower_offset, upper_offset) from an array of signed
    residuals (actual - predicted)."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 5:
        return 0.0, 0.0
    return float(np.quantile(residuals, lower_q)), float(np.quantile(residuals, upper_q))


def apply_probabilistic_bands(point_forecast: np.ndarray, lower_offset: float,
                               upper_offset: float) -> pd.DataFrame:
    """Turn a point-forecast array into a P10/P50/P90 DataFrame, clipped
    at zero (demand cannot be negative)."""
    p50 = np.clip(point_forecast, 0, None)
    p10 = np.clip(point_forecast + lower_offset, 0, None)
    p90 = np.clip(point_forecast + upper_offset, 0, None)
    # Guard against quantile crossing (can happen with small samples)
    p10, p90 = np.minimum(p10, p90), np.maximum(p10, p90)
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90})


def build_probabilistic_forecast(point_forecast: np.ndarray, backtest_residuals: np.ndarray,
                                  lower_q: float = 0.10, upper_q: float = 0.90) -> pd.DataFrame:
    lower_offset, upper_offset = compute_residual_quantiles(backtest_residuals, lower_q, upper_q)
    return apply_probabilistic_bands(point_forecast, lower_offset, upper_offset)
