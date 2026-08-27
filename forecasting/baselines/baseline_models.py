"""
Baseline forecasters. Every model in the benchmarking framework must beat
these to be worth using -- they are the "can a trivial rule do just as
well" sanity check.

All baseline functions take a historical series (indexed by date, one
store) and a horizon, and return a forecast array of that length. They
are intentionally simple and dependency-free (no statsmodels/sklearn).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for the whole horizon."""
    last_value = history.iloc[-1]
    return np.full(horizon, last_value, dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Repeat the value from `season_length` periods ago (here: same
    weekday last week), cycling through the last full season if the
    horizon exceeds it."""
    if len(history) < season_length:
        return naive_forecast(history, horizon)
    last_season = history.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon].astype(float)


def moving_average_forecast(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Flat forecast at the mean of the last `window` observations."""
    window = min(window, len(history))
    avg = history.iloc[-window:].mean()
    return np.full(horizon, avg, dtype=float)
