"""
Time-series backtesting framework.

Uses rolling-origin ("walk-forward") cross-validation, NOT random
train/test splits, which would leak future information into training and
produce misleadingly good validation scores for time-dependent data.

Each fold:
    train:  [start ... cutoff]
    test:   (cutoff ... cutoff + horizon]

Successive folds move the cutoff forward, so every fold's test window is
strictly after its train window, mimicking how forecasts are actually
generated in production (using only data available at forecast time).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestFold:
    fold: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def generate_rolling_folds(dates: pd.Series, horizon: int, n_folds: int,
                            step: int | None = None) -> list[BacktestFold]:
    """Generate `n_folds` rolling-origin folds ending at the most recent
    data and stepping backward by `step` days (default = horizon, i.e.
    non-overlapping test windows)."""
    step = step or horizon
    max_date = pd.Timestamp(dates.max())
    folds = []
    for i in range(n_folds):
        test_end = max_date - pd.Timedelta(days=step * i)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        folds.append(BacktestFold(
            fold=n_folds - i,  # fold 1 = earliest, fold n = most recent
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=test_end,
        ))
    return list(reversed(folds))
