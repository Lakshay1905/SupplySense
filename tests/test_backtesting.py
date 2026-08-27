"""Tests for forecasting.evaluation.backtesting."""
from __future__ import annotations

import pandas as pd

from forecasting.evaluation.backtesting import generate_rolling_folds


def test_generate_rolling_folds_count_and_order():
    dates = pd.Series(pd.date_range("2015-01-01", periods=200, freq="D"))
    folds = generate_rolling_folds(dates, horizon=42, n_folds=3)
    assert len(folds) == 3
    # folds returned in chronological order (earliest fold first)
    assert folds[0].test_end_date < folds[1].test_end_date < folds[2].test_end_date


def test_generate_rolling_folds_no_leakage_train_before_test():
    dates = pd.Series(pd.date_range("2015-01-01", periods=200, freq="D"))
    folds = generate_rolling_folds(dates, horizon=42, n_folds=3)
    for fold in folds:
        assert fold.train_end_date < fold.test_start_date
        assert fold.test_start_date <= fold.test_end_date


def test_generate_rolling_folds_test_window_length_matches_horizon():
    dates = pd.Series(pd.date_range("2015-01-01", periods=300, freq="D"))
    horizon = 30
    folds = generate_rolling_folds(dates, horizon=horizon, n_folds=2)
    for fold in folds:
        assert (fold.test_end_date - fold.test_start_date).days == horizon - 1


def test_generate_rolling_folds_last_fold_ends_at_max_date():
    dates = pd.Series(pd.date_range("2015-01-01", periods=150, freq="D"))
    folds = generate_rolling_folds(dates, horizon=42, n_folds=2)
    assert folds[-1].test_end_date == dates.max()


def test_generate_rolling_folds_non_overlapping_by_default():
    dates = pd.Series(pd.date_range("2015-01-01", periods=300, freq="D"))
    folds = generate_rolling_folds(dates, horizon=42, n_folds=3)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end_date < later.test_start_date
