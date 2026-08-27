"""Tests for forecasting.evaluation.model_selection."""
from __future__ import annotations

import pandas as pd
import pytest

from forecasting.evaluation.model_selection import (
    select_best_model_per_store, summarize_model_performance,
)


def _sample_evaluations() -> pd.DataFrame:
    rows = []
    # store 1: xgboost clearly best
    for fold, wmape_xgb, wmape_naive in [(1, 8.0, 20.0), (2, 9.0, 22.0), (3, 7.5, 19.0)]:
        rows.append({"store_id": 1, "model_name": "xgboost", "fold": fold,
                     "mae": 100, "rmse": 150, "mape": wmape_xgb, "wmape": wmape_xgb, "bias": 0.1})
        rows.append({"store_id": 1, "model_name": "naive", "fold": fold,
                     "mae": 300, "rmse": 400, "mape": wmape_naive, "wmape": wmape_naive, "bias": 2.0})
    # store 2: naive happens to win (edge case, e.g. very stable series)
    for fold, wmape_xgb, wmape_naive in [(1, 15.0, 10.0), (2, 14.0, 9.0), (3, 16.0, 11.0)]:
        rows.append({"store_id": 2, "model_name": "xgboost", "fold": fold,
                     "mae": 100, "rmse": 150, "mape": wmape_xgb, "wmape": wmape_xgb, "bias": 0.1})
        rows.append({"store_id": 2, "model_name": "naive", "fold": fold,
                     "mae": 80, "rmse": 100, "mape": wmape_naive, "wmape": wmape_naive, "bias": 0.5})
    return pd.DataFrame(rows)


def test_select_best_model_per_store_picks_lowest_avg_wmape():
    evaluations = _sample_evaluations()
    best = select_best_model_per_store(evaluations)
    assert len(best) == 2  # one row per store
    store1_pick = best[best["store_id"] == 1]["model_name"].iloc[0]
    store2_pick = best[best["store_id"] == 2]["model_name"].iloc[0]
    assert store1_pick == "xgboost"
    assert store2_pick == "naive"


def test_select_best_model_per_store_averages_across_folds():
    evaluations = _sample_evaluations()
    best = select_best_model_per_store(evaluations)
    store1_row = best[best["store_id"] == 1].iloc[0]
    assert store1_row["n_folds"] == 3
    assert store1_row["avg_wmape"] == pytest.approx((8.0 + 9.0 + 7.5) / 3)


def test_summarize_model_performance_sorted_by_wmape_ascending():
    evaluations = _sample_evaluations()
    summary = summarize_model_performance(evaluations)
    assert summary.iloc[0]["avg_wmape"] <= summary.iloc[-1]["avg_wmape"]
    assert set(summary["model_name"]) == {"xgboost", "naive"}


def test_summarize_model_performance_counts_evaluations():
    evaluations = _sample_evaluations()
    summary = summarize_model_performance(evaluations)
    xgb_row = summary[summary["model_name"] == "xgboost"].iloc[0]
    assert xgb_row["n_evaluations"] == 6  # 2 stores x 3 folds
