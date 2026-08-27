"""Tests for analytics.features.feature_engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.features.feature_engineering import (
    add_calendar_features, add_lag_and_rolling_features,
    classify_demand_segment, compute_demand_segments, build_feature_table,
)


def test_add_calendar_features_adds_expected_columns(sample_fact_sales_df):
    df = add_calendar_features(sample_fact_sales_df)
    for col in ["day_of_week", "is_weekend", "week_of_year", "month"]:
        assert col in df.columns
    assert df["day_of_week"].between(1, 7).all()


def test_lag_features_have_no_leakage(sample_fact_sales_df):
    """lag_1 for a given day must equal the previous day's sales for that
    store, and must be NaN on the very first observed day (no leakage)."""
    df = add_lag_and_rolling_features(sample_fact_sales_df)
    store1 = df[df["store_id"] == 1].sort_values("date_id").reset_index(drop=True)
    assert pd.isna(store1.loc[0, "lag_1"])
    # lag_1 on day 2 should equal day 1's actual sales
    assert abs(store1.loc[1, "lag_1"] - store1.loc[0, "sales"]) < 1e-9


def test_rolling_mean_excludes_current_day(sample_fact_sales_df):
    """Rolling mean at row t must be computed only from rows before t."""
    df = add_lag_and_rolling_features(sample_fact_sales_df)
    store1 = df[df["store_id"] == 1].sort_values("date_id").reset_index(drop=True)
    # Manually compute rolling_mean_7 at index 10 using data[3:10] (shift by 1, window 7)
    manual = store1.loc[3:9, "sales"].mean()
    computed = store1.loc[10, "rolling_mean_7"]
    assert abs(manual - computed) < 1e-6


def test_classify_demand_segment_intermittent():
    s = pd.Series([0] * 20 + [100] * 10)  # 66% zeros
    assert classify_demand_segment(s) == "intermittent"


def test_classify_demand_segment_volatile():
    rng = np.random.default_rng(0)
    s = pd.Series(np.abs(rng.normal(1000, 900, size=40)))
    result = classify_demand_segment(s)
    assert result in ("volatile", "stable", "seasonal")  # high variance data should usually be volatile
    cv = s.std() / s.mean()
    if cv > 0.5:
        assert result == "volatile"


def test_classify_demand_segment_stable():
    rng = np.random.default_rng(1)
    s = pd.Series(1000 + rng.normal(0, 10, size=40))  # tiny noise
    assert classify_demand_segment(s) == "stable"


def test_classify_demand_segment_insufficient_data():
    s = pd.Series([100, 200, 300])  # too few points
    assert classify_demand_segment(s) == "insufficient_data"


def test_compute_demand_segments_returns_one_row_per_store(sample_fact_sales_df):
    segments = compute_demand_segments(sample_fact_sales_df)
    assert set(segments["store_id"]) == set(sample_fact_sales_df["store_id"].unique())
    assert len(segments) == sample_fact_sales_df["store_id"].nunique()


def test_build_feature_table_no_leakage_and_correct_shape(sample_fact_sales_df):
    dim_store = pd.DataFrame({
        "store_id": [1, 2, 3],
        "competition_open_since_year": [2010, np.nan, 2012],
        "competition_open_since_month": [5, np.nan, 1],
    })
    features = build_feature_table(sample_fact_sales_df, dim_store)
    assert len(features) == len(sample_fact_sales_df)
    assert "demand_segment" in features.columns
    assert features["demand_segment"].notna().all()
    # first row per store must have NaN lag_1 (no leakage into first obs)
    first_rows = features.sort_values("date_id").groupby("store_id").head(1)
    assert first_rows["lag_1"].isna().all()
