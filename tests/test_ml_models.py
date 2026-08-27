"""Tests for forecasting.ml.ml_models (unit-level, no DB dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.ml.ml_models import prepare_ml_features, train_xgboost, FEATURE_COLUMNS


@pytest.fixture
def synthetic_features_and_store():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    store_ids = rng.choice([1, 2], size=n)
    df = pd.DataFrame({
        "date_id": dates, "store_id": store_ids,
        "sales": rng.uniform(100, 1000, size=n),
        "lag_1": rng.uniform(100, 1000, size=n),
        "lag_7": rng.uniform(100, 1000, size=n),
        "lag_14": rng.uniform(100, 1000, size=n),
        "lag_28": rng.uniform(100, 1000, size=n),
        "rolling_mean_7": rng.uniform(100, 1000, size=n),
        "rolling_mean_14": rng.uniform(100, 1000, size=n),
        "rolling_mean_28": rng.uniform(100, 1000, size=n),
        "rolling_std_7": rng.uniform(0, 100, size=n),
        "rolling_std_28": rng.uniform(0, 100, size=n),
        "day_of_week": dates.dayofweek + 1,
        "is_weekend": (dates.dayofweek >= 5),
        "week_of_year": dates.isocalendar().week.values,
        "month": dates.month,
        "is_promo": rng.choice([True, False], size=n),
        "is_school_holiday": rng.choice([True, False], size=n),
        "is_state_holiday": rng.choice([True, False], size=n),
        "days_since_competition": rng.uniform(0, 1000, size=n),
        "demand_segment": "stable",
    })
    dim_store = pd.DataFrame({"store_id": [1, 2], "store_type": ["a", "b"], "assortment": ["a", "c"]})
    return df, dim_store


def test_prepare_ml_features_encodes_categoricals(synthetic_features_and_store):
    features, dim_store = synthetic_features_and_store
    prepared = prepare_ml_features(features, dim_store)
    assert "store_type_code" in prepared.columns
    assert "assortment_code" in prepared.columns
    assert prepared["store_type_code"].notna().all()
    for col in ["is_weekend", "is_promo", "is_school_holiday", "is_state_holiday"]:
        assert prepared[col].dtype in (int, np.int64)


def test_train_xgboost_predictions_are_nonnegative(synthetic_features_and_store):
    features, dim_store = synthetic_features_and_store
    prepared = prepare_ml_features(features, dim_store)
    model = train_xgboost(prepared)
    preds = model.predict(prepared)
    assert (preds >= 0).all()
    assert len(preds) == len(prepared)


def test_train_xgboost_handles_missing_features_via_median_fill(synthetic_features_and_store):
    features, dim_store = synthetic_features_and_store
    prepared = prepare_ml_features(features, dim_store)
    model = train_xgboost(prepared)

    test_row = prepared.iloc[[0]].copy()
    test_row["lag_1"] = np.nan
    preds = model.predict(test_row)
    assert np.isfinite(preds[0])
