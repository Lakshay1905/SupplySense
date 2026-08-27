"""Tests for pipelines.transformation.transform."""
from __future__ import annotations

import pandas as pd

from pipelines.transformation.clean import clean_train, clean_store, clean_store_states
from pipelines.transformation.transform import (
    build_dim_date, build_dim_region, build_dim_store, build_fact_sales,
)


def test_build_dim_date_covers_full_range_with_no_gaps():
    dates = pd.Series(pd.to_datetime(["2015-01-01", "2015-01-10"]))
    dim = build_dim_date(dates)
    assert len(dim) == 10  # inclusive range Jan 1 - Jan 10
    assert dim["date_id"].is_monotonic_increasing
    assert dim["date_id"].nunique() == len(dim)


def test_build_dim_date_flags_weekends_correctly():
    dates = pd.Series(pd.to_datetime(["2015-01-01", "2015-01-04"]))  # Thu, Sun
    dim = build_dim_date(dates)
    sunday_row = dim[dim["date_id"] == pd.Timestamp("2015-01-04")].iloc[0]
    assert sunday_row["is_weekend"] == True  # noqa: E712
    assert sunday_row["day_of_week"] == 7


def test_build_dim_region_maps_known_state_names(sample_store_states_df):
    dim = build_dim_region(sample_store_states_df)
    assert set(dim["state_code"]) == {"BE", "NW"}
    be_row = dim[dim["state_code"] == "BE"].iloc[0]
    assert be_row["state_name"] == "Berlin"
    assert dim["region_id"].is_unique


def test_build_dim_store_joins_region_correctly(sample_store_df, sample_store_states_df):
    dim_region = build_dim_region(sample_store_states_df)
    dim_store = build_dim_store(sample_store_df, sample_store_states_df, dim_region)
    assert len(dim_store) == len(sample_store_df)
    assert dim_store["region_id"].notna().all()
    # store 1 is in Berlin (BE)
    berlin_id = dim_region[dim_region["state_code"] == "BE"]["region_id"].iloc[0]
    assert dim_store[dim_store["store_id"] == 1]["region_id"].iloc[0] == berlin_id


def test_build_fact_sales_row_count_matches_input(sample_train_df):
    clean, _ = clean_train(sample_train_df)
    fact = build_fact_sales(clean)
    assert len(fact) == len(clean)
    assert set(["date_id", "store_id", "sales", "customers", "is_open",
                "is_promo", "state_holiday", "school_holiday",
                "sales_per_customer"]).issubset(fact.columns)


def test_build_fact_sales_types(sample_train_df):
    clean, _ = clean_train(sample_train_df)
    fact = build_fact_sales(clean)
    assert fact["is_open"].dtype == bool
    assert fact["is_promo"].dtype == bool
    assert pd.api.types.is_integer_dtype(fact["store_id"])
