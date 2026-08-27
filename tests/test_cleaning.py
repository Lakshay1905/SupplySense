"""Tests for pipelines.transformation.clean."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.transformation.clean import clean_train, clean_store, clean_store_states


def test_clean_train_removes_exact_duplicates(sample_train_df):
    clean, stats = clean_train(sample_train_df)
    assert stats["dropped_duplicates"] == 1
    # no duplicate (Store, Date) pairs remain
    assert not clean.duplicated(subset=["Store", "Date"]).any()


def test_clean_train_clips_negative_sales(sample_train_df):
    clean, stats = clean_train(sample_train_df)
    assert stats["negative_sales_clipped"] == 1
    assert (clean["Sales"] >= 0).all()


def test_clean_train_zeroes_closed_stores_with_stray_sales(sample_train_df):
    clean, stats = clean_train(sample_train_df)
    assert stats["closed_with_nonzero_sales"] == 1
    closed_rows = clean[clean["Open"] == 0]
    assert (closed_rows["Sales"] == 0).all()


def test_clean_train_computes_sales_per_customer(sample_train_df):
    clean, _ = clean_train(sample_train_df)
    row = clean[(clean["Store"] == 1) & (clean["Open"] == 1)].iloc[0]
    if row["Customers"] > 0:
        expected = row["Sales"] / row["Customers"]
        assert abs(row["SalesPerCustomer"] - expected) < 1e-6


def test_clean_train_handles_zero_customers_without_division_error(sample_train_df):
    clean, _ = clean_train(sample_train_df)
    # closed store row has 0 customers -> SalesPerCustomer must be 0, not NaN/inf
    closed_row = clean[clean["Open"] == 0].iloc[0]
    assert closed_row["SalesPerCustomer"] == 0
    assert np.isfinite(closed_row["SalesPerCustomer"])


def test_clean_train_output_rows_no_data_loss_beyond_documented(sample_train_df):
    clean, stats = clean_train(sample_train_df)
    assert stats["output_rows"] == stats["input_rows"] - stats["dropped_duplicates"] - stats["dropped_missing_key"]


def test_clean_store_deduplicates(sample_store_df):
    dup_df = pd.concat([sample_store_df, sample_store_df.iloc[[0]]], ignore_index=True)
    clean, stats = clean_store(dup_df)
    assert stats["dropped_duplicates"] == 1
    assert clean["Store"].is_unique


def test_clean_store_lowercases_categoricals(sample_store_df):
    df = sample_store_df.copy()
    df["StoreType"] = df["StoreType"].str.upper()
    clean, _ = clean_store(df)
    assert (clean["StoreType"].str.islower()).all()


def test_clean_store_preserves_null_competition_distance_as_unknown(sample_store_df):
    clean, stats = clean_store(sample_store_df)
    assert stats["missing_competition_distance"] == 1
    # Store 2 had NaN competition distance -- must remain NaN (not fabricated)
    assert pd.isna(clean.loc[clean["Store"] == 2, "CompetitionDistance"].iloc[0])


def test_clean_store_states_deduplicates(sample_store_states_df):
    dup_df = pd.concat([sample_store_states_df, sample_store_states_df.iloc[[0]]], ignore_index=True)
    clean, stats = clean_store_states(dup_df)
    assert stats["dropped_duplicates"] == 1
