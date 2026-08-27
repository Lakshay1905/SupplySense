"""Shared pytest fixtures for SupplySense tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable when running `pytest` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """A small, realistic-looking raw train.csv-shaped DataFrame with
    deliberate data-quality issues (duplicates, nulls, negative value,
    closed store with sales) for validation/cleaning tests."""
    dates = pd.date_range("2015-01-01", periods=10, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "Store": 1,
            "DayOfWeek": d.dayofweek + 1,
            "Date": d.strftime("%Y-%m-%d"),
            "Sales": 1000 + i * 10,
            "Customers": 100 + i,
            "Open": 1,
            "Promo": 1 if i % 2 == 0 else 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        })
    # closed store with stray nonzero sales
    rows.append({
        "Store": 1, "DayOfWeek": 3, "Date": "2015-01-11", "Sales": 50,
        "Customers": 0, "Open": 0, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0,
    })
    # negative sales (impossible)
    rows.append({
        "Store": 2, "DayOfWeek": 4, "Date": "2015-01-01", "Sales": -20,
        "Customers": 5, "Open": 1, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0,
    })
    # exact duplicate of first row (same Store+Date)
    rows.append(rows[0].copy())
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def sample_store_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Store": 1, "StoreType": "a", "Assortment": "a", "CompetitionDistance": 500,
         "CompetitionOpenSinceMonth": 3, "CompetitionOpenSinceYear": 2010,
         "Promo2": 0, "Promo2SinceWeek": np.nan, "Promo2SinceYear": np.nan, "PromoInterval": np.nan},
        {"Store": 2, "StoreType": "b", "Assortment": "c", "CompetitionDistance": np.nan,
         "CompetitionOpenSinceMonth": np.nan, "CompetitionOpenSinceYear": np.nan,
         "Promo2": 1, "Promo2SinceWeek": 10, "Promo2SinceYear": 2014, "PromoInterval": "Jan,Apr,Jul,Oct"},
    ])


@pytest.fixture
def sample_store_states_df() -> pd.DataFrame:
    return pd.DataFrame([{"Store": 1, "State": "BE"}, {"Store": 2, "State": "NW"}])


@pytest.fixture
def sample_fact_sales_df() -> pd.DataFrame:
    """A clean, longer synthetic fact_sales-shaped DataFrame for feature
    engineering / segmentation tests (needs enough history for lags)."""
    n_days = 60
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(42)

    rows = []
    for store_id, pattern in [(1, "stable"), (2, "intermittent"), (3, "volatile")]:
        for i, d in enumerate(dates):
            if pattern == "stable":
                sales = 1000 + rng.normal(0, 20)
            elif pattern == "intermittent":
                sales = 0 if rng.random() < 0.5 else rng.uniform(50, 150)
            else:  # volatile
                sales = max(0, rng.normal(1000, 900))
            rows.append({
                "date_id": d, "store_id": store_id, "sales": max(0, sales),
                "customers": 100, "is_open": True, "is_promo": bool(i % 3 == 0),
                "state_holiday": "0", "school_holiday": False,
                "sales_per_customer": max(0, sales) / 100,
            })
    return pd.DataFrame(rows)
