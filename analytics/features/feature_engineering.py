"""
Feature engineering for the forecasting engine (Phase 2 consumes this
output). Computed here, in Phase 1, and materialized to
`fact_sales_features` so forecasting code never has to recompute lags from
scratch and every model (baseline, statistical, ML) sees the same features.

All features are computed causally (only using information available at
or before the row's date) to avoid leakage into backtests.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date_id") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek + 1
    df["is_weekend"] = df["day_of_week"].isin([6, 7])
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, group_col: str = "store_id",
                                  date_col: str = "date_id",
                                  target_col: str = "sales") -> pd.DataFrame:
    """Add lag + rolling mean/std features per store, sorted by date.

    Rolling/lag windows are computed on `shift(1)` so that the feature for
    a given day never includes that day's own sales (no leakage).
    """
    df = df.sort_values([group_col, date_col]).copy()
    grouped = df.groupby(group_col, sort=False)[target_col]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)  # exclude current day from rolling stats
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window, min_periods=max(2, window // 2))
            .mean().reset_index(level=0, drop=True)
        )
    for window in [7, 28]:
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window, min_periods=max(2, window // 2))
            .std().reset_index(level=0, drop=True)
        )
    return df


def add_days_since_competition(df: pd.DataFrame, dim_store: pd.DataFrame) -> pd.DataFrame:
    """Number of days since a competitor opened nearby (NaN if unknown)."""
    df = df.copy()
    comp = dim_store[["store_id", "competition_open_since_year", "competition_open_since_month"]].copy()
    comp["competition_open_date"] = pd.to_datetime(
        dict(year=comp["competition_open_since_year"].fillna(1900).astype(int),
             month=comp["competition_open_since_month"].fillna(1).astype(int),
             day=1),
        errors="coerce",
    )
    comp.loc[comp["competition_open_since_year"].isna(), "competition_open_date"] = pd.NaT
    df = df.merge(comp[["store_id", "competition_open_date"]], on="store_id", how="left")
    df["days_since_competition"] = (pd.to_datetime(df["date_id"]) - df["competition_open_date"]).dt.days
    df.loc[df["days_since_competition"] < 0, "days_since_competition"] = np.nan
    df = df.drop(columns=["competition_open_date"])
    return df


def classify_demand_segment(series: pd.Series) -> str:
    """Classify a store's demand history into stable/seasonal/volatile/intermittent.

    Heuristic (standard supply-chain segmentation, ADI/CV^2-style):
      - intermittent: >30% of active days have zero demand
      - volatile: coefficient of variation (std/mean) > 0.5
      - seasonal: strong weekly autocorrelation (lag-7) relative to noise
      - stable: everything else
    """
    s = series.dropna()
    if len(s) < 14 or s.mean() == 0:
        return "insufficient_data"

    zero_share = (s == 0).mean()
    if zero_share > 0.30:
        return "intermittent"

    cv = s.std() / s.mean() if s.mean() else np.inf
    if cv > 0.5:
        return "volatile"

    # Weekly seasonality strength via autocorrelation at lag 7
    if len(s) > 21:
        s_shift = s.shift(7)
        valid = s.notna() & s_shift.notna()
        if valid.sum() > 10 and s[valid].std() > 0 and s_shift[valid].std() > 0:
            corr = np.corrcoef(s[valid], s_shift[valid])[0, 1]
        else:
            corr = 0.0
        if corr > 0.4:
            return "seasonal"

    return "stable"


def compute_demand_segments(fact_sales: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of store_id -> demand_segment using only open-day sales."""
    active = fact_sales[fact_sales["is_open"]]
    segments = (
        active.groupby("store_id")["sales"]
        .apply(classify_demand_segment)
        .reset_index()
        .rename(columns={"sales": "demand_segment"})
    )
    logger.info("Demand segment distribution: %s",
                segments["demand_segment"].value_counts().to_dict())
    return segments


def build_feature_table(fact_sales: pd.DataFrame, dim_store: pd.DataFrame) -> pd.DataFrame:
    """Full Phase-1 feature engineering pipeline producing fact_sales_features."""
    df = fact_sales.copy()
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_days_since_competition(df, dim_store)

    segments = compute_demand_segments(fact_sales)
    df = df.merge(segments, on="store_id", how="left")

    df = df.rename(columns={
        "is_promo": "is_promo",
        "school_holiday": "is_school_holiday",
    })
    df["is_state_holiday"] = df["state_holiday"] != "0"

    feature_cols = [
        "date_id", "store_id", "sales",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_28",
        "day_of_week", "is_weekend", "week_of_year", "month",
        "is_promo", "is_school_holiday", "is_state_holiday",
        "days_since_competition", "demand_segment",
    ]
    out = df[feature_cols].copy()
    logger.info("Built feature table with %d rows, %d columns", *out.shape)
    return out
