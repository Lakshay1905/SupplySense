"""
Transform cleaned source tables into the dimensional model
(dim_date, dim_region, dim_store, fact_sales) that gets loaded to
PostgreSQL.
"""
from __future__ import annotations

import pandas as pd

from data.schemas.raw_schema import GERMAN_STATE_NAMES
from config.logging_config import get_logger

logger = get_logger(__name__)


def build_dim_date(dates: pd.Series) -> pd.DataFrame:
    """Build a calendar dimension covering the min..max date range seen in data."""
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    dim = pd.DataFrame({"date_id": full_range})
    dim["year"] = dim["date_id"].dt.year
    dim["quarter"] = dim["date_id"].dt.quarter
    dim["month"] = dim["date_id"].dt.month
    dim["week_of_year"] = dim["date_id"].dt.isocalendar().week.astype(int)
    dim["day_of_month"] = dim["date_id"].dt.day
    dim["day_of_week"] = dim["date_id"].dt.dayofweek + 1  # 1=Mon .. 7=Sun
    dim["day_name"] = dim["date_id"].dt.day_name()
    dim["is_weekend"] = dim["day_of_week"].isin([6, 7])
    dim["month_name"] = dim["date_id"].dt.month_name()
    logger.info("Built dim_date with %d calendar days", len(dim))
    return dim


def build_dim_region(store_states: pd.DataFrame) -> pd.DataFrame:
    states = sorted(store_states["State"].dropna().unique().tolist())
    dim = pd.DataFrame({"state_code": states})
    dim["state_name"] = dim["state_code"].map(GERMAN_STATE_NAMES).fillna(dim["state_code"])
    dim.insert(0, "region_id", range(1, len(dim) + 1))
    logger.info("Built dim_region with %d regions (German federal states)", len(dim))
    return dim


def build_dim_store(store: pd.DataFrame, store_states: pd.DataFrame,
                     dim_region: pd.DataFrame) -> pd.DataFrame:
    df = store.merge(store_states, on="Store", how="left")
    df = df.merge(dim_region[["region_id", "state_code"]], left_on="State",
                  right_on="state_code", how="left")

    dim = pd.DataFrame({
        "store_id": df["Store"],
        "store_type": df["StoreType"],
        "assortment": df["Assortment"],
        "competition_distance_m": df["CompetitionDistance"],
        "competition_open_since_month": df["CompetitionOpenSinceMonth"],
        "competition_open_since_year": df["CompetitionOpenSinceYear"],
        "promo2_active": df["Promo2"].astype(bool),
        "promo2_since_week": df["Promo2SinceWeek"],
        "promo2_since_year": df["Promo2SinceYear"],
        "promo_interval": df["PromoInterval"],
        "region_id": df["region_id"],
    })
    n_missing_region = int(dim["region_id"].isna().sum())
    if n_missing_region:
        logger.warning("%d stores have no mapped region (state) and will have NULL region_id",
                        n_missing_region)
    logger.info("Built dim_store with %d stores", len(dim))
    return dim


def build_fact_sales(train: pd.DataFrame) -> pd.DataFrame:
    fact = pd.DataFrame({
        "date_id": train["Date"],
        "store_id": train["Store"].astype(int),
        "sales": train["Sales"].astype(float),
        "customers": train["Customers"].astype(int),
        "is_open": train["Open"].astype(bool),
        "is_promo": train["Promo"].astype(bool),
        "state_holiday": train["StateHoliday"],
        "school_holiday": train["SchoolHoliday"].astype(bool),
        "sales_per_customer": train["SalesPerCustomer"].astype(float),
    })
    logger.info("Built fact_sales with %d rows", len(fact))
    return fact
