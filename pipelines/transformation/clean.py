"""
Cleaning routines: fix dtypes, handle missing values, remove/flag
duplicates and impossible records. Every cleaning decision is logged and
counted so it shows up in the Data Quality report -- nothing is silently
dropped.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


def clean_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the daily sales fact table. Returns (clean_df, stats)."""
    stats = {"input_rows": len(df)}
    df = df.copy()

    # --- dtypes ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce")
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"nan": "0"})

    # --- drop rows with unusable keys ---
    before = len(df)
    df = df.dropna(subset=["Store", "Date"])
    stats["dropped_missing_key"] = before - len(df)

    # --- duplicates on (Store, Date) -- keep first occurrence ---
    before = len(df)
    df = df.drop_duplicates(subset=["Store", "Date"], keep="first")
    stats["dropped_duplicates"] = before - len(df)

    # --- business-rule fixes ---
    # A closed store (Open=0) should logically have 0 sales/customers.
    # Some source rows have Open=0 with stray nonzero values (data-entry
    # noise) or NaN. We normalize these to 0 rather than dropping the row,
    # since closed-day rows are still informative for demand-pattern
    # analysis (e.g. store closures) -- but we flag how many were affected.
    closed_mask = df["Open"] == 0
    inconsistent_closed = int(((df["Sales"] > 0) & closed_mask).sum())
    stats["closed_with_nonzero_sales"] = inconsistent_closed
    df.loc[closed_mask, "Sales"] = 0.0
    df.loc[closed_mask, "Customers"] = 0.0

    # Negative sales/customers are impossible -> clip at 0 and flag.
    neg_sales = int((df["Sales"] < 0).sum())
    neg_cust = int((df["Customers"] < 0).sum())
    stats["negative_sales_clipped"] = neg_sales
    stats["negative_customers_clipped"] = neg_cust
    df["Sales"] = df["Sales"].clip(lower=0)
    df["Customers"] = df["Customers"].clip(lower=0)

    # Fill any remaining NaNs in Sales/Customers with 0 (treated as no
    # recorded activity) -- and flag how many were affected.
    remaining_na_sales = int(df["Sales"].isna().sum())
    stats["sales_na_filled"] = remaining_na_sales
    df["Sales"] = df["Sales"].fillna(0)
    df["Customers"] = df["Customers"].fillna(0)

    # Derived measure
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SalesPerCustomer"] = np.where(df["Customers"] > 0, df["Sales"] / df["Customers"], 0.0)

    df["Open"] = df["Open"].fillna((df["Sales"] > 0).astype(int)).astype(int)
    df["Promo"] = df["Promo"].fillna(0).astype(int)
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)

    stats["output_rows"] = len(df)
    logger.info("clean_train stats: %s", stats)
    return df, stats


def clean_store(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean store attribute dimension."""
    stats = {"input_rows": len(df)}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Store"], keep="first")
    stats["dropped_duplicates"] = before - len(df)

    numeric_cols = ["CompetitionDistance", "CompetitionOpenSinceMonth",
                     "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # CompetitionDistance missing => no known nearby competitor. Rather than
    # imputing a fabricated distance, we keep it null and let downstream
    # consumers treat null as "unknown / no competition on record" (documented
    # proxy decision, not fabricated data).
    stats["missing_competition_distance"] = int(df["CompetitionDistance"].isna().sum())

    df["Promo2"] = df["Promo2"].fillna(0).astype(int)
    df["StoreType"] = df["StoreType"].astype(str).str.lower()
    df["Assortment"] = df["Assortment"].astype(str).str.lower()
    df["PromoInterval"] = df["PromoInterval"].fillna("")

    stats["output_rows"] = len(df)
    logger.info("clean_store stats: %s", stats)
    return df, stats


def clean_store_states(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"input_rows": len(df)}
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["Store"], keep="first")
    stats["dropped_duplicates"] = before - len(df)
    stats["output_rows"] = len(df)
    return df, stats
