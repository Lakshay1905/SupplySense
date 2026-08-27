"""
Exploratory data analysis and demand-pattern analysis.

Produces a JSON summary (for programmatic use by the Streamlit app / AI
copilot later) and a small set of PNG charts (for the README / portfolio
write-up). Every number here is computed directly from the loaded data --
nothing is hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import REPORTS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)

EDA_DIR = REPORTS_DIR / "eda"


def _save_fig(fig, name: str) -> str:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_eda(fact_sales: pd.DataFrame, dim_store: pd.DataFrame,
            dim_region: pd.DataFrame, features: pd.DataFrame) -> dict:
    summary: dict = {}

    active = fact_sales[fact_sales["is_open"]]

    # --- overview ---
    summary["overview"] = {
        "n_stores": int(fact_sales["store_id"].nunique()),
        "date_min": str(fact_sales["date_id"].min().date()),
        "date_max": str(fact_sales["date_id"].max().date()),
        "n_store_days": int(len(fact_sales)),
        "n_open_store_days": int(len(active)),
        "pct_closed": float(1 - len(active) / len(fact_sales)),
        "total_sales": float(fact_sales["sales"].sum()),
        "avg_daily_sales_per_open_store": float(active["sales"].mean()),
        "median_daily_sales_per_open_store": float(active["sales"].median()),
    }

    # --- demand distribution ---
    summary["sales_distribution"] = {
        "mean": float(active["sales"].mean()),
        "std": float(active["sales"].std()),
        "p10": float(active["sales"].quantile(0.10)),
        "p50": float(active["sales"].quantile(0.50)),
        "p90": float(active["sales"].quantile(0.90)),
        "skew": float(active["sales"].skew()),
    }

    # --- day-of-week seasonality ---
    dow = active.copy()
    dow["day_of_week"] = pd.to_datetime(dow["date_id"]).dt.dayofweek + 1
    dow_avg = dow.groupby("day_of_week")["sales"].mean()
    summary["day_of_week_avg_sales"] = {int(k): float(v) for k, v in dow_avg.items()}

    # --- monthly seasonality ---
    monthly = active.copy()
    monthly["month"] = pd.to_datetime(monthly["date_id"]).dt.month
    monthly_avg = monthly.groupby("month")["sales"].mean()
    summary["monthly_avg_sales"] = {int(k): float(v) for k, v in monthly_avg.items()}

    # --- promo effect (correlational, not causal) ---
    promo_avg = active.groupby("is_promo")["sales"].mean()
    summary["promo_effect"] = {
        "avg_sales_no_promo": float(promo_avg.get(False, np.nan)),
        "avg_sales_with_promo": float(promo_avg.get(True, np.nan)),
        "note": "Descriptive/correlational comparison only; not a causal uplift estimate.",
    }

    # --- store type (category proxy) comparison ---
    merged = active.merge(dim_store[["store_id", "store_type", "assortment", "region_id"]],
                           on="store_id", how="left")
    by_type = merged.groupby("store_type")["sales"].agg(["mean", "median", "count"])
    summary["sales_by_store_type"] = by_type.round(2).to_dict(orient="index")

    # --- region comparison ---
    merged_region = merged.merge(dim_region, on="region_id", how="left")
    by_region = merged_region.groupby("state_name")["sales"].mean().sort_values(ascending=False)
    summary["avg_sales_by_region"] = {k: float(v) for k, v in by_region.items()}

    # --- demand segmentation distribution ---
    seg_counts = features.drop_duplicates("store_id")["demand_segment"].value_counts()
    summary["demand_segment_distribution"] = {k: int(v) for k, v in seg_counts.items()}

    # --- charts ---
    charts = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    dow_avg.plot(kind="bar", ax=ax, color="#3b6fa0")
    ax.set_title("Average Daily Sales by Day of Week (open stores)")
    ax.set_xlabel("Day of week (1=Mon)")
    ax.set_ylabel("Avg sales")
    charts["day_of_week"] = _save_fig(fig, "sales_by_day_of_week.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    monthly_avg.plot(kind="line", marker="o", ax=ax, color="#3b6fa0")
    ax.set_title("Average Daily Sales by Month (seasonality)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg sales")
    charts["monthly_seasonality"] = _save_fig(fig, "sales_by_month.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sample_store = int(active["store_id"].iloc[0])
    ts = active[active["store_id"] == sample_store].sort_values("date_id")
    ax.plot(ts["date_id"], ts["sales"], linewidth=0.8, color="#3b6fa0")
    ax.set_title(f"Daily Sales History -- Store {sample_store}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    charts["sample_store_timeseries"] = _save_fig(fig, "sample_store_timeseries.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    seg_counts.plot(kind="bar", ax=ax, color="#a05a3b")
    ax.set_title("Store Count by Demand Segment")
    ax.set_ylabel("Number of stores")
    charts["demand_segments"] = _save_fig(fig, "demand_segments.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    active["sales"].clip(upper=active["sales"].quantile(0.99)).hist(bins=60, ax=ax, color="#3b6fa0")
    ax.set_title("Distribution of Daily Sales (open stores, 99th pct clipped)")
    ax.set_xlabel("Sales")
    charts["sales_histogram"] = _save_fig(fig, "sales_histogram.png")

    summary["charts"] = charts

    out_path = EDA_DIR / "eda_summary.json"
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("EDA summary written to %s", out_path)

    return summary
