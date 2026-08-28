"""
Recommendation engine: the glue between stored data (forecasts, current
inventory, supplier lead times) and the optimization engine. This is what
the Streamlit "Inventory Recommendations" tab and the AI copilot will
call in Phase 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BUSINESS_DEFAULTS
from config.logging_config import get_logger
from database.connection import read_sql
from optimization.cost_model import build_store_cost_profile, StoreCostProfile
from optimization.inventory_optimizer import (
    OptimizationConstraints, OptimizationResult, optimize_order_quantity,
)

logger = get_logger(__name__)


def get_store_cost_profile(store_id: int) -> StoreCostProfile:
    row = read_sql(
        "SELECT AVG(NULLIF(sales_per_customer,0)) AS avg_spc FROM fact_sales "
        "WHERE store_id = :sid AND is_open = true",
        {"sid": store_id},
    ).iloc[0]
    avg_spc = row["avg_spc"] if pd.notna(row["avg_spc"]) else 10.0
    return build_store_cost_profile(store_id, avg_spc)


def get_decision_horizon_forecast(store_id: int, horizon_days: int) -> pd.DataFrame:
    """Pull the first `horizon_days` of this store's stored forecast
    (lead time + review period), converted to unit-equivalents."""
    df = read_sql(
        "SELECT target_date, p10, p50, p90 FROM forecasts "
        "WHERE store_id = :sid ORDER BY target_date LIMIT :h",
        {"sid": store_id, "h": horizon_days},
    )
    if len(df) < horizon_days:
        logger.warning("Store %d has only %d forecast days available (%d requested)",
                        store_id, len(df), horizon_days)
    return df


def get_current_inventory(store_id: int) -> tuple[float, float]:
    df = read_sql(
        "SELECT on_hand_units, incoming_units FROM inventory_snapshot "
        "WHERE store_id = :sid ORDER BY date_id DESC LIMIT 1",
        {"sid": store_id},
    )
    if df.empty:
        logger.warning("No inventory snapshot for store %d; assuming 0 on-hand", store_id)
        return 0.0, 0.0
    return float(df["on_hand_units"].iloc[0]), float(df["incoming_units"].iloc[0])


def get_supplier_lead_time(store_id: int) -> float:
    df = read_sql("""
        SELECT ds.lead_time_days FROM dim_supplier ds
        WHERE ds.supplier_name = :name
    """, {"name": f"Supplier for Store {store_id}"})
    if df.empty:
        return BUSINESS_DEFAULTS["default_lead_time_days"]
    return float(df["lead_time_days"].iloc[0])


def get_recent_avg_daily_units(store_id: int, cost_profile: StoreCostProfile, days: int = 28) -> float:
    df = read_sql("""
        SELECT AVG(sales) AS avg_sales FROM (
            SELECT sales FROM fact_sales WHERE store_id = :sid AND is_open = true
            ORDER BY date_id DESC LIMIT :d
        ) recent
    """, {"sid": store_id, "d": days})
    avg_sales = df["avg_sales"].iloc[0] if pd.notna(df["avg_sales"].iloc[0]) else 0.0
    return cost_profile.revenue_to_units(avg_sales)


def generate_recommendation(
    store_id: int,
    review_period_days: int = BUSINESS_DEFAULTS["review_period_days"],
    target_service_level: float = BUSINESS_DEFAULTS["target_service_level"],
    warehouse_capacity_units: float | None = None,
    procurement_budget: float | None = None,
    demand_multiplier: float = 1.0,
    lead_time_override: float | None = None,
    promo_uplift_pct: float = 0.0,
    promo_duration_days: int = 0,
    n_simulations: int = 5000,
) -> OptimizationResult:
    """Generate a full inventory recommendation for one store. Supports
    scenario overrides (demand_multiplier, lead_time_override, promo
    uplift) so this same function powers both the baseline recommendation
    and the scenario/what-if engine (Phase 3, scenarios module)."""
    cost_profile = get_store_cost_profile(store_id)
    lead_time_days = lead_time_override if lead_time_override is not None else get_supplier_lead_time(store_id)
    horizon_days = int(round(lead_time_days + review_period_days))

    forecast_df = get_decision_horizon_forecast(store_id, horizon_days)
    if forecast_df.empty:
        raise ValueError(f"No forecast available for store {store_id}")

    daily_p10 = np.array([cost_profile.revenue_to_units(v) for v in forecast_df["p10"]]) * demand_multiplier
    daily_p50 = np.array([cost_profile.revenue_to_units(v) for v in forecast_df["p50"]]) * demand_multiplier
    daily_p90 = np.array([cost_profile.revenue_to_units(v) for v in forecast_df["p90"]]) * demand_multiplier

    if promo_uplift_pct and promo_duration_days > 0:
        n = min(promo_duration_days, len(daily_p50))
        uplift_factor = 1 + promo_uplift_pct / 100
        daily_p10[:n] *= uplift_factor
        daily_p50[:n] *= uplift_factor
        daily_p90[:n] *= uplift_factor

    starting_inventory, incoming_inventory = get_current_inventory(store_id)
    recent_avg_daily_units = get_recent_avg_daily_units(store_id, cost_profile)

    constraints = OptimizationConstraints(
        warehouse_capacity_units=warehouse_capacity_units,
        procurement_budget=procurement_budget,
        target_service_level=target_service_level,
    )

    result = optimize_order_quantity(
        store_id=store_id,
        daily_p10_units=daily_p10, daily_p50_units=daily_p50, daily_p90_units=daily_p90,
        starting_inventory=starting_inventory, incoming_inventory=incoming_inventory,
        cost_profile=cost_profile, constraints=constraints,
        recent_avg_daily_units=recent_avg_daily_units, n_simulations=n_simulations,
    )
    result.drivers["lead_time_days"] = round(lead_time_days, 1)
    result.drivers["review_period_days"] = review_period_days
    return result
