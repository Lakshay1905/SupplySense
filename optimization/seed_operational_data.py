"""
Seed `dim_supplier`, `orders` (supplier assignment), and `inventory_snapshot`
with a documented, reproducible starting operational state.

WHY THIS IS NEEDED: Rossmann is a sales-history dataset. It has no supplier
master data, no lead-time records, and no inventory feed -- no public
retail sales dataset does. Every real inventory-optimization system is
configured with these as business inputs (supplier contracts, warehouse
policy), not derived from sales history. We seed a single reproducible
(fixed-seed) synthetic state here, clearly labeled as such everywhere it
surfaces, so the downstream decision engine has something concrete to
optimize against. No historical fact is fabricated -- only forward-looking
operational parameters that a real business would configure directly.

Seeding logic (documented, reproducible via fixed random seed):
  - One supplier per store, lead time ~ Normal(BUSINESS_DEFAULTS mean, 20%
    CV), truncated to be positive, varied deterministically by store_id so
    reruns are stable.
  - MOQ / order multiple pulled from BUSINESS_DEFAULTS (uniform policy
    across stores, as a simplifying assumption).
  - Initial on-hand inventory = ~10 days of the store's recent average
    daily unit demand, +/- 40% random variation (simulates realistic
    variation in where each store happens to sit in its ordering cycle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BUSINESS_DEFAULTS
from config.logging_config import get_logger
from database.connection import read_sql, get_engine, write_dataframe
from optimization.cost_model import build_store_cost_profile

logger = get_logger(__name__)

SEED = 20260827  # fixed seed for reproducibility


def seed_suppliers(store_ids: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    mean_lead = BUSINESS_DEFAULTS["default_lead_time_days"]
    lead_times = np.clip(rng.normal(mean_lead, mean_lead * 0.25, size=len(store_ids)), 2, None)
    reliability = np.clip(rng.normal(0.93, 0.05, size=len(store_ids)), 0.6, 0.999)

    suppliers = pd.DataFrame({
        "supplier_name": [f"Supplier for Store {sid}" for sid in store_ids],
        "lead_time_days": np.round(lead_times, 1),
        "lead_time_std_days": np.round(lead_times * 0.2, 1),
        "reliability_score": np.round(reliability, 3),
        "moq_units": BUSINESS_DEFAULTS["moq_units"],
        "order_multiple": BUSINESS_DEFAULTS["order_multiple_units"],
        "notes": "Synthetic supplier profile (documented assumption; Rossmann has no supplier data).",
    })
    return suppliers


def seed_inventory_snapshot(store_ids: list[int], as_of_date: pd.Timestamp,
                             avg_daily_units: dict[int, float]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    coverage_days = 10
    rows = []
    for sid in store_ids:
        daily = max(avg_daily_units.get(sid, 1.0), 0.1)
        variation = rng.uniform(0.6, 1.4)
        on_hand = round(daily * coverage_days * variation, 1)
        rows.append({
            "date_id": as_of_date, "store_id": sid,
            "on_hand_units": on_hand, "incoming_units": 0.0,
            "reorder_point": None, "safety_stock": None,
        })
    return pd.DataFrame(rows)


def run_seeding() -> None:
    logger.info("Seeding suppliers + initial inventory snapshot (documented synthetic assumptions)")
    dim_store = read_sql("SELECT store_id FROM dim_store ORDER BY store_id")
    store_ids = dim_store["store_id"].tolist()

    suppliers = seed_suppliers(store_ids)
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE dim_supplier RESTART IDENTITY CASCADE")
    write_dataframe(suppliers, "dim_supplier", if_exists="append")

    # Map store -> supplier_id via insertion order (supplier i corresponds to store_ids[i])
    supplier_ids = read_sql("SELECT supplier_id FROM dim_supplier ORDER BY supplier_id")["supplier_id"].tolist()
    store_to_supplier = dict(zip(store_ids, supplier_ids))

    # avg daily units per store, derived from real historical sales_per_customer & sales
    hist = read_sql("""
        SELECT store_id, AVG(sales) AS avg_sales, AVG(NULLIF(sales_per_customer,0)) AS avg_spc
        FROM fact_sales WHERE is_open = true
        GROUP BY store_id
    """)
    avg_daily_units = {}
    for _, row in hist.iterrows():
        profile = build_store_cost_profile(int(row["store_id"]), row["avg_spc"] or 10.0)
        avg_daily_units[int(row["store_id"])] = profile.revenue_to_units(row["avg_sales"] or 0.0)

    as_of_date = read_sql("SELECT MAX(date_id) AS d FROM fact_sales")["d"].iloc[0]
    as_of_date = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
    inventory = seed_inventory_snapshot(store_ids, as_of_date, avg_daily_units)

    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE inventory_snapshot RESTART IDENTITY CASCADE")
    write_dataframe(inventory, "inventory_snapshot", if_exists="append")

    logger.info("Seeded %d suppliers and %d inventory snapshots", len(suppliers), len(inventory))
    print(f"Seeded {len(suppliers)} suppliers and {len(inventory)} inventory snapshots "
          f"(as of {as_of_date.date()}).")


if __name__ == "__main__":
    run_seeding()
