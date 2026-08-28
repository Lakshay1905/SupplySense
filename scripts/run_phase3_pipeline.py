"""
Phase 3 orchestrator.

Runs the full decision-engine pipeline:
  1. Seed operational data (suppliers, initial inventory) -- idempotent.
  2. Generate baseline inventory recommendations for a representative
     sample of stores (running the Monte Carlo optimizer for all 1,115
     stores is feasible compute-wise, but we sample here to keep the
     orchestrator run fast and because Phase 4's UI computes
     recommendations on-demand per store anyway).
  3. Run a portfolio-level budget allocation example for one region.
  4. Run every preset scenario for a handful of example stores and store
     the comparisons.

Usage:
    python -m scripts.run_phase3_pipeline
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from database.connection import get_engine, read_sql, write_dataframe
from optimization.seed_operational_data import run_seeding
from optimization.recommendation_engine import generate_recommendation
from optimization.portfolio_optimizer import optimize_portfolio_allocation
from scenarios.scenario_engine import run_scenario, PRESET_SCENARIOS
from config.settings import REPORTS_DIR

logger = get_logger(__name__)

N_SAMPLE_STORES = 120


def get_sample_stores(n: int = N_SAMPLE_STORES, seed: int = 7) -> list[int]:
    df = read_sql("""
        SELECT DISTINCT f.store_id, f.demand_segment, s.store_type
        FROM fact_sales_features f JOIN dim_store s ON f.store_id = s.store_id
    """)
    rng = np.random.default_rng(seed)
    sampled = []
    groups = df.groupby(["demand_segment", "store_type"])
    per_group = max(1, n // len(groups))
    for _, group in groups:
        k = min(len(group), per_group)
        sampled.extend(rng.choice(group["store_id"].values, size=k, replace=False).tolist())
    return sorted(set(int(s) for s in sampled))


def run_baseline_recommendations(store_ids: list[int]):
    rows = []
    curves = {}
    for i, sid in enumerate(store_ids):
        try:
            result = generate_recommendation(store_id=sid)
        except ValueError as exc:
            logger.warning("Skipping store %d: %s", sid, exc)
            continue
        rows.append({
            "store_id": sid,
            "run_date": pd.Timestamp.today().normalize(),
            "recommended_order_qty": result.recommended_order_qty,
            "expected_cost": result.expected_total_cost,
            "stockout_probability": result.stockout_probability,
            "service_level_target": result.drivers["target_service_level_pct"] / 100,
            "drivers_json": json.dumps(result.drivers),
        })
        curves[sid] = result.full_cost_curve
        if (i + 1) % 30 == 0:
            logger.info("  computed recommendations for %d/%d stores", i + 1, len(store_ids))
    return pd.DataFrame(rows), curves


def run_portfolio_example(curves: dict, store_ids: list[int]) -> dict:
    """Demonstrate shared-budget portfolio allocation across the stores in
    one region (first region found among the sample)."""
    region_df = read_sql("SELECT store_id, region_id FROM dim_store WHERE store_id = ANY(:ids)",
                          {"ids": store_ids})
    top_region = region_df["region_id"].value_counts().idxmax()
    region_store_ids = region_df[region_df["region_id"] == top_region]["store_id"].tolist()
    region_curves = {sid: curves[sid] for sid in region_store_ids if sid in curves}

    if len(region_curves) < 2:
        logger.warning("Not enough stores in a single sampled region for a portfolio demo; skipping")
        return {}

    unconstrained_need = sum(
        c.loc[c["expected_total_cost"].idxmin(), "procurement_cost"] for c in region_curves.values()
    )
    constrained_budget = unconstrained_need * 0.6

    result = optimize_portfolio_allocation(region_curves, total_budget=constrained_budget, total_capacity=None)
    logger.info("Portfolio example: region_id=%s, %d stores, budget=%.0f (60%% of %.0f unconstrained need)",
                top_region, len(region_curves), constrained_budget, unconstrained_need)

    return {
        "region_id": int(top_region),
        "n_stores": len(region_curves),
        "unconstrained_procurement_need": float(unconstrained_need),
        "constrained_budget": float(constrained_budget),
        "status": result.status,
        "total_cost": result.total_cost,
        "total_procurement_spend": result.total_procurement_spend,
        "budget_utilization_pct": result.budget_utilization_pct,
        "allocations": result.allocations.to_dict(orient="records"),
    }


def run_scenario_examples(store_ids: list[int], n_stores: int = 8) -> pd.DataFrame:
    example_stores = store_ids[:n_stores]
    rows = []
    for sid in example_stores:
        for key, scenario in PRESET_SCENARIOS.items():
            try:
                comparison = run_scenario(sid, scenario)
            except ValueError as exc:
                logger.warning("Scenario %s skipped for store %d: %s", key, sid, exc)
                continue
            rows.append({
                "scenario_name": f"{comparison.scenario_name} (store {sid})",
                "parameters_json": json.dumps(scenario.__dict__),
                "result_json": json.dumps({
                    "baseline": comparison.baseline, "scenario": comparison.scenario,
                    "deltas": comparison.deltas,
                }),
            })
    return pd.DataFrame(rows)


def main() -> None:
    logger.info("=== Phase 3: Optimization, Simulation & Decision Engine ===")

    logger.info("Step 1/4: Seeding operational data")
    run_seeding()

    logger.info("Step 2/4: Generating baseline recommendations for %d sample stores", N_SAMPLE_STORES)
    store_ids = get_sample_stores()
    recs, curves = run_baseline_recommendations(store_ids)

    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE optimization_results RESTART IDENTITY")
    write_dataframe(recs, "optimization_results", if_exists="append")

    out_dir = REPORTS_DIR / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_dir / "baseline_recommendations.csv", index=False)

    logger.info("Step 3/4: Portfolio budget-allocation example")
    portfolio_example = run_portfolio_example(curves, store_ids)
    if portfolio_example:
        with open(out_dir / "portfolio_allocation_example.json", "w") as f:
            json.dump(portfolio_example, f, indent=2, default=str)

    logger.info("Step 4/4: Running preset scenarios for example stores")
    scenario_results = run_scenario_examples(store_ids)
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE scenarios RESTART IDENTITY")
    write_dataframe(scenario_results, "scenarios", if_exists="append")
    scenario_results.to_csv(out_dir / "scenario_examples.csv", index=False)

    print("\n=== Phase 3 complete ===")
    print(f"Baseline recommendations computed for {len(recs)} stores")
    print(f"  stores recommending an order (qty > 0): {(recs['recommended_order_qty'] > 0).sum()}")
    print(f"  stores recommending no order:            {(recs['recommended_order_qty'] == 0).sum()}")
    print(f"  avg stockout probability:                {recs['stockout_probability'].mean()*100:.2f}%")
    if portfolio_example:
        print(f"\nPortfolio allocation demo (region {portfolio_example['region_id']}, "
              f"{portfolio_example['n_stores']} stores):")
        print(f"  budget: {portfolio_example['constrained_budget']:.0f} "
              f"(60% of {portfolio_example['unconstrained_procurement_need']:.0f} unconstrained need)")
        print(f"  budget utilization: {portfolio_example['budget_utilization_pct']:.1f}%")
    print(f"\nScenario comparisons stored: {len(scenario_results)}")


if __name__ == "__main__":
    main()
