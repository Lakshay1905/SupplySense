"""
Grounded tool functions for the AI Analytics Copilot.

Every function here either (a) queries real data already computed and
stored by Phases 1-3, or (b) invokes the actual forecasting/optimization/
scenario engines live. The copilot is never allowed to answer from
"what it thinks a plausible metric might be" -- these functions are the
only source of numbers it can cite.
"""
from __future__ import annotations

import pandas as pd

from database.connection import read_sql
from optimization.recommendation_engine import generate_recommendation
from scenarios.scenario_engine import run_scenario, PRESET_SCENARIOS, ScenarioDefinition
from forecasting.evaluation.model_selection import summarize_model_performance


def _df_to_records(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    return df.head(max_rows).to_dict(orient="records")


def get_store_forecast(store_id: int, days: int = 14) -> dict:
    """Real stored P10/P50/P90 forecast for a store's next N days."""
    df = read_sql(
        "SELECT target_date, p10, p50, p90, model_name FROM forecasts "
        "WHERE store_id = :sid ORDER BY target_date LIMIT :d",
        {"sid": store_id, "d": days},
    )
    if df.empty:
        return {"error": f"No forecast found for store {store_id}"}
    return {"store_id": store_id, "forecast_days": _df_to_records(df, days)}


def get_store_history(store_id: int, days: int = 30) -> dict:
    """Real recent historical sales for a store."""
    df = read_sql("""
        SELECT date_id, sales, is_open, is_promo, state_holiday FROM fact_sales
        WHERE store_id = :sid ORDER BY date_id DESC LIMIT :d
    """, {"sid": store_id, "d": days})
    if df.empty:
        return {"error": f"No sales history found for store {store_id}"}
    return {"store_id": store_id, "recent_history": _df_to_records(df.sort_values("date_id"), days)}


def get_store_recommendation(store_id: int, procurement_budget: float | None = None,
                              warehouse_capacity_units: float | None = None,
                              target_service_level: float = 0.95) -> dict:
    """Live-computed inventory recommendation (runs the real optimizer, not stored data)."""
    try:
        result = generate_recommendation(
            store_id=store_id, procurement_budget=procurement_budget,
            warehouse_capacity_units=warehouse_capacity_units,
            target_service_level=target_service_level,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "store_id": store_id,
        "recommended_order_qty": round(result.recommended_order_qty, 1),
        "expected_total_cost": round(result.expected_total_cost, 2),
        "expected_holding_cost": round(result.expected_holding_cost, 2),
        "expected_stockout_cost": round(result.expected_stockout_cost, 2),
        "procurement_cost": round(result.procurement_cost, 2),
        "stockout_probability_pct": round(result.stockout_probability * 100, 2),
        "achieved_service_level_pct": round(result.achieved_service_level * 100, 2),
        "within_target_service_level": bool(result.within_target_service_level),
        "drivers": result.drivers,
    }


def run_what_if_scenario(store_id: int, scenario_preset: str | None = None,
                          demand_multiplier: float = 1.0, lead_time_override: float | None = None,
                          budget_multiplier: float | None = None, capacity_multiplier: float | None = None,
                          promo_uplift_pct: float = 0.0, promo_duration_days: int = 0) -> dict:
    """Live-computed baseline-vs-scenario comparison. Accepts either a
    named preset (e.g. 'demand_up_20') or custom parameters."""
    if scenario_preset:
        if scenario_preset not in PRESET_SCENARIOS:
            return {"error": f"Unknown preset '{scenario_preset}'. Available: {list(PRESET_SCENARIOS.keys())}"}
        scenario = PRESET_SCENARIOS[scenario_preset]
    else:
        scenario = ScenarioDefinition(
            name="Custom scenario", demand_multiplier=demand_multiplier,
            lead_time_override=lead_time_override, budget_multiplier=budget_multiplier,
            capacity_multiplier=capacity_multiplier, promo_uplift_pct=promo_uplift_pct,
            promo_duration_days=promo_duration_days,
        )
    try:
        comparison = run_scenario(store_id, scenario)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "scenario_name": comparison.scenario_name, "store_id": store_id,
        "baseline": comparison.baseline, "scenario": comparison.scenario, "deltas": comparison.deltas,
    }


def get_model_performance_summary() -> dict:
    """Real benchmark results: why a given model was selected."""
    df = read_sql("SELECT * FROM model_evaluations")
    if df.empty:
        return {"error": "No model evaluation data available. Run scripts.run_phase2_benchmark first."}
    summary = summarize_model_performance(df)
    return {"model_comparison": _df_to_records(summary, 20)}


def get_stockout_risk_ranking(limit: int = 10, ascending: bool = False) -> dict:
    """Real stored optimization results ranked by stockout probability."""
    order = "ASC" if ascending else "DESC"
    df = read_sql(f"""
        SELECT store_id, recommended_order_qty, stockout_probability, expected_cost
        FROM optimization_results ORDER BY stockout_probability {order} LIMIT :lim
    """, {"lim": limit})
    if df.empty:
        return {"error": "No optimization results available. Run scripts.run_phase3_pipeline first."}
    return {"ranking": _df_to_records(df, limit)}


def get_data_quality_summary() -> dict:
    """Real logged data-quality check results from the most recent pipeline run."""
    df = read_sql("""
        SELECT run_id, stage, status, COUNT(*) AS n_checks, SUM(records_failed) AS total_failed
        FROM data_quality_log
        WHERE run_id = (SELECT run_id FROM pipeline_runs ORDER BY started_at DESC LIMIT 1)
        GROUP BY run_id, stage, status ORDER BY stage, status
    """)
    if df.empty:
        return {"error": "No data quality log entries found."}
    return {"data_quality_summary": _df_to_records(df, 100)}


def get_top_products_by_metric(metric: str = "stockout_probability", limit: int = 10) -> dict:
    """Generic store ranking by a metric present in optimization_results."""
    allowed_metrics = {"stockout_probability", "expected_cost", "recommended_order_qty"}
    if metric not in allowed_metrics:
        return {"error": f"metric must be one of {allowed_metrics}"}
    df = read_sql(f"""
        SELECT store_id, recommended_order_qty, stockout_probability, expected_cost
        FROM optimization_results ORDER BY {metric} DESC LIMIT :lim
    """, {"lim": limit})
    return {"ranking": _df_to_records(df, limit)}
