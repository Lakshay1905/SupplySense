"""
End-to-end integration test simulating the platform's full intended user
journey (see README "Final Success Criteria"):

    view inventory health -> select a store -> inspect historical demand
    -> view probabilistic forecast -> inspect model performance
    -> view recommended order -> see stockout/cost implications
    -> modify an assumption -> run a scenario -> compare baseline vs scenario
    -> inspect optimized portfolio decision -> ask the AI copilot "why"
       (tool-dispatch layer only, mocked LLM -- no live API call)

This test exercises the real database and real engines end-to-end in a
single coherent flow, rather than each phase's modules in isolation.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from database.connection import get_engine, table_row_count, read_sql
from app.components.data_loaders import (
    load_overview_kpis, load_store_history, load_store_forecast, load_model_comparison,
)
from optimization.recommendation_engine import generate_recommendation
from optimization.portfolio_optimizer import optimize_portfolio_allocation
from scenarios.scenario_engine import run_scenario, PRESET_SCENARIOS
from ai import copilot as ai_copilot


def _full_stack_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return all(table_row_count(t) > 0 for t in
                   ["fact_sales", "forecasts", "model_evaluations", "optimization_results",
                    "dim_supplier", "inventory_snapshot"])
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _full_stack_ready(), reason="Full Phase 1-3 data not populated")

STORE_ID = 1


def test_full_user_journey_end_to_end():
    # 1. View overall inventory health (Overview dashboard KPIs)
    kpis = load_overview_kpis()
    assert kpis["total_stores"] > 0
    assert kpis["best_model"] is not None

    # 2. Select a store, inspect historical demand
    history = load_store_history(STORE_ID, days=60)
    assert len(history) > 0
    assert "sales" in history.columns

    # 3. View probabilistic forecast
    forecast = load_store_forecast(STORE_ID)
    assert len(forecast) > 0
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()

    # 4. Inspect model performance
    model_comparison = load_model_comparison()
    assert len(model_comparison) > 0
    best_model_name = model_comparison.iloc[0]["model_name"]
    assert best_model_name == kpis["best_model"]

    # 5. View recommended order quantity + stockout/cost implications
    baseline_recommendation = generate_recommendation(store_id=STORE_ID)
    assert baseline_recommendation.recommended_order_qty >= 0
    assert 0 <= baseline_recommendation.stockout_probability <= 1
    assert baseline_recommendation.expected_total_cost >= 0

    # 6. Modify an assumption and run a scenario; compare baseline vs scenario
    comparison = run_scenario(STORE_ID, PRESET_SCENARIOS["demand_up_20"])
    assert comparison.baseline["recommended_order_qty"] == pytest.approx(
        baseline_recommendation.recommended_order_qty, rel=0.01
    )
    assert "recommended_order_qty" in comparison.deltas

    # 7. Inspect an optimized portfolio decision across multiple stores
    region_df = read_sql("SELECT store_id FROM dim_store LIMIT 5")
    store_ids = region_df["store_id"].tolist()
    curves = {}
    for sid in store_ids:
        try:
            result = generate_recommendation(store_id=sid)
            curves[sid] = result.full_cost_curve
        except ValueError:
            continue
    assert len(curves) >= 2
    total_need = sum(c.loc[c["expected_total_cost"].idxmin(), "procurement_cost"] for c in curves.values())
    allocation = optimize_portfolio_allocation(curves, total_budget=total_need * 0.5, total_capacity=None)
    assert allocation.status == "Optimal"
    assert len(allocation.allocations) == len(curves)

    # 8. Ask the AI copilot "why" -- verify the tool-dispatch layer resolves
    #    to real, grounded data (LLM call itself is mocked; we're testing
    #    that the tool the copilot would call returns real backing data)
    tool_result = ai_copilot._dispatch_tool_call(
        "get_store_recommendation", {"store_id": STORE_ID}
    )
    assert "error" not in tool_result
    assert tool_result["recommended_order_qty"] == pytest.approx(
        baseline_recommendation.recommended_order_qty, rel=0.01
    )
    assert "drivers" in tool_result


def test_full_user_journey_scenario_reflects_in_copilot_tool():
    """The copilot's run_what_if_scenario tool must return results
    consistent with directly calling the scenario engine -- i.e. the UI,
    the engine, and the copilot are all reading from the same source of
    truth, not three different code paths that could silently diverge."""
    direct_comparison = run_scenario(STORE_ID, PRESET_SCENARIOS["budget_cut_15"])
    tool_result = ai_copilot._dispatch_tool_call(
        "run_what_if_scenario", {"store_id": STORE_ID, "scenario_preset": "budget_cut_15"}
    )
    assert "error" not in tool_result
    assert tool_result["scenario"]["recommended_order_qty"] == pytest.approx(
        direct_comparison.scenario["recommended_order_qty"], rel=0.01
    )
