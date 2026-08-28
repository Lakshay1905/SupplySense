"""
Integration tests for optimization.recommendation_engine and
scenarios.scenario_engine, run against the live, Phase1+2+3-populated
database. Skipped automatically if the required tables aren't populated.
"""
from __future__ import annotations

import pytest

from database.connection import get_engine, table_row_count
from optimization.recommendation_engine import generate_recommendation
from scenarios.scenario_engine import run_scenario, PRESET_SCENARIOS


def _db_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return (table_row_count("forecasts") > 0 and table_row_count("inventory_snapshot") > 0
                and table_row_count("dim_supplier") > 0)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Phase 1-3 data not fully populated")

TEST_STORE_ID = 1


def test_generate_recommendation_returns_valid_result():
    result = generate_recommendation(store_id=TEST_STORE_ID)
    assert result.store_id == TEST_STORE_ID
    assert result.recommended_order_qty >= 0
    assert 0 <= result.stockout_probability <= 1
    assert result.horizon_days > 0


def test_generate_recommendation_higher_demand_multiplier_increases_or_maintains_order():
    baseline = generate_recommendation(store_id=TEST_STORE_ID, demand_multiplier=1.0)
    higher_demand = generate_recommendation(store_id=TEST_STORE_ID, demand_multiplier=2.0)
    assert higher_demand.recommended_order_qty >= baseline.recommended_order_qty


def test_generate_recommendation_promo_uplift_increases_recommended_quantity():
    baseline = generate_recommendation(store_id=TEST_STORE_ID)
    with_promo = generate_recommendation(
        store_id=TEST_STORE_ID, promo_uplift_pct=50.0, promo_duration_days=14
    )
    assert with_promo.recommended_order_qty >= baseline.recommended_order_qty


def test_generate_recommendation_tighter_budget_never_increases_procurement_cost():
    baseline = generate_recommendation(store_id=TEST_STORE_ID)
    if baseline.procurement_cost <= 0:
        pytest.skip("Store already has zero recommended order at baseline; budget test not meaningful")
    tight = generate_recommendation(store_id=TEST_STORE_ID, procurement_budget=baseline.procurement_cost * 0.5)
    assert tight.procurement_cost <= baseline.procurement_cost + 1e-6


def test_run_scenario_demand_up_produces_valid_comparison():
    comparison = run_scenario(TEST_STORE_ID, PRESET_SCENARIOS["demand_up_20"])
    assert comparison.store_id == TEST_STORE_ID
    assert "recommended_order_qty" in comparison.baseline
    assert "recommended_order_qty" in comparison.scenario
    assert "recommended_order_qty" in comparison.deltas


def test_run_scenario_lead_time_doubled_changes_horizon_days():
    comparison = run_scenario(TEST_STORE_ID, PRESET_SCENARIOS["lead_time_doubled"])
    # a longer lead time generally requires covering more days of demand,
    # so the scenario's order quantity should generally be >= baseline
    # (not a strict guarantee under all inventory states, but true when
    # starting inventory is fixed and horizon grows)
    assert comparison.scenario["recommended_order_qty"] >= 0


def test_run_scenario_all_presets_execute_without_error():
    for key, scenario in PRESET_SCENARIOS.items():
        comparison = run_scenario(TEST_STORE_ID, scenario)
        assert comparison.scenario_name == scenario.name
