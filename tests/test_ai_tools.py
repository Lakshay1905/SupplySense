"""
Integration tests for ai.tools -- verifying every grounded tool function
returns real, correctly-shaped data (or a clear error) from the live
database, never a fabricated placeholder.
"""
from __future__ import annotations

import pytest

from database.connection import get_engine, table_row_count
from ai import tools


def _db_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return table_row_count("forecasts") > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Phase 1-3 data not fully populated")

TEST_STORE_ID = 1


def test_get_store_forecast_returns_real_rows():
    result = tools.get_store_forecast(TEST_STORE_ID, days=5)
    assert "error" not in result
    assert len(result["forecast_days"]) == 5
    for day in result["forecast_days"]:
        assert day["p10"] <= day["p50"] <= day["p90"]


def test_get_store_forecast_unknown_store_returns_error():
    result = tools.get_store_forecast(store_id=999999, days=5)
    assert "error" in result


def test_get_store_history_returns_real_rows():
    result = tools.get_store_history(TEST_STORE_ID, days=10)
    assert "error" not in result
    assert len(result["recent_history"]) == 10


def test_get_store_recommendation_returns_grounded_numbers():
    result = tools.get_store_recommendation(TEST_STORE_ID)
    assert "error" not in result
    assert result["recommended_order_qty"] >= 0
    assert 0 <= result["stockout_probability_pct"] <= 100
    assert "drivers" in result


def test_run_what_if_scenario_with_preset():
    result = tools.run_what_if_scenario(TEST_STORE_ID, scenario_preset="demand_up_20")
    assert "error" not in result
    assert "baseline" in result and "scenario" in result and "deltas" in result


def test_run_what_if_scenario_unknown_preset_returns_error():
    result = tools.run_what_if_scenario(TEST_STORE_ID, scenario_preset="not_a_real_preset")
    assert "error" in result


def test_get_model_performance_summary_includes_xgboost():
    result = tools.get_model_performance_summary()
    assert "error" not in result
    model_names = {row["model_name"] for row in result["model_comparison"]}
    assert "xgboost" in model_names


def test_get_stockout_risk_ranking_sorted_descending_by_default():
    result = tools.get_stockout_risk_ranking(limit=10)
    assert "error" not in result
    risks = [row["stockout_probability"] for row in result["ranking"]]
    assert risks == sorted(risks, reverse=True)


def test_get_stockout_risk_ranking_ascending():
    result = tools.get_stockout_risk_ranking(limit=10, ascending=True)
    assert "error" not in result
    risks = [row["stockout_probability"] for row in result["ranking"]]
    assert risks == sorted(risks)


def test_get_data_quality_summary_returns_real_checks():
    result = tools.get_data_quality_summary()
    assert "error" not in result
    assert len(result["data_quality_summary"]) > 0


def test_get_top_products_by_metric_rejects_unknown_metric():
    result = tools.get_top_products_by_metric(metric="not_a_real_metric")
    assert "error" in result


def test_get_top_products_by_metric_valid_metric():
    result = tools.get_top_products_by_metric(metric="stockout_probability", limit=5)
    assert "ranking" in result
