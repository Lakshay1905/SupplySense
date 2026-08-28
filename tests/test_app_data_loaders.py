"""
Tests for app.components.data_loaders. These test the underlying data
logic (the SQL/shape correctness), not Streamlit rendering -- calling
`@st.cache_data`-decorated functions works fine outside a real Streamlit
session (it just skips caching), so we can test them directly.
"""
from __future__ import annotations

import pytest

from database.connection import get_engine, table_row_count
from app.components.data_loaders import (
    load_store_list, load_overview_kpis, load_store_history, load_store_forecast,
    load_model_comparison, load_optimization_results, load_data_quality_log, load_pipeline_runs,
)


def _db_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return table_row_count("forecasts") > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Phase 1-3 data not fully populated")


def test_load_store_list_has_expected_columns():
    df = load_store_list()
    assert {"store_id", "store_type", "assortment", "state_name"}.issubset(df.columns)
    assert len(df) > 0


def test_load_overview_kpis_returns_expected_keys():
    kpis = load_overview_kpis()
    for key in ["total_stores", "forecast_range", "best_model", "avg_stockout_risk",
                "total_expected_cost", "n_stores_evaluated", "n_stores_ordering", "data_quality"]:
        assert key in kpis
    assert kpis["total_stores"] == 1115


def test_load_overview_kpis_best_model_is_xgboost():
    kpis = load_overview_kpis()
    assert kpis["best_model"] == "xgboost"


def test_load_store_history_returns_sorted_ascending():
    df = load_store_history(1, days=10)
    dates = df["date_id"].tolist()
    assert dates == sorted(dates)


def test_load_store_forecast_probabilistic_bands_ordered():
    df = load_store_forecast(1)
    assert (df["p10"] <= df["p50"]).all()
    assert (df["p50"] <= df["p90"]).all()


def test_load_model_comparison_sorted_by_wmape_ascending():
    df = load_model_comparison()
    wmapes = df["avg_wmape"].tolist()
    assert wmapes == sorted(wmapes)


def test_load_optimization_results_sorted_by_stockout_descending():
    df = load_optimization_results()
    risks = df["stockout_probability"].tolist()
    assert risks == sorted(risks, reverse=True)


def test_load_data_quality_log_only_latest_run():
    df = load_data_quality_log()
    if len(df):
        assert df["run_id"].nunique() == 1


def test_load_pipeline_runs_returns_at_most_10():
    df = load_pipeline_runs()
    assert len(df) <= 10
