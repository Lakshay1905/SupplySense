"""
Integration tests against the live PostgreSQL database populated by
scripts/run_phase1_pipeline.py.

These are skipped automatically if the database is unreachable or has not
yet been populated, so the suite still runs in environments without a
live Postgres (e.g. a bare CI checkout before `docker-compose up`).
"""
from __future__ import annotations

import pytest
import pandas as pd

from database.connection import get_engine, read_sql, table_row_count


def _db_available() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return True
    except Exception:
        return False


def _data_loaded() -> bool:
    try:
        return table_row_count("fact_sales") > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


def test_dim_store_row_count_matches_expected():
    n = table_row_count("dim_store")
    assert n == 1115


def test_dim_region_has_no_duplicate_state_codes():
    df = read_sql("SELECT state_code, COUNT(*) AS n FROM dim_region GROUP BY state_code HAVING COUNT(*) > 1")
    assert len(df) == 0


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_fact_sales_has_no_orphan_store_ids():
    df = read_sql("""
        SELECT COUNT(*) AS n FROM fact_sales fs
        LEFT JOIN dim_store ds ON fs.store_id = ds.store_id
        WHERE ds.store_id IS NULL
    """)
    assert int(df["n"].iloc[0]) == 0


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_fact_sales_no_orphan_dates():
    df = read_sql("""
        SELECT COUNT(*) AS n FROM fact_sales fs
        LEFT JOIN dim_date dd ON fs.date_id = dd.date_id
        WHERE dd.date_id IS NULL
    """)
    assert int(df["n"].iloc[0]) == 0


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_fact_sales_no_negative_values():
    df = read_sql("SELECT COUNT(*) AS n FROM fact_sales WHERE sales < 0 OR customers < 0")
    assert int(df["n"].iloc[0]) == 0


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_fact_sales_features_row_count_matches_fact_sales():
    n_fact = table_row_count("fact_sales")
    n_feat = table_row_count("fact_sales_features")
    assert n_fact == n_feat


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_no_duplicate_store_date_pairs_in_fact_sales():
    df = read_sql("""
        SELECT store_id, date_id, COUNT(*) AS n
        FROM fact_sales GROUP BY store_id, date_id HAVING COUNT(*) > 1
    """)
    assert len(df) == 0


@pytest.mark.skipif(not _data_loaded(), reason="fact_sales not yet populated")
def test_data_quality_log_has_no_unexpected_failures():
    df = read_sql("SELECT * FROM data_quality_log WHERE status = 'fail'")
    assert len(df) == 0


def _forecasts_loaded() -> bool:
    try:
        return table_row_count("forecasts") > 0
    except Exception:
        return False


def _evaluations_loaded() -> bool:
    try:
        return table_row_count("model_evaluations") > 0
    except Exception:
        return False


@pytest.mark.skipif(not _evaluations_loaded(), reason="model_evaluations not yet populated")
def test_model_evaluations_has_no_null_metrics():
    df = read_sql("SELECT * FROM model_evaluations WHERE wmape IS NULL OR mae IS NULL")
    assert len(df) == 0


@pytest.mark.skipif(not _evaluations_loaded(), reason="model_evaluations not yet populated")
def test_model_evaluations_contains_multiple_model_families():
    df = read_sql("SELECT DISTINCT model_name FROM model_evaluations")
    models = set(df["model_name"])
    # at least one baseline, one statistical, one ML model present
    assert {"naive", "seasonal_naive"} & models
    assert {"holt_winters", "sarima"} & models
    assert {"xgboost", "random_forest"} & models


@pytest.mark.skipif(not _forecasts_loaded(), reason="forecasts not yet populated")
def test_forecasts_probabilistic_bands_are_ordered():
    df = read_sql("SELECT * FROM forecasts WHERE p10 > p50 OR p50 > p90")
    assert len(df) == 0


@pytest.mark.skipif(not _forecasts_loaded(), reason="forecasts not yet populated")
def test_forecasts_no_negative_values():
    df = read_sql("SELECT * FROM forecasts WHERE p10 < 0 OR p50 < 0 OR p90 < 0")
    assert len(df) == 0


@pytest.mark.skipif(not _forecasts_loaded(), reason="forecasts not yet populated")
def test_forecasts_cover_all_stores_with_full_horizon():
    df = read_sql("SELECT store_id, COUNT(*) AS n FROM forecasts GROUP BY store_id")
    # every forecasted store should have exactly one row per horizon day
    assert df["n"].nunique() == 1  # all stores have the same number of forecast days
    assert df["n"].iloc[0] > 0


def _suppliers_loaded() -> bool:
    try:
        return table_row_count("dim_supplier") > 0
    except Exception:
        return False


def _optimization_results_loaded() -> bool:
    try:
        return table_row_count("optimization_results") > 0
    except Exception:
        return False


@pytest.mark.skipif(not _suppliers_loaded(), reason="dim_supplier not yet populated")
def test_dim_supplier_has_positive_lead_times():
    df = read_sql("SELECT * FROM dim_supplier WHERE lead_time_days <= 0")
    assert len(df) == 0


@pytest.mark.skipif(not _suppliers_loaded(), reason="dim_supplier not yet populated")
def test_dim_supplier_one_per_store():
    n_suppliers = table_row_count("dim_supplier")
    n_stores = table_row_count("dim_store")
    assert n_suppliers == n_stores


@pytest.mark.skipif(not _optimization_results_loaded(), reason="optimization_results not yet populated")
def test_optimization_results_have_no_negative_order_quantities():
    df = read_sql("SELECT * FROM optimization_results WHERE recommended_order_qty < 0")
    assert len(df) == 0


@pytest.mark.skipif(not _optimization_results_loaded(), reason="optimization_results not yet populated")
def test_optimization_results_stockout_probability_in_valid_range():
    df = read_sql("SELECT * FROM optimization_results WHERE stockout_probability < 0 OR stockout_probability > 1")
    assert len(df) == 0


@pytest.mark.skipif(not _optimization_results_loaded(), reason="optimization_results not yet populated")
def test_optimization_results_have_drivers_json():
    df = read_sql("SELECT * FROM optimization_results WHERE drivers_json IS NULL")
    assert len(df) == 0
