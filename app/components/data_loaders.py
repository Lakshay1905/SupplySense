"""
Cached data-loading helpers shared across every Streamlit page. All data
comes from the real database or real engines -- nothing here is mocked
or hand-typed. `st.cache_data` keeps the app responsive without
recomputing expensive queries/simulations on every widget interaction.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from database.connection import read_sql, get_engine


@st.cache_data(ttl=300)
def load_store_list() -> pd.DataFrame:
    return read_sql("""
        SELECT s.store_id, s.store_type, s.assortment, r.state_name
        FROM dim_store s LEFT JOIN dim_region r ON s.region_id = r.region_id
        ORDER BY s.store_id
    """)


@st.cache_data(ttl=300)
def load_overview_kpis() -> dict:
    total_stores = read_sql("SELECT COUNT(*) n FROM dim_store")["n"].iloc[0]

    forecast_dates = read_sql("SELECT MIN(target_date) mn, MAX(target_date) mx FROM forecasts")
    model_perf = read_sql("""
        SELECT model_name, AVG(wmape) avg_wmape FROM model_evaluations
        GROUP BY model_name ORDER BY avg_wmape ASC LIMIT 1
    """)
    opt_results = read_sql("""
        SELECT AVG(stockout_probability) avg_risk, SUM(expected_cost) total_cost,
               COUNT(*) n_stores, SUM(CASE WHEN recommended_order_qty > 0 THEN 1 ELSE 0 END) n_ordering
        FROM optimization_results
    """)
    dq = read_sql("""
        SELECT status, COUNT(*) n FROM data_quality_log
        WHERE run_id = (SELECT run_id FROM pipeline_runs ORDER BY started_at DESC LIMIT 1)
        GROUP BY status
    """)

    return {
        "total_stores": int(total_stores),
        "forecast_range": (forecast_dates["mn"].iloc[0], forecast_dates["mx"].iloc[0])
                          if len(forecast_dates) and pd.notna(forecast_dates["mn"].iloc[0]) else (None, None),
        "best_model": model_perf["model_name"].iloc[0] if len(model_perf) else None,
        "best_model_wmape": float(model_perf["avg_wmape"].iloc[0]) if len(model_perf) else None,
        "avg_stockout_risk": float(opt_results["avg_risk"].iloc[0]) if len(opt_results) and pd.notna(opt_results["avg_risk"].iloc[0]) else None,
        "total_expected_cost": float(opt_results["total_cost"].iloc[0]) if len(opt_results) and pd.notna(opt_results["total_cost"].iloc[0]) else None,
        "n_stores_evaluated": int(opt_results["n_stores"].iloc[0]) if len(opt_results) else 0,
        "n_stores_ordering": int(opt_results["n_ordering"].iloc[0]) if len(opt_results) and pd.notna(opt_results["n_ordering"].iloc[0]) else 0,
        "data_quality": {row["status"]: int(row["n"]) for _, row in dq.iterrows()} if len(dq) else {},
    }


@st.cache_data(ttl=300)
def load_store_history(store_id: int, days: int = 180) -> pd.DataFrame:
    return read_sql("""
        SELECT date_id, sales, is_open, is_promo, state_holiday, school_holiday
        FROM fact_sales WHERE store_id = :sid
        ORDER BY date_id DESC LIMIT :d
    """, {"sid": store_id, "d": days}).sort_values("date_id")


@st.cache_data(ttl=300)
def load_store_forecast(store_id: int) -> pd.DataFrame:
    return read_sql("""
        SELECT target_date, p10, p50, p90, model_name FROM forecasts
        WHERE store_id = :sid ORDER BY target_date
    """, {"sid": store_id})


@st.cache_data(ttl=300)
def load_model_comparison() -> pd.DataFrame:
    return read_sql("""
        SELECT model_name, AVG(mae) avg_mae, AVG(rmse) avg_rmse, AVG(mape) avg_mape,
               AVG(wmape) avg_wmape, AVG(bias) avg_bias, COUNT(*) n_evaluations
        FROM model_evaluations GROUP BY model_name ORDER BY avg_wmape ASC
    """)


@st.cache_data(ttl=300)
def load_per_store_best_model() -> pd.DataFrame:
    return read_sql("""
        SELECT store_id, model_name, AVG(wmape) avg_wmape
        FROM model_evaluations GROUP BY store_id, model_name
    """)


@st.cache_data(ttl=300)
def load_optimization_results() -> pd.DataFrame:
    return read_sql("SELECT * FROM optimization_results ORDER BY stockout_probability DESC")


@st.cache_data(ttl=300)
def load_data_quality_log() -> pd.DataFrame:
    return read_sql("""
        SELECT * FROM data_quality_log
        WHERE run_id = (SELECT run_id FROM pipeline_runs ORDER BY started_at DESC LIMIT 1)
        ORDER BY stage, check_name
    """)


@st.cache_data(ttl=300)
def load_pipeline_runs() -> pd.DataFrame:
    return read_sql("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 10")


@st.cache_data(ttl=300)
def load_scenario_examples() -> pd.DataFrame:
    df = read_sql("SELECT * FROM scenarios ORDER BY scenario_id")
    if len(df):
        df["result_parsed"] = df["result_json"].apply(lambda x: json.loads(x) if x else {})
    return df


def clear_all_caches() -> None:
    st.cache_data.clear()
